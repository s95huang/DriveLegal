import asyncio
import argparse
import json
import os
import pathlib
import sys
import numpy as np
import re
import math
from typing import Dict, List, Any, Optional, Tuple
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

# Ensure package imports work when running as a script
if __package__ in (None, ""):
    project_root = str(pathlib.Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.append(project_root)

from Evaluation.metrics import compute_context_relevance, compute_evidence_recall, compute_context_relevance_v2
from langchain_ollama import OllamaEmbeddings
from Evaluation.llm import OllamaClient,OllamaWrapper

SEED = 42


def get_openai_api_key() -> Optional[str]:
    """
    If api.txt exists in CWD, read its single-line API key and set OPENAI_API_KEY.
    Returns the key if set, else whatever is currently in env.
    """
    try:
        with open("api.txt", "r", encoding="utf-8") as f:
            key = f.read().strip()
            if key:
                os.environ["LLM_API_KEY"] = key  # in-process env set
                print(f"Loaded API key from api.txt")
                return key
    except FileNotFoundError:
        pass
    return os.getenv("LLM_API_KEY")  # return existing if already set

async def evaluate_dataset(
    dataset: Dataset,
    llm: Any,
    embeddings: Embeddings,
    max_concurrent: int = 1,
    detailed_output: bool = False,
    relevance_version: str = "v1"
) -> Dict[str, Any]:
    """Evaluate context relevance and recall for a dataset"""
    results = {
        "context_relevancy": [],
        "evidence_recall": []
    }
    detailed_results = [] if detailed_output else None
    
    ids = dataset["id"]
    questions = dataset["question"]
    contexts_list = dataset["contexts"]
    evidences = dataset["evidences"]

    total_samples = len(questions)
    print(f"\nStarting evaluation of {total_samples} samples...")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def evaluate_with_semaphore(i):
        async with semaphore:
            sample_metrics = await evaluate_sample(
                sample_id=str(ids[i]),
                question=questions[i],
                contexts=contexts_list[i],
                evidences=evidences[i],
                llm=llm,
                embeddings=embeddings,
                relevance_version=relevance_version
            )
            if detailed_output:
                return {
                    "id": ids[i],
                    "question": questions[i],
                    "contexts": contexts_list[i],
                    "evidences": evidences[i],
                    "metrics": sample_metrics
                }
            return sample_metrics

    tasks = [evaluate_with_semaphore(i) for i in range(total_samples)]
    sample_results = []
    completed = 0
    
    for future in asyncio.as_completed(tasks):
        try:
            result = await future
            if detailed_output and detailed_results is not None:
                detailed_results.append(result)
                # metrics aggregation (guard types for linters)
                if isinstance(result, dict):
                    metrics_dict = result.get("metrics")
                    if isinstance(metrics_dict, dict):
                        for metric, score in metrics_dict.items():
                            if isinstance(score, (int, float)) and not np.isnan(score):
                                results[metric].append(score)
            else:
                sample_results.append(result)
                if isinstance(result, dict):
                    for metric, score in result.items():
                        if isinstance(score, (int, float)) and not np.isnan(score):
                            results[metric].append(score)
            completed += 1
            print(f"✅ Completed sample {completed}/{total_samples} - {(completed/total_samples)*100:.1f}%")
        except Exception as e:
            print(f"❌ Sample failed: {e}")
            completed += 1
    
    avg_results = {
        "context_relevancy": np.nanmean(results["context_relevancy"]),
        "evidence_recall": np.nanmean(results["evidence_recall"])
    }
    
    if detailed_output:
        return {
            "average_scores": avg_results,
            "detailed": detailed_results
        }
    else:
        return avg_results


async def evaluate_sample(
    sample_id: str,
    question: str,
    contexts: List[str],
    evidences: List[str],
    llm: Any,
    embeddings: Embeddings,
    relevance_version: str = "v1"
) -> Dict[str, float]:
    """Evaluate retrieval metrics for a single sample"""

    def _ensure_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return ["" if v is None else str(v) for v in value]
        if value is None:
            return []
        return [str(value)]

    def _normalize_contexts(value: List[str]) -> List[str]:
        normalized: List[str] = []
        for item in value:
            if not item:
                continue
            text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized

    def _normalize_evidences(value: List[str]) -> List[str]:
        normalized: List[str] = []
        for raw in value:
            if not raw:
                continue
            text = str(raw).strip()
            if not text:
                continue
            parts = re.split(r"\s*\|\|\s*", text)
            for part in parts:
                cleaned = part.strip()
                if cleaned.startswith('"') and cleaned.endswith('"'):
                    cleaned = cleaned[1:-1].strip()
                if cleaned:
                    normalized.append(cleaned)
        return normalized

    norm_contexts = _normalize_contexts(_ensure_list(contexts))
    norm_evidences = _normalize_evidences(_ensure_list(evidences))

    # Evaluate both metrics in parallel
    if relevance_version == "v2":
        relevance_task = compute_context_relevance_v2(question, norm_contexts, norm_evidences, llm)
    else:
        relevance_task = compute_context_relevance(question, norm_contexts, llm)
    recall_task = compute_evidence_recall(question, norm_contexts, norm_evidences, llm)
    
    # Wait for both tasks to complete
    relevance_score, recall_score = await asyncio.gather(relevance_task, recall_task)

    if isinstance(relevance_score, float) and math.isnan(relevance_score):
        print(f"[WARN] Context relevance unavailable for sample '{sample_id}'; defaulting to 0.")
        relevance_score = 0.0
    if isinstance(recall_score, float) and math.isnan(recall_score):
        print(f"[WARN] Evidence recall unavailable for sample '{sample_id}'; defaulting to 0.")
        recall_score = 0.0

    print(f"Relevance Score: {relevance_score}, Recall Score: {recall_score}")

    return {
        "context_relevancy": relevance_score,
        "evidence_recall": recall_score
    }

async def evaluate_prediction_file(
    data_file: str,
    output_file: Optional[str],
    llm: Any,
    embedding: Embeddings,
    num_samples: Optional[int],
    detailed_output: bool,
    max_concurrent: int,
    relevance_version: str
) -> Dict[str, Any]:
    """Load a prediction file, group by question type, and run retrieval evaluation."""
    print(f"Loading evaluation data from {data_file}...")
    with open(data_file, 'r') as f:
        file_data = json.load(f)

    grouped_data: Dict[str, List[Dict[str, Any]]] = {}
    for item in file_data:
        q_type = item.get("question_type", "Uncategorized")
        grouped_data.setdefault(q_type, []).append(item)

    all_results: Dict[str, Any] = {}

    for question_type in list(grouped_data.keys()):
        print(f"\n{'='*50}")
        print(f"Evaluating question type: {question_type}")
        print(f"{'='*50}")

        group_items = grouped_data[question_type]

        ids = [item['id'] for item in group_items]
        questions = [item['question'] for item in group_items]
        evidences = [item['evidence'] for item in group_items]
        contexts = [item['context'] for item in group_items]

        data = {
            "id": ids,
            "question": questions,
            "contexts": contexts,
            "evidences": evidences
        }
        dataset = Dataset.from_dict(data)

        if num_samples:
            dataset = dataset.select([i for i in list(range(num_samples))])

        results = await evaluate_dataset(
            dataset=dataset,
            llm=llm,
            embeddings=embedding,
            detailed_output=detailed_output,
            max_concurrent=max_concurrent,
            relevance_version=relevance_version
        )

        all_results[question_type] = results
        print(f"\nResults for {question_type}:")
        if detailed_output:
            for metric, score in results["average_scores"].items():
                print(f"  {metric}: {score:.4f}")
        else:
            for metric, score in results.items():
                print(f"  {metric}: {score:.4f}")

    if output_file:
        print(f"\nSaving results to {output_file}...")
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

    print(f"\nEvaluation complete for {data_file}.")
    return all_results

async def main(args: argparse.Namespace):
    """Main retrieval evaluation function"""
    if args.mode == "API":
        # Check API key
        if not os.getenv("LLM_API_KEY"):
            raise ValueError("LLM_API_KEY environment variable is not set")
        
        # Initialize models
        # Wrap API key in SecretStr to satisfy type hints
        from pydantic import SecretStr
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise ValueError("LLM_API_KEY environment variable is not set")
        llm = ChatOpenAI(
            model=args.model,
            base_url=args.base_url,
            api_key=SecretStr(api_key),
            temperature=0.0,
            max_retries=3,
            timeout=30,
            model_kwargs={
                "top_p": 1,
                "seed": SEED,
                "presence_penalty": 0,
                "frequency_penalty": 0
            }
        )
        
        # Initialize the embedding model
        embedding = HuggingFaceBgeEmbeddings(model_name=args.embedding_model)

    elif args.mode == "ollama":
        ollama_client = OllamaClient(base_url=args.base_url)
        llm = OllamaWrapper(
            ollama_client,
            args.model,
            default_options={
                "temperature": 0.0,
                "top_p": 1,
                "num_ctx": 32768,
                "seed": SEED
            }
        )
        ollama_embeddings = OllamaEmbeddings(
            model=args.embedding_model,
            base_url=args.base_url
        )
        embedding = LangchainEmbeddingsWrapper(embeddings=ollama_embeddings)
        
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    data_path = pathlib.Path(args.data_file)
    eval_targets: List[Tuple[str, Optional[str]]] = []

    if data_path.is_dir():
        output_name = os.path.basename(args.output_file) if args.output_file else "retrieval_results.json"
        for sub_path in sorted(p for p in data_path.iterdir() if p.is_dir()):
            prediction_files = sorted(sub_path.glob("predictions_*.json"))
            if not prediction_files:
                print(f"Warning: No prediction files found in {sub_path}")
                continue
            for prediction_file in prediction_files:
                output_path = sub_path / output_name if output_name else None
                eval_targets.append((str(prediction_file), str(output_path) if output_path else None))
    else:
        eval_targets.append((str(data_path), args.output_file))

    if not eval_targets:
        raise ValueError("No data files found for evaluation. Provide a prediction file or directory containing them.")

    if args.max_concurrent < 1:
        raise ValueError("max_concurrent must be at least 1")

    for data_file, output_file in eval_targets:
        await evaluate_prediction_file(
            data_file=data_file,
            output_file=output_file,
            llm=llm,
            embedding=embedding,
            num_samples=args.num_samples,
            detailed_output=args.detailed_output,
            max_concurrent=args.max_concurrent,
            relevance_version=args.relevance_version
        )

    if args.mode == "ollama":
        await llm.close()

    print('\nEvaluation complete.')

if __name__ == "__main__":
    
    api_key = get_openai_api_key()

    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add command-line arguments
    parser.add_argument(
        "--mode", 
        # required=True,
        choices=["API", "ollama"],
        type=str,
        default="API",
        help="Use API or ollama for LLM"
    )

    parser.add_argument(
        "--model", 
        type=str,
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="OpenAI model to use for evaluation"
    )
    
    parser.add_argument(
        "--base_url", 
        type=str,
        default="http://0.0.0.0:8000/v1",
        help="Base URL for the OpenAI API"
    )
    
    parser.add_argument(
        "--embedding_model", 
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="HuggingFace model for embeddings"
    )
    
    parser.add_argument(
        "--data_file", 
        type=str,
        required=True,
        help="Path to JSON file containing evaluation data"
    )
    
    parser.add_argument(
        "--output_file", 
        type=str,
        default="retrieval_results.json",
        help="Path to save evaluation results"
    )
    
    parser.add_argument(
        "--num_samples", 
        type=int,
        default=None,
        help="Number of samples per question type to evaluate (optional)"
    )

    parser.add_argument(
        "--detailed_output",
        action="store_true",
        help="Whether to include detailed output"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Maximum number of samples to evaluate concurrently"
    )
    parser.add_argument(
        "--relevance-version",
        choices=["v1", "v2"],
        default="v1",
        help="Context relevance metric version to use"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the main function
    asyncio.run(main(args))
