import asyncio
import argparse
import json
import logging
import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Any
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# append the sys.path to include parent directory for imports
import sys
import pathlib
# add parent directory to sys.path to avod no module error
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from Evaluation.metrics import compute_answer_correctness, compute_coverage_score, compute_faithfulness_score, compute_rouge_score
from langchain_ollama import OllamaEmbeddings
from Evaluation.llm import OllamaClient, OllamaWrapper

import logging

os.environ["LLM_API_KEY"] = "000"

SEED = 42

METRIC_CONFIG = {
    'Fact Retrieval': ["rouge_score", "answer_correctness"],
    'Complex Reasoning': ["rouge_score", "answer_correctness"],
    'Contextual Summarize': ["answer_correctness", "coverage_score"],
    'Creative Generation': ["answer_correctness", "coverage_score", "faithfulness"]
}

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
    metrics: List[str],
    llm: Any,
    embeddings: Embeddings,
    max_concurrent: int = 3,  # Limit concurrent evaluations
    detailed_output: bool = False
) -> Dict[str, Any]:
    """Evaluate the metric scores on the entire dataset."""
    results = {metric: [] for metric in metrics}
    detailed_results = [] if detailed_output else None
    
    ids = dataset["id"]
    questions = dataset["question"]
    answers = dataset["answer"]
    contexts_list = dataset["contexts"]
    ground_truths = dataset["ground_truth"]
    
    total_samples = len(questions)
    print(f"\nStarting evaluation of {total_samples} samples...")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def evaluate_with_semaphore(i):
        async with semaphore:
            sample_metrics = await evaluate_sample(
                question=questions[i],
                answer=answers[i],
                contexts=contexts_list[i],
                ground_truth=ground_truths[i],
                metrics=metrics,
                llm=llm,
                embeddings=embeddings
            )
            if detailed_output:
                return {
                    "id": ids[i],
                    "question": questions[i],
                    "ground_truth": ground_truths[i],
                    "generated_answer": answers[i],
                    "contexts": contexts_list[i],
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
    
    avg_results = {metric: np.nanmean(scores) for metric, scores in results.items()}
    
    if detailed_output:
        return {
            "average_scores": avg_results,
            "detailed": detailed_results
        }
    else:
        return avg_results

async def evaluate_sample(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str,
    metrics: List[str],
    llm: Any,
    embeddings: Embeddings
) -> Dict[str, float]:
    """Evaluate the metric scores for a single sample."""
    results = {}
    
    tasks = {}
    if "rouge_score" in metrics:
        tasks["rouge_score"] = compute_rouge_score(answer, ground_truth)
    
    if "answer_correctness" in metrics:
        tasks["answer_correctness"] = compute_answer_correctness(
            question, answer, ground_truth, llm, embeddings
        )
    
    if "coverage_score" in metrics:
        tasks["coverage_score"] = compute_coverage_score(
            question, ground_truth, answer, llm
        )
    
    if "faithfulness" in metrics:
        tasks["faithfulness"] = compute_faithfulness_score(
            question, answer, contexts, llm
        )
    
    task_results = await asyncio.gather(*tasks.values())
    
    for i, metric in enumerate(tasks.keys()):
        results[metric] = task_results[i]
    
    return results

async def evaluate_prediction_file(
    data_file: str,
    output_file: Optional[str],
    llm: Any,
    embedding: Embeddings,
    num_samples: Optional[int],
    detailed_output: bool,
    max_concurrent: int
) -> Dict[str, Any]:
    """Run evaluation for a single prediction file."""
    print(f"Loading evaluation data from {data_file}...")
    with open(data_file, 'r') as f:
        file_data = json.load(f)

    grouped_data: Dict[str, List[Dict[str, Any]]] = {}
    for item in file_data:
        q_type = item.get("question_type", "Uncategorized")
        grouped_data.setdefault(q_type, []).append(item)

    all_results: Dict[str, Any] = {}

    for question_type in list(grouped_data.keys()):
        if question_type not in METRIC_CONFIG:
            print(f"Skipping undefined question type: {question_type}")
            continue

        print(f"\n{'='*50}")
        print(f"Evaluating question type: {question_type}")
        print(f"{'='*50}")

        group_items = grouped_data[question_type]
        ids = [item['id'] for item in group_items]
        questions = [item['question'] for item in group_items]
        ground_truths = [item['ground_truth'] for item in group_items]
        answers = [item['generated_answer'] for item in group_items]
        contexts = [item['context'] for item in group_items]

        data = {
            "id": ids,
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        dataset = Dataset.from_dict(data)

        if num_samples:
            dataset = dataset.select([i for i in list(range(num_samples))])

        results = await evaluate_dataset(
            dataset=dataset,
            metrics=METRIC_CONFIG[question_type],
            llm=llm,
            embeddings=embedding,
            detailed_output=detailed_output,
            max_concurrent=max_concurrent
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
    """Main evaluation function that accepts command-line arguments."""
    if args.mode == "API":
    # Check if the API key is set

        if not os.getenv("LLM_API_KEY"):
            raise ValueError("LLM_API_KEY environment variable is not set")
    
        # Initialize the model
        # Wrap API key in SecretStr to satisfy type hints
        from pydantic import SecretStr
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise ValueError("LLM_API_KEY environment variable is not set")
        llm = ChatOpenAI(
            model=args.model,
            base_url=args.base_url,
            api_key=SecretStr(api_key),
            temperature=0.2,
            max_retries=3,
            timeout=60,
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
        embedding = OllamaEmbeddings(
            model=args.embedding_model,
            base_url=args.base_url
        )
    
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    data_path = pathlib.Path(args.data_file)
    eval_targets: List[Tuple[str, Optional[str]]] = []

    if data_path.is_dir():
        output_name = os.path.basename(args.output_file) if args.output_file else "evaluation_results.json"
        for sub_path in sorted(p for p in data_path.iterdir() if p.is_dir()):
            prediction_files = sorted(sub_path.glob("predictions_*.json"))
            if not prediction_files:
                logging.warning(f"No prediction files found in {sub_path}")
                continue
            for prediction_file in prediction_files:
                output_path = sub_path / output_name if output_name else None
                eval_targets.append((str(prediction_file), str(output_path) if output_path else None))
    else:
        eval_targets.append((str(data_path), args.output_file))

    if not eval_targets:
        raise ValueError("No data files found for evaluation. Provide a prediction file or directory containing them.")

    for data_file, output_file in eval_targets:
        await evaluate_prediction_file(
            data_file=data_file,
            output_file=output_file,
            llm=llm,
            embedding=embedding,
            num_samples=args.num_samples,
            detailed_output=args.detailed_output,
            max_concurrent=args.max_concurrent
        )

    if args.mode == "ollama":
        await llm.close()

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # Handle API key security
    api_key = get_openai_api_key()
    if not api_key:
        logging.warning("No API key found! Requests may fail.")
    parser = argparse.ArgumentParser(
        description="Evaluate RAG performance using various metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
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
        default="BAAI/bge-large-en-v1.5",
        help="HuggingFace model for embeddings"
    )
    
    parser.add_argument(
        "--data_file", 
        type=str,
        required=True,
        help="Path to JSON file or directory of prediction files for evaluation"
    )
    
    parser.add_argument(
        "--output_file", 
        type=str,
        default="evaluation_results.json",
        help="Path to save evaluation results"
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to use for evaluation"
    )

    parser.add_argument(
        "--detailed_output",
        action="store_true",
        help="Whether to include detailed output"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Maximum number of samples to evaluate concurrently"
    )
    
    args = parser.parse_args()
    
    if args.max_concurrent < 1:
        raise ValueError("max_concurrent must be at least 1")

    asyncio.run(main(args))
