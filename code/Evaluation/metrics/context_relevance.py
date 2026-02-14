from typing import List, Callable, Awaitable, Optional, Union
import numpy as np
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import Callbacks
import re
import logging
from Evaluation.metrics.utils import JSONHandler

CONTEXT_RELEVANCE_PROMPT = """
### Instructions
You are a world class expert designed to evaluate the relevance score of a Context in order to answer the Question.
Your task is to determine if the Context contains proper information to answer the Question.
Do not rely on your previous knowledge about the Question.
Use only what is written in the Context and in the Question.

Scoring rules:
0. If the context does not contain any relevant information to answer the question, score 0.
1. If the context partially contains relevant information to answer the question, score 1.
2. If the context fully contains relevant information to answer the question, score 2.

Output format:
You must output strictly in JSON format with a single key "score".
No explanation, no additional text.

Example:
Question: What is the capital of France?
Context: Paris is the capital of France.
Output:
{{ "score": 2 }}

Now evaluate the following:
Question: {question}
Context: {context}
"""

 
async def compute_context_relevance(
    question: str,
    contexts: List[str],
    llm: BaseLanguageModel,
    callbacks: Callbacks = None,
    max_retries: int = 2
) -> float:
    """
    Evaluate the relevance of retrieved contexts for answering a question.
    Returns a score between 0.0 (irrelevant) and 1.0 (fully relevant).
    """
    # Handle edge cases
    if not question.strip() or not contexts or not any(c.strip() for c in contexts):
        return 0.0
    
    context_str = "\n".join(contexts)[:30000]  # Truncate long contexts
    
    # Check for exact matches (often indicate low relevance)
    if context_str.strip() == question.strip() or context_str.strip() in question:
        return 0.0
    
    # Get two independent ratings from LLM
    rating1 = await _get_llm_rating(question, context_str, llm, callbacks, max_retries)
    rating2 = await _get_llm_rating(question, context_str, llm, callbacks, max_retries)
    
    # Process ratings (0-2 scale) and convert to 0-1 scale
    scores = [r/2 for r in [rating1, rating2] if r is not None]
    
    # Calculate final score
    if not scores:
        return 0.0
    return sum(scores) / len(scores)  # Average of valid scores


async def _get_llm_rating(
    question: str,
    context: str,
    llm: BaseLanguageModel,
    callbacks: Callbacks,
    max_retries: int,
    self_healing: bool = False
) -> Optional[float]:
    """
    Get a single relevance rating from LLM with retries.
    """
    parser = JSONHandler(max_retries=max_retries, self_healing=self_healing)

    prompt = CONTEXT_RELEVANCE_PROMPT.format(question=question, context=context)

    for attempt in range(max_retries):
        try:
            response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
            content = getattr(response, "content", response)
            parsed = await parser.parse_with_fallbacks(
                content,
                llm=llm if self_healing else None,
                callbacks=callbacks
            )
            return _normalize_rating(parsed)
        except Exception as exc:
            snippet = _safe_response_snippet(locals().get("response"))
            logging.warning(
                "Context relevance parse failed (attempt %s/%s): %s | response=%s",
                attempt + 1,
                max_retries,
                exc,
                snippet
            )
            continue

    print("[WARN] Unable to obtain context relevance rating after retries.")
    return None


def _safe_response_snippet(response: Optional[object], max_len: int = 200) -> str:
    if response is None:
        return "<no response>"
    content = getattr(response, "content", None)
    if content is None:
        content = str(response)
    if isinstance(content, list):
        content = " ".join(str(part) for part in content)
    text = str(content).strip()
    return text[:max_len] + ("..." if len(text) > max_len else "")


def _normalize_rating(parsed: Union[dict, list, str, None]) -> Optional[float]:
    """
    Normalize parsed content to extract a valid rating (0-2).
    """
    # Case 1: JSON dict with "rating" or "score"
    if isinstance(parsed, dict):
        score = parsed.get("rating", parsed.get("score"))
        if _is_valid_rating(score):
            return float(score)

    # Case 2: JSON list with a single number
    if isinstance(parsed, list) and len(parsed) == 1:
        if _is_valid_rating(parsed[0]):
            return float(parsed[0])

    # Case 3: Raw string - try parse as JSON first
    if isinstance(parsed, str):
        stripped = parsed.strip()
        try:
            import json
            data = json.loads(stripped)
            if isinstance(data, dict):
                score = data.get("rating", data.get("score"))
                if _is_valid_rating(score):
                    return float(score)
        except Exception:
            pass

        # Case 4: fallback to direct float
        try:
            value = float(stripped)
            if _is_valid_rating(value):
                return value
        except ValueError:
            pass

        # Case 5: fallback token scan
        for token in stripped.split()[:8]:
            try:
                value = float(token)
                if _is_valid_rating(value):
                    return value
            except ValueError:
                continue

    return None

def _is_valid_rating(value) -> bool:
    """Check if value is an integer between 0 and 2."""
    try:
        ivalue = float(value)
        return 0 <= ivalue <= 2
    except (TypeError, ValueError):
        return False
