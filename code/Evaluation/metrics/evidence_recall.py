import json
import numpy as np
from typing import List, Dict, Optional
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import Callbacks
import logging
from Evaluation.metrics.utils import JSONHandler

EVIDENCE_RECALL_PROMPT = """
### Task
You are a factual verifier. For each evidence statement, determine whether it is **explicitly supported or logically entailed** by the Context.

#### Decision Principles
- **attributed = 1**:
  - The Context explicitly states, or unmistakably implies, the same fact.
  - All key factual elements (actor, action, object, condition, date/number, and polarity) match.
  - Minor rewording, synonyms, or grammatical changes are acceptable if the *meaning* is unchanged.
  - General statements can support specific evidence **only if** the specific fact is fully covered (not partially).
- **attributed = 0**:
  - The Context is silent, ambiguous, or only loosely related.
  - Any key element is missing, contradicted, or requires assumption or background knowledge.
  - The evidence generalizes or narrows the Context beyond what is stated.
  - Any negation or uncertainty mismatch (e.g., “must” vs. “may”, “not allowed” vs. “allowed”).

Be conservative: if factual equivalence is not clear and complete, mark 0.


Output:
{{
  "classifications": [
    {{
      "statement": "Einstein received the Nobel Prize",
      "reason": "Matches context about Nobel Prize",
      "attributed": 1
    }},
    {{
      "statement": "He was born in Germany",
      "reason": "Birth information not in context",
      "attributed": 0
    }}
  ]
}}

#### Examples
Context: "Einstein won the Nobel Prize in Physics in 1921."
Evidence:
- "Einstein won the Nobel Prize in 1921"                -> 1 (direct)
- "Einstein received the 1921 physics prize"            -> 1 (paraphrase)
- "Einstein studied physics"                            -> 0 (not supported here)
- "Einstein won a Nobel Prize in 1922"                  -> 0 (date mismatch)
- "Einstein probably won a big award around 1920"       -> 0 (vague + wrong)

### Actual Input
Context: "{context}"

Evidence: {evidence}

Question: "{question}" (for reference only)

### Your Response:
"""

async def compute_evidence_recall(
    question: str,
    contexts: List[str],
    reference_evidence: List[str],
    llm: BaseLanguageModel,
    callbacks: Callbacks = None,
    max_retries: int = 1
) -> float:
    """
    Calculate context recall score (0.0-1.0) by measuring what percentage of 
    reference evidence are supported by the context.
    """
    # Handle edge cases
    
    context_str = "\n".join(contexts or [])
    if not context_str.strip():
        return 0.0  # No context means no attribution

    # Normalise evidence items
    if not isinstance(reference_evidence, list):
        reference_evidence = [reference_evidence]
    cleaned_evidence = []
    for item in reference_evidence:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        cleaned_evidence.append(text)
    if not cleaned_evidence:
        return 0.0
    
    # Format prompt with actual data
    prompt = EVIDENCE_RECALL_PROMPT.format(
        question=question,
        context=context_str[:30000],  # Truncate long contexts
        evidence=cleaned_evidence
    )
    
    # Get LLM classification with retries
    classifications = await _get_classifications(
        prompt, llm, callbacks, max_retries
    )
    
    # Calculate recall score
    if classifications:
        attributed = [c["attributed"] for c in classifications]
        return sum(attributed) / len(attributed)
    return 0.0  # Return 0 if no valid classifications

async def _get_classifications(
    prompt: str,
    llm: BaseLanguageModel,
    callbacks: Callbacks,
    max_retries: int,
    self_healing: bool = False
) -> List[Dict]:
    """
    Get valid classifications from LLM with retries using RobustJSONHandler.
    """
    parser = JSONHandler(max_retries=max_retries, self_healing=self_healing)

    for attempt in range(max_retries + 1):
        try:
            response = await llm.ainvoke(prompt, config={"callbacks": callbacks})

            classifications = await parser.parse_with_fallbacks(
                response.content,
                key="classifications",
                llm=llm if self_healing else None,
                callbacks=callbacks
            )
            return _validate_classifications(classifications)
        except Exception as exc:
            snippet = _safe_response_snippet(locals().get("response"))
            logging.warning(
                "Evidence recall parse failed (attempt %s/%s): %s | response=%s",
                attempt + 1,
                max_retries + 1,
                exc,
                snippet
            )
            continue
    print("[WARN] Unable to obtain evidence recall classifications after retries.")
    return []


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

def _validate_classifications(classifications: List) -> List[Dict]:
    """
    Ensure classifications have required fields and proper types.
    """
    valid = []
    for item in classifications:
        try:
            if (
                isinstance(item, dict)
                and "statement" in item
                and "reason" in item
                and "attributed" in item
                and item["attributed"] in {0, 1}
            ):
                valid.append({
                    "statement": str(item["statement"]),
                    "reason": str(item["reason"]),
                    "attributed": int(item["attributed"])
                })
        except (TypeError, ValueError):
            continue
    return valid
