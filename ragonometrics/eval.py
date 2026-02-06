from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class GoldenExample:
    """Represents a single golden evaluation example.

    Attributes:
        paper_path: Path to the paper file.
        question: Question to evaluate.
        expected_pages: Expected relevant page numbers.
        expected_chunk_ids: Optional expected chunk identifiers.
        expected_citations: Optional expected citation identifiers.
    """

    paper_path: str
    question: str
    expected_pages: List[int]
    expected_chunk_ids: List[str] | None = None
    expected_citations: List[str] | None = None


def load_golden_set(path: Path) -> List[GoldenExample]:
    """Load a golden set from a JSON list file.

    Args:
        path: Path to the JSON file.

    Returns:
        List[GoldenExample]: Parsed golden examples.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    out: List[GoldenExample] = []
    for row in data:
        out.append(
            GoldenExample(
                paper_path=str(row.get("paper_path", "")),
                question=str(row.get("question", "")),
                expected_pages=[int(x) for x in row.get("expected_pages", [])],
                expected_chunk_ids=row.get("expected_chunk_ids"),
                expected_citations=row.get("expected_citations"),
            )
        )
    return out


def recall_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int) -> float:
    """Compute recall@k.

    Args:
        retrieved_ids: Ranked list of retrieved identifiers.
        expected_ids: Expected relevant identifiers.
        k: Cutoff.

    Returns:
        float: Recall at k.
    """
    if not expected_ids or k <= 0:
        return 0.0
    retrieved_k = set(retrieved_ids[:k])
    hits = sum(1 for eid in expected_ids if eid in retrieved_k)
    return hits / len(expected_ids)


def mrr_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int) -> float:
    """Compute mean reciprocal rank at k.

    Args:
        retrieved_ids: Ranked list of retrieved identifiers.
        expected_ids: Expected relevant identifiers.
        k: Cutoff.

    Returns:
        float: MRR at k.
    """
    if not expected_ids or k <= 0:
        return 0.0
    for rank, rid in enumerate(retrieved_ids[:k], start=1):
        if rid in expected_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    retrieved_meta: List[Dict],
    *,
    expected_pages: List[int],
    expected_chunk_ids: Optional[List[str]] = None,
    k: int = 6,
) -> Dict[str, float]:
    """Compute retrieval metrics from retrieved chunk metadata.

    Args:
        retrieved_meta: Retrieved chunk metadata dicts.
        expected_pages: Expected relevant pages.
        expected_chunk_ids: Optional expected chunk ids.
        k: Cutoff for metrics.

    Returns:
        Dict[str, float]: Retrieval metrics.
    """
    retrieved_page_ids = [str(m.get("page")) for m in retrieved_meta if m.get("page") is not None]
    expected_page_ids = [str(p) for p in expected_pages]

    metrics = {
        "recall_at_k": recall_at_k(retrieved_page_ids, expected_page_ids, k),
        "mrr_at_k": mrr_at_k(retrieved_page_ids, expected_page_ids, k),
    }
    if expected_chunk_ids:
        retrieved_chunk_ids = [str(m.get("chunk_id")) for m in retrieved_meta if m.get("chunk_id") is not None]
        metrics["recall_at_k_chunks"] = recall_at_k(retrieved_chunk_ids, expected_chunk_ids, k)
        metrics["mrr_at_k_chunks"] = mrr_at_k(retrieved_chunk_ids, expected_chunk_ids, k)
    return metrics


_CITATION_PATTERNS = [
    re.compile(r"\(page\s+\d+", re.IGNORECASE),
    re.compile(r"\bpage\s+\d+\b", re.IGNORECASE),
    re.compile(r"\bwords\s+\d+\s*-\s*\d+\b", re.IGNORECASE),
]


def answer_has_citation(answer: str) -> bool:
    """Return True if the answer appears to contain a provenance citation.

    Args:
        answer: Answer string.

    Returns:
        bool: True if a citation-like pattern is found.
    """
    return any(p.search(answer or "") for p in _CITATION_PATTERNS)


def normalize_answer(text: str) -> str:
    """Normalize an answer for comparison."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def self_consistency_rate(answers: Iterable[str]) -> float:
    """Compute a simple self-consistency rate.

    Args:
        answers: Iterable of answer strings.

    Returns:
        float: Fraction of the most common answer.
    """
    answers_list = [normalize_answer(a) for a in answers if a]
    if not answers_list:
        return 0.0
    counts: Dict[str, int] = {}
    for ans in answers_list:
        counts[ans] = counts.get(ans, 0) + 1
    return max(counts.values()) / len(answers_list)


def evaluate_answers(answers: Iterable[str]) -> Dict[str, float]:
    """Compute answer-quality proxy metrics.

    Args:
        answers: Iterable of answer strings.

    Returns:
        Dict[str, float]: Answer-quality proxy metrics.
    """
    answers_list = [a for a in answers if a is not None]
    if not answers_list:
        return {"citation_coverage": 0.0, "hallucination_rate_proxy": 0.0, "self_consistency": 0.0}

    cited = sum(1 for a in answers_list if answer_has_citation(a))
    coverage = cited / len(answers_list)
    return {
        "citation_coverage": coverage,
        "hallucination_rate_proxy": 1.0 - coverage,
        "self_consistency": self_consistency_rate(answers_list),
    }
