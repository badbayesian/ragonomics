"""Evaluation and benchmarking utilities for retrieval quality and pipeline performance."""

from .eval import (
    GoldenExample,
    load_golden_set,
    recall_at_k,
    mrr_at_k,
    evaluate_retrieval,
    evaluate_answers,
    answer_has_citation,
)

from .benchmark import benchmark_chunking, benchmark_indexing, bench_papers

__all__ = [
    "GoldenExample",
    "load_golden_set",
    "recall_at_k",
    "mrr_at_k",
    "evaluate_retrieval",
    "evaluate_answers",
    "answer_has_citation",
    "benchmark_chunking",
    "benchmark_indexing",
    "bench_papers",
]
