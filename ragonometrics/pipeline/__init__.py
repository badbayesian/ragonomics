"""LLM pipeline routines, caching, and usage tracking used to produce summaries, citations, and answers."""

from .pipeline import (
    DEFAULT_MODEL,
    DEFAULT_MAX_OUTPUT_TOKENS,
    call_openai,
    summarize_paper,
    extract_citations,
    rank_citations,
    extract_metadata,
)

from .query_cache import DEFAULT_CACHE_PATH, get_cached_answer, make_cache_key, set_cached_answer
from .token_usage import DEFAULT_USAGE_DB, get_recent_usage, get_usage_by_model, get_usage_summary, record_usage

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "call_openai",
    "summarize_paper",
    "extract_citations",
    "rank_citations",
    "extract_metadata",
    "DEFAULT_CACHE_PATH",
    "get_cached_answer",
    "make_cache_key",
    "set_cached_answer",
    "DEFAULT_USAGE_DB",
    "get_recent_usage",
    "get_usage_by_model",
    "get_usage_summary",
    "record_usage",
]
