"""Query cache tests for sqlite roundtrip and keying behavior."""

from pathlib import Path

from ragonometrics.pipeline.query_cache import get_cached_answer, make_cache_key, set_cached_answer


def test_query_cache_roundtrip(tmp_path):
    db_path = tmp_path / "cache.sqlite"
    query = "What is the research question?"
    paper = "paper.pdf"
    model = "gpt-test"
    context = "(page 1 words 0-10)\nSome text."
    key = make_cache_key(query, paper, model, context)

    assert get_cached_answer(db_path, key) is None
    set_cached_answer(
        db_path,
        cache_key=key,
        query=query,
        paper_path=paper,
        model=model,
        context=context,
        answer="Answer",
    )
    assert get_cached_answer(db_path, key) == "Answer"

