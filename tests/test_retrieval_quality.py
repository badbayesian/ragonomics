import os
from pathlib import Path

from ragonometrics.core.io_loaders import chunk_pages
from ragonometrics.core.main import expand_queries, Settings


def test_section_aware_chunking_adds_section(monkeypatch):
    monkeypatch.setenv("SECTION_AWARE_CHUNKING", "1")
    pages = [
        "Abstract\nThis is the abstract section.",
        "Introduction\nThis is the intro section.",
    ]
    chunks = chunk_pages(pages, chunk_words_count=5, overlap_words=0)
    assert chunks
    assert chunks[0].get("section") == "abstract"
    # find a chunk from page 2
    page2 = [c for c in chunks if c.get("page") == 2]
    assert page2
    assert page2[0].get("section") == "introduction"


def test_expand_queries_default(monkeypatch):
    monkeypatch.delenv("QUERY_EXPANSION", raising=False)
    settings = Settings(
        papers_dir=Path("."),
        max_papers=1,
        max_words=1000,
        chunk_words=200,
        chunk_overlap=20,
        top_k=3,
        batch_size=32,
        embedding_model="emb",
        chat_model="chat",
    )

    class DummyClient:
        pass

    out = expand_queries("test query", DummyClient(), settings)
    assert out == ["test query"]

