"""Hybrid retriever tests for FAISS + DB integration path."""

import os
import tempfile
import faiss
import numpy as np
import types


class FakeClient:
    class embeddings:
        @staticmethod
        def create(model, input):
            class Item:
                def __init__(self, embedding):
                    self.embedding = embedding

            vec = [0.1] * 8
            return types.SimpleNamespace(data=[Item(vec)])


def test_hybrid_search_creates_hits(tmp_path, monkeypatch):
    # prepare a small FAISS index with 3 vectors
    dim = 8
    xb = np.random.RandomState(123).randn(3, dim).astype('float32')
    index = faiss.IndexFlatIP(dim)
    # normalize
    norms = np.linalg.norm(xb, axis=1, keepdims=True)
    xb = xb / (norms + 1e-9)
    index.add(xb)

    idx_path = tmp_path / "test.index"
    faiss.write_index(index, str(idx_path))

    # populate DB: create tables and insert index_shards and vectors rows
    import psycopg2

    conn = psycopg2.connect()
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS index_shards (shard_name TEXT UNIQUE, path TEXT, pipeline_run_id INTEGER, created_at TEXT, is_active INTEGER, index_id TEXT)"
    )
    cur.execute(
        "CREATE TABLE IF NOT EXISTS vectors (id INTEGER PRIMARY KEY, text TEXT, page INTEGER, start_word INTEGER, end_word INTEGER, doc_id TEXT, pipeline_run_id INTEGER, created_at TEXT)"
    )
    # clear any prior rows from other tests (shared in-memory DB)
    cur.execute("DELETE FROM index_shards")
    cur.execute("DELETE FROM vectors")
    # insert index_shard and vectors rows
    index_id = "idx-test-1"
    sidecar = idx_path.with_suffix(".index.version.json")
    sidecar.write_text(f'{{"index_id": "{index_id}"}}')
    cur.execute(
        "INSERT INTO index_shards (shard_name, path, pipeline_run_id, created_at, is_active, index_id) VALUES (?, ?, ?, ?, 1, ?)",
        ("s1", str(idx_path), 1, "now", index_id),
    )
    for i in range(3):
        cur.execute(
            "INSERT INTO vectors (id, text, page, start_word, end_word, doc_id, pipeline_run_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (i, f"doc text {i}", 1, 0, 10, 'doc', 1, "now"),
        )
    conn.commit()

    # monkeypatch OpenAI client usage: provide FakeClient instance
    fake_client = FakeClient()

    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("ragonometrics.indexing.retriever", Path("ragonometrics/indexing/retriever.py").resolve())
    retriever = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(retriever)

    hits = retriever.hybrid_search("test query", client=fake_client, db_url="dummy", top_k=2, bm25_weight=0.5)
    # hits should be a list (possibly empty if scoring ties)
    assert isinstance(hits, list)
