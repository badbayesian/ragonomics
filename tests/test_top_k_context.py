"""Top-k context tests for the hybrid retrieval path and output content."""

import os
import types
import psycopg2


class FakeClient:
    class embeddings:
        @staticmethod
        def create(model, input):
            class Item:
                def __init__(self, embedding):
                    self.embedding = embedding

            return types.SimpleNamespace(data=[Item([0.01] * 8)])


def test_top_k_context_uses_hybrid(monkeypatch):
    # ensure DATABASE_URL env is set to trigger hybrid path
    os.environ["DATABASE_URL"] = "dummy"

    # populate DB with a vectors row matching id 0
    conn = psycopg2.connect()
    cur = conn.cursor()
    # ensure vectors table exists and is clean
    cur.execute("CREATE TABLE IF NOT EXISTS vectors (id INTEGER PRIMARY KEY, text TEXT, page INTEGER, start_word INTEGER, end_word INTEGER, doc_id TEXT, pipeline_run_id INTEGER, created_at TEXT)")
    cur.execute("DELETE FROM vectors")
    cur.execute("INSERT INTO vectors (id, text, page, start_word, end_word, doc_id, pipeline_run_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))", (0, "hello world", 1, 0, 5, 'd1', 1))
    conn.commit()
    # ensure index_shards exists (hybrid loader queries it)
    cur.execute("CREATE TABLE IF NOT EXISTS index_shards (shard_name TEXT UNIQUE, path TEXT, pipeline_run_id INTEGER, created_at TEXT, is_active INTEGER)")
    cur.execute("DELETE FROM index_shards")
    conn.commit()

    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("ragonometrics.core.main", Path("ragonometrics/core/main.py").resolve())
    main = importlib.util.module_from_spec(spec)
    import sys
    sys.modules["ragonometrics.core.main"] = main
    spec.loader.exec_module(main)

    top_k_context = main.top_k_context
    Settings = main.Settings

    chunks = [{"text": "hello world", "page": 1, "start_word": 0, "end_word": 5}]
    embeddings = [[0.01] * 8]
    fake_client = FakeClient()
    settings = Settings(papers_dir=None, max_papers=1, max_words=1000, chunk_words=256, chunk_overlap=32, top_k=1, batch_size=1, embedding_model="e", chat_model="c")

    ctx = top_k_context(chunks, embeddings, "hello", client=fake_client, settings=settings)
    assert "hello world" in ctx
