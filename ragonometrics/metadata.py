from __future__ import annotations

import time
from datetime import datetime, timezone
import json
from typing import Optional

import psycopg2


def init_metadata_db(db_url: str):
    """Initialize metadata tables for pipeline runs and vectors.

    Args:
        db_url: Postgres database URL.

    Returns:
        connection: Open psycopg2 connection.
    """
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    # pipeline runs (audit manifest for a build)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id SERIAL PRIMARY KEY,
            git_sha TEXT,
            extractor_version TEXT,
            embedding_model TEXT,
            chunk_words INTEGER,
            chunk_overlap INTEGER,
            normalized BOOLEAN,
            created_at TIMESTAMP
        )
        """
    )
    # idempotency key (for deduping runs with same inputs)
    try:
        cur.execute("ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS idempotency_key TEXT")
    except Exception:
        try:
            cur.execute("ALTER TABLE pipeline_runs ADD COLUMN idempotency_key TEXT")
        except Exception:
            pass

    # index shards manifest
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS index_shards (
            id SERIAL PRIMARY KEY,
            shard_name TEXT UNIQUE,
            path TEXT,
            pipeline_run_id INTEGER REFERENCES pipeline_runs(id),
            created_at TIMESTAMP,
            is_active BOOLEAN DEFAULT FALSE
        )
        """
    )
    # add index_id column if missing (guard for older tables)
    try:
        cur.execute("ALTER TABLE index_shards ADD COLUMN IF NOT EXISTS index_id TEXT")
    except Exception:
        # sqlite or older versions may not support IF NOT EXISTS
        try:
            cur.execute("ALTER TABLE index_shards ADD COLUMN index_id TEXT")
        except Exception:
            pass

    # index versions
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS index_versions (
            index_id TEXT PRIMARY KEY,
            created_at TIMESTAMP,
            embedding_model TEXT,
            chunk_words INTEGER,
            chunk_overlap INTEGER,
            corpus_fingerprint TEXT,
            index_path TEXT,
            shard_path TEXT
        )
        """
    )

    # failure logging for replay/debug
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS request_failures (
            id SERIAL PRIMARY KEY,
            component TEXT,
            error TEXT,
            context_json TEXT,
            created_at TIMESTAMP
        )
        """
    )

    # vectors metadata (per-chunk)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS vectors (
            id BIGINT PRIMARY KEY,
            doc_id TEXT,
            paper_path TEXT,
            page INTEGER,
            start_word INTEGER,
            end_word INTEGER,
            text TEXT,
            pipeline_run_id INTEGER REFERENCES pipeline_runs(id),
            created_at TIMESTAMP
        )
        """
    )

    # simple documents table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            path TEXT,
            title TEXT,
            author TEXT,
            extracted_at TIMESTAMP
        )
        """
    )

    conn.commit()
    return conn


def create_pipeline_run(
    conn,
    *,
    git_sha: Optional[str],
    extractor_version: Optional[str],
    embedding_model: str,
    chunk_words: int,
    chunk_overlap: int,
    normalized: bool,
    idempotency_key: Optional[str] = None,
) -> int:
    """Create a pipeline run record and return its id.

    Args:
        conn: Open database connection.
        git_sha: Optional git SHA for the run.
        extractor_version: Optional extractor version string.
        embedding_model: Embedding model name.
        chunk_words: Chunk size in words.
        chunk_overlap: Overlap size in words.
        normalized: Whether embeddings were normalized.

    Returns:
        int: Pipeline run id if available.
    """
    cur = conn.cursor()
    if idempotency_key:
        try:
            cur.execute("SELECT COALESCE(id, rowid) FROM pipeline_runs WHERE idempotency_key = %s LIMIT 1", (idempotency_key,))
            row = cur.fetchone()
            if row and row[0] is not None:
                return int(row[0])
        except Exception:
            # ignore lookup failures (e.g., sqlite without column)
            pass
    cur.execute(
        "INSERT INTO pipeline_runs (git_sha, extractor_version, embedding_model, chunk_words, chunk_overlap, normalized, created_at, idempotency_key) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        (
            git_sha,
            extractor_version,
            embedding_model,
            chunk_words,
            chunk_overlap,
            normalized,
            datetime.now(timezone.utc).isoformat(),
            idempotency_key,
        ),
    )
    # attempt to find the inserted id
    try:
        # prefer explicit id, fall back to rowid for sqlite
        cur.execute("SELECT COALESCE(id, rowid) FROM pipeline_runs ORDER BY rowid DESC LIMIT 1")
        row = cur.fetchone()
        run_id = row[0] if row else None
    except Exception:
        run_id = None
    conn.commit()
    return run_id


def publish_shard(conn, shard_name: str, path: str, pipeline_run_id: int, index_id: str | None = None) -> int:
    """Upsert an index shard and mark it active.

    Args:
        conn: Open database connection.
        shard_name: Unique shard name.
        path: Filesystem path to the shard.
        pipeline_run_id: Associated pipeline run id.
        index_id: Optional index version id.

    Returns:
        int: Shard id if available.
    """
    cur = conn.cursor()
    # deactivate any previous active shard(s)
    cur.execute("UPDATE index_shards SET is_active = FALSE WHERE is_active = TRUE")
    # upsert shard (Postgres-style ON CONFLICT used in production). For sqlite testing, attempt insert then select.
    try:
        cur.execute(
            "INSERT INTO index_shards (shard_name, path, pipeline_run_id, created_at, is_active, index_id) VALUES (%s, %s, %s, %s, TRUE, %s) ON CONFLICT (shard_name) DO UPDATE SET path = EXCLUDED.path, pipeline_run_id = EXCLUDED.pipeline_run_id, created_at = EXCLUDED.created_at, is_active = TRUE, index_id = EXCLUDED.index_id",
            (shard_name, path, pipeline_run_id, datetime.now(timezone.utc).isoformat(), index_id),
        )
    except Exception:
        # fallback for sqlite: try simple insert or replace
        try:
            cur.execute(
                "REPLACE INTO index_shards (shard_name, path, pipeline_run_id, created_at, is_active, index_id) VALUES (%s, %s, %s, %s, 1, %s)",
                (shard_name, path, pipeline_run_id, datetime.now(timezone.utc).isoformat(), index_id),
            )
        except Exception:
            pass
    # find the shard id
    try:
        cur.execute("SELECT COALESCE(id, rowid) FROM index_shards WHERE shard_name = %s LIMIT 1", (shard_name,))
        r = cur.fetchone()
        shard_id = r[0] if r else None
    except Exception:
        shard_id = None
    conn.commit()
    return shard_id


def get_active_shards(conn):
    """Fetch active index shards ordered by creation time.

    Args:
        conn: Open database connection.

    Returns:
        list[tuple[str, str]]: (shard_name, path) rows.
    """
    cur = conn.cursor()
    cur.execute("SELECT shard_name, path FROM index_shards WHERE is_active = TRUE ORDER BY created_at DESC")
    return cur.fetchall()


def record_failure(conn, component: str, error: str, context: dict | None = None) -> None:
    """Record a failure for later replay/debug."""
    cur = conn.cursor()
    payload = json.dumps(context or {}, ensure_ascii=False)
    cur.execute(
        "INSERT INTO request_failures (component, error, context_json, created_at) VALUES (%s, %s, %s, %s)",
        (component, error, payload, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def create_index_version(
    conn,
    *,
    index_id: str,
    embedding_model: str,
    chunk_words: int,
    chunk_overlap: int,
    corpus_fingerprint: str,
    index_path: str,
    shard_path: str,
) -> str:
    """Insert an index version row and return the index id."""
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO index_versions (
            index_id, created_at, embedding_model, chunk_words, chunk_overlap,
            corpus_fingerprint, index_path, shard_path
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (index_id) DO NOTHING
        """,
        (
            index_id,
            datetime.now(timezone.utc).isoformat(),
            embedding_model,
            chunk_words,
            chunk_overlap,
            corpus_fingerprint,
            index_path,
            shard_path,
        ),
    )
    conn.commit()
    return index_id
