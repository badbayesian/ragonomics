"""SQLite cache for question answers. Used by CLI and Streamlit to avoid recomputation across runs."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_PATH = PROJECT_ROOT / "ragonometrics_query_cache.sqlite"


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS query_cache (
            cache_key TEXT PRIMARY KEY,
            query TEXT,
            paper_path TEXT,
            model TEXT,
            context_hash TEXT,
            answer TEXT,
            created_at TEXT
        )
        """
    )
    conn.commit()
    return conn


def make_cache_key(query: str, paper_path: str, model: str, context: str) -> str:
    payload = f"{paper_path}||{model}||{query}||{hashlib.sha256(context.encode('utf-8')).hexdigest()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_cached_answer(db_path: Path, cache_key: str) -> Optional[str]:
    conn = _connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT answer FROM query_cache WHERE cache_key = ?", (cache_key,))
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def set_cached_answer(
    db_path: Path,
    *,
    cache_key: str,
    query: str,
    paper_path: str,
    model: str,
    context: str,
    answer: str,
) -> None:
    conn = _connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO query_cache
            (cache_key, query, paper_path, model, context_hash, answer, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cache_key,
                query,
                paper_path,
                model,
                hashlib.sha256(context.encode("utf-8")).hexdigest(),
                answer,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()

