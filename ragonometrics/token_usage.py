from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ragonometrics.config import PROJECT_ROOT


DEFAULT_USAGE_DB = PROJECT_ROOT / "ragonometrics_token_usage.sqlite"


@dataclass(frozen=True)
class UsageSummary:
    """Aggregated token usage statistics."""

    calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS token_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            model TEXT,
            operation TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            total_tokens INTEGER,
            session_id TEXT,
            request_id TEXT,
            meta TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_token_usage_created_at ON token_usage(created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_token_usage_session ON token_usage(session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_token_usage_request ON token_usage(request_id)")
    conn.commit()
    return conn


def record_usage(
    *,
    db_path: Path = DEFAULT_USAGE_DB,
    model: str,
    operation: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist a token usage record to SQLite."""
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO token_usage
            (created_at, model, operation, input_tokens, output_tokens, total_tokens, session_id, request_id, meta)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                model,
                operation,
                int(input_tokens),
                int(output_tokens),
                int(total_tokens),
                session_id,
                request_id,
                json.dumps(meta or {}, ensure_ascii=False),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _where_clauses(
    *,
    session_id: Optional[str],
    request_id: Optional[str],
    since: Optional[str],
) -> tuple[str, List[Any]]:
    clauses = []
    params: List[Any] = []
    if session_id:
        clauses.append("session_id = ?")
        params.append(session_id)
    if request_id:
        clauses.append("request_id = ?")
        params.append(request_id)
    if since:
        clauses.append("created_at >= ?")
        params.append(since)
    if clauses:
        return " WHERE " + " AND ".join(clauses), params
    return "", params


def get_usage_summary(
    *,
    db_path: Path = DEFAULT_USAGE_DB,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
    since: Optional[str] = None,
) -> UsageSummary:
    """Return aggregate usage stats."""
    conn = _connect(db_path)
    try:
        where_sql, params = _where_clauses(session_id=session_id, request_id=request_id, since=since)
        row = conn.execute(
            f"""
            SELECT COUNT(*),
                   COALESCE(SUM(input_tokens), 0),
                   COALESCE(SUM(output_tokens), 0),
                   COALESCE(SUM(total_tokens), 0)
            FROM token_usage
            {where_sql}
            """,
            params,
        ).fetchone()
        return UsageSummary(
            calls=int(row[0] or 0),
            input_tokens=int(row[1] or 0),
            output_tokens=int(row[2] or 0),
            total_tokens=int(row[3] or 0),
        )
    finally:
        conn.close()


def get_usage_by_model(
    *,
    db_path: Path = DEFAULT_USAGE_DB,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
    since: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return aggregated usage totals grouped by model."""
    conn = _connect(db_path)
    try:
        where_sql, params = _where_clauses(session_id=session_id, request_id=request_id, since=since)
        rows = conn.execute(
            f"""
            SELECT model,
                   COUNT(*) AS calls,
                   COALESCE(SUM(total_tokens), 0) AS total_tokens
            FROM token_usage
            {where_sql}
            GROUP BY model
            ORDER BY total_tokens DESC
            """,
            params,
        ).fetchall()
        return [
            {"model": row[0], "calls": int(row[1] or 0), "total_tokens": int(row[2] or 0)}
            for row in rows
        ]
    finally:
        conn.close()


def get_recent_usage(
    *,
    db_path: Path = DEFAULT_USAGE_DB,
    limit: int = 200,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return the most recent usage rows."""
    conn = _connect(db_path)
    try:
        where_sql, params = _where_clauses(session_id=session_id, request_id=request_id, since=None)
        params.append(int(limit))
        rows = conn.execute(
            f"""
            SELECT created_at, model, operation, input_tokens, output_tokens, total_tokens, session_id, request_id
            FROM token_usage
            {where_sql}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [
            {
                "created_at": row[0],
                "model": row[1],
                "operation": row[2],
                "input_tokens": int(row[3] or 0),
                "output_tokens": int(row[4] or 0),
                "total_tokens": int(row[5] or 0),
                "session_id": row[6],
                "request_id": row[7],
            }
            for row in rows
        ]
    finally:
        conn.close()
