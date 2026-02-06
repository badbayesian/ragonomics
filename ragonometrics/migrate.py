from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Iterable, List

import psycopg2

from . import metadata


def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    """Return column names for a SQLite table.

    Args:
        conn: Open SQLite connection.
        table: Table name.

    Returns:
        List[str]: Column names in order.
    """
    cur = conn.execute(f"PRAGMA table_info('{table}')")
    return [row[1] for row in cur.fetchall()]


def migrate_vectors(sqlite_path: Path, pg_url: str) -> int:
    """Migrate vectors-like rows from SQLite into Postgres.

    Args:
        sqlite_path: Path to the SQLite database file.
        pg_url: Postgres database URL.

    Returns:
        int: Number of vector rows migrated.
    """
    if not sqlite_path.exists():
        return 0
    sconn = sqlite3.connect(str(sqlite_path))
    scol = _table_columns(sconn, "vectors")
    cur = sconn.cursor()
    # read all rows, mapping by commonly used column names
    select_cols = [c for c in ["id", "doc_id", "paper_path", "page", "start_word", "end_word", "text", "pipeline_run_id", "created_at"] if c in scol]
    if not select_cols:
        return 0
    cur.execute(f"SELECT {', '.join(select_cols)} FROM vectors")
    rows = cur.fetchall()
    pg = psycopg2.connect(pg_url)
    metadata.init_metadata_db(pg_url)
    pcur = pg.cursor()
    migrated = 0
    for r in rows:
        vals = dict(zip(select_cols, r))
        # ensure minimal fields
        if "id" not in vals or "text" not in vals:
            continue
        pcur.execute(
            """
            INSERT INTO vectors (id, doc_id, paper_path, page, start_word, end_word, text, pipeline_run_id, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                doc_id = EXCLUDED.doc_id,
                paper_path = EXCLUDED.paper_path,
                page = EXCLUDED.page,
                start_word = EXCLUDED.start_word,
                end_word = EXCLUDED.end_word,
                text = EXCLUDED.text,
                pipeline_run_id = EXCLUDED.pipeline_run_id,
                created_at = EXCLUDED.created_at
            """,
            (
                vals.get("id"),
                vals.get("doc_id"),
                vals.get("paper_path"),
                vals.get("page"),
                vals.get("start_word"),
                vals.get("end_word"),
                vals.get("text"),
                vals.get("pipeline_run_id"),
                vals.get("created_at"),
            ),
        )
        migrated += 1
    pg.commit()
    pg.close()
    sconn.close()
    return migrated


def migrate_doi_network(sqlite_path: Path, pg_url: str) -> int:
    """Migrate DOI network tables from SQLite into Postgres.

    Args:
        sqlite_path: Path to the SQLite database file.
        pg_url: Postgres database URL.

    Returns:
        int: Number of citation edges migrated.
    """
    if not sqlite_path.exists():
        return 0
    sconn = sqlite3.connect(str(sqlite_path))
    scol = _table_columns(sconn, "citations")
    cur = sconn.cursor()
    # ensure tables exist in sqlite
    try:
        cur.execute("SELECT source_doi, target_doi FROM citations")
    except sqlite3.DatabaseError:
        sconn.close()
        return 0

    rows = cur.fetchall()
    pg = psycopg2.connect(pg_url)
    # ensure doi tables exist
    metadata.init_metadata_db(pg_url)
    pcur = pg.cursor()
    migrated = 0
    # upsert dois and citations
    for src, tgt in rows:
        if not src or not tgt:
            continue
        pcur.execute("INSERT INTO dois (doi) VALUES (%s) ON CONFLICT (doi) DO NOTHING", (src,))
        pcur.execute("INSERT INTO dois (doi) VALUES (%s) ON CONFLICT (doi) DO NOTHING", (tgt,))
        pcur.execute(
            "INSERT INTO citations (source_doi, target_doi) VALUES (%s, %s) ON CONFLICT (source_doi, target_doi) DO NOTHING",
            (src, tgt),
        )
        migrated += 1
    pg.commit()
    pg.close()
    sconn.close()
    return migrated


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entry point for migrating SQLite metadata to Postgres.

    Args:
        argv: Optional argument list for testing.
    """
    p = argparse.ArgumentParser(description="Migrate SQLite vectors/doi-network to Postgres")
    p.add_argument("--sqlite-vectors", type=str, default="vectors_meta.sqlite", help="Path to SQLite file containing vectors table")
    p.add_argument("--sqlite-dois", type=str, default="doi_network.sqlite", help="Path to SQLite file containing DOI network tables")
    p.add_argument("--pg-url", type=str, required=True, help="Postgres DATABASE_URL to migrate into")
    args = p.parse_args(args=list(argv) if argv else None)

    vcount = migrate_vectors(Path(args.sqlite_vectors), args.pg_url)
    print(f"Migrated {vcount} vector rows")
    ccount = migrate_doi_network(Path(args.sqlite_dois), args.pg_url)
    print(f"Migrated {ccount} citation edges")


if __name__ == "__main__":
    main()
