from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import backoff
import requests
import psycopg2

CACHE_TTL = 60 * 60 * 24 * 30  # 30 days


def init_crossref_cache_db(db_url: str):
    """Initialize the Crossref cache table.

    Args:
        db_url: Postgres database URL.

    Returns:
        connection: Open psycopg2 connection.
    """
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS crossref_cache (
            doi TEXT PRIMARY KEY,
            fetched_at BIGINT,
            response TEXT
        )
        """
    )
    conn.commit()
    return conn


def get_cached(conn, doi: str) -> Optional[str]:
    """Return a cached Crossref response for a DOI if fresh.

    Args:
        conn: Open database connection.
        doi: DOI to look up.

    Returns:
        Optional[str]: Cached response text if present and not expired.
    """
    cur = conn.cursor()
    cur.execute("SELECT fetched_at, response FROM crossref_cache WHERE doi = %s", (doi,))
    row = cur.fetchone()
    if not row:
        return None
    fetched_at, response = row
    if time.time() - fetched_at > CACHE_TTL:
        return None
    return response


def set_cached(conn, doi: str, response_text: str) -> None:
    """Store a Crossref response in the cache.

    Args:
        conn: Open database connection.
        doi: DOI for the cached response.
        response_text: Raw response body.
    """
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO crossref_cache (doi, fetched_at, response) VALUES (%s, %s, %s) ON CONFLICT (doi) DO UPDATE SET fetched_at = EXCLUDED.fetched_at, response = EXCLUDED.response",
        (doi, int(time.time()), response_text),
    )
    conn.commit()


@backoff.on_exception(backoff.expo, (requests.RequestException,), max_tries=5)
def fetch_crossref_raw(doi: str, timeout: int = 10) -> Optional[str]:
    """Fetch a Crossref response for a DOI with retries.

    Args:
        doi: DOI to fetch.
        timeout: Request timeout in seconds.

    Returns:
        Optional[str]: Raw response text.
    """
    url = f"https://api.crossref.org/works/{requests.utils.requote_uri(doi)}"
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Ragonometrics/0.1"})
    resp.raise_for_status()
    return resp.text


def fetch_crossref_with_cache(doi: str, db_url: str) -> Optional[str]:
    """Fetch Crossref data for a DOI using a persistent cache.

    Args:
        doi: DOI to fetch.
        db_url: Postgres database URL for cache storage.

    Returns:
        Optional[str]: Raw response text if available.
    """
    conn = init_crossref_cache_db(db_url)
    try:
        cached = get_cached(conn, doi)
        if cached:
            return cached
        raw = fetch_crossref_raw(doi)
        if raw:
            set_cached(conn, doi, raw)
            return raw
    finally:
        conn.close()
    return None

