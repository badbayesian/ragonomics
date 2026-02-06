"""Semantic Scholar API integration and lightweight caching for paper metadata."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_PATH = PROJECT_ROOT / "ragonometrics_semantic_scholar.sqlite"
DEFAULT_FIELDS = ",".join(
    [
        "title",
        "abstract",
        "authors",
        "year",
        "venue",
        "doi",
        "url",
        "citationCount",
        "referenceCount",
        "influentialCitationCount",
        "externalIds",
    ]
)


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS semantic_scholar_cache (
            cache_key TEXT PRIMARY KEY,
            paper_id TEXT,
            query TEXT,
            response TEXT,
            fetched_at INTEGER
        )
        """
    )
    conn.commit()
    return conn


def _cache_ttl_seconds() -> int:
    try:
        days = int(os.environ.get("SEMANTIC_SCHOLAR_CACHE_TTL_DAYS", "30"))
    except Exception:
        days = 30
    return max(days, 1) * 24 * 60 * 60


def make_cache_key(
    *,
    doi: Optional[str],
    title: Optional[str],
    author: Optional[str],
    year: Optional[int],
) -> str:
    payload = f"{(doi or '').lower()}||{(title or '').lower()}||{(author or '').lower()}||{year or ''}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_cached_metadata(db_path: Path, cache_key: str) -> Optional[Dict[str, Any]]:
    conn = _connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT fetched_at, response FROM semantic_scholar_cache WHERE cache_key = ?", (cache_key,))
        row = cur.fetchone()
        if not row:
            return None
        fetched_at, response = row
        if time.time() - int(fetched_at) > _cache_ttl_seconds():
            return None
        try:
            return json.loads(response)
        except Exception:
            return None
    finally:
        conn.close()


def set_cached_metadata(
    db_path: Path,
    *,
    cache_key: str,
    paper_id: Optional[str],
    query: Optional[str],
    response: Dict[str, Any],
) -> None:
    conn = _connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO semantic_scholar_cache
            (cache_key, paper_id, query, response, fetched_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                cache_key,
                paper_id,
                query,
                json.dumps(response, ensure_ascii=False),
                int(time.time()),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _headers() -> Dict[str, str]:
    headers = {"User-Agent": "Ragonometrics/0.1"}
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or os.environ.get("S2_API_KEY")
    if key:
        headers["x-api-key"] = key
    return headers


def _request_json(url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 10) -> Optional[Dict[str, Any]]:
    max_retries = int(os.environ.get("SEMANTIC_SCHOLAR_MAX_RETRIES", "2"))
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, params=params, headers=_headers(), timeout=timeout)
            if resp.status_code == 404:
                return None
            if resp.status_code == 429:
                raise requests.RequestException("rate_limited")
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt >= max_retries:
                return None
            try:
                time.sleep(0.5 * (attempt + 1))
            except Exception:
                pass
    return None


def fetch_paper_by_id(paper_id: str, fields: str = DEFAULT_FIELDS, timeout: int = 10) -> Optional[Dict[str, Any]]:
    if not paper_id:
        return None
    encoded = requests.utils.quote(paper_id, safe="")
    url = f"https://api.semanticscholar.org/graph/v1/paper/{encoded}"
    return _request_json(url, params={"fields": fields}, timeout=timeout)


def search_paper(query: str, fields: str = DEFAULT_FIELDS, limit: int = 1, timeout: int = 10) -> Optional[Dict[str, Any]]:
    if not query:
        return None
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    data = _request_json(url, params={"query": query, "limit": limit, "fields": fields}, timeout=timeout)
    if not data:
        return None
    items = data.get("data") or []
    if not items:
        return None
    if isinstance(items, list):
        return items[0]
    return None


def fetch_semantic_scholar_metadata(
    *,
    title: Optional[str],
    author: Optional[str],
    year: Optional[int] = None,
    doi: Optional[str] = None,
    cache_path: Path = DEFAULT_CACHE_PATH,
    timeout: int = 10,
) -> Optional[Dict[str, Any]]:
    """Fetch Semantic Scholar metadata for a paper, using DOI when possible."""
    if os.environ.get("SEMANTIC_SCHOLAR_DISABLE", "").strip() == "1":
        return None

    cache_key = make_cache_key(doi=doi, title=title, author=author, year=year)
    cached = get_cached_metadata(cache_path, cache_key)
    if cached:
        return cached

    data = None
    paper_id = None

    if doi:
        for candidate in (doi, f"DOI:{doi}"):
            data = fetch_paper_by_id(candidate, timeout=timeout)
            if data:
                paper_id = candidate
                break

    if not data and title:
        query_parts = [title]
        if author:
            query_parts.append(author)
        if year:
            query_parts.append(str(year))
        query = " ".join(query_parts)
        data = search_paper(query, timeout=timeout)
        paper_id = data.get("paperId") if isinstance(data, dict) else None

    if data:
        set_cached_metadata(
            cache_path,
            cache_key=cache_key,
            paper_id=paper_id,
            query=title or "",
            response=data,
        )
    return data


def format_semantic_scholar_context(
    meta: Optional[Dict[str, Any]],
    *,
    max_abstract_chars: int = 1200,
    max_authors: int = 8,
) -> str:
    """Format Semantic Scholar metadata into a compact context block."""
    if not meta:
        return ""
    lines = ["Semantic Scholar Metadata:"]

    title = meta.get("title")
    if title:
        lines.append(f"Title: {title}")

    authors = meta.get("authors") or []
    names = []
    for author in authors:
        if isinstance(author, dict) and author.get("name"):
            names.append(author["name"])
    if names:
        suffix = " et al." if len(names) > max_authors else ""
        lines.append(f"Authors: {', '.join(names[:max_authors])}{suffix}")

    year = meta.get("year")
    if year:
        lines.append(f"Year: {year}")

    venue = meta.get("venue")
    if venue:
        lines.append(f"Venue: {venue}")

    doi = meta.get("doi") or (meta.get("externalIds") or {}).get("DOI")
    if doi:
        lines.append(f"DOI: {doi}")

    url = meta.get("url")
    if url:
        lines.append(f"URL: {url}")

    citation_count = meta.get("citationCount")
    if citation_count is not None:
        lines.append(f"Citation Count: {citation_count}")

    reference_count = meta.get("referenceCount")
    if reference_count is not None:
        lines.append(f"Reference Count: {reference_count}")

    influential = meta.get("influentialCitationCount")
    if influential is not None:
        lines.append(f"Influential Citations: {influential}")

    abstract = meta.get("abstract")
    if abstract:
        if len(abstract) > max_abstract_chars:
            abstract = abstract[: max_abstract_chars - 3].rstrip() + "..."
        lines.append(f"Abstract: {abstract}")

    if len(lines) <= 1:
        return ""
    return "\n".join(lines)
