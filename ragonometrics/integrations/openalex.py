"""OpenAlex API integration and lightweight caching for paper metadata."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from ragonometrics.core.config import SQLITE_DIR


DEFAULT_CACHE_PATH = SQLITE_DIR / "ragonometrics_openalex.sqlite"
DEFAULT_SELECT = ",".join(
    [
        "id",
        "display_name",
        "publication_year",
        "primary_location",
        "host_venue",
        "doi",
        "authorships",
        "cited_by_count",
        "referenced_works_count",
        "abstract_inverted_index",
    ]
)


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS openalex_cache (
            cache_key TEXT PRIMARY KEY,
            work_id TEXT,
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
        days = int(os.environ.get("OPENALEX_CACHE_TTL_DAYS", "30"))
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
        cur.execute("SELECT fetched_at, response FROM openalex_cache WHERE cache_key = ?", (cache_key,))
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
    work_id: Optional[str],
    query: Optional[str],
    response: Dict[str, Any],
) -> None:
    conn = _connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO openalex_cache
            (cache_key, work_id, query, response, fetched_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                cache_key,
                work_id,
                query,
                json.dumps(response, ensure_ascii=False),
                int(time.time()),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _request_json(url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 10) -> Optional[Dict[str, Any]]:
    max_retries = int(os.environ.get("OPENALEX_MAX_RETRIES", "2"))
    payload = dict(params or {})
    api_key = os.environ.get("OPENALEX_API_KEY", "").strip()
    if api_key:
        payload["api_key"] = api_key
    mailto = (os.environ.get("OPENALEX_MAILTO") or os.environ.get("OPENALEX_EMAIL") or "").strip()
    if mailto:
        payload["mailto"] = mailto
    headers = {"User-Agent": "Ragonometrics/0.1"}
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, params=payload, headers=headers, timeout=timeout)
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


def fetch_work_by_doi(doi: str, select: str = DEFAULT_SELECT, timeout: int = 10) -> Optional[Dict[str, Any]]:
    if not doi:
        return None
    doi_url = doi.strip()
    if not doi_url.lower().startswith("http"):
        doi_url = f"https://doi.org/{doi_url}"
    encoded = requests.utils.quote(doi_url, safe=":/")
    url = f"https://api.openalex.org/works/{encoded}"
    return _request_json(url, params={"select": select}, timeout=timeout)


def search_work(query: str, select: str = DEFAULT_SELECT, limit: int = 1, timeout: int = 10) -> Optional[Dict[str, Any]]:
    if not query:
        return None
    url = "https://api.openalex.org/works"
    data = _request_json(
        url,
        params={"search": query, "per-page": limit, "select": select},
        timeout=timeout,
    )
    if not data:
        return None
    items = data.get("results") or []
    if not items:
        return None
    if isinstance(items, list):
        return items[0]
    return None


def fetch_openalex_metadata(
    *,
    title: Optional[str],
    author: Optional[str],
    year: Optional[int] = None,
    doi: Optional[str] = None,
    cache_path: Path = DEFAULT_CACHE_PATH,
    timeout: int = 10,
) -> Optional[Dict[str, Any]]:
    """Fetch OpenAlex metadata for a paper, using DOI when possible."""
    if os.environ.get("OPENALEX_DISABLE", "").strip() == "1":
        return None

    cache_key = make_cache_key(doi=doi, title=title, author=author, year=year)
    cached = get_cached_metadata(cache_path, cache_key)
    if cached:
        return cached

    data = None
    work_id = None

    if doi:
        data = fetch_work_by_doi(doi, timeout=timeout)
        if data:
            work_id = data.get("id")

    if not data and title:
        query_parts = [title]
        if author:
            query_parts.append(author)
        if year:
            query_parts.append(str(year))
        query = " ".join(query_parts)
        data = search_work(query, timeout=timeout)
        work_id = data.get("id") if isinstance(data, dict) else None

    if data:
        set_cached_metadata(
            cache_path,
            cache_key=cache_key,
            work_id=work_id,
            query=title or "",
            response=data,
        )
    return data


def _abstract_from_inverted_index(inv: Optional[Dict[str, Any]]) -> str:
    if not inv or not isinstance(inv, dict):
        return ""
    positions = []
    for vals in inv.values():
        if isinstance(vals, list):
            positions.extend([v for v in vals if isinstance(v, int)])
    if not positions:
        return ""
    max_pos = max(positions)
    words: List[str] = [""] * (max_pos + 1)
    for token, offsets in inv.items():
        if not isinstance(offsets, list):
            continue
        for pos in offsets:
            if isinstance(pos, int) and 0 <= pos <= max_pos:
                words[pos] = token
    return " ".join([w for w in words if w])


def _get_venue(meta: Dict[str, Any]) -> Optional[str]:
    primary = meta.get("primary_location") or {}
    source = primary.get("source") or {}
    venue = source.get("display_name")
    if venue:
        return venue
    host = meta.get("host_venue") or {}
    return host.get("display_name")


def format_openalex_context(
    meta: Optional[Dict[str, Any]],
    *,
    max_abstract_chars: int = 1200,
    max_authors: int = 8,
) -> str:
    """Format OpenAlex metadata into a compact context block."""
    if not meta:
        return ""
    lines = ["OpenAlex Metadata:"]

    title = meta.get("display_name") or meta.get("title")
    if title:
        lines.append(f"Title: {title}")

    authors = meta.get("authorships") or []
    names: List[str] = []
    for author in authors:
        if isinstance(author, dict):
            author_obj = author.get("author") or {}
            name = author_obj.get("display_name")
            if name:
                names.append(name)
    if names:
        suffix = " et al." if len(names) > max_authors else ""
        lines.append(f"Authors: {', '.join(names[:max_authors])}{suffix}")

    year = meta.get("publication_year")
    if year:
        lines.append(f"Year: {year}")

    venue = _get_venue(meta)
    if venue:
        lines.append(f"Venue: {venue}")

    doi = meta.get("doi")
    if doi:
        lines.append(f"DOI: {doi}")

    url = meta.get("id")
    landing = (meta.get("primary_location") or {}).get("landing_page_url")
    if landing:
        lines.append(f"URL: {landing}")
    elif url:
        lines.append(f"URL: {url}")

    citation_count = meta.get("cited_by_count")
    if citation_count is not None:
        lines.append(f"Citation Count: {citation_count}")

    reference_count = meta.get("referenced_works_count")
    if reference_count is None and isinstance(meta.get("referenced_works"), list):
        reference_count = len(meta.get("referenced_works"))
    if reference_count is not None:
        lines.append(f"Reference Count: {reference_count}")

    abstract = _abstract_from_inverted_index(meta.get("abstract_inverted_index"))
    if abstract:
        if len(abstract) > max_abstract_chars:
            abstract = abstract[: max_abstract_chars - 3].rstrip() + "..."
        lines.append(f"Abstract: {abstract}")

    if len(lines) <= 1:
        return ""
    return "\n".join(lines)
