"""Preparation-phase helpers for profiling and validating a corpus before ingestion."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ragonometrics.core.io_loaders import run_pdftotext_pages


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truthy(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    text = value.strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return default


def _hash_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _pdfinfo_pages(path: Path) -> Optional[int]:
    try:
        result = subprocess.run(
            ["pdfinfo", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    for raw_line in result.stdout.splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        if key.strip().lower() == "pages":
            try:
                return int(value.strip())
            except Exception:
                return None
    return None


def _write_manifest(report_dir: Path, run_id: str, payload: Dict[str, Any]) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"prep-manifest-{run_id}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def prep_corpus(
    pdfs: List[Path],
    *,
    report_dir: Path,
    run_id: str,
) -> Dict[str, Any]:
    """Profile and validate a corpus of PDF files before ingestion.

    Environment toggles:
        PREP_HASH_FILES: Enable SHA-256 hashing per file (default True).
        PREP_VALIDATE_TEXT: Run pdftotext to estimate text coverage (default False).
        PREP_FAIL_ON_EMPTY: Fail if no PDFs or no extractable text (default False).
        PREP_VALIDATE_ONLY: If set, workflows should exit after prep (default False).
    """
    hash_files = _truthy(os.environ.get("PREP_HASH_FILES"), default=True)
    validate_text = _truthy(os.environ.get("PREP_VALIDATE_TEXT"), default=False)
    fail_on_empty = _truthy(os.environ.get("PREP_FAIL_ON_EMPTY"), default=False)

    total_size = 0
    pdfinfo_available = False
    file_rows: List[Dict[str, Any]] = []
    hash_index: Dict[str, List[str]] = {}
    empty_files: List[str] = []
    empty_text_files: List[str] = []
    text_page_total = 0
    text_word_total = 0

    for path in pdfs:
        entry: Dict[str, Any] = {"path": str(path)}
        try:
            stat = path.stat()
            entry["size_bytes"] = stat.st_size
            entry["mtime"] = stat.st_mtime
            total_size += stat.st_size
            if stat.st_size == 0:
                empty_files.append(str(path))
        except Exception as exc:
            entry["error"] = f"stat_failed: {exc}"
            file_rows.append(entry)
            continue

        pages = _pdfinfo_pages(path)
        if pages is not None:
            pdfinfo_available = True
            entry["pages"] = pages

        if hash_files:
            try:
                digest = _hash_file(path)
                entry["sha256"] = digest
                hash_index.setdefault(digest, []).append(str(path))
            except Exception as exc:
                entry["hash_error"] = str(exc)

        if validate_text:
            try:
                page_texts = run_pdftotext_pages(path)
                words_per_page = [len((p or "").split()) for p in page_texts]
                nonempty_pages = sum(1 for w in words_per_page if w > 0)
                total_words = sum(words_per_page)
                entry["nonempty_pages"] = nonempty_pages
                entry["words_total"] = total_words
                text_page_total += nonempty_pages
                text_word_total += total_words
                if total_words == 0:
                    empty_text_files.append(str(path))
            except Exception as exc:
                entry["text_error"] = str(exc)

        file_rows.append(entry)

    duplicates = [
        {"sha256": digest, "paths": paths}
        for digest, paths in hash_index.items()
        if len(paths) > 1
    ]

    corpus_hash = hashlib.sha256()
    for entry in sorted(file_rows, key=lambda e: e.get("path", "")):
        if hash_files and entry.get("sha256"):
            corpus_hash.update(str(entry["sha256"]).encode("utf-8"))
        else:
            fingerprint = f"{entry.get('path')}|{entry.get('size_bytes')}|{entry.get('mtime')}"
            corpus_hash.update(fingerprint.encode("utf-8"))
    corpus_hash_hex = corpus_hash.hexdigest()

    warnings: List[str] = []
    if not pdfs:
        warnings.append("no_pdfs_found")
    if empty_files:
        warnings.append("empty_files_detected")
    if empty_text_files:
        warnings.append("empty_text_detected")
    if duplicates:
        warnings.append("duplicate_files_detected")
    if not pdfinfo_available:
        warnings.append("pdfinfo_unavailable")

    stats: Dict[str, Any] = {
        "num_files": len(pdfs),
        "total_size_bytes": total_size,
        "corpus_hash": corpus_hash_hex,
        "hashing_enabled": hash_files,
        "validate_text": validate_text,
        "duplicate_count": len(duplicates),
        "empty_file_count": len(empty_files),
        "empty_text_count": len(empty_text_files),
        "pdfinfo_available": pdfinfo_available,
    }
    if validate_text:
        stats["text_pages_nonempty"] = text_page_total
        stats["text_words_total"] = text_word_total

    manifest = {
        "run_id": run_id,
        "created_at": _utc_now(),
        "corpus_hash": corpus_hash_hex,
        "hashing_enabled": hash_files,
        "validate_text": validate_text,
        "stats": stats,
        "warnings": warnings,
        "duplicates": duplicates,
        "empty_files": empty_files,
        "empty_text_files": empty_text_files,
        "files": file_rows,
    }
    manifest_path = _write_manifest(report_dir, run_id, manifest)

    status = "completed"
    reason = None
    if fail_on_empty and (not pdfs or (validate_text and text_word_total == 0)):
        status = "failed"
        reason = "empty_corpus"

    return {
        "status": status,
        "reason": reason,
        "stats": stats,
        "warnings": warnings,
        "manifest_path": str(manifest_path),
    }
