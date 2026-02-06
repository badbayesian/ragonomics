from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from ragonometrics.config import PROJECT_ROOT
from ragonometrics.main import Settings


def get_git_sha(repo_root: Path = PROJECT_ROOT) -> Optional[str]:
    """Return the current git SHA if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
        sha = result.stdout.strip()
        return sha or None
    except Exception:
        return None


def build_run_manifest(
    *,
    settings: Settings,
    paper_paths: Iterable[Path],
    index_path: Path,
    shard_path: Path,
    pipeline_run_id: Optional[int],
) -> dict:
    """Build a run manifest for an indexing run."""
    return {
        "git_sha": get_git_sha(),
        "config_path": str(settings.config_path) if settings.config_path else None,
        "config_hash": settings.config_hash,
        "embedding_model": settings.embedding_model,
        "chat_model": settings.chat_model,
        "chunk_words": settings.chunk_words,
        "chunk_overlap": settings.chunk_overlap,
        "batch_size": settings.batch_size,
        "top_k": settings.top_k,
        "papers": [str(p) for p in paper_paths],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "index_path": str(index_path),
        "index_version_path": str(shard_path),
        "pipeline_run_id": pipeline_run_id,
    }


def write_run_manifest(shard_path: Path, manifest: dict) -> Path:
    """Write a manifest JSON file next to the index shard."""
    manifest_path = shard_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def build_index_version(
    *,
    index_id: str,
    embedding_model: str,
    chunk_words: int,
    chunk_overlap: int,
    corpus_fingerprint: str,
    created_at: Optional[str] = None,
) -> dict:
    """Build the index version payload for sidecar storage."""
    return {
        "index_id": index_id,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "embedding_model": embedding_model,
        "chunk_words": chunk_words,
        "chunk_overlap": chunk_overlap,
        "corpus_fingerprint": corpus_fingerprint,
    }


def write_index_version_sidecar(shard_path: Path, payload: dict) -> Path:
    """Write an index version sidecar JSON next to the FAISS artifact."""
    sidecar = shard_path.with_suffix(".index.version.json")
    sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return sidecar

