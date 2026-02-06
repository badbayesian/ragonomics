from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping

try:
    import tomllib
except ImportError:  # pragma: no cover - fallback for older Python
    import tomli as tomllib  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.toml"


def load_config(path: Path | None) -> Dict[str, Any]:
    """Load a TOML config file and return the ragonometrics section or top-level dict."""
    if not path or not path.exists():
        return {}
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "ragonometrics" in data and isinstance(data["ragonometrics"], dict):
        return data["ragonometrics"]
    return data or {}


def hash_config_dict(config: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 hash of a config mapping."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _env_or_config(env: Mapping[str, str], config: Mapping[str, Any], env_key: str, config_key: str, default: Any) -> Any:
    if env_key in env and env[env_key] != "":
        return env[env_key]
    if config_key in config:
        return config[config_key]
    return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def build_effective_config(
    config: Mapping[str, Any],
    env: Mapping[str, str],
    *,
    project_root: Path,
) -> Dict[str, Any]:
    """Build the effective config with env overrides applied."""
    papers_dir_val = _env_or_config(env, config, "PAPERS_DIR", "papers_dir", project_root / "papers")
    papers_dir = Path(papers_dir_val)
    if not papers_dir.is_absolute():
        papers_dir = project_root / papers_dir

    effective = {
        "papers_dir": str(papers_dir),
        "max_papers": _coerce_int(_env_or_config(env, config, "MAX_PAPERS", "max_papers", 3), 3),
        "max_words": _coerce_int(_env_or_config(env, config, "MAX_WORDS", "max_words", 12000), 12000),
        "chunk_words": _coerce_int(_env_or_config(env, config, "CHUNK_WORDS", "chunk_words", 350), 350),
        "chunk_overlap": _coerce_int(_env_or_config(env, config, "CHUNK_OVERLAP", "chunk_overlap", 50), 50),
        "top_k": _coerce_int(_env_or_config(env, config, "TOP_K", "top_k", 6), 6),
        "batch_size": _coerce_int(_env_or_config(env, config, "EMBED_BATCH", "batch_size", 64), 64),
        "embedding_model": _env_or_config(env, config, "EMBEDDING_MODEL", "embedding_model", "text-embedding-3-small"),
        "chat_model": _env_or_config(env, config, "OPENAI_MODEL", "chat_model", None)
        or _env_or_config(env, config, "CHAT_MODEL", "chat_model", "gpt-5-nano"),
    }
    return effective

