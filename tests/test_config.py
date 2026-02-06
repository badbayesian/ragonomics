import os
from pathlib import Path

from ragonometrics.core.config import build_effective_config, hash_config_dict, load_config
from ragonometrics.core import main as main_mod
from ragonometrics.core.main import load_settings


def test_load_settings_from_config_and_env(tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        """
papers_dir = "papers_cfg"
max_papers = 5
chunk_words = 123
chat_model = "model-from-config"
""",
        encoding="utf-8",
    )

    monkeypatch.setenv("MAX_PAPERS", "7")
    monkeypatch.setenv("PAPERS_DIR", str(tmp_path / "papers_env"))
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("CHAT_MODEL", raising=False)
    monkeypatch.setattr(main_mod, "DOTENV_PATH", tmp_path / "missing.env")

    settings = load_settings(config_path=cfg)
    assert settings.max_papers == 7
    assert settings.chunk_words == 123
    assert settings.chat_model == "model-from-config"
    assert settings.papers_dir == Path(tmp_path / "papers_env")
    assert settings.config_path == cfg
    assert settings.config_hash is not None
    assert len(settings.config_hash) == 64


def test_config_hash_stable(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text("max_papers = 2\n", encoding="utf-8")
    data = load_config(cfg)
    effective = build_effective_config(data, os.environ, project_root=tmp_path)
    h1 = hash_config_dict(effective)
    h2 = hash_config_dict(effective)
    assert h1 == h2

