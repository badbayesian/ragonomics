from pathlib import Path

from ragonometrics.indexing.manifest import build_run_manifest, write_run_manifest
from ragonometrics.core.main import Settings


def test_write_run_manifest(tmp_path):
    settings = Settings(
        papers_dir=Path(tmp_path / "papers"),
        max_papers=1,
        max_words=1000,
        chunk_words=200,
        chunk_overlap=20,
        top_k=3,
        batch_size=32,
        embedding_model="emb",
        chat_model="chat",
        config_path=Path(tmp_path / "config.toml"),
        config_hash="abc123",
    )
    index_path = tmp_path / "vectors.index"
    shard_path = tmp_path / "indexes" / "vectors-123.index"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.write_text("", encoding="utf-8")

    manifest = build_run_manifest(
        settings=settings,
        paper_paths=[Path("a.pdf"), Path("b.pdf")],
        index_path=index_path,
        shard_path=shard_path,
        pipeline_run_id=42,
    )
    manifest_path = write_run_manifest(shard_path, manifest)

    assert manifest_path.exists()
    data = manifest_path.read_text(encoding="utf-8")
    assert '"pipeline_run_id": 42' in data
    assert '"embedding_model": "emb"' in data
    assert '"config_hash": "abc123"' in data

