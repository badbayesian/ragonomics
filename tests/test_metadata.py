"""Metadata DB helper tests for pipeline run creation and shard publishing."""

import importlib.util
from pathlib import Path


def _load_mod(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path).resolve())
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


metadata = _load_mod("ragonometrics/indexing/metadata.py", "ragonometrics.indexing.metadata")


def test_pipeline_run_and_shard_publish():
    db_url = "dummy"
    conn = metadata.init_metadata_db(db_url)
    run_id = metadata.create_pipeline_run(conn, git_sha="deadbeef", extractor_version="poppler-23", embedding_model="emb-1", chunk_words=256, chunk_overlap=32, normalized=True)
    assert isinstance(run_id, int)

    shard_id = metadata.publish_shard(conn, "shard-test", "/tmp/shard-test.index", run_id)
    assert isinstance(shard_id, int)

    shards = metadata.get_active_shards(conn)
    assert len(shards) == 1
    assert shards[0][0] == "shard-test"


def test_pipeline_run_idempotency():
    db_url = "dummy"
    conn = metadata.init_metadata_db(db_url)
    run_id1 = metadata.create_pipeline_run(
        conn,
        git_sha="deadbeef",
        extractor_version="poppler-23",
        embedding_model="emb-1",
        chunk_words=256,
        chunk_overlap=32,
        normalized=True,
        idempotency_key="same-key",
    )
    run_id2 = metadata.create_pipeline_run(
        conn,
        git_sha="deadbeef",
        extractor_version="poppler-23",
        embedding_model="emb-1",
        chunk_words=256,
        chunk_overlap=32,
        normalized=True,
        idempotency_key="same-key",
    )
    assert run_id1 == run_id2
