from __future__ import annotations

from typing import List, Tuple, Dict
import json
import os
import hashlib
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi
import psycopg2


def _load_index_sidecar(path: str) -> Dict:
    sidecar = Path(path).with_suffix(".index.version.json")
    if not sidecar.exists():
        raise RuntimeError(f"Index sidecar not found: {sidecar}")
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to read index sidecar: {sidecar}") from exc


def _verify_index_version(path: str, db_index_id: str | None) -> None:
    if os.environ.get("ALLOW_UNVERIFIED_INDEX"):
        return
    sidecar = _load_index_sidecar(path)
    sidecar_id = sidecar.get("index_id")
    if not sidecar_id:
        raise RuntimeError(f"Index sidecar missing index_id: {path}")
    if not db_index_id:
        raise RuntimeError(f"DB index_id missing for shard: {path}")
    if sidecar_id != db_index_id:
        raise RuntimeError(f"Index id mismatch for shard {path}: sidecar={sidecar_id} db={db_index_id}")


def _load_active_indexes(db_url: str) -> List[Tuple[str, faiss.Index]]:
    """Load active FAISS indexes from metadata.

    Args:
        db_url: Postgres database URL.

    Returns:
        List[Tuple[str, faiss.Index]]: (shard_name, index) pairs.
    """
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute("SELECT shard_name, path, index_id FROM index_shards WHERE is_active = TRUE ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()
    res = []
    for shard_name, path, index_id in rows:
        _verify_index_version(path, index_id)
        idx = faiss.read_index(path)
        res.append((shard_name, idx))
    return res


def _load_texts_for_shards(db_url: str) -> Tuple[List[str], List[int]]:
    """Load vector texts and ids from Postgres.

    Args:
        db_url: Postgres database URL.

    Returns:
        Tuple[List[str], List[int]]: Texts and corresponding vector ids.
    """
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute("SELECT id, text FROM vectors ORDER BY id")
    rows = cur.fetchall()
    conn.close()
    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    return texts, ids


def hybrid_search(query: str, client: OpenAI, db_url: str, top_k: int = 6, bm25_weight: float = 0.5) -> List[Tuple[int, float]]:
    """Perform hybrid BM25 + embedding search over stored vectors.

    Args:
        query: Search query string.
        client: OpenAI client for embeddings.
        db_url: Postgres database URL.
        top_k: Number of results to return.
        bm25_weight: Weight for BM25 score in the hybrid blend.

    Returns:
        List[Tuple[int, float]]: (vector_id, score) results.
    """
    # 1. BM25 over stored texts
    texts, ids = _load_texts_for_shards(db_url)
    if not texts:
        return []
    tokenized = [t.split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    q_tokens = query.split()
    bm25_scores = bm25.get_scores(q_tokens)

    # 2. embedding search via FAISS across active indexes (concatenate results)
    emb = client.embeddings.create(model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"), input=[query]).data[0].embedding
    vec = np.array([emb], dtype=np.float32)
    # normalize
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    vec = vec / norm

    # load active index (assume single index for now)
    indexes = _load_active_indexes(db_url)
    if not indexes:
        return []
    # for simplicity, search first index and map ids directly
    _, index = indexes[0]
    D, I = index.search(vec, top_k * 5)
    emb_hits = list(zip(I[0].tolist(), D[0].tolist()))

    # combine scores: map bm25 by position in ids
    id_to_bm = {doc_id: float(bm25_scores[i]) for i, doc_id in enumerate(ids)}

    combined: Dict[int, float] = {}
    for doc_id, score in emb_hits:
        bm = id_to_bm.get(doc_id, 0.0)
        combined[doc_id] = combined.get(doc_id, 0.0) + (1.0 - bm25_weight) * float(score) + bm25_weight * bm

    # sort by combined score
    sorted_hits = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return sorted_hits
