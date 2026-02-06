# Ragonometrics Architecture: Decisions and Rationale

This document summarizes the current Ragonometrics architecture, design tradeoffs, and operational guidance.

Overview
--------
Ragonometrics ingests PDFs, extracts per-page text for provenance, chunks with overlap, embeds chunks, indexes embeddings in FAISS, and serves retrieval + LLM summaries via CLI and a Streamlit UI. The system enriches papers with external metadata (Semantic Scholar, CitEc, Crossref) when available. DOI metadata can be retrieved from Crossref and cached. The system is designed to be reproducible, auditable, and scalable from local runs to a Postgres-backed deployment.

Architecture Diagram
--------------------
```mermaid
flowchart LR
  subgraph Ingestion
    PDFs[PDFs] -->|pdftotext/pdfinfo| Extract[Text + Metadata]
    Extract -->|chunk + overlap| Chunks[Provenance Chunks]
  end

  subgraph Enrichment
    Chunks -->|DOI/RePEc extract| Ids[Identifiers]
    Ids -->|Crossref| Crossref[Crossref Cache]
    Ids -->|Semantic Scholar| S2[Semantic Scholar Cache]
    Ids -->|CitEc| CitEc[CitEc Cache]
  end

  subgraph Indexing
    Chunks -->|embeddings| Embeds[Embeddings]
    Embeds -->|FAISS| Faiss[FAISS Index]
    Chunks -->|metadata| PG[(Postgres)]
  end

  subgraph Retrieval
    Query[User Query] -->|expand/rerank| Retrieve[Hybrid Retrieval]
    Retrieve --> Context[Context Chunks]
    Context --> LLM[LLM Answer]
    S2 --> LLM
    CitEc --> LLM
  end

  subgraph Interfaces
    CLI[CLI] --> Query
    UI[Streamlit UI] --> Query
  end

  PG --> Retrieve
  Faiss --> Retrieve
```

Key Components
--------------
- Config and prompts
  - `config.toml` (optional) is the primary configuration surface with env-var overrides.
  - Centralized prompts live in `ragonometrics/core/prompts.py`.
- Package layout (logical groupings)
  - `ragonometrics/core/`: settings, ingestion, extraction, core prompts, logging.
  - `ragonometrics/pipeline/`: LLM call wrapper, query cache, token usage accounting.
  - `ragonometrics/indexing/`: FAISS indexing, Postgres metadata, hybrid retrieval, migrations.
  - `ragonometrics/integrations/`: Crossref cache, Semantic Scholar, CitEc, Redis/RQ jobs.
  - `ragonometrics/ui/`: Streamlit app.
  - `ragonometrics/eval/`: eval + benchmark tooling.
- PDF extraction and preprocessing
  - `pdftotext` + `pdfinfo` (Poppler) with OCR fallback.
  - Per-page extraction supports provenance (page + word offsets).
  - Optional section-aware chunking (title/abstract/introduction/methods/results) via `SECTION_AWARE_CHUNKING`.
- Embeddings and retrieval
  - OpenAI embeddings via `embed_texts`.
  - Hybrid BM25 + FAISS retrieval when `DATABASE_URL` is configured.
  - Optional query expansion (`QUERY_EXPANSION`) and LLM reranking (`RERANKER_MODEL`, `RERANK_TOP_N`).
- Indexing
  - FAISS `IndexFlatIP` with normalized vectors.
  - Index versions are tracked with `index_id` and a sidecar JSON next to FAISS artifacts.
  - Postgres metadata stores vectors, index shards, and index version rows.
  - Idempotent indexing based on a deterministic key (same corpus + params).
- UI and CLI
  - Streamlit UI (`ragonometrics/ui/streamlit_app.py`) provides Chat, DOI Network, and Usage tabs.
  - External metadata (Semantic Scholar + CitEc) is shown in a UI expander and injected into prompts.
  - Console entrypoints: `ragonometrics index | query | ui | benchmark`.
- Caching
  - Crossref responses cached in Postgres when a cache DB is provided.
  - Semantic Scholar metadata cached in SQLite (`ragonometrics_semantic_scholar.sqlite`).
  - CitEc metadata cached in SQLite (`ragonometrics_citec.sqlite`).
  - Query/answer cache stored in local SQLite (`ragonometrics_query_cache.sqlite`).
  - Token usage captured in SQLite (`ragonometrics_usage.sqlite`).

Data and Metadata Stores
------------------------
- Postgres (`DATABASE_URL`):
  - `vectors`, `index_shards`, `index_versions`, `pipeline_runs`, and failure logs.
- Local artifacts:
  - FAISS indexes in `vectors.index` and versioned shards in `indexes/`.
  - Index version sidecar JSON next to each shard.
  - Query cache in `ragonometrics_query_cache.sqlite`.
  - Semantic Scholar cache in `ragonometrics_semantic_scholar.sqlite`.
  - CitEc cache in `ragonometrics_citec.sqlite`.
  - Usage tracking in `ragonometrics_usage.sqlite`.

Reproducibility
---------------
- A config hash is computed from effective settings (config + env overrides).
- Each indexing run writes a manifest JSON next to the index shard containing:
  - git SHA, config hash, model names, chunk params, paper list, timestamps, and artifact paths.

Retrieval Quality Controls
--------------------------
- Optional query expansion and LLM reranking to improve relevance.
- Section-aware chunking enriches chunk metadata and retrieval provenance.
- Guardrails prevent retrieval when the FAISS shard and DB `index_id` disagree.

Operational Hardening
---------------------
- Idempotent indexing: same corpus + params does not double-insert.
- Structured JSON logging for key operations.
- OpenAI and Crossref calls include retries and failure recording in Postgres.
- Semantic Scholar and CitEc calls include retries and local caching.

Evaluation
----------
- `ragonometrics/eval/eval.py` provides retrieval metrics (recall@k, MRR) and answer proxies
  (citation coverage, hallucination proxy, self-consistency).
- Golden-set format supports curated Q/A and expected citations.

Queueing
--------
- Redis + RQ (`ragonometrics/integrations/rq_queue.py`) for async indexing jobs.

Benchmarks
----------
- `ragonometrics/eval/benchmark.py` and `tools/benchmark.py` measure indexing, chunking, and retrieval timing.

Entrypoints
-----------
- `ragonometrics index` builds FAISS indexes.
- `ragonometrics query` runs a question against a paper.
- `ragonometrics ui` launches the Streamlit UI.
- `ragonometrics benchmark` runs the benchmark suite.

