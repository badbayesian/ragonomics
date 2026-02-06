Ragonometrics - RAG pipeline for economics papers
============================================

Overview
--------
Ragonometrics ingests PDFs, extracts per-page text for provenance, chunks with overlap, embeds chunks, indexes with FAISS, and serves retrieval + LLM summaries via CLI and a Streamlit UI. DOI metadata can be fetched from Crossref and cached. The system is designed to be reproducible, auditable, and scalable from local runs to a Postgres-backed deployment.

Quick Start
-----------
1. Install dependencies.

```bash
python -m pip install -e .
```

2. Install Poppler (provides `pdftotext` and `pdfinfo`). On Windows, add Poppler `bin` to PATH.

3. Set your OpenAI API key.

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

4. Place PDFs in `papers/` (e.g., `papers/example.pdf`) or set `PAPERS_DIR`.

5. Run the summarizer.

```bash
python -m ragonometrics.main
```

Console Entrypoints
-------------------
After installation, use:

```bash
ragonometrics index --papers-dir papers/ --index-path vectors.index --meta-db-url "postgres://user:pass@localhost:5432/ragonometrics"
ragonometrics query --paper papers/example.pdf --question "What is the research question?" --model gpt-5-nano
ragonometrics ui
ragonometrics benchmark --papers-dir papers/ --out bench/benchmark.csv --limit 5
```

Streamlit App
-------------
Run the local UI:

```bash
streamlit run ragonometrics/streamlit_app.py
```

Notes:
- The app includes a Chat tab and a DOI Network tab.
- Answers are concise and researcher-focused, with citations and snapshots.
- Optional page snapshots require `pdf2image` + Poppler and benefit from `pytesseract` for highlight overlays.

Configuration (Env Vars)
------------------------
- `PAPERS_DIR`: directory with PDFs. Default `PROJECT_ROOT/papers`.
- `MAX_PAPERS`: number of PDFs to process (default `3`).
- `MAX_WORDS`: maximum words per paper (default `12000`).
- `CHUNK_WORDS`: words per chunk (default `350`).
- `CHUNK_OVERLAP`: overlap between chunks (default `50`).
- `TOP_K`: number of chunks retrieved for context (default `6`).
- `EMBED_BATCH`: batch size for embeddings (default `64`).
- `EMBEDDING_MODEL`: embedding model name (default `text-embedding-3-small`).
- `OPENAI_MODEL` or `CHAT_MODEL`: chat model used for summaries (default `gpt-5-nano`).
- `LLM_MODELS`: comma-separated model list for Streamlit dropdown.
- `DATABASE_URL`: Postgres URL for metadata + hybrid retrieval.
- `BM25_WEIGHT`: blend weight for hybrid retrieval (default `0.5`).
- `FORCE_OCR`: set to any value to bypass `pdftotext` and use OCR (if available).

Components and Files
--------------------
- Prompts: `ragonometrics/prompts.py`.
- IO + preprocessing: `ragonometrics/io_loaders.py`.
- Summarizer CLI: `ragonometrics/main.py`.
- Pipeline tools (chunk summaries, citations, metadata): `ragonometrics/pipeline.py`.
- Indexing: `ragonometrics/indexer.py` (FAISS + Postgres metadata).
- Retrieval: `ragonometrics/retriever.py` (hybrid BM25 + FAISS).
- Streamlit UI: `ragonometrics/streamlit_app.py`.
- Queue worker: `ragonometrics/rq_queue.py` (Redis + RQ).
- Crossref cache: `ragonometrics/crossref_cache.py` (Postgres cache).
- Benchmarks: `ragonometrics/benchmark.py` and wrapper `tools/benchmark.py`.

Indexing and Retrieval
----------------------
- FAISS index uses `IndexFlatIP` with normalized vectors for cosine similarity.
- Index files are written to `vectors.index` and versioned in `indexes/`.
- Metadata and vector text are stored in Postgres (requires `DATABASE_URL`).

DOI Network
-----------
- `build_doi_network_from_paper()` extracts DOIs from text and queries Crossref.
- Optional persistence in Postgres with `build_and_store_doi_network()`.

Queueing
--------
- `rq_queue.py` enqueues indexing jobs.
- Use Redis + RQ worker for async indexing.

Benchmarks
----------
- `tools/benchmark.py` runs indexing and chunking benchmarks against sample PDFs.
- `ragonometrics/benchmark.py` supports retrieval benchmarks and OCR forcing.

Troubleshooting
---------------
- Ensure Poppler is installed for `pdftotext` and `pdfinfo`.
- Set `OPENAI_API_KEY` before running.
- If using hybrid retrieval or indexing, configure `DATABASE_URL`.
- If PDF snapshots are not shown, install `pdf2image` and `pytesseract`.

Architecture Diagram (Mermaid)
------------------------------
```mermaid
flowchart LR
    A["PDFs in papers/"] --> B["io_loaders: pdftotext/pdfinfo/OCR"]
    B --> C["Normalize & Chunk (provenance, optional section tags)"]
    C --> D["Embeddings (OpenAI)"]
    D --> E[FAISS Index]
    C --> F["Metadata DB (Postgres)"]
    E --> G["Retriever (BM25 + FAISS)"]
    F --> G
    G --> H[Context]
    H --> I["LLM Response (call_openai)"]

    I --> J["CLI Summary / Query"]
    I --> K["Streamlit Chat Tab"]

    C --> L["DOI Extraction"]
    L --> M["Crossref API + Cache"]
    M --> N["DOI Network Tab"]

    Q["Redis + RQ"] --> E
    R["Query Cache (SQLite)"] --> I
    S["Index Version Sidecar + Manifest"] --> E
```

Architecture
------------
See `docs/architecture.md` for design rationale, scaling notes, and operational guidance.

