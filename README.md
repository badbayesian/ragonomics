Ragonometrics - RAG pipeline for economics papers
============================================

Overview
--------
Ragonometrics ingests PDFs, extracts per-page text for provenance, chunks with overlap, embeds chunks, indexes with FAISS, and serves retrieval + LLM summaries via CLI and a Streamlit UI. External metadata is enriched via Semantic Scholar and CitEc when available, and DOI metadata can be fetched from Crossref and cached. The system is designed to be reproducible, auditable, and scalable from local runs to a Postgres-backed deployment.

This repo is a combination of coding + vibe coding.

Quick Start
-----------
1. Install dependencies in your enviroment.

```bash
python -m pip install -e .
```

2. Install Poppler (provides `pdftotext` and `pdfinfo`). On Windows, add Poppler `bin` to PATH.

3. Set your OpenAI API key in your environment (see [Configuration](https://github.com/badbayesian/ragonometrics/blob/main/docs/configuration/configuration.md)).

4. Place PDFs in [`papers/`](https://github.com/badbayesian/ragonometrics/tree/main/papers) (e.g., `papers/your.pdf`) or set `PAPERS_DIR`.

5. Run the summarizer.

```bash
python -m ragonometrics.core.main
```

Docs
-------------
Docs root: [docs/](https://github.com/badbayesian/ragonometrics/tree/main/docs)
- [Architecture](https://github.com/badbayesian/ragonometrics/blob/main/docs/architecture/architecture.md): System design, tradeoffs, and reproducibility.
- [Workflow Architecture](https://github.com/badbayesian/ragonometrics/blob/main/docs/architecture/workflow_architecture.md): Workflow steps, artifacts, and state.
- [Configuration](https://github.com/badbayesian/ragonometrics/blob/main/docs/configuration/configuration.md): [`config.toml`](https://github.com/badbayesian/ragonometrics/blob/main/config.toml) + env override reference.
- [Workflow and CLI](https://github.com/badbayesian/ragonometrics/blob/main/docs/guides/workflow.md): CLI commands and workflow usage.
- [Docker](https://github.com/badbayesian/ragonometrics/blob/main/docs/deployment/docker.md): Compose usage and container notes.
- [Indexing and Retrieval](https://github.com/badbayesian/ragonometrics/blob/main/docs/components/indexing.md): FAISS, Postgres metadata, DOI network, queueing.
- [Streamlit UI](https://github.com/badbayesian/ragonometrics/blob/main/docs/guides/ui.md): UI launch and behavior.
- [Troubleshooting](https://github.com/badbayesian/ragonometrics/blob/main/docs/guides/troubleshooting.md): Common setup and runtime fixes.
- [Agentic workflow](https://github.com/badbayesian/ragonometrics/blob/main/docs/guides/agentic.md): Agentic mode overview and notes.
- [Econ schema](https://github.com/badbayesian/ragonometrics/blob/main/docs/data/econ_schema.md): Time-series schema and econ data notes.
- [Cloud deployment](https://github.com/badbayesian/ragonometrics/blob/main/docs/deployment/cloud.md): Deployment scaffolding and guidance.
- [Onboarding](https://github.com/badbayesian/ragonometrics/blob/main/docs/guides/onboarding.md): Getting started for contributors.
- [Contributing](https://github.com/badbayesian/ragonometrics/blob/main/docs/guides/contributing.md): Contribution guidelines.
- [ADRs](https://github.com/badbayesian/ragonometrics/tree/main/docs/adr): Architecture decision records.
