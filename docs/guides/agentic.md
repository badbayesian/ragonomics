# Agentic Workflow and State Management

This repo includes a lightweight, multi-step workflow runner that demonstrates:

- Orchestrating multi-step AI pipelines
- Persisting workflow state and step outputs
- Async execution through Redis + RQ
- API-driven enrichment (Semantic Scholar, CitEc, Crossref, FRED/World Bank)

Workflow Steps
--------------
1. **ingest**: load PDFs and extract normalized text
2. **enrich**: collect external metadata (Semantic Scholar + CitEc)
3. **econ_data**: optional FRED/World Bank pulls for econ context
4. **agentic**: LLM-driven sub-question planning, retrieval, and synthesis
5. **index**: build FAISS index + Postgres metadata (if configured)
6. **evaluate**: compute lightweight quality stats
7. **report**: emit a JSON report for downstream review

State Persistence
-----------------
State is persisted to SQLite in [`sqlite/ragonometrics_workflow_state.sqlite`](https://github.com/badbayesian/ragonometrics/blob/main/sqlite/ragonometrics_workflow_state.sqlite):

- `workflow_runs` stores overall run metadata and status.
- `workflow_steps` stores per-step status, timestamps, and output JSON.

Code locations:
- [`ragonometrics/pipeline/state.py`](https://github.com/badbayesian/ragonometrics/blob/main/ragonometrics/pipeline/state.py)
- [`ragonometrics/pipeline/workflow.py`](https://github.com/badbayesian/ragonometrics/blob/main/ragonometrics/pipeline/workflow.py)

Async Execution (RQ)
--------------------
Use the queue helper in [`ragonometrics/integrations/rq_queue.py`](https://github.com/badbayesian/ragonometrics/blob/main/ragonometrics/integrations/rq_queue.py):

```python
from pathlib import Path
from ragonometrics.integrations.rq_queue import enqueue_workflow

job = enqueue_workflow(Path("papers/"), redis_url="redis://localhost:6379")
print("queued job:", job.id)
```

Example input path: [`papers/`](https://github.com/badbayesian/ragonometrics/tree/main/papers).

Operational Notes
-----------------
- For long-running workflows, keep state DB on durable storage.
- Use [`reports/`](https://github.com/badbayesian/ragonometrics/tree/main/reports) for archived summaries or push to object storage.

Key Environment Variables
-------------------------
| Name | Description | Default | Type | Notes |
| --- | --- | --- | --- | --- |
| `DATABASE_URL` | Postgres URL for metadata + hybrid retrieval. | empty | string (URL) | Required for indexing/hybrid retrieval. |
| `WORKFLOW_AGENTIC` | Enable agentic step. | `0` | bool | Use `1` or `--agentic`. |
| `WORKFLOW_QUESTION` | Main agentic question. | `Summarize the paper's research question, methods, and key findings.` | string | CLI override `--question`. |
| `WORKFLOW_AGENTIC_CITATIONS` | Enable citation enrichment. | `0` | bool | Use `1` or `--agentic-citations`. |

See [Configuration](https://github.com/badbayesian/ragonometrics/blob/main/docs/configuration/configuration.md) for the full list.
