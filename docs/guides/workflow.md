# Workflow and CLI Usage

Console Entrypoints
-------------------
After installation, use:

```bash
ragonometrics query --paper papers/example.pdf --question "What is the research question?" --model gpt-5-nano
ragonometrics benchmark --papers-dir papers/ --out bench/benchmark.csv --limit 5
ragonometrics workflow --papers papers/
ragonometrics workflow --papers papers/ --agentic --question "What is the key contribution?"
ragonometrics workflow --papers papers/ --agentic --agentic-citations --question "What is the key contribution?"
```

Example inputs live in [`papers/`](https://github.com/badbayesian/ragonometrics/tree/main/papers).

Commands that require Docker (Postgres):
```bash
ragonometrics index --papers-dir papers/ --index-path vectors.index --meta-db-url "postgres://user:pass@localhost:5432/ragonometrics"
```

Index artifacts are written to [`vectors.index`](https://github.com/badbayesian/ragonometrics/blob/main/vectors.index) and versioned in [`indexes/`](https://github.com/badbayesian/ragonometrics/tree/main/indexes).

Workflow Notes
--------------
- `--papers` accepts a directory or a single PDF file.
- Use `--report-question-set structured|agentic|both|none` to control report questions.
- For faster runs, reduce `TOP_K` or use `report-question-set agentic`.
- Each run writes a prep manifest to [`reports/prep-manifest-<run_id>.json`](https://github.com/badbayesian/ragonometrics/tree/main/reports) for corpus profiling.
