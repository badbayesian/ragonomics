"""Multi-step agentic workflow orchestration for ingest -> enrich -> index -> evaluate -> report."""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openai import OpenAI

from ragonometrics.core.main import (
    embed_texts,
    load_papers,
    load_settings,
    prepare_chunks_for_paper,
    top_k_context,
)
from ragonometrics.core.prompts import RESEARCHER_QA_PROMPT
from ragonometrics.indexing.indexer import build_index
from ragonometrics.pipeline import call_openai, extract_citations as llm_extract_citations
from ragonometrics.pipeline.pipeline import extract_json
from ragonometrics.pipeline.state import (
    DEFAULT_STATE_DB,
    create_workflow_run,
    record_step,
    set_workflow_status,
)
from ragonometrics.pipeline.token_usage import DEFAULT_USAGE_DB
from ragonometrics.pipeline.prep import prep_corpus
from ragonometrics.integrations.econ_data import fetch_fred_series


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_paper_paths(papers_path: Path) -> List[Path]:
    if papers_path.is_file():
        if papers_path.suffix.lower() == ".pdf":
            return [papers_path]
        return []
    if papers_path.is_dir():
        return sorted(papers_path.glob("*.pdf"))
    return []


def _progress_iter(items, desc: str):
    try:
        from tqdm import tqdm

        return tqdm(items, desc=desc)
    except Exception:
        return items


def _can_connect_db(db_url: str) -> bool:
    try:
        import psycopg2

        conn = psycopg2.connect(db_url, connect_timeout=3)
        conn.close()
        return True
    except Exception:
        return False


def _write_report(report_dir: Path, run_id: str, payload: Dict[str, Any]) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"workflow-report-{run_id}.json"
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


def _parse_subquestions(raw: str, max_items: int) -> List[str]:
    items: List[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        line = line.lstrip("-*•").strip()
        line = line.lstrip("0123456789. ").strip()
        if not line:
            continue
        if line not in items:
            items.append(line)
        if len(items) >= max_items:
            break
    return items


def _agentic_plan(client: OpenAI, question: str, *, model: str, max_items: int) -> List[str]:
    instructions = (
        "You are a research analyst. Generate a short list of sub-questions that would help "
        "answer the main question. Return one sub-question per line, no extra text."
    )
    raw = call_openai(
        client,
        model=model,
        instructions=instructions,
        user_input=f"Main question: {question}",
        max_output_tokens=200,
        usage_context="agent_plan",
    )
    items = _parse_subquestions(raw, max_items)
    if not items:
        items = [question]
    return items


def _agentic_summarize(
    client: OpenAI,
    *,
    model: str,
    question: str,
    sub_answers: List[Dict[str, str]],
) -> str:
    bullets = "\n".join([f"- {item['question']}: {item['answer']}" for item in sub_answers])
    synthesis_prompt = (
        "Synthesize a concise, researcher-grade answer based on the sub-answers below. "
        "Keep it factual and avoid speculation.\n\n"
        f"Main question: {question}\n\nSub-answers:\n{bullets}"
    )
    return call_openai(
        client,
        model=model,
        instructions=RESEARCHER_QA_PROMPT,
        user_input=synthesis_prompt,
        max_output_tokens=None,
        usage_context="agent_synthesis",
    ).strip()


def _format_citations_context(citations: List[Dict[str, Any]], max_items: int) -> str:
    if not citations:
        return ""
    items = citations[:max_items]
    lines = ["Extracted citations (preview):"]
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            lines.append(f"{idx}. {item}")
            continue
        title = item.get("title") or item.get("citation") or item.get("ref") or "Unknown title"
        year = item.get("year")
        authors = item.get("authors")
        author_txt = ""
        if isinstance(authors, list) and authors:
            author_txt = ", ".join([str(a) for a in authors[:3]])
            if len(authors) > 3:
                author_txt += " et al."
        elif isinstance(authors, str):
            author_txt = authors
        suffix = []
        if author_txt:
            suffix.append(author_txt)
        if year:
            suffix.append(str(year))
        meta = f" ({'; '.join(suffix)})" if suffix else ""
        lines.append(f"{idx}. {title}{meta}")
    return "\n".join(lines)


def _split_context_chunks(context: str) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    if not context:
        return chunks
    parts = [p for p in context.split("\n\n") if p.strip()]
    pattern = re.compile(r"^\(page\s+(?P<page>\d+)\s+words\s+(?P<start>\d+)-(?P<end>\d+)(?:\s+section\s+(?P<section>[^)]+))?\)\s*$")
    for part in parts:
        lines = part.splitlines()
        if not lines:
            continue
        meta = lines[0].strip()
        match = pattern.match(meta)
        if not match:
            continue
        text = "\n".join(lines[1:]).strip()
        chunks.append(
            {
                "page": int(match.group("page")),
                "start_word": int(match.group("start")),
                "end_word": int(match.group("end")),
                "section": (match.group("section") or "").strip() or None,
                "text": text,
            }
        )
    return chunks


def _build_report_prompt(
    *,
    question: str,
    context: str,
    anchors: List[Dict[str, Any]],
) -> str:
    anchor_lines = []
    for item in anchors:
        section = f" section {item['section']}" if item.get("section") else ""
        anchor_lines.append(
            f"- page {item['page']} words {item['start_word']}-{item['end_word']}{section}"
        )
    anchor_block = "\n".join(anchor_lines) if anchor_lines else "None"
    return (
        "Answer the question using only the provided context. Return a single JSON object with keys:\n"
        "answer (string), evidence_type (string), confidence (high|medium|low),\n"
        "citation_anchors (list of {page,start_word,end_word,section,note}),\n"
        "quote_snippet (short verbatim snippet from context, <= 200 chars),\n"
        "table_figure (string or null), data_source (string or null),\n"
        "assumption_flag (boolean or null), assumption_notes (string or null),\n"
        "related_questions (list of ids, or empty list).\n\n"
        f"Available anchors:\n{anchor_block}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )


REPORT_QUESTION_SECTIONS: List[tuple[str, str, List[str]]] = [
    (
        "A",
        "Research question / contribution",
        [
            "What is the main research question of the paper?",
            "What is the paper's primary contribution relative to the existing literature?",
            "What is the central hypothesis being tested?",
            "What are the main outcomes of interest (dependent variables)?",
            "What are the key treatment/exposure variables (independent variables)?",
            "What setting/context does the paper study (country, market, period)?",
            "What is the main mechanism proposed by the authors?",
            "What alternative mechanisms are discussed?",
            "What are the main policy implications claimed by the paper?",
            "What is the welfare interpretation (if any) of the results?",
            "What are the main limitations acknowledged by the authors?",
            "What does the paper claim is novel about its data or identification?",
        ],
    ),
    (
        "B",
        "Identification strategy / causal design",
        [
            "What is the identification strategy (in one sentence)?",
            "Is the design experimental, quasi-experimental, or observational?",
            "What is the source of exogenous variation used for identification?",
            "What is the treatment definition and timing?",
            "What is the control/comparison group definition?",
            "What is the estimating equation / baseline regression specification?",
            "What fixed effects are included (unit, time, two-way, higher dimensional)?",
            "What standard errors are used (robust, clustered; at what level)?",
            "What is the key identifying assumption (parallel trends, exclusion restriction, ignorability)?",
            "What evidence is provided to support the identifying assumption?",
            "Are there event-study or pre-trend tests? What do they show?",
            "What instruments are used (if IV)? Define instrument and first stage.",
            "What is the first-stage strength (F-stat, partial R^2, relevance evidence)?",
            "If RDD: what is the running variable and cutoff? bandwidth choice?",
            "If DiD: what is the timing variation (staggered adoption)? estimator used?",
        ],
    ),
    (
        "C",
        "Data, sample, and measurement",
        [
            "What dataset(s) are used? (name sources explicitly)",
            "What is the unit of observation (individual, household, firm, county, transaction, product)?",
            "What is the sample period and geographic coverage?",
            "What are the sample restrictions / inclusion criteria?",
            "What is the sample size (N) in the main analysis?",
            "How is the key outcome measured? Any transformations (logs, z-scores, indices)?",
            "How is treatment/exposure measured? Any constructed variables?",
            "Are there key covariates/controls? Which ones are always included?",
            "How are missing data handled (dropping, imputation, weighting)?",
            "Are weights used (survey weights, propensity weights)? How?",
            "Are data linked/merged across sources? How is linkage performed?",
            "What summary statistics are reported for main variables?",
            "Are there descriptive figures/maps that establish baseline patterns?",
        ],
    ),
    (
        "D",
        "Results, magnitudes, heterogeneity, robustness",
        [
            "What is the headline main effect estimate (sign and magnitude)?",
            "What is the preferred specification and why is it preferred?",
            "How economically meaningful is the effect (percent change, elasticity, dollars)?",
            "What are the key robustness checks and do results survive them?",
            "What placebo tests are run and what do they show?",
            "What falsification outcomes are tested (unaffected outcomes)?",
            "What heterogeneity results are reported (by income, size, baseline exposure, region)?",
            "What mechanism tests are performed and what do they imply?",
            "How sensitive are results to alternative samples/bandwidths/controls?",
            "What are the main takeaways in the conclusion (bullet summary)?",
        ],
    ),
    (
        "E",
        "Citations and related literature",
        [
            "What are the most important prior papers cited and why are they central here?",
            "Which papers does this work most directly build on or extend?",
            "Which papers are used as benchmarks or comparisons in the results?",
            "What data sources or datasets are cited and how are they used?",
            "What methodological or econometric references are cited (e.g., DiD, IV, RDD methods)?",
            "Are there any seminal or classic references the paper positions itself against?",
            "Are there citations to code, data repositories, or appendices that are essential to the claims?",
            "What gaps in the literature do the authors say these citations leave open?",
        ],
    ),
    (
        "F",
        "Replication and transparency",
        [
            "Are replication files or code provided? If so, where?",
            "Is there a pre-analysis plan or registered trial? Provide details if mentioned.",
            "Are data access constraints disclosed (restricted access, proprietary data, NDAs)?",
            "Are key steps in data cleaning and construction documented?",
            "Are robustness and sensitivity analyses fully reported or partially omitted?",
        ],
    ),
    (
        "G",
        "External validity and generalization",
        [
            "What populations or settings are most likely to generalize from this study?",
            "What populations or settings are least likely to generalize?",
            "Do the authors discuss boundary conditions or scope limits?",
            "How might the results change in different time periods or markets?",
        ],
    ),
    (
        "H",
        "Measurement validity",
        [
            "Are key variables measured directly or via proxies?",
            "What measurement error risks are acknowledged or likely?",
            "Are there validation checks for key measures?",
            "Do the authors discuss construct validity for core outcomes?",
        ],
    ),
    (
        "I",
        "Policy counterfactuals and welfare",
        [
            "What policy counterfactuals are considered or implied?",
            "What are the main welfare tradeoffs or distributional impacts discussed?",
            "Are cost-benefit or incidence analyses provided?",
            "What policy recommendations are stated or implied?",
        ],
    ),
    (
        "J",
        "Data quality and integrity",
        [
            "What missingness or attrition patterns are reported?",
            "How are outliers handled (winsorization, trimming, exclusions)?",
            "Are there data audits or validation steps described?",
            "Is there evidence of reporting bias or selective sample inclusion?",
        ],
    ),
    (
        "K",
        "Model fit and diagnostics",
        [
            "What goodness-of-fit or diagnostic metrics are reported?",
            "Are functional form choices tested (logs, levels, nonlinearities)?",
            "Are residual checks or specification tests reported?",
            "How sensitive are results to alternative specifications or estimators?",
        ],
    ),
]


def _normalize_report_question_set(value: str | None, enabled_default: bool) -> str:
    if not value:
        return "structured" if enabled_default else "none"
    text = value.strip().lower()
    if text in {"structured", "full", "fixed"}:
        return "structured"
    if text in {"agentic", "previous", "subquestions", "legacy"}:
        return "agentic"
    if text in {"both", "all"}:
        return "both"
    if text in {"none", "off", "0"}:
        return "none"
    return "structured"


def _build_report_questions() -> List[Dict[str, str]]:
    questions: List[Dict[str, str]] = []
    for section_key, section_title, items in REPORT_QUESTION_SECTIONS:
        for idx, question in enumerate(items, start=1):
            questions.append(
                {
                    "id": f"{section_key}{idx:02d}",
                    "category": f"{section_key}) {section_title}",
                    "question": question,
                }
            )
    return questions


def _report_questions_from_sub_answers(sub_answers: List[Dict[str, str]]) -> List[Dict[str, str]]:
    report: List[Dict[str, str]] = []
    for idx, item in enumerate(sub_answers, start=1):
        report.append(
            {
                "id": f"P{idx:02d}",
                "category": "P) Previous questions",
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
            }
        )
    return report


def _answer_report_question_item(
    *,
    client: OpenAI,
    model: str,
    settings,
    chunks: List[Dict[str, Any]] | List[str],
    chunk_embeddings: List[List[float]],
    citations_context: str,
    item: Dict[str, str],
) -> Dict[str, Any]:
    context = top_k_context(
        chunks,
        chunk_embeddings,
        query=item["question"],
        client=client,
        settings=settings,
    )
    if citations_context:
        context = f"{context}\n\n{citations_context}"
    parsed_chunks = _split_context_chunks(context)
    anchor_defaults = [
        {
            "page": c["page"],
            "start_word": c["start_word"],
            "end_word": c["end_word"],
            "section": c.get("section"),
            "note": None,
        }
        for c in parsed_chunks[: settings.top_k]
    ]
    prompt = _build_report_prompt(
        question=item["question"],
        context=context,
        anchors=parsed_chunks[: settings.top_k],
    )
    raw = call_openai(
        client,
        model=model,
        instructions="Return JSON only.",
        user_input=prompt,
        max_output_tokens=None,
        usage_context="agent_report_question",
    ).strip()
    parsed: Dict[str, Any] = {}
    try:
        parsed_json = extract_json(raw)
        if isinstance(parsed_json, dict):
            parsed = parsed_json
    except Exception:
        parsed = {}
    answer_text = str(parsed.get("answer") or raw).strip()
    citation_anchors = parsed.get("citation_anchors")
    if not isinstance(citation_anchors, list) or not citation_anchors:
        citation_anchors = anchor_defaults
    quote_snippet = parsed.get("quote_snippet")
    if not isinstance(quote_snippet, str) or not quote_snippet.strip():
        quote_snippet = ""
        if parsed_chunks and parsed_chunks[0].get("text"):
            quote_snippet = str(parsed_chunks[0]["text"])[:200]
    return {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "answer": answer_text,
        "evidence_type": parsed.get("evidence_type") or "unspecified",
        "confidence": parsed.get("confidence") or "medium",
        "citation_anchors": citation_anchors,
        "quote_snippet": quote_snippet,
        "table_figure": parsed.get("table_figure"),
        "data_source": parsed.get("data_source"),
        "assumption_flag": parsed.get("assumption_flag"),
        "assumption_notes": parsed.get("assumption_notes"),
        "related_questions": parsed.get("related_questions") if isinstance(parsed.get("related_questions"), list) else [],
    }


def _answer_report_questions(
    client: OpenAI,
    *,
    model: str,
    settings,
    chunks: List[Dict[str, Any]] | List[str],
    chunk_embeddings: List[List[float]],
    citations_context: str,
) -> List[Dict[str, str]]:
    questions = _build_report_questions()
    if not questions:
        return []
    try:
        worker_cap = int(os.environ.get("WORKFLOW_REPORT_QUESTION_WORKERS", "8"))
    except Exception:
        worker_cap = 8
    max_workers = max(1, min(worker_cap, len(questions)))
    if max_workers == 1:
        return [
            _answer_report_question_item(
                client=client,
                model=model,
                settings=settings,
                chunks=chunks,
                chunk_embeddings=chunk_embeddings,
                citations_context=citations_context,
                item=item,
            )
            for item in _progress_iter(questions, "Report questions")
        ]
    results: List[Dict[str, Any] | None] = [None] * len(questions)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(
                _answer_report_question_item,
                client=client,
                model=model,
                settings=settings,
                chunks=chunks,
                chunk_embeddings=chunk_embeddings,
                citations_context=citations_context,
                item=item,
            ): idx
            for idx, item in enumerate(questions)
        }
        for future in _progress_iter(list(as_completed(future_map)), "Report questions"):
            idx = future_map[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                item = questions[idx]
                results[idx] = {
                    "id": item["id"],
                    "category": item["category"],
                    "question": item["question"],
                    "answer": f"ERROR: {exc}",
                    "evidence_type": "unspecified",
                    "confidence": "low",
                    "citation_anchors": [],
                    "quote_snippet": "",
                    "table_figure": None,
                    "data_source": None,
                    "assumption_flag": None,
                    "assumption_notes": None,
                    "related_questions": [],
                }
    return [item for item in results if item is not None]


def _answer_subquestion(
    *,
    client: OpenAI,
    model: str,
    settings,
    chunks: List[Dict[str, Any]] | List[str],
    chunk_embeddings: List[List[float]],
    citations_context: str,
    subq: str,
) -> Dict[str, str]:
    context = top_k_context(
        chunks,
        chunk_embeddings,
        query=subq,
        client=client,
        settings=settings,
    )
    if citations_context:
        context = f"{context}\n\n{citations_context}"
    answer = call_openai(
        client,
        model=model,
        instructions=RESEARCHER_QA_PROMPT,
        user_input=f"Context:\n{context}\n\nQuestion: {subq}",
        max_output_tokens=None,
        usage_context="agent_answer",
    ).strip()
    return {"question": subq, "answer": answer}


def run_workflow(
    *,
    papers_dir: Path,
    config_path: Optional[Path] = None,
    meta_db_url: Optional[str] = None,
    report_dir: Optional[Path] = None,
    state_db: Path = DEFAULT_STATE_DB,
    agentic: Optional[bool] = None,
    question: Optional[str] = None,
    agentic_model: Optional[str] = None,
    agentic_max_subquestions: Optional[int] = None,
    agentic_citations: Optional[bool] = None,
    agentic_citations_max_items: Optional[int] = None,
    report_question_set: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the multi-step workflow and persist state transitions.

    Steps:
    1) ingest: load papers + extract text
    2) enrich: tally external metadata availability
    3) index: build FAISS + metadata tables (if DATABASE_URL is configured)
    4) evaluate: lightweight stats for provenance quality
    5) report: write a JSON report
    """
    settings = load_settings(config_path=config_path)
    run_id = uuid4().hex
    create_workflow_run(
        state_db,
        run_id=run_id,
        papers_dir=str(papers_dir),
        config_hash=settings.config_hash,
        metadata={"config_path": str(settings.config_path) if settings.config_path else None},
    )

    report_dir = report_dir or Path("reports")
    summary: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": _utc_now(),
        "papers_dir": str(papers_dir),
        "config": asdict(settings),
        "usage_db": str(DEFAULT_USAGE_DB),
    }

    # Step 0: Prep (corpus profiling)
    prep_start = _utc_now()
    record_step(state_db, run_id=run_id, step="prep", status="running", started_at=prep_start)
    pdfs = _resolve_paper_paths(Path(papers_dir))
    prep_out = prep_corpus(pdfs, report_dir=report_dir, run_id=run_id)
    record_step(
        state_db,
        run_id=run_id,
        step="prep",
        status="completed" if prep_out.get("status") == "completed" else "failed",
        started_at=prep_start,
        finished_at=_utc_now(),
        output=prep_out,
    )
    summary["prep"] = prep_out

    validate_only = os.environ.get("PREP_VALIDATE_ONLY", "").strip() == "1"
    if prep_out.get("status") == "failed":
        summary["finished_at"] = _utc_now()
        report_path = _write_report(report_dir, run_id, summary)
        record_step(
            state_db,
            run_id=run_id,
            step="report",
            status="completed",
            started_at=_utc_now(),
            finished_at=_utc_now(),
            output={"report_path": str(report_path)},
        )
        summary["report_path"] = str(report_path)
        set_workflow_status(state_db, run_id, "failed")
        return summary

    if validate_only:
        summary["finished_at"] = _utc_now()
        report_path = _write_report(report_dir, run_id, summary)
        record_step(
            state_db,
            run_id=run_id,
            step="report",
            status="completed",
            started_at=_utc_now(),
            finished_at=_utc_now(),
            output={"report_path": str(report_path)},
        )
        summary["report_path"] = str(report_path)
        set_workflow_status(state_db, run_id, "completed")
        return summary

    # Step 1: Ingest
    ingest_start = _utc_now()
    record_step(state_db, run_id=run_id, step="ingest", status="running", started_at=ingest_start)
    papers = load_papers(pdfs, progress=True, progress_desc="Ingesting papers")
    ingest_out = {"num_pdfs": len(pdfs), "num_papers": len(papers)}
    record_step(
        state_db,
        run_id=run_id,
        step="ingest",
        status="completed",
        started_at=ingest_start,
        finished_at=_utc_now(),
        output=ingest_out,
    )
    summary["ingest"] = ingest_out

    # Step 2: Enrich
    enrich_start = _utc_now()
    record_step(state_db, run_id=run_id, step="enrich", status="running", started_at=enrich_start)
    openalex_count = sum(1 for p in papers if getattr(p, "openalex", None))
    citec_count = sum(1 for p in papers if getattr(p, "citec", None))
    enrich_out = {"openalex": openalex_count, "citec": citec_count}
    record_step(
        state_db,
        run_id=run_id,
        step="enrich",
        status="completed",
        started_at=enrich_start,
        finished_at=_utc_now(),
        output=enrich_out,
    )
    summary["enrich"] = enrich_out

    # Step 3: Econ data (optional)
    econ_start = _utc_now()
    record_step(state_db, run_id=run_id, step="econ_data", status="running", started_at=econ_start)
    econ_out: Dict[str, Any] = {"status": "skipped"}
    series_env = os.environ.get("ECON_SERIES_IDS", "").strip()
    series_ids = [s.strip() for s in series_env.split(",") if s.strip()] if series_env else []
    if os.environ.get("FRED_API_KEY") or series_ids:
        if not series_ids:
            series_ids = ["GDPC1", "FEDFUNDS"]
        series_counts = {}
        for series_id in series_ids:
            obs = fetch_fred_series(series_id, limit=120)
            series_counts[series_id] = len(obs)
        econ_out = {"status": "fetched", "series_counts": series_counts}
    record_step(
        state_db,
        run_id=run_id,
        step="econ_data",
        status="completed",
        started_at=econ_start,
        finished_at=_utc_now(),
        output=econ_out,
    )
    summary["econ_data"] = econ_out

    # Step 4: Agentic workflow (optional)
    agentic_enabled = agentic if agentic is not None else os.environ.get("WORKFLOW_AGENTIC", "").strip() == "1"
    question = (question or os.environ.get("WORKFLOW_QUESTION") or "").strip()
    if not question:
        question = "Summarize the paper's research question, methods, and key findings."
    agentic_model = agentic_model or os.environ.get("WORKFLOW_AGENTIC_MODEL") or settings.chat_model
    try:
        max_subq = int(agentic_max_subquestions or os.environ.get("WORKFLOW_AGENTIC_MAX_SUBQUESTIONS", "3"))
    except Exception:
        max_subq = 3
    citations_enabled = agentic_citations
    if citations_enabled is None:
        citations_enabled = os.environ.get("WORKFLOW_AGENTIC_CITATIONS", "").strip() == "1"
    try:
        max_citations = int(agentic_citations_max_items or os.environ.get("WORKFLOW_AGENTIC_CITATIONS_MAX", "12"))
    except Exception:
        max_citations = 12
    agentic_start = _utc_now()
    record_step(state_db, run_id=run_id, step="agentic", status="running", started_at=agentic_start)
    agentic_out: Dict[str, Any] = {"status": "skipped"}
    if agentic_enabled:
        if not papers:
            agentic_out = {"status": "skipped", "reason": "no_papers"}
        else:
            try:
                client = OpenAI()
                target_paper = papers[0]
                citations_context = ""
                citations_preview: List[Dict[str, Any]] = []
                citations_error = None
                if citations_enabled:
                    try:
                        citations_full = llm_extract_citations(
                            paper_path=target_paper.path,
                            model=agentic_model,
                            api_key=os.environ.get("OPENAI_API_KEY"),
                        )
                        if isinstance(citations_full, list):
                            citations_preview = citations_full[:max_citations]
                            citations_context = _format_citations_context(citations_full, max_citations)
                    except Exception as exc:
                        citations_error = str(exc)
                chunks = prepare_chunks_for_paper(target_paper, settings)
                chunk_texts = [c["text"] if isinstance(c, dict) else str(c) for c in chunks]
                chunk_embeddings = embed_texts(client, chunk_texts, settings.embedding_model, settings.batch_size)
                subquestions = _agentic_plan(client, question, model=agentic_model, max_items=max_subq)
                sub_answers: List[Dict[str, str]] = []
                for subq in _progress_iter(subquestions, "Agentic sub-questions"):
                    sub_answers.append(
                        _answer_subquestion(
                            client=client,
                            model=agentic_model,
                            settings=settings,
                            chunks=chunks,
                            chunk_embeddings=chunk_embeddings,
                            citations_context=citations_context,
                            subq=subq,
                        )
                    )
                report_questions_enabled = os.environ.get("WORKFLOW_REPORT_QUESTIONS", "1").strip() != "0"
                report_question_mode = _normalize_report_question_set(
                    report_question_set or os.environ.get("WORKFLOW_REPORT_QUESTIONS_SET"),
                    report_questions_enabled,
                )
                report_questions_enabled = report_question_mode != "none"
                report_questions: List[Dict[str, str]] = []
                report_questions_error = None
                if report_question_mode in {"structured", "both"}:
                    try:
                        report_questions = _answer_report_questions(
                            client,
                            model=agentic_model,
                            settings=settings,
                            chunks=chunks,
                            chunk_embeddings=chunk_embeddings,
                            citations_context=citations_context,
                        )
                    except Exception as exc:
                        report_questions_error = str(exc)
                if report_question_mode in {"agentic", "both"}:
                    report_questions.extend(_report_questions_from_sub_answers(sub_answers))
                final_answer = _agentic_summarize(
                    client,
                    model=agentic_model,
                    question=question,
                    sub_answers=sub_answers,
                )
                agentic_out = {
                    "status": "completed",
                    "question": question,
                    "subquestions": subquestions,
                    "sub_answers": sub_answers,
                    "final_answer": final_answer,
                    "report_questions_enabled": report_questions_enabled,
                    "report_questions_set": report_question_mode,
                    "report_questions": report_questions,
                    "report_questions_error": report_questions_error,
                    "citations_enabled": citations_enabled,
                    "citations_preview": citations_preview,
                    "citations_error": citations_error,
                }
            except Exception as exc:
                agentic_out = {"status": "failed", "error": str(exc)}
    record_step(
        state_db,
        run_id=run_id,
        step="agentic",
        status="completed",
        started_at=agentic_start,
        finished_at=_utc_now(),
        output=agentic_out,
    )
    summary["agentic"] = agentic_out

    # Step 5: Index (optional)
    index_start = _utc_now()
    record_step(state_db, run_id=run_id, step="index", status="running", started_at=index_start)
    db_url = meta_db_url or os.environ.get("DATABASE_URL")
    db_ok = bool(db_url) and _can_connect_db(db_url)
    index_out: Dict[str, Any] = {"database_url": bool(db_url), "database_reachable": db_ok}
    if db_ok and pdfs:
        try:
            build_index(settings, pdfs, index_path=Path("vectors.index"), meta_db_url=db_url)
            index_out["status"] = "indexed"
        except Exception as exc:
            index_out["status"] = "failed"
            index_out["error"] = str(exc)
    else:
        index_out["status"] = "skipped"
        if db_url and not db_ok:
            index_out["reason"] = "db_unreachable"
    record_step(
        state_db,
        run_id=run_id,
        step="index",
        status="completed",
        started_at=index_start,
        finished_at=_utc_now(),
        output=index_out,
    )
    summary["index"] = index_out

    # Step 6: Evaluate (lightweight stats)
    eval_start = _utc_now()
    record_step(state_db, run_id=run_id, step="evaluate", status="running", started_at=eval_start)
    chunk_counts = []
    for paper in papers:
        chunks = prepare_chunks_for_paper(paper, settings)
        chunk_counts.append(len(chunks))
    eval_out = {
        "avg_chunks_per_paper": (sum(chunk_counts) / len(chunk_counts)) if chunk_counts else 0,
        "max_chunks": max(chunk_counts) if chunk_counts else 0,
        "min_chunks": min(chunk_counts) if chunk_counts else 0,
    }
    record_step(
        state_db,
        run_id=run_id,
        step="evaluate",
        status="completed",
        started_at=eval_start,
        finished_at=_utc_now(),
        output=eval_out,
    )
    summary["evaluate"] = eval_out

    # Step 7: Report
    report_start = _utc_now()
    record_step(state_db, run_id=run_id, step="report", status="running", started_at=report_start)
    summary["finished_at"] = _utc_now()
    report_path = _write_report(report_dir, run_id, summary)
    record_step(
        state_db,
        run_id=run_id,
        step="report",
        status="completed",
        started_at=report_start,
        finished_at=_utc_now(),
        output={"report_path": str(report_path)},
    )
    summary["report_path"] = str(report_path)
    set_workflow_status(state_db, run_id, "completed")
    return summary


def workflow_entrypoint(
    papers_dir: str,
    config_path: Optional[str] = None,
    meta_db_url: Optional[str] = None,
    agentic: Optional[bool] = None,
    question: Optional[str] = None,
    agentic_model: Optional[str] = None,
    agentic_citations: Optional[bool] = None,
    report_question_set: Optional[str] = None,
) -> str:
    """Helper for queue execution. Returns run_id for logging."""
    summary = run_workflow(
        papers_dir=Path(papers_dir),
        config_path=Path(config_path) if config_path else None,
        meta_db_url=meta_db_url,
        agentic=agentic,
        question=question,
        agentic_model=agentic_model,
        agentic_citations=agentic_citations,
        report_question_set=report_question_set,
    )
    return summary.get("run_id", "")
