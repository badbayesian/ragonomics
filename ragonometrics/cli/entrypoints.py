"""Primary CLI entrypoints for indexing, querying, UI, and benchmarks. Wires top-level commands to core pipeline components."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from openai import OpenAI

from ragonometrics.eval.benchmark import bench_papers
from ragonometrics.indexing.indexer import build_index
from ragonometrics.core.main import (
    embed_texts,
    load_papers,
    load_settings,
    prepare_chunks_for_paper,
    top_k_context,
)
from ragonometrics.pipeline import call_openai
from ragonometrics.core.prompts import RESEARCHER_QA_PROMPT
from ragonometrics.pipeline.query_cache import DEFAULT_CACHE_PATH, get_cached_answer, make_cache_key, set_cached_answer
from ragonometrics.integrations.openalex import format_openalex_context
from ragonometrics.integrations.citec import format_citec_context
from ragonometrics.pipeline.workflow import run_workflow
from ragonometrics.integrations.rq_queue import enqueue_workflow


def cmd_index(args: argparse.Namespace) -> int:
    """Build a FAISS index from PDFs.

    Args:
        args: Parsed CLI arguments.

    Returns:
        int: Exit code (0 on success).
    """
    settings = load_settings()
    papers_dir = Path(args.papers_dir) if args.papers_dir else settings.papers_dir
    pdfs = sorted(papers_dir.glob("*.pdf"))
    if args.limit and args.limit > 0:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        print("No PDFs found to index.")
        return 1
    build_index(settings, pdfs, index_path=Path(args.index_path), meta_db_url=args.meta_db_url)
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    """Run a single query against a paper.

    Args:
        args: Parsed CLI arguments.

    Returns:
        int: Exit code (0 on success).
    """
    settings = load_settings()
    paper_path = Path(args.paper)
    papers = load_papers([paper_path])
    if not papers:
        print("No paper text extracted.")
        return 1
    paper = papers[0]
    chunks = prepare_chunks_for_paper(paper, settings)
    if not chunks:
        print("No chunks extracted.")
        return 1
    client = OpenAI()
    chunk_texts = [c["text"] if isinstance(c, dict) else str(c) for c in chunks]
    chunk_embeddings = embed_texts(client, chunk_texts, settings.embedding_model, settings.batch_size)
    context = top_k_context(
        chunks,
        chunk_embeddings,
        query=args.question,
        client=client,
        settings=settings,
    )
    model = args.model or settings.chat_model
    cache_key = make_cache_key(args.question, str(paper_path), model, context)
    cached = get_cached_answer(DEFAULT_CACHE_PATH, cache_key)
    if cached is not None:
        answer = cached
    else:
        openalex_context = format_openalex_context(paper.openalex)
        citec_context = format_citec_context(paper.citec)
        user_input = f"Context:\n{context}\n\nQuestion: {args.question}"
        prefix_parts = [ctx for ctx in (openalex_context, citec_context) if ctx]
        if prefix_parts:
            prefix = "\n\n".join(prefix_parts)
            user_input = f"{prefix}\n\n{user_input}"
        answer = call_openai(
            client,
            model=model,
            instructions=RESEARCHER_QA_PROMPT,
            user_input=user_input,
            max_output_tokens=None,
        ).strip()
        set_cached_answer(
            DEFAULT_CACHE_PATH,
            cache_key=cache_key,
            query=args.question,
            paper_path=str(paper_path),
            model=model,
            context=context,
            answer=answer,
        )
    print(answer)
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    """Launch the Streamlit UI.

    Args:
        args: Parsed CLI arguments.

    Returns:
        int: Exit code (0 on success).
    """
    app_path = Path(__file__).resolve().parents[1] / "ui" / "streamlit_app.py"
    try:
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path)])
    except FileNotFoundError:
        print("Streamlit is not installed.")
        return 1


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run benchmark suite.

    Args:
        args: Parsed CLI arguments.

    Returns:
        int: Exit code (0 on success).
    """
    settings = load_settings()
    papers_dir = Path(args.papers_dir) if args.papers_dir else settings.papers_dir
    bench_papers(
        Path(papers_dir),
        Path(args.out),
        limit=args.limit,
        use_openai=args.use_openai,
        db_url=args.db_url,
        force_ocr=args.force_ocr,
    )
    return 0


def cmd_workflow(args: argparse.Namespace) -> int:
    """Run or enqueue the multi-step workflow.

    Args:
        args: Parsed CLI arguments.

    Returns:
        int: Exit code (0 on success).
    """
    papers_dir = Path(args.papers) if args.papers else load_settings().papers_dir
    if args.async_mode:
        job = enqueue_workflow(
            papers_dir,
            redis_url=args.redis_url,
            config_path=Path(args.config_path) if args.config_path else None,
            meta_db_url=args.meta_db_url,
            agentic=args.agentic,
            question=args.question,
            agentic_model=args.agentic_model,
            agentic_citations=args.agentic_citations,
            report_question_set=args.report_question_set,
        )
        print(f"Enqueued workflow job: {job.id}")
        return 0

    summary = run_workflow(
        papers_dir=papers_dir,
        config_path=Path(args.config_path) if args.config_path else None,
        meta_db_url=args.meta_db_url,
        agentic=args.agentic,
        question=args.question,
        agentic_model=args.agentic_model,
        agentic_citations=args.agentic_citations,
        report_question_set=args.report_question_set,
    )
    print(f"Workflow run completed: {summary.get('run_id')}")
    print(f"Report: {summary.get('report_path')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    p = argparse.ArgumentParser(prog="ragonometrics")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("index", help="Build FAISS index from PDFs")
    s.add_argument("--papers-dir", type=str, default=None)
    s.add_argument("--index-path", type=str, default="vectors.index")
    s.add_argument("--meta-db-url", type=str, default=None)
    s.add_argument("--limit", type=int, default=0)
    s.set_defaults(func=cmd_index)

    q = sub.add_parser("query", help="Ask a question against a paper")
    q.add_argument("--paper", type=str, required=True)
    q.add_argument("--question", type=str, required=True)
    q.add_argument("--model", type=str, default=None)
    q.set_defaults(func=cmd_query)

    u = sub.add_parser("ui", help="Launch the Streamlit UI")
    u.set_defaults(func=cmd_ui)

    b = sub.add_parser("benchmark", help="Run benchmark suite")
    b.add_argument("--papers-dir", type=str, default=None)
    b.add_argument("--out", type=str, default="bench/benchmark.csv")
    b.add_argument("--limit", type=int, default=0)
    b.add_argument("--use-openai", action="store_true")
    b.add_argument("--force-ocr", action="store_true")
    b.add_argument("--db-url", type=str, default=None)
    b.set_defaults(func=cmd_benchmark)

    w = sub.add_parser("workflow", help="Run or enqueue a multi-step workflow")
    w.add_argument("--papers", type=str, default=None, help="PDF file or directory")
    w.add_argument("--papers-dir", dest="papers", type=str, help=argparse.SUPPRESS)
    w.add_argument("--config-path", type=str, default=None)
    w.add_argument("--meta-db-url", type=str, default=None)
    w.add_argument("--redis-url", type=str, default="redis://redis:6379")
    w.add_argument("--async", dest="async_mode", action="store_true", help="Enqueue workflow via Redis/RQ")
    w.add_argument("--agentic", action="store_true", help="Enable agentic LLM sub-question workflow")
    w.add_argument("--question", type=str, default=None, help="Main question for agentic workflow")
    w.add_argument("--agentic-model", type=str, default=None, help="Model override for agentic workflow")
    w.add_argument("--agentic-citations", action="store_true", help="Use citations API to enrich agentic context")
    w.add_argument(
        "--report-question-set",
        type=str,
        default=None,
        help="Report questions: structured|agentic|both|none (overrides WORKFLOW_REPORT_QUESTIONS_SET).",
    )
    w.set_defaults(func=cmd_workflow)

    return p


def main() -> int:
    """CLI entrypoint for ragonometrics.

    Returns:
        int: Exit code.
    """
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

