from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .pipeline import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    extract_metadata,
    extract_citations,
    rank_citations,
    summarize_paper,
)


def write_output(text: str, out_path: str | None) -> None:
    """Write output to a file or stdout.

    Args:
        text: Output text to write.
        out_path: Optional file path. If None, prints to stdout.
    """
    if out_path:
        Path(out_path).write_text(text, encoding="utf-8")
    else:
        print(text)


def load_env(path: Path) -> None:
    """Load environment variables from a .env-style file.

    Existing environment variables are not overwritten.

    Args:
        path: Path to the .env file.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def add_common_args(p: argparse.ArgumentParser) -> None:
    """Add common CLI arguments shared across subcommands.

    Args:
        p: Argument parser to configure.
    """
    p.add_argument("--paper", type=str, required=True, help="Path to paper (.pdf/.md/.txt)")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenAI model name")
    p.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key (or set OPENAI_API_KEY)",
    )
    p.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Max tokens per OpenAI response",
    )
    p.add_argument("--out", type=str, default=None, help="Write output to file")


def main() -> None:
    """CLI entry point for the RAG pipeline commands."""
    load_env(Path(__file__).resolve().parents[1] / ".env")
    ap = argparse.ArgumentParser(prog="rag-pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("summarize", help="Summarize a paper with OpenAI")
    add_common_args(s)
    s.add_argument("--chunk-words", type=int, default=800, help="Chunk size (words)")
    s.add_argument("--overlap-words", type=int, default=120, help="Chunk overlap (words)")

    c = sub.add_parser("citations", help="Extract citations from a paper with OpenAI")
    add_common_args(c)

    r = sub.add_parser("rank", help="Rank cited papers by importance with OpenAI")
    add_common_args(r)

    m = sub.add_parser("metadata", help="Extract paper metadata + citations")
    add_common_args(m)
    m.add_argument(
        "--no-citations",
        action="store_true",
        help="Skip citation extraction",
    )
    m.add_argument(
        "--rank",
        action="store_true",
        help="Include ranked citations (extra LLM call)",
    )

    args = ap.parse_args()
    paper_path = Path(args.paper)

    if args.cmd == "summarize":
        summary = summarize_paper(
            paper_path=paper_path,
            model=args.model,
            api_key=args.api_key,
            chunk_words=args.chunk_words,
            overlap_words=args.overlap_words,
            max_output_tokens=args.max_output_tokens,
        )
        write_output(summary, args.out)
        return

    if args.cmd == "citations":
        citations = extract_citations(
            paper_path=paper_path,
            model=args.model,
            api_key=args.api_key,
            max_output_tokens=args.max_output_tokens,
        )
        write_output(json.dumps(citations, indent=2, ensure_ascii=False), args.out)
        return

    if args.cmd == "rank":
        ranked = rank_citations(
            paper_path=paper_path,
            model=args.model,
            api_key=args.api_key,
            max_output_tokens=args.max_output_tokens,
        )
        write_output(json.dumps(ranked, indent=2, ensure_ascii=False), args.out)
        return

    if args.cmd == "metadata":
        metadata = extract_metadata(
            paper_path=paper_path,
            model=args.model,
            api_key=args.api_key,
            max_output_tokens=args.max_output_tokens,
            include_citations=not args.no_citations,
            include_ranked_citations=args.rank,
        )
        write_output(json.dumps(metadata, indent=2, ensure_ascii=False), args.out)
        return

    raise SystemExit(2)


if __name__ == "__main__":
    main()
