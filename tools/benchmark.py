from __future__ import annotations

import argparse
import json
from pathlib import Path

from ragonometrics.benchmark import benchmark_chunking, benchmark_indexing
from ragonometrics.main import load_settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default="bench_results.json")
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    settings = load_settings()
    papers_dir = Path(args.papers_dir) if args.papers_dir else settings.papers_dir
    pdfs = sorted(papers_dir.glob("*.pdf"))[:5]

    res = {"indexing": benchmark_indexing(pdfs, runs=args.runs)}
    if pdfs:
        res["chunking"] = benchmark_chunking(pdfs[0])

    Path(args.output).write_text(json.dumps(res, indent=2))
    print("Wrote", args.output)


if __name__ == "__main__":
    main()

