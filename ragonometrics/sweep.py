from __future__ import annotations

import itertools
from pathlib import Path
from typing import List

from .benchmark import bench_papers


def sweep(
    papers_dir: str | Path,
    out_dir: str | Path,
    chunk_words_vals: List[int],
    chunk_overlap_vals: List[int],
    bm25_vals: List[float],
    limit: int = 0,
    use_openai: bool = False,
):
    """Run a parameter sweep and write benchmark CSVs.

    Args:
        papers_dir: Directory containing PDF files.
        out_dir: Output directory for sweep results.
        chunk_words_vals: Chunk sizes to evaluate.
        chunk_overlap_vals: Overlap sizes to evaluate.
        bm25_vals: BM25 weights to evaluate.
        limit: Maximum number of papers to process (0 = all).
        use_openai: Whether to use the real OpenAI client.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combos = list(itertools.product(chunk_words_vals, chunk_overlap_vals, bm25_vals))
    summary = []
    for cw, co, bm in combos:
        fname = out_dir / f"sweep_cw{cw}_co{co}_bm{bm:.2f}.csv"
        print(f"Running sweep: chunk_words={cw}, chunk_overlap={co}, bm25_weight={bm}")
        bench_papers(papers_dir, fname, limit=limit, use_openai=use_openai, db_url=None, chunk_words=cw, chunk_overlap=co, bm25_weight=bm, force_ocr=True)
        # quick summary: count rows
        with fname.open("r", encoding="utf-8") as fh:
            lines = sum(1 for _ in fh) - 1
        summary.append((cw, co, bm, lines))
    # write summary
    sfile = out_dir / "sweep_summary.csv"
    with sfile.open("w", encoding="utf-8") as fh:
        fh.write("chunk_words,chunk_overlap,bm25_weight,papers\n")
        for cw, co, bm, cnt in summary:
            fh.write(f"{cw},{co},{bm},{cnt}\n")
    print(f"Sweep complete — results in {out_dir}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--papers-dir", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="bench/sweep")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--use-openai", action="store_true")
    args = p.parse_args()
    pw = args.papers_dir
    if not pw:
        from .main import load_settings

        pw = load_settings().papers_dir
    sweep(pw, args.out_dir, chunk_words_vals=[200,300,350], chunk_overlap_vals=[30,50], bm25_vals=[0.0,0.5,1.0], limit=args.limit, use_openai=args.use_openai)
