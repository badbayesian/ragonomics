from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

from ragonometrics.indexer import build_index
from ragonometrics.main import (
    Paper,
    Settings,
    embed_texts,
    extract_dois_from_text,
    load_papers,
    load_settings,
    prepare_chunks_for_paper,
    top_k_context,
)


class FakeClient:
    """Lightweight OpenAI client stub for deterministic benchmarks."""

    class embeddings:
        @staticmethod
        def create(model, input):
            """Return deterministic embedding vectors for testing."""
            # return a fixed-length small vector for deterministic dry runs
            class Item:
                def __init__(self, embedding):
                    self.embedding = embedding

            vec = [0.01] * 8
            return type("Resp", (), {"data": [Item(vec) for _ in input]})


def benchmark_indexing(papers: List[Path], runs: int = 1) -> List[dict]:
    """Benchmark vector index building across runs.

    Args:
        papers: PDF paths to index.
        runs: Number of benchmark runs.

    Returns:
        list[dict]: Per-run timing results.
    """
    settings = load_settings()
    results = []
    for r in range(runs):
        start = time.time()
        build_index(settings, papers, index_path=Path(f"vectors_{r}.index"), meta_db_url=None)
        elapsed = time.time() - start
        results.append({"run": r, "elapsed_seconds": elapsed, "num_papers": len(papers)})
    return results


def benchmark_chunking(paper: Path) -> dict:
    """Benchmark chunk preparation for a single paper.

    Args:
        paper: PDF path to process.

    Returns:
        dict: Chunk count and timing results for the paper.
    """
    settings = load_settings()
    papers = load_papers([paper])
    if not papers:
        return {"paper": str(paper), "chunks": 0, "elapsed_seconds": 0.0}
    paper_obj = papers[0]
    start = time.time()
    chunks = prepare_chunks_for_paper(paper_obj, settings)
    elapsed = time.time() - start
    return {"paper": str(paper_obj.path), "chunks": len(chunks), "elapsed_seconds": elapsed}


def bench_papers(
    papers_dir: Path,
    out_csv: Path,
    limit: int = 0,
    use_openai: bool = False,
    db_url: Optional[str] = None,
    chunk_words: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    bm25_weight: Optional[float] = None,
    force_ocr: bool = False,
):
    """Benchmark chunking, embedding, and retrieval across papers.

    Args:
        papers_dir: Directory of PDF files.
        out_csv: Output CSV path.
        limit: Maximum number of papers to process (0 = all).
        use_openai: Whether to use the real OpenAI client.
        db_url: Optional Postgres URL for hybrid retrieval.
        chunk_words: Override chunk size in words.
        chunk_overlap: Override overlap size in words.
        bm25_weight: Optional BM25 weight for hybrid retrieval.
        force_ocr: Whether to force OCR instead of pdftotext.
    """
    settings = load_settings()
    settings = Settings(
        papers_dir=settings.papers_dir,
        max_papers=settings.max_papers,
        max_words=settings.max_words,
        chunk_words=chunk_words or settings.chunk_words,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        top_k=settings.top_k,
        batch_size=settings.batch_size,
        embedding_model=settings.embedding_model,
        chat_model=settings.chat_model,
    )

    files = sorted(Path(papers_dir).glob("*.pdf"))
    if limit > 0:
        files = files[:limit]

    client = OpenAI() if use_openai else FakeClient()

    rows = []
    for p in files:
        # optionally bypass `pdftotext` and use OCR directly
        if force_ocr:
            try:
                from pdf2image import convert_from_path
                import pytesseract
            except Exception:
                raise RuntimeError("FORCE_OCR requested but pdf2image/pytesseract not available")
            try:
                images = convert_from_path(str(p), timeout=10)
                texts = [pytesseract.image_to_string(im) for im in images]
                text = "\n\n".join(texts)
            except Exception:
                # if OCR fails or times out, skip this paper with empty text
                text = ""
            paper = Paper(path=p, title=p.stem, author="Unknown", text=text, pages=None)
        else:
            paper_list = load_papers([p])
            if not paper_list:
                continue
            paper = paper_list[0]
        row = {
            "path": str(p),
            "title": paper.title,
            "author": paper.author,
        }

        # DOI extraction
        dois = extract_dois_from_text(paper.text)
        row["dois_found"] = len(dois)

        # chunking
        start = time.perf_counter()
        chunks = prepare_chunks_for_paper(paper, settings)
        row["num_chunks"] = len(chunks)
        words = [len((c["text"] if isinstance(c, dict) else str(c)).split()) for c in chunks]
        row["avg_chunk_words"] = (sum(words) / len(words)) if words else 0
        row["chunk_time_s"] = time.perf_counter() - start

        # embeddings (batched)
        start = time.perf_counter()
        chunk_texts = [c["text"] if isinstance(c, dict) else str(c) for c in chunks]
        if chunk_texts:
            _embs = embed_texts(client, chunk_texts, settings.embedding_model, settings.batch_size)
        else:
            _embs = []
        row["embedding_time_s"] = time.perf_counter() - start

        # retrieval: hybrid (if db_url provided) or local cosine
        start = time.perf_counter()
        try:
            # set BM25 weight via env so hybrid_search will pick it up
            if bm25_weight is not None:
                import os

                os.environ["BM25_WEIGHT"] = str(bm25_weight)
            context = top_k_context(chunks, _embs, query="What is the research question?", client=client, settings=settings)
            row["retrieval_time_s"] = time.perf_counter() - start
            row["context_len_chars"] = len(context)
        except Exception as e:
            row["retrieval_err"] = str(e)
            row["retrieval_time_s"] = time.perf_counter() - start
            row["context_len_chars"] = 0

        rows.append(row)

    # write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = ["path", "title", "author", "dois_found", "num_chunks", "avg_chunk_words", "chunk_time_s", "embedding_time_s", "retrieval_time_s", "context_len_chars", "retrieval_err"]
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in keys})

    print(f"Wrote benchmark results for {len(rows)} papers to {out_csv}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Benchmark RAG pipeline across papers")
    p.add_argument("--papers-dir", type=str, default=None)
    p.add_argument("--out", type=str, default="bench/benchmark.csv")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--use-openai", action="store_true")
    p.add_argument("--force-ocr", action="store_true", help="Force OCR instead of pdftotext")
    p.add_argument("--db-url", type=str, default=None)
    args = p.parse_args()

    papers_dir = args.papers_dir or load_settings().papers_dir
    bench_papers(Path(papers_dir), Path(args.out), limit=args.limit, use_openai=args.use_openai, db_url=args.db_url, force_ocr=args.force_ocr)

