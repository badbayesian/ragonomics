"""RQ/Redis queue wrapper for asynchronous indexing jobs. Used to run pipeline indexing in the background."""

from __future__ import annotations

from rq import Queue
from redis import Redis
from pathlib import Path
from typing import List

from ragonometrics.indexing.indexer import build_index
from ragonometrics.core.main import load_settings


def enqueue_index(papers: List[Path], redis_url: str = "redis://redis:6379"):
    """Enqueue an indexing job in Redis.

    Args:
        papers: PDF paths to index.
        redis_url: Redis connection URL.

    Returns:
        rq.job.Job: Enqueued job handle.
    """
    redis_conn = Redis.from_url(redis_url)
    q = Queue("default", connection=redis_conn)
    settings = load_settings()
    job = q.enqueue(build_index, settings, papers)
    return job


if __name__ == "__main__":
    import sys
    from pathlib import Path

    settings = load_settings()
    papers_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else settings.papers_dir
    pdfs = sorted(papers_dir.glob("*.pdf"))
    print("Enqueuing indexing job for", len(pdfs), "papers")
    job = enqueue_index(pdfs)
    print("Enqueued job:", job.id)

