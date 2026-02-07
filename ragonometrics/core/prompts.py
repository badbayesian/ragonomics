"""Centralized prompt strings used by the pipeline and UI."""

MAIN_SUMMARY_PROMPT = (
    "Summarize the economics paper for a researcher using only the provided context. "
    "Return 4-6 succinct bullet points covering: research question, method, "
    "key findings, and implications. If the context is thin, say what is missing."
)


RESEARCHER_QA_PROMPT = (
    "You are answering a researcher. Use only the provided context. "
    "Respond succinctly, generally in bullet points. "
    "Cite the context chunk(s) you used, referencing their provenance (e.g., page/word range)."
)

PIPELINE_SUMMARY_CHUNK_INSTRUCTIONS = (
    "You are a concise academic summarizer. Summarize the provided chunk of a paper. "
    "Capture key ideas, methods, results, and limitations mentioned in the chunk. "
    "Return plain text bullets (no JSON)."
)

PIPELINE_SUMMARY_MERGE_INSTRUCTIONS = (
    "You are a concise academic summarizer. Combine the chunk summaries into a single "
    "coherent summary of the paper. Use the structure: "
    "1) Problem, 2) Approach/Methods, 3) Key Results, 4) Limitations, 5) Future Work. "
    "Keep it brief and factual."
)

PIPELINE_CITATION_EXTRACT_INSTRUCTIONS = (
    "You extract bibliographic references from a paper. "
    "Return ONLY valid JSON: a list of objects with keys "
    "`citation_id` (string or null), `title`, `authors` (list of strings), "
    "`year` (int or null), `venue` (string or null), and `raw` (string). "
    "If data is missing, use null or an empty list. Do not include any extra keys."
)

PIPELINE_CITATION_RANK_INSTRUCTIONS = (
    "You rank cited papers by importance to the source paper. "
    "Use mention counts as the primary signal. If counts are similar, favor citations "
    "that seem foundational or central based on title/venue. "
    "Return ONLY valid JSON: a list sorted by importance descending, each item with keys "
    "`citation_id`, `title`, `authors`, `year`, `venue`, `mention_count` (int), "
    "`importance_rank` (int, 1 is highest), `importance_score` (0-100), and `rationale`."
)

QUERY_EXPANSION_PROMPT = (
    "Generate up to 3 alternative search queries that capture the same intent as the input. "
    "Return one query per line with no numbering."
)

RERANK_PROMPT = (
    "You are a retrieval reranker. Given a query and a list of chunk IDs with text, "
    "return the IDs ordered by relevance (most relevant first). "
    "Return only a flat list of IDs, one per line."
)

