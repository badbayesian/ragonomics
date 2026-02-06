"""LLM pipeline routines for summarization, citation extraction, and metadata. Wraps OpenAI calls and is used by CLI and main flows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import os
import re

from openai import OpenAI

from ragonometrics.core.io_loaders import chunk_words, load_pdf, load_text_file
from ragonometrics.core.prompts import (
    PIPELINE_CITATION_EXTRACT_INSTRUCTIONS,
    PIPELINE_CITATION_RANK_INSTRUCTIONS,
    PIPELINE_SUMMARY_CHUNK_INSTRUCTIONS,
    PIPELINE_SUMMARY_MERGE_INSTRUCTIONS,
)


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")
_MAX_TOKENS_ENV = os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "").strip()
try:
    DEFAULT_MAX_OUTPUT_TOKENS = int(_MAX_TOKENS_ENV) if _MAX_TOKENS_ENV else 800
except ValueError:
    DEFAULT_MAX_OUTPUT_TOKENS = 800




@dataclass(frozen=True)
class PaperText:
    """Container for full and segmented paper text.

    Attributes:
        text: Full extracted text.
        body_text: Body text excluding references.
        references_text: References section text.
    """

    text: str
    body_text: str
    references_text: str


def load_paper_text(path: Path) -> PaperText:
    """Load a paper file and split it into body and references sections.

    Args:
        path: Path to a PDF or text file.

    Returns:
        PaperText: Full text plus body and references segments.
    """
    if path.suffix.lower() == ".pdf":
        raw = load_pdf(path).text
    else:
        raw = load_text_file(path).text

    body, refs = split_references(raw)
    if not refs:
        refs = tail_text(raw, max_chars=40_000)

    return PaperText(text=raw, body_text=body, references_text=refs)


def split_references(text: str) -> Tuple[str, str]:
    """Split a paper into body and references using heading heuristics.

    Args:
        text: Full paper text.

    Returns:
        Tuple[str, str]: (body_text, references_text).
    """
    pattern = re.compile(
        r"^\s*(references|bibliography|works cited)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return text, ""
    start = matches[-1].start()
    return text[:start], text[start:]


def tail_text(text: str, max_chars: int) -> str:
    """Return the last N characters of text.

    Args:
        text: Input text.
        max_chars: Maximum characters to keep.

    Returns:
        str: Tail of the text.
    """
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def head_text(text: str, max_chars: int) -> str:
    """Return the first N characters of text.

    Args:
        text: Input text.
        max_chars: Maximum characters to keep.

    Returns:
        str: Head of the text.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _clean_lines(text: str, max_lines: int = 80) -> List[str]:
    """Normalize and truncate lines for header parsing.

    Args:
        text: Source text to clean.
        max_lines: Maximum number of lines to return.

    Returns:
        List[str]: Cleaned lines.
    """
    lines = [re.sub(r"\s+", " ", ln.strip()) for ln in text.splitlines() if ln.strip()]
    return lines[:max_lines]


def _is_abstract_line(line: str) -> bool:
    """Return True if a line looks like an abstract heading."""
    return bool(re.match(r"^abstract\b", line.strip(), flags=re.IGNORECASE))


def _is_keywords_line(line: str) -> bool:
    """Return True if a line looks like a keywords heading."""
    return bool(re.match(r"^keywords?\b", line.strip(), flags=re.IGNORECASE))


def _is_jel_line(line: str) -> bool:
    """Return True if a line looks like a JEL heading."""
    return bool(re.match(r"^jel\b", line.strip(), flags=re.IGNORECASE))


def _is_noise_line(line: str) -> bool:
    """Return True if a line is likely boilerplate or noise."""
    low = line.lower()
    noise_tokens = [
        "working paper",
        "working paper series",
        "nber",
        "national bureau",
        "journal",
        "review",
        "proceedings",
        "department",
        "university",
        "school",
        "institute",
        "doi",
        "issn",
        "isbn",
        "http",
        "www.",
        "copyright",
    ]
    if any(tok in low for tok in noise_tokens):
        return True
    if len(line) < 4:
        return True
    return False


def _looks_like_author_line(line: str) -> bool:
    """Heuristically determine whether a line contains author names."""
    low = line.lower()
    if low.startswith("by "):
        return True
    if any(ch.isdigit() for ch in line):
        return False
    if "," in line or " and " in low or " & " in line:
        words = line.split()
        caps = sum(1 for w in words if w[:1].isupper())
        if words and caps / len(words) >= 0.6:
            return True
    return False


def _looks_like_title_line(line: str) -> bool:
    """Heuristically determine whether a line could be a title line."""
    if _is_noise_line(line):
        return False
    if _looks_like_author_line(line):
        return False
    if _is_abstract_line(line):
        return False
    words = line.split()
    if len(words) < 3 or len(words) > 20:
        return False
    letters = sum(c.isalpha() for c in line)
    if letters < 8:
        return False
    return True


def _looks_like_title_continuation(line: str) -> bool:
    """Heuristically determine whether a line continues a title."""
    if _is_noise_line(line) or _looks_like_author_line(line):
        return False
    if _is_abstract_line(line):
        return False
    words = line.split()
    if len(words) < 2 or len(words) > 15:
        return False
    letters = sum(c.isalpha() for c in line)
    if letters < 6:
        return False
    return True


def _extract_title(lines: List[str]) -> Tuple[Optional[str], int]:
    """Extract a title candidate and its end index from header lines.

    Args:
        lines: Cleaned header lines.

    Returns:
        Tuple[Optional[str], int]: (title, end_index).
    """
    title_lines: List[str] = []
    end_idx = 0
    for i, line in enumerate(lines):
        if _is_abstract_line(line):
            break
        if _is_noise_line(line):
            continue
        if not title_lines:
            if _looks_like_title_line(line):
                title_lines.append(line)
                end_idx = i + 1
                continue
            continue
        if _looks_like_title_continuation(line):
            title_lines.append(line)
            end_idx = i + 1
            continue
        break
    if not title_lines:
        return None, 0
    title = " ".join(title_lines).strip()
    title = re.sub(r"[†∗*‡]", "", title).strip()
    return title or None, end_idx


def _extract_keywords(lines: List[str]) -> List[str]:
    """Extract keyword tokens from header lines.

    Args:
        lines: Cleaned header lines.

    Returns:
        List[str]: Extracted keywords.
    """
    keywords: List[str] = []
    for line in lines:
        m = re.match(r"^keywords?\s*[:\-]?\s*(.*)$", line, flags=re.IGNORECASE)
        if m:
            tail = m.group(1).strip()
            if tail:
                parts = re.split(r"[;,]", tail)
                keywords = [p.strip() for p in parts if p.strip()]
            break
    return keywords


def _extract_abstract(lines: List[str]) -> Optional[str]:
    """Extract an abstract block from header lines.

    Args:
        lines: Cleaned header lines.

    Returns:
        Optional[str]: Abstract text if found.
    """
    start = None
    for i, line in enumerate(lines):
        if _is_abstract_line(line):
            start = i + 1
            break
    if start is None:
        return None
    collected: List[str] = []
    for line in lines[start:]:
        low = line.lower()
        if _is_keywords_line(line) or _is_jel_line(line):
            break
        if re.match(r"^(introduction|i\.)\b", low):
            break
        if re.match(r"^\d+(\.| )", line):
            break
        collected.append(line)
    abstract = " ".join(collected).strip()
    return abstract or None


def _preclean_author_segment(segment: str) -> str:
    """Remove common noise characters from an author segment.

    Args:
        segment: Raw author segment.

    Returns:
        str: Cleaned author segment.
    """
    segment = re.sub(r"^by\s+", "", segment, flags=re.IGNORECASE)
    segment = re.sub(r"[†∗*‡]", "", segment)
    segment = re.sub(r"\s+", " ", segment).strip()
    return segment


def _clean_name_segment(segment: str) -> str:
    """Normalize a name segment by removing degrees and punctuation.

    Args:
        segment: Raw name segment.

    Returns:
        str: Cleaned name segment.
    """
    segment = re.sub(
        r"\b(Ph\.?D\.?|M\.?D\.?|M\.?P\.?H\.?|M\.?A\.?|B\.?A\.?|"
        r"M\.?S\.?|B\.?S\.?|J\.?D\.?|MBA|D\.?Phil\.?|Sc\.?D\.?)\b",
        "",
        segment,
        flags=re.IGNORECASE,
    )
    segment = re.sub(r"[^A-Za-z0-9\s\.-]", "", segment)
    segment = re.sub(r"\s+", " ", segment).strip().strip(",")
    return segment


def _looks_like_name(name: str) -> bool:
    """Heuristically determine whether a string looks like a person name."""
    words = name.split()
    if len(words) < 2 or len(words) > 10:
        return False
    if any(word.islower() for word in words):
        # avoid full lowercase words in names
        return False
    caps = sum(1 for w in words if w[:1].isupper())
    if caps / len(words) < 0.8:
        return False
    return True


def _split_author_blob(text: str) -> List[str]:
    """Split a string into possible author name chunks.

    Args:
        text: Raw author text.

    Returns:
        List[str]: Candidate author blobs.
    """
    if not text:
        return []
    # If there are no obvious separators, try to split by repeated name patterns.
    if "," not in text and " and " not in text.lower() and " & " not in text:
        pattern = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-z]+)+\b")
        matches = pattern.findall(text)
        if len(matches) > 1:
            return matches
    return [text]


def _extract_authors(lines: List[str], start_idx: int) -> List[str]:
    """Extract author names from header lines.

    Args:
        lines: Cleaned header lines.
        start_idx: Line index to start scanning from.

    Returns:
        List[str]: Deduplicated author names.
    """
    authors: List[str] = []
    for raw in lines[start_idx : start_idx + 12]:
        if _is_abstract_line(raw):
            break
        segments = [seg.strip() for seg in re.split(r"\s{2,}", raw) if seg.strip()]
        if not segments:
            segments = [raw.strip()]
        for seg in segments:
            seg = _preclean_author_segment(seg)
            if not seg:
                continue
            for blob in _split_author_blob(seg):
                parts = re.split(r"\s+and\s+|\s*&\s*|,", blob)
                parts = [_clean_name_segment(p) for p in parts if p.strip()]
                if len(parts) == 1:
                    if _looks_like_name(parts[0]):
                        authors.append(parts[0])
                else:
                    for p in parts:
                        if _looks_like_name(p):
                            authors.append(p)
        if authors and _looks_like_author_line(raw):
            # stop after the main author line(s)
            continue
    dedup: List[str] = []
    seen = set()
    for name in authors:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(name)
    return dedup


def _extract_year(lines: List[str]) -> Optional[int]:
    """Extract a publication year from header lines.

    Args:
        lines: Cleaned header lines.

    Returns:
        Optional[int]: Year if found.
    """
    for line in lines[:40]:
        if "doi" in line.lower() or "http" in line.lower():
            continue
        for m in re.finditer(r"\b(19\d{2}|20\d{2})\b", line):
            try:
                return int(m.group(1))
            except ValueError:
                continue
    return None


def _extract_dois(text: str) -> List[str]:
    """Extract DOI strings from text.

    Args:
        text: Input text.

    Returns:
        List[str]: DOI strings in order of appearance.
    """
    doi_pattern = re.compile(r"\b10\.\d{4,9}/[^\s\"<>]+", re.IGNORECASE)
    matches = doi_pattern.findall(text)
    cleaned: List[str] = []
    for raw in matches:
        doi = raw.rstrip(").,;")
        if doi not in cleaned:
            cleaned.append(doi)
    return cleaned




def build_client(api_key: Optional[str]) -> OpenAI:
    """Build an OpenAI client using an explicit or env API key.

    Args:
        api_key: Optional API key override.

    Returns:
        OpenAI: Configured OpenAI client.

    Raises:
        ValueError: If no API key is provided or configured.
    """
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=key)


def response_text(response: Any) -> str:
    """Extract text content from an OpenAI response object.

    Args:
        response: OpenAI response object.

    Returns:
        str: Extracted text.

    Raises:
        ValueError: If no text content is found.
    """
    parts: List[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) == "output_text":
                parts.append(content.text)
    if parts:
        return "\n".join(parts).strip()

    fallback = getattr(response, "output_text", None) or getattr(response, "text", None)
    if fallback:
        return str(fallback).strip()
    raise ValueError("No text in OpenAI response.")


def call_openai(
    client: OpenAI,
    *,
    model: str,
    instructions: str,
    user_input: str,
    max_output_tokens: Optional[int],
    temperature: Optional[float] = None,
    usage_context: str = "llm",
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> str:
    """Call OpenAI Responses API and return text.

    Args:
        client: OpenAI client.
        model: Model name.
        instructions: System instructions.
        user_input: User input text.
        max_output_tokens: Maximum output tokens.
        temperature: Optional sampling temperature for response variation.
        usage_context: Label for the type of request (e.g., "answer", "rerank").
        session_id: Optional session identifier for grouping usage.
        request_id: Optional request identifier for per-query usage.

    Returns:
        str: Response text.
    """
    if max_output_tokens is not None and max_output_tokens < 16:
        max_output_tokens = 16
    max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "2"))
    for attempt in range(max_retries + 1):
        try:
            payload = dict(
                model=model,
                instructions=instructions,
                input=user_input,
                max_output_tokens=max_output_tokens,
            )
            if temperature is not None:
                payload["temperature"] = temperature
            resp = client.responses.create(**payload)
            try:
                from ragonometrics.pipeline.token_usage import record_usage

                usage = getattr(resp, "usage", None)
                input_tokens = output_tokens = total_tokens = 0
                if usage is not None:
                    if isinstance(usage, dict):
                        input_tokens = int(usage.get("input_tokens") or 0)
                        output_tokens = int(usage.get("output_tokens") or 0)
                        total_tokens = int(usage.get("total_tokens") or 0)
                    else:
                        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
                        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
                        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
                record_usage(
                    model=model,
                    operation=usage_context,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    session_id=session_id,
                    request_id=request_id,
                )
            except Exception:
                pass
            return response_text(resp)
        except Exception as exc:
            if attempt >= max_retries:
                db_url = os.environ.get("DATABASE_URL")
                if db_url:
                    try:
                        import psycopg2
                        from ragonometrics.indexing import metadata

                        conn = psycopg2.connect(db_url)
                        metadata.record_failure(
                            conn,
                            "openai",
                            str(exc),
                            {"model": model, "max_output_tokens": max_output_tokens},
                        )
                        conn.close()
                    except Exception:
                        pass
                raise
            try:
                import time

                time.sleep(0.5 * (attempt + 1))
            except Exception:
                pass


def extract_json(text: str) -> Any:
    """Extract JSON from a string, tolerating markdown fences.

    Args:
        text: Raw model output text.

    Returns:
        Any: Parsed JSON value.
    """
    candidate = text.strip()
    if "```" in candidate:
        blocks = re.findall(r"```(?:json)?\s*(.*?)```", candidate, flags=re.DOTALL)
        if blocks:
            candidate = blocks[0].strip()

    start = min([i for i in [candidate.find("["), candidate.find("{")] if i != -1], default=-1)
    if start >= 0:
        candidate = candidate[start:]

    end_bracket = candidate.rfind("]")
    end_brace = candidate.rfind("}")
    end = max(end_bracket, end_brace)
    if end >= 0:
        candidate = candidate[: end + 1]

    return json.loads(candidate)


def summarize_paper(
    *,
    paper_path: Path,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    chunk_words: int = 800,
    overlap_words: int = 120,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> str:
    """Summarize a paper by chunking and merging summaries.

    Args:
        paper_path: Path to the paper file.
        model: OpenAI model name.
        api_key: Optional API key override.
        chunk_words: Chunk size in words.
        overlap_words: Overlap size in words.
        max_output_tokens: Maximum tokens per model response.

    Returns:
        str: Final summary.
    """
    paper = load_paper_text(paper_path)
    chunks = chunk_words(paper.text, chunk_words, overlap_words)
    if not chunks:
        raise ValueError("Paper text is empty.")

    client = build_client(api_key)

    chunk_summaries: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        prompt = f"Chunk {i} of {len(chunks)}:\n\n{chunk}"
        chunk_summary = call_openai(
            client,
            model=model,
            instructions=PIPELINE_SUMMARY_CHUNK_INSTRUCTIONS,
            user_input=prompt,
            max_output_tokens=max_output_tokens,
        )
        chunk_summaries.append(chunk_summary)

    merged_input = "\n\n".join(
        [f"Chunk summary {i}:\n{s}" for i, s in enumerate(chunk_summaries, start=1)]
    )
    final_summary = call_openai(
        client,
        model=model,
        instructions=PIPELINE_SUMMARY_MERGE_INSTRUCTIONS,
        user_input=merged_input,
        max_output_tokens=max_output_tokens,
    )
    return final_summary


def extract_citations(
    *,
    paper_path: Path,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> List[Dict[str, Any]]:
    """Extract citation metadata from a paper using OpenAI.

    Args:
        paper_path: Path to the paper file.
        model: OpenAI model name.
        api_key: Optional API key override.
        max_output_tokens: Maximum tokens per model response.

    Returns:
        List[Dict[str, Any]]: Citation objects.
    """
    paper = load_paper_text(paper_path)
    client = build_client(api_key)

    prompt = (
        "Extract all references from the text below.\n\n"
        f"{paper.references_text}"
    )
    raw = call_openai(
        client,
        model=model,
        instructions=PIPELINE_CITATION_EXTRACT_INSTRUCTIONS,
        user_input=prompt,
        max_output_tokens=max_output_tokens,
    )
    try:
        data = extract_json(raw)
        if not isinstance(data, list):
            raise ValueError("Citations output is not a list.")
        return data
    except Exception:
        # Retry once with stricter instructions, then fall back to empty list.
        retry_instructions = (
            PIPELINE_CITATION_EXTRACT_INSTRUCTIONS
            + " If you are unsure, return an empty JSON list: []."
        )
        raw_retry = call_openai(
            client,
            model=model,
            instructions=retry_instructions,
            user_input=prompt,
            max_output_tokens=max_output_tokens,
        )
        try:
            data = extract_json(raw_retry)
            if not isinstance(data, list):
                raise ValueError("Citations output is not a list.")
            return data
        except Exception:
            return []


def _count_numeric_citations(text: str) -> Dict[str, int]:
    """Count numeric citation occurrences like [12] or [3-5].

    Args:
        text: Body text to scan.

    Returns:
        Dict[str, int]: Citation id to count mapping.
    """
    counts: Dict[str, int] = {}
    pattern = re.compile(r"\[(\d{1,4}(?:\s*[-,]\s*\d{1,4})*)\]")
    for match in pattern.finditer(text):
        body = match.group(1)
        for part in re.split(r"\s*,\s*", body):
            if "-" in part:
                a, b = part.split("-", 1)
                try:
                    start = int(a.strip())
                    end = int(b.strip())
                except ValueError:
                    continue
                for i in range(start, end + 1):
                    key = str(i)
                    counts[key] = counts.get(key, 0) + 1
            else:
                try:
                    key = str(int(part.strip()))
                except ValueError:
                    continue
                counts[key] = counts.get(key, 0) + 1
    return counts


def _count_author_year(text: str, authors: List[str], year: Optional[int]) -> int:
    """Count author-year style citations in text.

    Args:
        text: Body text to scan.
        authors: Citation authors list.
        year: Citation year.

    Returns:
        int: Count of matches.
    """
    if not authors or not year:
        return 0
    last = authors[0].split()[-1]
    if not last:
        return 0
    pattern = re.compile(
        rf"\b{re.escape(last)}\b.{0,40}\b{year}\b",
        re.IGNORECASE | re.DOTALL,
    )
    return len(pattern.findall(text))


def add_mention_counts(body_text: str, citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach mention counts to extracted citations.

    Args:
        body_text: Paper body text.
        citations: Citation objects to enrich.

    Returns:
        List[Dict[str, Any]]: Enriched citation objects.
    """
    numeric_counts = _count_numeric_citations(body_text)
    enriched: List[Dict[str, Any]] = []
    for c in citations:
        citation_id = c.get("citation_id")
        citation_id_norm = None
        if citation_id is not None:
            digits = re.findall(r"\d+", str(citation_id))
            if digits:
                citation_id_norm = digits[0]
        count = 0
        if citation_id_norm and citation_id_norm in numeric_counts:
            count = numeric_counts.get(citation_id_norm, 0)
        else:
            authors = c.get("authors") or []
            year = c.get("year")
            if isinstance(year, str) and year.isdigit():
                year = int(year)
            count = _count_author_year(body_text, authors, year)
        out = dict(c)
        out["mention_count"] = int(count)
        enriched.append(out)
    return enriched


def rank_citations(
    *,
    paper_path: Path,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> List[Dict[str, Any]]:
    """Rank citations by importance using OpenAI.

    Args:
        paper_path: Path to the paper file.
        model: OpenAI model name.
        api_key: Optional API key override.
        max_output_tokens: Maximum tokens per model response.

    Returns:
        List[Dict[str, Any]]: Ranked citations.
    """
    paper = load_paper_text(paper_path)
    citations = extract_citations(
        paper_path=paper_path,
        model=model,
        api_key=api_key,
        max_output_tokens=max_output_tokens,
    )
    citations_with_counts = add_mention_counts(paper.body_text, citations)
    payload = json.dumps(citations_with_counts, ensure_ascii=False)

    client = build_client(api_key)
    raw = call_openai(
        client,
        model=model,
        instructions=PIPELINE_CITATION_RANK_INSTRUCTIONS,
        user_input=f"Rank these citations:\n\n{payload}",
        max_output_tokens=max_output_tokens,
    )
    data = extract_json(raw)
    if not isinstance(data, list):
        raise ValueError("Rank output is not a list.")
    return data


def extract_metadata(
    *,
    paper_path: Path,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    include_citations: bool = True,
    include_ranked_citations: bool = False,
) -> Dict[str, Any]:
    """Extract paper metadata and optional citations.

    Args:
        paper_path: Path to the paper file.
        model: OpenAI model name.
        api_key: Optional API key override.
        max_output_tokens: Maximum tokens per model response.
        include_citations: Whether to include extracted citations.
        include_ranked_citations: Whether to include ranked citations.

    Returns:
        Dict[str, Any]: Extracted metadata.
    """
    paper = load_paper_text(paper_path)
    header_text = head_text(paper.text, max_chars=12_000)
    lines = _clean_lines(header_text, max_lines=80)

    title, title_end = _extract_title(lines)
    if not title:
        title = paper_path.stem
        title_end = 0

    authors = _extract_authors(lines, title_end)
    year = _extract_year(lines)
    dois = _extract_dois(paper.text)
    doi = dois[0] if dois else None
    abstract = _extract_abstract(lines)
    keywords = _extract_keywords(lines)

    metadata: Dict[str, Any] = {
        "title": title,
        "authors": authors,
        "year": year,
        "doi": doi,
        "dois": dois,
        "abstract": abstract,
        "keywords": keywords,
    }

    if include_citations:
        citations = extract_citations(
            paper_path=paper_path,
            model=model,
            api_key=api_key,
            max_output_tokens=max_output_tokens,
        )
        citations = add_mention_counts(paper.body_text, citations)
        metadata["citations"] = citations
        metadata["citations_used"] = [
            c for c in citations if int(c.get("mention_count", 0)) > 0
        ]

    if include_ranked_citations:
        metadata["citations_ranked"] = rank_citations(
            paper_path=paper_path,
            model=model,
            api_key=api_key,
            max_output_tokens=max_output_tokens,
        )

    return metadata

