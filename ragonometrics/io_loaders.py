from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import subprocess
from typing import Dict, List, Optional

try:
    from pdf2image import convert_from_path
    import pytesseract
    from PIL import Image
except Exception:
    convert_from_path = None
    pytesseract = None
    Image = None


@dataclass(frozen=True)
class LoadedText:
    """Container for loaded text and its source path.

    Attributes:
        text: Extracted text content.
        path: Source file path.
    """

    text: str
    path: Path


def _ocr_pdf(path: Path, *, first_page: int | None = None, last_page: int | None = None) -> List[str]:
    if not convert_from_path or not pytesseract:
        raise RuntimeError("OCR tools are not available. Install pdf2image+pytesseract to enable OCR.")
    images = convert_from_path(str(path), first_page=first_page, last_page=last_page)
    return [pytesseract.image_to_string(im) for im in images]


def run_pdftotext(path: Path) -> str:
    """Extract text from a PDF using `pdftotext`, with OCR fallback.

    Uses OCR if `FORCE_OCR` is set or if `pdftotext` fails or times out.

    Args:
        path: PDF file path.

    Returns:
        str: Extracted text.

    Raises:
        RuntimeError: If extraction fails and OCR tools are not available.
    """
    if os.environ.get("FORCE_OCR"):
        texts = _ocr_pdf(path)
        return "\n\n".join(texts)

    try:
        result = subprocess.run(
            ["pdftotext", "-layout", "-q", str(path), "-"],
            check=True,
            capture_output=True,
            text=False,
            timeout=30,
        )
    except FileNotFoundError as exc:
        try:
            return "\n\n".join(_ocr_pdf(path))
        except Exception:
            raise RuntimeError(
                "pdftotext is not available. Install Poppler/TeXLive or add pdf2image+pytesseract for OCR fallback."
            ) from exc
    except subprocess.TimeoutExpired as exc:
        try:
            return "\n\n".join(_ocr_pdf(path))
        except Exception:
            raise RuntimeError("pdftotext timed out and OCR tools are not available") from exc
    return result.stdout.decode("utf-8", errors="replace")


def run_pdftotext_pages(path: Path) -> List[str]:
    """Extract text from each page of a PDF.

    Falls back to returning the full text as a single page if per-page
    extraction is not possible.

    Args:
        path: PDF file path.

    Returns:
        List[str]: Per-page text in order.
    """
    if os.environ.get("FORCE_OCR"):
        return _ocr_pdf(path)

    try:
        info = subprocess.run(["pdfinfo", str(path)], check=True, capture_output=True, text=True)
    except FileNotFoundError:
        return [run_pdftotext(path)]

    pages = 0
    for raw_line in info.stdout.splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        if key.strip().lower() == "pages":
            try:
                pages = int(value.strip())
            except Exception:
                pages = 0
            break

    if pages <= 0:
        return [run_pdftotext(path)]

    page_texts: List[str] = []
    for p in range(1, pages + 1):
        try:
            result = subprocess.run(
                ["pdftotext", "-layout", "-q", "-f", str(p), "-l", str(p), str(path), "-"],
                check=True,
                capture_output=True,
                text=False,
                timeout=30,
            )
        except FileNotFoundError:
            return [run_pdftotext(path)]
        except subprocess.TimeoutExpired:
            try:
                texts = _ocr_pdf(path, first_page=p, last_page=p)
                return texts or [run_pdftotext(path)]
            except Exception:
                return [run_pdftotext(path)]
        page_texts.append(result.stdout.decode("utf-8", errors="replace"))

    return page_texts


def load_pdf(path: Path) -> LoadedText:
    """Load a PDF file and return extracted text with metadata.

    Args:
        path: PDF file path.

    Returns:
        LoadedText: Extracted text and source path.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    text = run_pdftotext(path)
    return LoadedText(text=text, path=path)


def load_text_file(path: Path, encoding: str = "utf-8") -> LoadedText:
    """Load a text file and return its contents with metadata.

    Args:
        path: Text file path.
        encoding: File encoding.

    Returns:
        LoadedText: Loaded text and source path.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding=encoding, errors="replace")
    return LoadedText(text=text, path=path)


def normalize_text(text: str) -> str:
    """Normalize extracted text by removing nulls and collapsing whitespace.

    Args:
        text: Raw extracted text.

    Returns:
        str: Normalized text.
    """
    text = text.replace("\u0000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def trim_words(text: str, max_words: int) -> str:
    """Trim text to a maximum number of words.

    Args:
        text: Input text.
        max_words: Maximum number of words to keep. If <= 0, return input.

    Returns:
        str: Trimmed text.
    """
    if max_words <= 0:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def chunk_words(text: str, chunk_words: int, overlap_words: int) -> List[str]:
    """Split text into overlapping word chunks.

    Args:
        text: Input text to split.
        chunk_words: Target words per chunk.
        overlap_words: Number of words to overlap between chunks.

    Returns:
        List[str]: Chunked text segments.
    """
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_words - overlap_words)
    chunks = []
    i = 0
    while i < len(words):
        j = min(len(words), i + chunk_words)
        chunks.append(" ".join(words[i:j]))
        if j == len(words):
            break
        i += step
    return chunks


def chunk_pages(page_texts: List[str], chunk_words_count: int, overlap_words: int) -> List[Dict]:
    """Chunk page texts into word windows with provenance metadata.

    Args:
        page_texts: List of per-page text.
        chunk_words_count: Target words per chunk.
        overlap_words: Number of words to overlap between chunks.

    Returns:
        List[Dict]: Dicts with "text", "page", "start_word", and "end_word".
    """
    sections: Optional[List[str]] = None
    if os.environ.get("SECTION_AWARE_CHUNKING"):
        sections = infer_sections(page_texts)
    chunks: List[Dict] = []
    for page_idx, page_text in enumerate(page_texts, start=1):
        words = page_text.split()
        if not words:
            continue
        step = max(1, chunk_words_count - overlap_words)
        i = 0
        while i < len(words):
            j = min(len(words), i + chunk_words_count)
            chunk_text = " ".join(words[i:j])
            chunk = {
                "text": chunk_text,
                "page": page_idx,
                "start_word": i,
                "end_word": j - 1,
            }
            if sections:
                chunk["section"] = sections[page_idx - 1]
            chunks.append(chunk)
            if j == len(words):
                break
            i += step
    return chunks


SECTION_PATTERNS = [
    ("abstract", re.compile(r"^\s*abstract\b", re.IGNORECASE)),
    ("introduction", re.compile(r"^\s*(introduction|background)\b", re.IGNORECASE)),
    ("methods", re.compile(r"^\s*(methods|methodology|data and methods)\b", re.IGNORECASE)),
    ("results", re.compile(r"^\s*(results|findings)\b", re.IGNORECASE)),
    ("discussion", re.compile(r"^\s*(discussion)\b", re.IGNORECASE)),
    ("conclusion", re.compile(r"^\s*(conclusion|conclusions)\b", re.IGNORECASE)),
    ("references", re.compile(r"^\s*(references|bibliography|works cited)\b", re.IGNORECASE)),
    ("appendix", re.compile(r"^\s*appendix\b", re.IGNORECASE)),
]


def infer_sections(page_texts: List[str]) -> List[str]:
    """Infer a coarse section label for each page based on headings."""
    current = "unknown"
    labels: List[str] = []
    for page_text in page_texts:
        section = current
        for raw_line in page_text.splitlines()[:80]:
            line = raw_line.strip()
            if not line:
                continue
            for name, pattern in SECTION_PATTERNS:
                if pattern.match(line):
                    section = name
                    break
            if section != current:
                break
        labels.append(section)
        current = section
    return labels
