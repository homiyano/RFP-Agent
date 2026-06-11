"""Parsing and section-aware chunking for PDF and DOCX RFP documents."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

import fitz  # PyMuPDF
from docx import Document as DocxDocument

FALLBACK_CHUNK_SIZE = 1000

# Patterns used to detect a line that introduces a new logical section.
_HEADING_PATTERNS = [
    re.compile(r"^[A-Z][A-Z0-9 \-/&,.()]{2,}$"),  # ALL CAPS heading
    re.compile(r"^.{1,80}:$"),  # short line ending in ':'
    re.compile(r"^\s*\d+(\.\d+)*[.)]\s+\S"),  # numbered section, e.g. "1." or "2.1)"
]


class UnsupportedFileTypeError(ValueError):
    """Raised when the uploaded file is neither a PDF nor a DOCX."""


@dataclass
class DocumentSection:
    """A chunk of an RFP document, optionally introduced by a heading."""

    heading: str | None
    content: str


def extract_text_from_pdf(data: bytes) -> str:
    """Extract raw text from a PDF file's bytes using PyMuPDF."""
    with fitz.open(stream=data, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


def extract_text_from_docx(data: bytes) -> str:
    """Extract raw text from a DOCX file's bytes using python-docx."""
    document = DocxDocument(io.BytesIO(data))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def is_heading(line: str) -> bool:
    """Heuristically determine whether a line introduces a new section."""
    stripped = line.strip()
    if not stripped or len(stripped) > 100:
        return False
    return any(pattern.match(stripped) for pattern in _HEADING_PATTERNS)


def chunk_by_section(text: str) -> list[DocumentSection]:
    """Split text into sections based on detected heading lines."""
    sections: list[DocumentSection] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        if is_heading(line):
            content = "\n".join(current_lines).strip()
            if content or current_heading is not None:
                sections.append(DocumentSection(heading=current_heading, content=content))
            current_heading = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    content = "\n".join(current_lines).strip()
    if content or current_heading is not None:
        sections.append(DocumentSection(heading=current_heading, content=content))

    return [s for s in sections if s.heading or s.content]


def chunk_fallback(text: str, chunk_size: int = FALLBACK_CHUNK_SIZE) -> list[DocumentSection]:
    """Split text into fixed-size chunks when no section structure is detected."""
    text = text.strip()
    if not text:
        return []
    return [
        DocumentSection(heading=None, content=text[i : i + chunk_size])
        for i in range(0, len(text), chunk_size)
    ]


def parse_document(filename: str, data: bytes) -> list[DocumentSection]:
    """Parse a PDF or DOCX file and chunk it by logical section.

    Falls back to fixed-size chunks if no heading structure is detected
    (i.e. the heuristics found at most one section).
    """
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if suffix == "pdf":
        text = extract_text_from_pdf(data)
    elif suffix == "docx":
        text = extract_text_from_docx(data)
    else:
        raise UnsupportedFileTypeError(f"Unsupported file type: {filename}")

    sections = chunk_by_section(text)

    headed_sections = [s for s in sections if s.heading is not None]
    if not headed_sections:
        return chunk_fallback(text)

    return sections


def sections_to_text(sections: list[DocumentSection]) -> str:
    """Render parsed sections back into a single text blob for LLM context."""
    parts: list[str] = []
    for section in sections:
        if section.heading:
            parts.append(f"## {section.heading}\n{section.content}".strip())
        else:
            parts.append(section.content)
    return "\n\n".join(p for p in parts if p)
