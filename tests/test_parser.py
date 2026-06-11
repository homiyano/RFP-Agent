"""Tests for app/parsers/document.py."""

from __future__ import annotations

import io

import fitz
import pytest
from docx import Document as DocxDocument

from app.parsers.document import (
    UnsupportedFileTypeError,
    chunk_by_section,
    chunk_fallback,
    extract_text_from_docx,
    extract_text_from_pdf,
    is_heading,
    parse_document,
    sections_to_text,
)


def _build_docx_bytes(paragraphs: list[str]) -> bytes:
    document = DocxDocument()
    for text in paragraphs:
        document.add_paragraph(text)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _build_pdf_bytes(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


class TestIsHeading:
    def test_all_caps_line_is_heading(self):
        assert is_heading("COMPANY OVERVIEW")

    def test_colon_terminated_line_is_heading(self):
        assert is_heading("Evaluation Criteria:")

    def test_numbered_section_is_heading(self):
        assert is_heading("1. Scope of Work")
        assert is_heading("2.1) Sub-section detail")

    def test_regular_sentence_is_not_heading(self):
        assert not is_heading("This is a normal sentence describing the project.")

    def test_blank_line_is_not_heading(self):
        assert not is_heading("   ")


class TestChunkBySection:
    def test_splits_on_detected_headings(self):
        text = "\n".join(
            [
                "COMPANY OVERVIEW",
                "This is the overview text.",
                "1. Scope of Work",
                "Details about scope.",
                "Evaluation Criteria:",
                "Criteria details.",
            ]
        )

        sections = chunk_by_section(text)

        assert [s.heading for s in sections] == [
            "COMPANY OVERVIEW",
            "1. Scope of Work",
            "Evaluation Criteria:",
        ]
        assert sections[0].content == "This is the overview text."
        assert sections[1].content == "Details about scope."
        assert sections[2].content == "Criteria details."

    def test_no_headings_yields_single_unheaded_section(self):
        text = "just some plain prose with no structure at all."
        sections = chunk_by_section(text)
        assert len(sections) == 1
        assert sections[0].heading is None


class TestChunkFallback:
    def test_chunks_into_fixed_size_pieces(self):
        text = "a" * 2500
        chunks = chunk_fallback(text, chunk_size=1000)
        assert len(chunks) == 3
        assert all(c.heading is None for c in chunks)
        assert len(chunks[0].content) == 1000
        assert len(chunks[1].content) == 1000
        assert len(chunks[2].content) == 500

    def test_empty_text_yields_no_chunks(self):
        assert chunk_fallback("   ") == []


class TestParseDocument:
    def test_unsupported_extension_raises(self):
        with pytest.raises(UnsupportedFileTypeError):
            parse_document("rfp.txt", b"hello")

    def test_docx_with_structure_is_chunked_by_section(self):
        data = _build_docx_bytes(
            [
                "COMPANY OVERVIEW",
                "Acme Corp is seeking a vendor.",
                "1. Scope of Work",
                "Build a widget.",
                "Evaluation Criteria:",
                "Price and quality.",
            ]
        )

        sections = parse_document("rfp.docx", data)

        headings = [s.heading for s in sections]
        assert "COMPANY OVERVIEW" in headings
        assert "1. Scope of Work" in headings
        assert "Evaluation Criteria:" in headings

    def test_docx_without_structure_falls_back_to_fixed_chunks(self):
        long_text = "this is unstructured prose. " * 100  # > 1000 chars, no headings
        data = _build_docx_bytes([long_text])

        sections = parse_document("rfp.docx", data)

        assert len(sections) > 1
        assert all(s.heading is None for s in sections)
        assert all(len(s.content) <= 1000 for s in sections)

    def test_extract_text_from_docx(self):
        data = _build_docx_bytes(["Hello", "World"])
        text = extract_text_from_docx(data)
        assert "Hello" in text
        assert "World" in text

    def test_extract_text_from_pdf(self):
        data = _build_pdf_bytes("Hello PDF World")
        text = extract_text_from_pdf(data)
        assert "Hello" in text

    def test_parse_pdf_document(self):
        data = _build_pdf_bytes("Hello PDF World")
        sections = parse_document("rfp.pdf", data)
        assert sections
        assert "Hello" in sections_to_text(sections)


class TestSectionsToText:
    def test_renders_headings_and_content(self):
        sections = chunk_by_section(
            "\n".join(["SECTION ONE", "Some content."])
        )
        text = sections_to_text(sections)
        assert "## SECTION ONE" in text
        assert "Some content." in text
