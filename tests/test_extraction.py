"""Tests for app/agents/extraction.py."""

from __future__ import annotations

import io
from unittest.mock import patch

from docx import Document as DocxDocument

from app.agents.extraction import parse_node, run_extraction, validate_node
from app.schemas.rfp import Priority, RFPExtraction, Requirement


def _build_docx_bytes(paragraphs: list[str]) -> bytes:
    document = DocxDocument()
    for text in paragraphs:
        document.add_paragraph(text)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


class _FakeStructuredLLM:
    def __init__(self, result):
        self._result = result

    def invoke(self, messages):
        return self._result


class _FakeChatModel:
    def __init__(self, result):
        self._result = result

    def with_structured_output(self, schema):
        return _FakeStructuredLLM(self._result)


class TestParseNode:
    def test_parse_node_sets_document_text(self):
        data = _build_docx_bytes(["SECTION ONE", "Some content."])
        result = parse_node({"filename": "rfp.docx", "file_bytes": data})
        assert "SECTION ONE" in result["document_text"]
        assert "Some content." in result["document_text"]


class TestValidateNode:
    def test_high_confidence_complete_extraction_has_no_new_flags(self):
        extraction = RFPExtraction(
            company_name="Acme",
            project_title="Widget Build",
            submission_deadline="2026-01-01",
            budget_range="$100k-$200k",
            requirements=[
                Requirement(id="REQ-1", description="Build widget", priority=Priority.high)
            ],
            evaluation_criteria=["Price"],
            confidence_score=0.9,
            flags=[],
        )

        result = validate_node({"extraction": extraction})

        assert result["extraction"].flags == []

    def test_low_confidence_adds_flag_but_does_not_raise(self):
        extraction = RFPExtraction(
            company_name="Acme",
            project_title="Widget Build",
            submission_deadline="2026-01-01",
            requirements=[
                Requirement(id="REQ-1", description="Build widget", priority=Priority.high)
            ],
            confidence_score=0.4,
        )

        result = validate_node({"extraction": extraction})

        flags = result["extraction"].flags
        assert any("Low confidence" in f for f in flags)
        assert result["extraction"].confidence_score == 0.4

    def test_missing_required_fields_are_flagged(self):
        extraction = RFPExtraction(
            confidence_score=0.9,
            requirements=[Requirement(id="REQ-1", description="x", priority=Priority.low)],
        )

        result = validate_node({"extraction": extraction})

        flags = result["extraction"].flags
        assert any("company_name" in f for f in flags)
        assert any("project_title" in f for f in flags)
        assert any("submission_deadline" in f for f in flags)

    def test_invalid_deadline_format_is_flagged(self):
        extraction = RFPExtraction(
            company_name="Acme",
            project_title="Widget",
            submission_deadline="next Friday",
            confidence_score=0.9,
            requirements=[Requirement(id="REQ-1", description="x", priority=Priority.low)],
        )

        result = validate_node({"extraction": extraction})

        assert any("not a valid ISO date" in f for f in result["extraction"].flags)

    def test_no_requirements_is_flagged(self):
        extraction = RFPExtraction(
            company_name="Acme",
            project_title="Widget",
            submission_deadline="2026-01-01",
            confidence_score=0.9,
        )

        result = validate_node({"extraction": extraction})

        assert any("No requirements" in f for f in result["extraction"].flags)


class TestRunExtraction:
    def test_run_extraction_end_to_end_with_mocked_llm(self):
        fake_extraction = RFPExtraction(
            company_name="Acme",
            project_title="Widget Build",
            submission_deadline="2026-01-01",
            budget_range="$100k-$200k",
            requirements=[
                Requirement(id="REQ-1", description="Build widget", priority=Priority.high)
            ],
            evaluation_criteria=["Price", "Quality"],
            confidence_score=0.85,
            flags=[],
        )
        data = _build_docx_bytes(["SECTION ONE", "Some content describing the RFP."])

        with patch(
            "app.agents.extraction._get_llm", return_value=_FakeChatModel(fake_extraction)
        ):
            result = run_extraction("rfp.docx", data)

        assert result.company_name == "Acme"
        assert result.confidence_score == 0.85
        assert result.flags == []

    def test_run_extraction_flags_low_confidence(self):
        fake_extraction = RFPExtraction(
            company_name="Acme",
            project_title="Widget Build",
            submission_deadline="2026-01-01",
            requirements=[
                Requirement(id="REQ-1", description="Build widget", priority=Priority.high)
            ],
            confidence_score=0.5,
            flags=[],
        )
        data = _build_docx_bytes(["SECTION ONE", "Some content."])

        with patch(
            "app.agents.extraction._get_llm", return_value=_FakeChatModel(fake_extraction)
        ):
            result = run_extraction("rfp.docx", data)

        assert any("Low confidence" in f for f in result.flags)
