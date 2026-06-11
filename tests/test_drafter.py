"""Tests for app/agents/drafter.py."""

from __future__ import annotations

from unittest.mock import patch

from app.agents.drafter import render_node, run_drafter
from app.schemas.rfp import SECTION_ORDER, SECTION_TITLES, Priority, ProposalSection, RFPExtraction, Requirement


class _FakeSectionLLM:
    """Stands in for ChatOpenAI.with_structured_output(ProposalSection)."""

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        human_content = messages[-1].content
        title = "Section"
        for line in human_content.splitlines():
            if line.startswith("Section to draft:"):
                title = line.split(":", 1)[1].strip()
                break
        return ProposalSection(title=title, content=f"Drafted content for {title}.")


def _sample_extraction() -> RFPExtraction:
    return RFPExtraction(
        company_name="Acme Corp",
        project_title="Widget Platform",
        submission_deadline="2026-01-01",
        budget_range="$100k-$200k",
        requirements=[
            Requirement(id="REQ-1", description="Build a widget", priority=Priority.high),
            Requirement(id="REQ-2", description="Provide support", priority=Priority.medium),
        ],
        evaluation_criteria=["Price", "Quality"],
        confidence_score=0.9,
        flags=[],
    )


class TestRunDrafter:
    def test_run_drafter_produces_all_sections_in_order(self):
        extraction = _sample_extraction()

        with patch("app.agents.drafter._get_llm", return_value=_FakeSectionLLM()):
            proposal, docx_bytes = run_drafter(extraction)

        assert [s.title for s in proposal.sections] == [
            SECTION_TITLES[key] for key in SECTION_ORDER
        ]
        for section in proposal.sections:
            assert section.content
        assert proposal.extraction == extraction

    def test_run_drafter_returns_valid_docx_bytes(self):
        extraction = _sample_extraction()

        with patch("app.agents.drafter._get_llm", return_value=_FakeSectionLLM()):
            _, docx_bytes = run_drafter(extraction)

        # .docx files are zip archives, which start with the "PK" signature.
        assert docx_bytes[:2] == b"PK"
        assert len(docx_bytes) > 0


class TestRenderNode:
    def test_render_node_orders_sections_and_renders_docx(self):
        extraction = _sample_extraction()
        sections = {
            "team": ProposalSection(title=SECTION_TITLES["team"], content="Team content."),
            "executive_summary": ProposalSection(
                title=SECTION_TITLES["executive_summary"], content="Summary content."
            ),
        }

        result = render_node({"extraction": extraction, "sections": sections})

        proposal = result["proposal"]
        assert [s.title for s in proposal.sections] == [
            SECTION_TITLES["executive_summary"],
            SECTION_TITLES["team"],
        ]
        assert result["docx_bytes"][:2] == b"PK"
