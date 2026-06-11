"""LangGraph drafting pipeline: one node per proposal section, then render."""

from __future__ import annotations

import os
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from app.schemas.rfp import SECTION_ORDER, SECTION_TITLES, Proposal, ProposalSection, RFPExtraction
from app.templates.proposal import render_proposal

GROUNDING_INSTRUCTIONS = """\
You are drafting a section of a project proposal in response to an RFP.

You are given ONLY the structured data extracted from the RFP, as JSON. You MUST
ground everything you write strictly in this data:
- Do not invent company names, statistics, certifications, past projects, client
  references, or team member names that are not present in the provided data.
- Reference requirements, evaluation criteria, deadlines, and budget figures only
  as they appear in the data.
- If information needed for a polished section is not available, write in general,
  professional terms rather than fabricating specifics.
- Write in a confident, persuasive, professional proposal tone.
- Return only the section content (no markdown headings) - the title is supplied
  separately.
"""

SECTION_INSTRUCTIONS: dict[str, str] = {
    "executive_summary": (
        "Write an executive summary that demonstrates understanding of the "
        "project_title and the issuing organization (company_name), and "
        "summarizes the proposed approach at a high level."
    ),
    "technical_approach": (
        "Write a technical approach section that addresses each item in "
        "'requirements', organized by priority, explaining how the proposed "
        "solution will meet each requirement."
    ),
    "timeline": (
        "Write a project timeline section that proposes a realistic, phased "
        "schedule for delivering the requirements, taking the "
        "submission_deadline into account if present."
    ),
    "pricing": (
        "Write a pricing section that frames the proposed cost relative to the "
        "budget_range if one is provided, and explains the value delivered "
        "for that investment."
    ),
    "team": (
        "Write a team section describing the roles and expertise needed to "
        "deliver the requirements and meet the evaluation_criteria."
    ),
}


class DrafterState(TypedDict, total=False):
    extraction: RFPExtraction
    sections: dict[str, ProposalSection]
    proposal: Proposal
    docx_bytes: bytes


def _get_llm() -> ChatOpenAI:
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    return ChatOpenAI(model=model, temperature=0.2)


def _make_section_node(section_key: str):
    """Create a LangGraph node that drafts a single proposal section."""

    def node(state: DrafterState) -> dict:
        extraction = state["extraction"]
        llm = _get_llm().with_structured_output(ProposalSection)

        messages = [
            SystemMessage(content=GROUNDING_INSTRUCTIONS),
            HumanMessage(
                content=(
                    f"EXTRACTED RFP DATA (JSON):\n{extraction.model_dump_json(indent=2)}\n\n"
                    f"Section to draft: {SECTION_TITLES[section_key]}\n"
                    f"Instructions: {SECTION_INSTRUCTIONS[section_key]}"
                )
            ),
        ]

        section = llm.invoke(messages)
        section = section.model_copy(update={"title": SECTION_TITLES[section_key]})

        sections = dict(state.get("sections", {}))
        sections[section_key] = section
        return {"sections": sections}

    return node


def render_node(state: DrafterState) -> dict:
    """Assemble drafted sections in order and render the final .docx."""
    extraction = state["extraction"]
    drafted = state.get("sections", {})

    ordered_sections = [drafted[key] for key in SECTION_ORDER if key in drafted]
    proposal = Proposal(sections=ordered_sections, extraction=extraction)
    docx_bytes = render_proposal(proposal)

    return {"proposal": proposal, "docx_bytes": docx_bytes}


def build_drafter_graph():
    """Build and compile the section-by-section drafting LangGraph graph."""
    graph = StateGraph(DrafterState)

    for section_key in SECTION_ORDER:
        graph.add_node(section_key, _make_section_node(section_key))
    graph.add_node("render", render_node)

    graph.set_entry_point(SECTION_ORDER[0])
    for current_key, next_key in zip(SECTION_ORDER, SECTION_ORDER[1:]):
        graph.add_edge(current_key, next_key)
    graph.add_edge(SECTION_ORDER[-1], "render")
    graph.add_edge("render", END)

    return graph.compile()


def run_drafter(extraction: RFPExtraction) -> tuple[Proposal, bytes]:
    """Run the full drafting pipeline, returning the proposal and .docx bytes."""
    graph = build_drafter_graph()
    result = graph.invoke({"extraction": extraction, "sections": {}})
    return result["proposal"], result["docx_bytes"]
