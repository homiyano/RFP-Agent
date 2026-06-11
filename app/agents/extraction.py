"""LangGraph extraction pipeline: parse -> extract -> validate."""

from __future__ import annotations

import os
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from app.parsers.document import parse_document, sections_to_text
from app.schemas.rfp import RFPExtraction

CONFIDENCE_THRESHOLD = 0.7
REQUIRED_FIELDS = ("company_name", "project_title", "submission_deadline")

EXTRACTION_SYSTEM_PROMPT = """\
You are an expert proposal analyst reviewing a Request for Proposal (RFP) document.

Extract the following information strictly from the provided document text:
- company_name: the organization issuing the RFP
- project_title: the title of the project or solicitation
- submission_deadline: the proposal submission deadline, normalized to ISO format
  (YYYY-MM-DD) if a date can be determined
- budget_range: the stated budget, funding ceiling, or estimated contract value
- requirements: a list of distinct requirements, each with a short id (e.g. "REQ-1"),
  a description, and a priority of "high", "medium", or "low" based on how the
  document characterizes it (e.g. "must", "shall" -> high; "should" -> medium;
  "may", "nice to have" -> low)
- evaluation_criteria: the criteria that will be used to evaluate proposals
- confidence_score: your confidence (0.0-1.0) that the extraction above is complete
  and accurate given the document text
- flags: any notes for a human reviewer, e.g. ambiguous or missing information

If a field cannot be determined from the document, leave it null (or an empty list,
as appropriate) rather than guessing. Do not invent information that is not present
in the document.
"""


class ExtractionState(TypedDict, total=False):
    filename: str
    file_bytes: bytes
    document_text: str
    extraction: RFPExtraction


def _get_llm() -> ChatOpenAI:
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    return ChatOpenAI(model=model, temperature=0)


def parse_node(state: ExtractionState) -> dict:
    """Parse the uploaded document into section-aware text."""
    sections = parse_document(state["filename"], state["file_bytes"])
    return {"document_text": sections_to_text(sections)}


def extract_node(state: ExtractionState) -> dict:
    """Run structured extraction over the parsed document text via the LLM."""
    llm = _get_llm().with_structured_output(RFPExtraction)
    messages = [
        SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=f"RFP DOCUMENT TEXT:\n\n{state['document_text']}"),
    ]
    extraction = llm.invoke(messages)
    return {"extraction": extraction}


def validate_node(state: ExtractionState) -> dict:
    """Validate extraction quality and append review flags as needed.

    Per spec, low confidence or missing fields never raise an error -
    they are surfaced as flags for human review while processing continues.
    """
    extraction = state["extraction"]
    flags = list(extraction.flags)

    if extraction.confidence_score < CONFIDENCE_THRESHOLD:
        flags.append(
            f"Low confidence score ({extraction.confidence_score:.2f}); "
            "manual review recommended."
        )

    for field_name in REQUIRED_FIELDS:
        if not getattr(extraction, field_name):
            flags.append(f"Missing required field: {field_name}")

    if extraction.submission_deadline and extraction.parsed_deadline() is None:
        flags.append(
            f"Submission deadline '{extraction.submission_deadline}' is not a "
            "valid ISO date (YYYY-MM-DD)."
        )

    if not extraction.requirements:
        flags.append("No requirements were extracted.")

    return {"extraction": extraction.model_copy(update={"flags": flags})}


def build_extraction_graph():
    """Build and compile the parse -> extract -> validate LangGraph graph."""
    graph = StateGraph(ExtractionState)
    graph.add_node("parse", parse_node)
    graph.add_node("extract", extract_node)
    graph.add_node("validate", validate_node)

    graph.set_entry_point("parse")
    graph.add_edge("parse", "extract")
    graph.add_edge("extract", "validate")
    graph.add_edge("validate", END)

    return graph.compile()


def run_extraction(filename: str, file_bytes: bytes) -> RFPExtraction:
    """Run the full extraction pipeline on an uploaded document."""
    graph = build_extraction_graph()
    result = graph.invoke({"filename": filename, "file_bytes": file_bytes})
    return result["extraction"]
