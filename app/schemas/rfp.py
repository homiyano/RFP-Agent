"""Pydantic schemas for RFP extraction and proposal generation."""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class Priority(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class Requirement(BaseModel):
    """A single requirement extracted from an RFP."""

    id: str = Field(description="Short identifier for the requirement, e.g. 'REQ-1'")
    description: str = Field(description="Full text of the requirement")
    priority: Priority = Field(description="Priority of the requirement")


class RFPExtraction(BaseModel):
    """Structured data extracted from an RFP document."""

    company_name: str | None = Field(
        default=None, description="Name of the organization issuing the RFP"
    )
    project_title: str | None = Field(
        default=None, description="Title of the project or solicitation"
    )
    submission_deadline: str | None = Field(
        default=None,
        description="Submission deadline as stated in the document, ISO format (YYYY-MM-DD) if possible",
    )
    budget_range: str | None = Field(
        default=None, description="Budget or estimated contract value, if stated"
    )
    requirements: list[Requirement] = Field(
        default_factory=list, description="List of requirements extracted from the RFP"
    )
    evaluation_criteria: list[str] = Field(
        default_factory=list, description="List of evaluation criteria used to score proposals"
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Model's confidence that the extraction is complete and accurate",
    )
    flags: list[str] = Field(
        default_factory=list,
        description="Notes for human review, e.g. missing fields or low confidence",
    )

    def parsed_deadline(self) -> date | None:
        """Attempt to parse submission_deadline into a date object."""
        if not self.submission_deadline:
            return None
        try:
            return date.fromisoformat(self.submission_deadline)
        except ValueError:
            return None


class ProposalSection(BaseModel):
    """A single section of the drafted proposal."""

    title: str = Field(description="Section heading")
    content: str = Field(description="Body text of the section")


class Proposal(BaseModel):
    """A full drafted proposal composed of grounded sections."""

    sections: list[ProposalSection] = Field(default_factory=list)
    extraction: RFPExtraction


SECTION_ORDER: list[str] = [
    "executive_summary",
    "technical_approach",
    "timeline",
    "pricing",
    "team",
]

SECTION_TITLES: dict[str, str] = {
    "executive_summary": "Executive Summary",
    "technical_approach": "Technical Approach",
    "timeline": "Project Timeline",
    "pricing": "Pricing",
    "team": "Team",
}
