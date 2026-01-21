"""Pydantic schemas for LLM extraction results."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class EvidenceType(str, Enum):
    """Types of evidence supporting findings."""

    EMPIRICAL = "empirical"
    THEORETICAL = "theoretical"
    METHODOLOGICAL = "methodological"
    CASE_STUDY = "case_study"
    SURVEY = "survey"
    EXPERIMENTAL = "experimental"
    QUALITATIVE = "qualitative"
    QUANTITATIVE = "quantitative"
    MIXED = "mixed"


class SignificanceLevel(str, Enum):
    """Significance level of findings."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SupportType(str, Enum):
    """Types of support for claims."""

    DATA = "data"
    CITATION = "citation"
    LOGIC = "logic"
    EXAMPLE = "example"
    AUTHORITY = "authority"


class Methodology(BaseModel):
    """Research methodology details."""

    approach: str | None = Field(
        default=None,
        description="Overall research approach (qualitative, quantitative, mixed)",
    )
    design: str | None = Field(
        default=None,
        description="Research design (case study, experiment, survey, etc.)",
    )
    data_sources: list[str] = Field(
        default_factory=list,
        description="Sources of data used in the research",
    )
    analysis_methods: list[str] = Field(
        default_factory=list,
        description="Methods used for data analysis",
    )
    sample_size: str | None = Field(
        default=None,
        description="Sample size or scope of data",
    )
    time_period: str | None = Field(
        default=None,
        description="Time period covered by the research",
    )


class KeyFinding(BaseModel):
    """A key finding from the research."""

    finding: str = Field(description="The finding statement")
    evidence_type: EvidenceType | None = Field(
        default=EvidenceType.EMPIRICAL,
        description="Type of evidence supporting this finding",
    )
    significance: SignificanceLevel | None = Field(
        default=SignificanceLevel.MEDIUM,
        description="Significance level of this finding",
    )
    page_reference: str | None = Field(
        default=None,
        description="Page reference in the source document",
    )


class KeyClaim(BaseModel):
    """A key claim made in the paper."""

    claim: str = Field(description="The claim statement")
    support_type: SupportType | None = Field(
        default=SupportType.LOGIC,
        description="How this claim is supported",
    )
    page_reference: str | None = Field(
        default=None,
        description="Page reference in the source document",
    )
    strength: SignificanceLevel | None = Field(
        default=SignificanceLevel.MEDIUM,
        description="Strength of the claim and its support",
    )


class PaperExtraction(BaseModel):
    """Complete extraction from an academic paper."""

    # Core content
    thesis_statement: str | None = Field(
        default=None,
        description="The main thesis or argument of the paper",
    )
    research_questions: list[str] = Field(
        default_factory=list,
        description="Research questions addressed by the paper",
    )
    theoretical_framework: str | None = Field(
        default=None,
        description="Theoretical framework or lens used",
    )

    # Methodology
    methodology: Methodology = Field(
        default_factory=Methodology,
        description="Research methodology details",
    )

    # Findings and claims
    key_findings: list[KeyFinding] = Field(
        default_factory=list,
        description="Key findings from the research",
    )
    key_claims: list[KeyClaim] = Field(
        default_factory=list,
        description="Key claims made in the paper",
    )

    # Summary sections
    conclusions: str | None = Field(
        default=None,
        description="Main conclusions of the paper",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Acknowledged limitations of the research",
    )
    future_directions: list[str] = Field(
        default_factory=list,
        description="Suggested future research directions",
    )

    # Classification
    contribution_summary: str | None = Field(
        default=None,
        description="Brief summary of the paper's contribution",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Searchable keywords: concepts, methods, theories, phenomena",
    )
    discipline_tags: list[str] = Field(
        default_factory=list,
        description="Relevant discipline or topic tags",
    )

    # Metadata
    extraction_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score for this extraction (0-1)",
    )
    extraction_notes: str | None = Field(
        default=None,
        description="Notes about extraction quality or issues",
    )

    def to_index_dict(self) -> dict:
        """Convert to dictionary for indexing."""
        return {
            "thesis_statement": self.thesis_statement,
            "research_questions": self.research_questions,
            "theoretical_framework": self.theoretical_framework,
            "methodology": self.methodology.model_dump(),
            "key_findings": [f.model_dump() for f in self.key_findings],
            "key_claims": [c.model_dump() for c in self.key_claims],
            "conclusions": self.conclusions,
            "limitations": self.limitations,
            "future_directions": self.future_directions,
            "contribution_summary": self.contribution_summary,
            "keywords": self.keywords,
            "discipline_tags": self.discipline_tags,
            "extraction_confidence": self.extraction_confidence,
        }


class ExtractionResult(BaseModel):
    """Result of an extraction attempt."""

    paper_id: str = Field(description="ID of the paper")
    success: bool = Field(description="Whether extraction succeeded")
    extraction: PaperExtraction | None = Field(
        default=None,
        description="Extracted content if successful",
    )
    error: str | None = Field(
        default=None,
        description="Error message if extraction failed",
    )
    duration_seconds: float = Field(
        default=0.0,
        description="Time taken for extraction",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When extraction was performed",
    )
    model_used: str | None = Field(
        default=None,
        description="LLM model used for extraction",
    )
    input_tokens: int = Field(
        default=0,
        description="Number of input tokens used",
    )
    output_tokens: int = Field(
        default=0,
        description="Number of output tokens generated",
    )
