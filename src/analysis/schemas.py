from __future__ import annotations

"""Pydantic schemas for LLM extraction results."""

from datetime import datetime
from enum import Enum
from typing import ClassVar

from pydantic import BaseModel, Field, model_validator

from src.analysis.dimensions import (
    DEFAULT_DIMENSION_PROFILE,
    LEGACY_PROFILE_ID,
    build_legacy_dimension_profile,
    get_default_dimension_registry,
)

_LEGACY_PROFILE = build_legacy_dimension_profile()


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


class ReferenceEntry(BaseModel):
    """A parsed entry from the paper's bibliography/reference list."""

    raw_text: str = Field(description="The raw bibliography entry text")
    parsed_title: str | None = Field(
        default=None,
        description="Parsed title of the referenced work",
    )
    parsed_authors: str | None = Field(
        default=None,
        description="Parsed author(s) of the referenced work",
    )
    parsed_year: int | None = Field(
        default=None,
        description="Parsed publication year",
    )
    parsed_doi: str | None = Field(
        default=None,
        description="Parsed DOI if present",
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

    # Bibliography / reference list
    reference_list: list[ReferenceEntry] = Field(
        default_factory=list,
        description="Parsed bibliography entries (up to 50) for citation graph matching",
    )

    # Document type classification
    document_type: str | None = Field(
        default=None,
        description="Classified document type (e.g., research_paper, book_monograph)",
    )

    # Quality assessment
    quality_rating: int | None = Field(
        default=None,
        ge=1,
        le=5,
        description="Paper quality rating (1-5): methodology rigor, evidence strength, contribution significance",
    )
    quality_explanation: str | None = Field(
        default=None,
        description="Brief explanation for the quality rating",
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
            "reference_list": [r.model_dump() for r in self.reference_list],
            "document_type": self.document_type,
            "quality_rating": self.quality_rating,
            "quality_explanation": self.quality_explanation,
            "extraction_confidence": self.extraction_confidence,
        }


class DimensionedExtraction(BaseModel):
    """Canonical extraction model keyed by stable dimension IDs."""

    paper_id: str = Field(description="Zotero item key")
    profile_id: str = Field(
        default=DEFAULT_DIMENSION_PROFILE,
        description="Active dimension profile identifier",
    )
    profile_version: str = Field(
        default=_LEGACY_PROFILE.version,
        description="Version of the dimension profile used for extraction",
    )
    profile_fingerprint: str = Field(
        default=_LEGACY_PROFILE.fingerprint,
        description="Stable fingerprint of the exact profile snapshot used",
    )
    prompt_version: str = Field(
        description="Semantic analysis prompt version, e.g. '2.0.0'",
    )
    extraction_model: str = Field(description="LLM model used for extraction")
    extracted_at: str = Field(description="ISO 8601 timestamp of extraction")
    dimensions: dict[str, str | None] = Field(
        default_factory=dict,
        description="Canonical dimension values keyed by stable dimension ID",
    )
    dimension_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of active dimensions with non-None answers",
    )
    coverage_flags: list[str] = Field(
        default_factory=list,
        description="Coverage flags derived from the active profile",
    )

    @model_validator(mode="after")
    def _populate_profile_defaults(self) -> "DimensionedExtraction":
        registry = get_default_dimension_registry()
        normalized = dict(self.dimensions)
        profile = registry.profiles.get(self.profile_id)
        if profile is not None:
            self.profile_id = profile.profile_id
            if not self.profile_version:
                self.profile_version = profile.version
            if not self.profile_fingerprint:
                self.profile_fingerprint = profile.fingerprint
            for dimension in profile.dimensions:
                normalized.setdefault(dimension.id, None)
        self.dimensions = normalized
        return self

    @property
    def dimension_map(self) -> dict[str, str | None]:
        return dict(self.dimensions)

    def get_dimension(self, identifier: str) -> str | None:
        if identifier in self.dimensions:
            return self.dimensions.get(identifier)

        registry = get_default_dimension_registry()
        profile = registry.profiles.get(self.profile_id)
        if profile is not None:
            resolved = registry.resolve_optional_dimension(
                identifier,
                profile_id=profile.profile_id,
            )
            if resolved:
                return self.dimensions.get(resolved.id)
        return None

    def get_role(self, role_name: str) -> str | None:
        registry = get_default_dimension_registry()
        resolved = registry.resolve_role(role_name, profile_id=self.profile_id)
        if resolved:
            return self.dimensions.get(resolved.id)
        return None

    def non_none_dimensions(self) -> dict[str, str]:
        return {
            dimension_id: value
            for dimension_id, value in self.dimensions.items()
            if value is not None
        }

    def to_index_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "profile_fingerprint": self.profile_fingerprint,
            "prompt_version": self.prompt_version,
            "extraction_model": self.extraction_model,
            "extracted_at": self.extracted_at,
            "dimensions": dict(self.dimensions),
            "dimension_coverage": self.dimension_coverage,
            "coverage_flags": list(self.coverage_flags),
        }

    @classmethod
    def from_record(
        cls,
        record: dict | "SemanticAnalysis" | "DimensionedExtraction",
    ) -> "DimensionedExtraction":
        if isinstance(record, cls):
            return record
        if hasattr(record, "to_dimensioned_extraction"):
            return record.to_dimensioned_extraction()
        if isinstance(record, dict) and "extraction" in record and isinstance(record["extraction"], dict):
            record = record["extraction"]
        if isinstance(record, dict):
            dimensions = record.get("dimensions")
            if isinstance(dimensions, dict):
                return cls(**record)
            return SemanticAnalysis(**record).to_dimensioned_extraction()
        raise TypeError(f"Unsupported extraction record type: {type(record)!r}")


class SemanticAnalysis(BaseModel):
    """Legacy adapter over the canonical map-based dimension model.

    This model preserves the original ``qNN_*`` fields for backward
    compatibility while also carrying canonical profile metadata and the
    portable ``dimensions`` mapping used by the new storage model.
    """

    # Identification & metadata
    paper_id: str = Field(description="Zotero item key")
    profile_id: str = Field(
        default=DEFAULT_DIMENSION_PROFILE,
        description="Active dimension profile identifier",
    )
    profile_version: str = Field(
        default=_LEGACY_PROFILE.version,
        description="Version of the dimension profile used for extraction",
    )
    profile_fingerprint: str = Field(
        default=_LEGACY_PROFILE.fingerprint,
        description="Stable fingerprint of the exact profile snapshot used",
    )
    prompt_version: str = Field(
        description="Semantic analysis prompt version, e.g. '2.0.0'",
    )
    extraction_model: str = Field(
        description="LLM model used for extraction",
    )
    extracted_at: str = Field(
        description="ISO 8601 timestamp of extraction",
    )

    # Pass 1: Research Core
    q01_research_question: str | None = Field(
        default=None,
        description="What research questions or objectives does this work address?",
    )
    q02_thesis: str | None = Field(
        default=None,
        description="What is the central thesis or main argument?",
    )
    q03_key_claims: str | None = Field(
        default=None,
        description="What are the key claims or propositions made?",
    )
    q04_evidence: str | None = Field(
        default=None,
        description="What evidence is presented and how strong is it?",
    )
    q05_limitations: str | None = Field(
        default=None,
        description="What limitations are acknowledged or apparent?",
    )

    # Pass 2: Methodology
    q06_paradigm: str | None = Field(
        default=None,
        description="What research paradigm underlies this work?",
    )
    q07_methods: str | None = Field(
        default=None,
        description="What methods and analytical techniques are used?",
    )
    q08_data: str | None = Field(
        default=None,
        description="What data sources, sample sizes, and time periods are involved?",
    )
    q09_reproducibility: str | None = Field(
        default=None,
        description="How reproducible is this work? Are methods, data, and code available?",
    )
    q10_framework: str | None = Field(
        default=None,
        description="What theoretical or conceptual framework is used?",
    )

    # Pass 3: Context & Discourse
    q11_traditions: str | None = Field(
        default=None,
        description="What intellectual traditions or schools of thought does this draw from?",
    )
    q12_key_citations: str | None = Field(
        default=None,
        description="What are the most influential works cited, and how do they shape this paper?",
    )
    q13_assumptions: str | None = Field(
        default=None,
        description="What assumptions (stated or unstated) underlie the analysis?",
    )
    q14_counterarguments: str | None = Field(
        default=None,
        description="What counterarguments or alternative interpretations are addressed?",
    )
    q15_novelty: str | None = Field(
        default=None,
        description="What is novel or original about this work?",
    )
    q16_stance: str | None = Field(
        default=None,
        description="What is the author's stance or perspective on the topic?",
    )

    # Pass 4: Meta & Audience
    q17_field: str | None = Field(
        default=None,
        description="What academic field(s) does this work belong to?",
    )
    q18_audience: str | None = Field(
        default=None,
        description="Who is the intended audience?",
    )
    q19_implications: str | None = Field(
        default=None,
        description="What are the broader theoretical or practical implications?",
    )
    q20_future_work: str | None = Field(
        default=None,
        description="What future research directions are suggested?",
    )
    q21_quality: str | None = Field(
        default=None,
        description="How would you rate the overall quality? (methodology rigor, evidence strength, contribution significance)",
    )
    q22_contribution: str | None = Field(
        default=None,
        description="What is the explicit contribution of this work to its field?",
    )
    q23_source_type: str | None = Field(
        default=None,
        description="What type of document is this? (empirical study, review, theoretical, report, etc.)",
    )
    q24_other: str | None = Field(
        default=None,
        description="What else is noteworthy that the above questions don't capture?",
    )

    # Pass 5: Scholarly Positioning
    q25_institutional_context: str | None = Field(
        default=None,
        description="What institutional or organizational context shaped this work?",
    )
    q26_historical_timing: str | None = Field(
        default=None,
        description="Why does this work appear now? What historical/temporal factors are relevant?",
    )
    q27_paradigm_influence: str | None = Field(
        default=None,
        description="How does this work relate to dominant paradigms in its field?",
    )
    q28_disciplines_bridged: str | None = Field(
        default=None,
        description="What disciplines does this work bridge or draw from?",
    )
    q29_cross_domain_insights: str | None = Field(
        default=None,
        description="What insights transfer to or from other domains?",
    )
    q30_cultural_scope: str | None = Field(
        default=None,
        description="What cultural, geographic, or demographic scope does this cover?",
    )
    q31_philosophical_assumptions: str | None = Field(
        default=None,
        description="What philosophical assumptions underlie the methodology or claims?",
    )

    # Pass 6: Impact, Gaps & Domain
    q32_deployment_gap: str | None = Field(
        default=None,
        description="What gap exists between this research and real-world application?",
    )
    q33_infrastructure_contribution: str | None = Field(
        default=None,
        description="Does this work contribute tools, datasets, frameworks, or infrastructure?",
    )
    q34_power_dynamics: str | None = Field(
        default=None,
        description="What power dynamics, inequities, or stakeholder tensions are relevant?",
    )
    q35_gaps_and_omissions: str | None = Field(
        default=None,
        description="What important aspects does this work fail to address?",
    )
    q36_dual_use_concerns: str | None = Field(
        default=None,
        description="Are there dual-use or ethical concerns with the findings or methods?",
    )
    q37_emergence_claims: str | None = Field(
        default=None,
        description="Does this work describe emergent phenomena or system-level behaviors?",
    )
    q38_remaining_other: str | None = Field(
        default=None,
        description="What else is significant that no prior question has captured?",
    )
    q39_network_properties: str | None = Field(
        default=None,
        description="What network structures, metrics, or graph algorithms are central?",
    )
    q40_policy_recommendations: str | None = Field(
        default=None,
        description="What specific policy recommendations or actionable guidance is proposed?",
    )

    # Coverage metadata (computed post-extraction)
    dimension_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of non-None dimension answers (0.0-1.0)",
    )
    coverage_flags: list[str] = Field(
        default_factory=list,
        description="Coverage flags: PARTIAL_COVERAGE, SPARSE_COVERAGE, CRITICAL_GAPS, CORE_GAPS",
    )
    dimensions: dict[str, str | None] = Field(
        default_factory=dict,
        description="Canonical dimension map keyed by stable dimension ID",
    )

    # Dimension field names for iteration
    @model_validator(mode="before")
    @classmethod
    def _expand_dimensions_map(cls, data: dict | "SemanticAnalysis") -> dict | "SemanticAnalysis":
        if not isinstance(data, dict):
            return data

        dimensions = data.get("dimensions")
        profile_id = data.get("profile_id", DEFAULT_DIMENSION_PROFILE)
        registry = get_default_dimension_registry()
        profile = registry.profiles.get(profile_id, _LEGACY_PROFILE)
        dimension_map = (
            {str(key): value for key, value in dimensions.items()}
            if isinstance(dimensions, dict)
            else {}
        )

        if isinstance(dimensions, dict):
            for dimension in profile.dimensions:
                if dimension.id in dimensions and dimension.legacy_field_name:
                    data.setdefault(dimension.legacy_field_name, dimensions.get(dimension.id))
                if (
                    dimension.legacy_short_name
                    and dimension.legacy_short_name in dimensions
                    and dimension.legacy_field_name
                ):
                    data.setdefault(
                        dimension.legacy_field_name,
                        dimensions.get(dimension.legacy_short_name),
                    )

        for dimension in profile.dimensions:
            if dimension.id in data:
                dimension_map.setdefault(dimension.id, data[dimension.id])
                if dimension.legacy_field_name:
                    data.setdefault(dimension.legacy_field_name, data[dimension.id])
            if (
                dimension.legacy_short_name
                and dimension.legacy_short_name in data
                and dimension.id not in dimension_map
            ):
                dimension_map[dimension.id] = data[dimension.legacy_short_name]

        if dimension_map:
            data["dimensions"] = dimension_map
        return data

    @model_validator(mode="after")
    def _sync_dimensions_map(self) -> "SemanticAnalysis":
        registry = get_default_dimension_registry()
        dimension_map = dict(self.dimensions)
        profile = registry.profiles.get(self.profile_id)
        if profile is not None:
            self.profile_id = profile.profile_id
            if not self.profile_version:
                self.profile_version = profile.version
            if not self.profile_fingerprint:
                self.profile_fingerprint = profile.fingerprint

            for dimension in profile.dimensions:
                value = dimension_map.get(dimension.id)
                if value is None and dimension.legacy_field_name:
                    value = getattr(self, dimension.legacy_field_name, None)
                dimension_map[dimension.id] = value
                if (
                    dimension.legacy_field_name
                    and value is not None
                    and getattr(self, dimension.legacy_field_name, None) is None
                ):
                    setattr(self, dimension.legacy_field_name, value)
        else:
            for field_name in self.DIMENSION_FIELDS:
                value = getattr(self, field_name, None)
                if value is not None:
                    dimension_map.setdefault(field_name, value)
        self.dimensions = dimension_map
        return self

    DIMENSION_FIELDS: ClassVar[list[str]] = [
        dimension.legacy_field_name
        for dimension in _LEGACY_PROFILE.ordered_dimensions
        if dimension.legacy_field_name
    ]

    CORE_FIELDS: ClassVar[list[str]] = [
        dimension.legacy_field_name
        for dimension in _LEGACY_PROFILE.core_dimensions
        if dimension.legacy_field_name
    ]

    DIMENSION_GROUPS: ClassVar[dict[str, list[str]]] = {
        group_name: [
            _LEGACY_PROFILE.dimension_map[dimension_id].legacy_field_name
            for dimension_id in dimension_ids
            if _LEGACY_PROFILE.dimension_map[dimension_id].legacy_field_name
        ]
        for group_name, dimension_ids in get_default_dimension_registry().get_dimension_groups(
            profile_id=LEGACY_PROFILE_ID
        ).items()
    }

    def get_dimension_value(self, field_name: str) -> str | None:
        """Get the value of a dimension field by name."""
        return self.get_dimension(field_name)

    @property
    def dimension_map(self) -> dict[str, str | None]:
        """Return canonical dimension values keyed by stable IDs."""
        return dict(self.dimensions)

    def get_dimension(self, identifier: str) -> str | None:
        """Resolve a canonical, legacy, or role-based identifier."""
        if identifier in self.dimensions:
            return self.dimensions.get(identifier)

        registry = get_default_dimension_registry()
        profile = registry.profiles.get(self.profile_id)
        if profile is not None:
            resolved = registry.resolve_optional_dimension(
                identifier,
                profile_id=profile.profile_id,
            )
            if resolved:
                return self.dimensions.get(resolved.id)

        return getattr(self, identifier, None)

    def get_role(self, role_name: str) -> str | None:
        """Resolve the value for a downstream semantic role."""
        registry = get_default_dimension_registry()
        resolved = registry.resolve_role(role_name, profile_id=self.profile_id)
        if resolved:
            return self.dimensions.get(resolved.id)
        return None

    def non_none_dimensions(self) -> dict[str, str]:
        """Return populated dimensions keyed by legacy field when available."""
        registry = get_default_dimension_registry()
        profile = registry.profiles.get(self.profile_id)
        if profile is None:
            return {
                field_name: getattr(self, field_name, None)
                for field_name in self.DIMENSION_FIELDS
                if getattr(self, field_name, None) is not None
            }

        populated: dict[str, str] = {}
        for dimension in profile.dimensions:
            value = self.dimensions.get(dimension.id)
            if value is None:
                continue
            key = dimension.legacy_field_name or dimension.id
            populated[key] = value
        return populated

    def to_dimensioned_extraction(self) -> DimensionedExtraction:
        """Convert to the canonical map-based extraction model."""
        return DimensionedExtraction(
            paper_id=self.paper_id,
            profile_id=self.profile_id,
            profile_version=self.profile_version,
            profile_fingerprint=self.profile_fingerprint,
            prompt_version=self.prompt_version,
            extraction_model=self.extraction_model,
            extracted_at=self.extracted_at,
            dimensions=dict(self.dimensions),
            dimension_coverage=self.dimension_coverage,
            coverage_flags=list(self.coverage_flags),
        )

    def to_legacy_q_dict(self) -> dict[str, str | None]:
        """Return legacy q-field answers for backward-compatible consumers."""
        return {
            field_name: getattr(self, field_name, None)
            for field_name in self.DIMENSION_FIELDS
        }

    def to_index_dict(self) -> dict:
        """Convert to a storage dictionary with canonical and legacy fields."""
        payload = self.model_dump()
        payload["dimensions"] = dict(self.dimensions)
        return payload


class ExtractionResult(BaseModel):
    """Result of an extraction attempt."""

    paper_id: str = Field(description="ID of the paper")
    success: bool = Field(description="Whether extraction succeeded")
    extraction: SemanticAnalysis | DimensionedExtraction | None = Field(
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
    document_type: str | None = Field(
        default=None,
        description="Classified document type",
    )
    type_confidence: float = Field(
        default=0.0,
        description="Classification confidence (0.0-1.0)",
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
    pass_errors: list[str] = Field(
        default_factory=list,
        description="Per-pass errors captured during multi-pass extraction.",
    )
    extraction_method: str | None = Field(
        default=None,
        description="Cascade tier that produced the text (e.g. companion, marker, pymupdf).",
    )
    text_snapshot: dict[str, object] | None = Field(
        default=None,
        description="Canonical cleaned full-text snapshot and metadata captured during extraction.",
    )
