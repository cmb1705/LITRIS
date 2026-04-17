"""Dimension profiles, alias resolution, and portability helpers.

This module defines the canonical representation for semantic dimensions.
Legacy ``qNN`` identifiers remain supported via aliases so existing indexes,
tests, and APIs continue to function while newer profiles use readable IDs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

LEGACY_PROFILE_ID = "legacy_semantic_v1"
LEGACY_PROFILE_VERSION = "1.0.0"
DEFAULT_DIMENSION_PROFILE = LEGACY_PROFILE_ID

NON_DIMENSION_CHUNK_TYPES = ["abstract", "raptor_overview", "raptor_core"]
EXTRACTION_METADATA_KEYS = {
    "paper_id",
    "profile_id",
    "profile_version",
    "profile_fingerprint",
    "prompt_version",
    "extraction_model",
    "extracted_at",
    "dimension_coverage",
    "coverage_flags",
    "document_type",
    "type_confidence",
    "dimensions",
}
LEGACY_PAPER_EXTRACTION_MARKER_KEYS = {
    "thesis_statement",
    "research_questions",
    "theoretical_framework",
    "key_findings",
    "conclusions",
    "future_directions",
    "contribution_summary",
    "discipline_tags",
    "extraction_confidence",
    "extraction_notes",
}
DIMENSION_VALUE_TEXT_KEYS = (
    "description",
    "summary",
    "analysis",
    "answer",
    "text",
    "explanation",
    "rationale",
    "value",
    "finding",
    "claim",
    "recommendation",
    "implication",
    "note",
)


def normalize_dimension_value(value: Any) -> str | None:
    """Convert a raw dimension payload into a compact string value."""

    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, Mapping):
        preferred: list[str] = []
        extras: list[str] = []

        for key in DIMENSION_VALUE_TEXT_KEYS:
            if key not in value:
                continue
            text = normalize_dimension_value(value.get(key))
            if text:
                preferred.append(text)

        for key, item in value.items():
            if key in DIMENSION_VALUE_TEXT_KEYS:
                continue
            text = normalize_dimension_value(item)
            if not text:
                continue
            extras.append(f"{str(key).replace('_', ' ')}: {text}")

        parts = list(dict.fromkeys([*preferred, *extras]))
        if parts:
            return "; ".join(parts)
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, (list, tuple, set)):
        parts = [
            normalized
            for item in value
            if (normalized := normalize_dimension_value(item)) is not None
        ]
        unique = list(dict.fromkeys(parts))
        return "; ".join(unique) if unique else None
    return str(value)


class DimensionRole(BaseModel):
    """Named semantic role used by downstream features."""

    id: str
    label: str | None = None
    aliases: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _normalize_id(cls, value: str) -> str:
        return value.strip()


class DimensionSection(BaseModel):
    """Logical grouping for dimensions."""

    id: str
    label: str
    order: int
    prompt_label: str | None = None
    enabled: bool = True
    aliases: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _normalize_id(cls, value: str) -> str:
        return value.strip()

    @property
    def display_label(self) -> str:
        return self.prompt_label or self.label


class DimensionDefinition(BaseModel):
    """Canonical definition of one semantic dimension."""

    id: str
    label: str
    question: str
    section: str
    order: int
    enabled: bool = True
    core: bool = False
    aliases: list[str] = Field(default_factory=list)
    reasoning_effort: str = "high"
    roles: list[str] = Field(default_factory=list)
    legacy_field_name: str | None = None
    legacy_short_name: str | None = None
    legacy_chunk_type: str | None = None
    replaces: list[str] = Field(default_factory=list)

    @field_validator("id", "section")
    @classmethod
    def _normalize_id(cls, value: str) -> str:
        return value.strip()

    @field_validator("reasoning_effort")
    @classmethod
    def _validate_reasoning_effort(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"none", "low", "medium", "high", "xhigh"}:
            raise ValueError(f"Unsupported reasoning effort: {value}")
        return normalized

    @model_validator(mode="after")
    def _populate_aliases(self) -> DimensionDefinition:
        seen: set[str] = set()
        merged: list[str] = []
        for candidate in [
            self.id,
            self.legacy_short_name,
            self.legacy_field_name,
            self.legacy_chunk_type,
            *self.roles,
            *self.aliases,
        ]:
            if not candidate:
                continue
            normalized = candidate.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                merged.append(normalized)
        self.aliases = merged
        return self

    @property
    def chunk_type(self) -> str:
        return self.legacy_chunk_type or f"dim_{self.id}"


class DimensionProfile(BaseModel):
    """Versioned collection of semantic dimensions."""

    profile_id: str
    version: str
    title: str
    summary: str | None = None
    sections: list[DimensionSection]
    dimensions: list[DimensionDefinition]
    roles: list[DimensionRole] = Field(default_factory=list)

    @field_validator("profile_id")
    @classmethod
    def _normalize_id(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def _validate_uniqueness(self) -> DimensionProfile:
        section_ids = [section.id for section in self.sections]
        if len(section_ids) != len(set(section_ids)):
            raise ValueError("Duplicate section IDs in dimension profile")

        dimension_ids = [dimension.id for dimension in self.dimensions]
        if len(dimension_ids) != len(set(dimension_ids)):
            raise ValueError("Duplicate dimension IDs in dimension profile")

        unknown_sections = sorted(
            {
                dimension.section
                for dimension in self.dimensions
                if dimension.section not in set(section_ids)
            }
        )
        if unknown_sections:
            raise ValueError(
                f"Dimensions reference unknown sections: {', '.join(unknown_sections)}"
            )
        return self

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def ordered_sections(self) -> list[DimensionSection]:
        return sorted(self.sections, key=lambda section: section.order)

    @property
    def ordered_dimensions(self) -> list[DimensionDefinition]:
        return sorted(self.dimensions, key=lambda dimension: dimension.order)

    @property
    def enabled_dimensions(self) -> list[DimensionDefinition]:
        return [dimension for dimension in self.ordered_dimensions if dimension.enabled]

    @property
    def core_dimensions(self) -> list[DimensionDefinition]:
        return [dimension for dimension in self.enabled_dimensions if dimension.core]

    @property
    def section_map(self) -> dict[str, DimensionSection]:
        return {section.id: section for section in self.sections}

    @property
    def role_map(self) -> dict[str, DimensionRole]:
        mapping: dict[str, DimensionRole] = {}
        for role in self.roles:
            for alias in [role.id, *role.aliases]:
                mapping[alias] = role
        return mapping

    @property
    def dimension_map(self) -> dict[str, DimensionDefinition]:
        return {dimension.id: dimension for dimension in self.dimensions}

    def dimensions_for_section(
        self,
        section_id: str,
        include_disabled: bool = False,
    ) -> list[DimensionDefinition]:
        return [
            dimension
            for dimension in self.ordered_dimensions
            if dimension.section == section_id and (include_disabled or dimension.enabled)
        ]

    def alias_map(self) -> dict[str, DimensionDefinition]:
        mapping: dict[str, DimensionDefinition] = {}
        for dimension in self.dimensions:
            for alias in dimension.aliases:
                mapping[alias] = dimension
        return mapping

    def legacy_field_map(self) -> dict[str, str]:
        return {
            dimension.legacy_field_name: dimension.id
            for dimension in self.dimensions
            if dimension.legacy_field_name
        }

    def legacy_short_map(self) -> dict[str, str]:
        return {
            dimension.legacy_short_name: dimension.id
            for dimension in self.dimensions
            if dimension.legacy_short_name
        }

    def chunk_type_map(self) -> dict[str, str]:
        return {dimension.chunk_type: dimension.id for dimension in self.dimensions}


class DimensionProfileDiffEntry(BaseModel):
    """One change between two profiles."""

    status: str
    dimension_id: str
    old_section: str | None = None
    new_section: str | None = None
    old_question: str | None = None
    new_question: str | None = None
    replaces: list[str] = Field(default_factory=list)


class DimensionProfileDiff(BaseModel):
    """Profile-level diff result used for migration and backfill planning."""

    from_profile_id: str
    to_profile_id: str
    entries: list[DimensionProfileDiffEntry]

    @property
    def changed_section_ids(self) -> list[str]:
        section_ids = {
            section_id
            for entry in self.entries
            for section_id in [entry.old_section, entry.new_section]
            if section_id
        }
        return sorted(section_ids)

    @property
    def affected_dimension_ids(self) -> list[str]:
        dimension_ids = {
            dimension_id
            for entry in self.entries
            for dimension_id in [entry.dimension_id, *entry.replaces]
            if dimension_id
        }
        return sorted(dimension_ids)


def _dedupe_ordered(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _legacy_role_definitions() -> list[DimensionRole]:
    return [
        DimensionRole(id="research_question", aliases=["q01"]),
        DimensionRole(id="thesis", aliases=["q02"]),
        DimensionRole(id="key_claims", aliases=["q03"]),
        DimensionRole(id="evidence", aliases=["q04"]),
        DimensionRole(id="limitations", aliases=["q05"]),
        DimensionRole(id="paradigm", aliases=["q06"]),
        DimensionRole(id="methods", aliases=["q07"]),
        DimensionRole(id="data", aliases=["q08"]),
        DimensionRole(id="reproducibility", aliases=["q09"]),
        DimensionRole(id="framework", aliases=["q10"]),
        DimensionRole(id="field", aliases=["q17"]),
        DimensionRole(id="implications", aliases=["q19"]),
        DimensionRole(id="future_work", aliases=["q20"]),
        DimensionRole(id="quality", aliases=["q21"]),
        DimensionRole(id="contribution", aliases=["q22"]),
    ]


def _legacy_dimensions() -> list[DimensionDefinition]:
    raw_dimensions = [
        (
            "research_question",
            "Research Question",
            "What research questions or objectives does this work address?",
            "research_core",
            1,
            True,
            "high",
            "q01_research_question",
            "q01",
            "dim_q01",
            ["research_question"],
        ),
        (
            "thesis",
            "Thesis",
            "What is the central thesis or main argument?",
            "research_core",
            2,
            True,
            "high",
            "q02_thesis",
            "q02",
            "dim_q02",
            ["thesis"],
        ),
        (
            "key_claims",
            "Key Claims",
            "What are the key claims or propositions made?",
            "research_core",
            3,
            True,
            "xhigh",
            "q03_key_claims",
            "q03",
            "dim_q03",
            ["key_claims"],
        ),
        (
            "evidence",
            "Evidence",
            "What evidence is presented and how strong is it?",
            "research_core",
            4,
            True,
            "xhigh",
            "q04_evidence",
            "q04",
            "dim_q04",
            ["evidence"],
        ),
        (
            "limitations",
            "Limitations",
            "What limitations are acknowledged or apparent?",
            "research_core",
            5,
            True,
            "xhigh",
            "q05_limitations",
            "q05",
            "dim_q05",
            ["limitations"],
        ),
        (
            "paradigm",
            "Paradigm",
            "What research paradigm underlies this work? (positivist, interpretivist, critical, pragmatist, etc.)",
            "methodology",
            6,
            False,
            "xhigh",
            "q06_paradigm",
            "q06",
            "dim_q06",
            ["paradigm"],
        ),
        (
            "methods",
            "Methods",
            "What methods and analytical techniques are used?",
            "methodology",
            7,
            False,
            "high",
            "q07_methods",
            "q07",
            "dim_q07",
            ["methods"],
        ),
        (
            "data",
            "Data",
            "What data sources, sample sizes, and time periods are involved?",
            "methodology",
            8,
            False,
            "high",
            "q08_data",
            "q08",
            "dim_q08",
            ["data"],
        ),
        (
            "reproducibility",
            "Reproducibility",
            "How reproducible is this work? Are methods, data, and code available?",
            "methodology",
            9,
            False,
            "high",
            "q09_reproducibility",
            "q09",
            "dim_q09",
            ["reproducibility"],
        ),
        (
            "framework",
            "Framework",
            "What theoretical or conceptual framework is used?",
            "methodology",
            10,
            False,
            "high",
            "q10_framework",
            "q10",
            "dim_q10",
            ["framework"],
        ),
        (
            "traditions",
            "Traditions",
            "What intellectual traditions or schools of thought does this draw from?",
            "context",
            11,
            False,
            "xhigh",
            "q11_traditions",
            "q11",
            "dim_q11",
            [],
        ),
        (
            "key_citations",
            "Key Citations",
            "What are the most influential works cited, and how do they shape this paper?",
            "context",
            12,
            False,
            "high",
            "q12_key_citations",
            "q12",
            "dim_q12",
            [],
        ),
        (
            "assumptions",
            "Assumptions",
            "What assumptions (stated or unstated) underlie the analysis?",
            "context",
            13,
            False,
            "xhigh",
            "q13_assumptions",
            "q13",
            "dim_q13",
            [],
        ),
        (
            "counterarguments",
            "Counterarguments",
            "What counterarguments or alternative interpretations are addressed?",
            "context",
            14,
            False,
            "xhigh",
            "q14_counterarguments",
            "q14",
            "dim_q14",
            [],
        ),
        (
            "novelty",
            "Novelty",
            "What is novel or original about this work?",
            "context",
            15,
            False,
            "xhigh",
            "q15_novelty",
            "q15",
            "dim_q15",
            [],
        ),
        (
            "stance",
            "Stance",
            "What is the author's stance or perspective on the topic?",
            "context",
            16,
            False,
            "xhigh",
            "q16_stance",
            "q16",
            "dim_q16",
            [],
        ),
        (
            "field",
            "Field",
            "What academic field(s) does this work belong to?",
            "meta",
            17,
            False,
            "high",
            "q17_field",
            "q17",
            "dim_q17",
            ["field"],
        ),
        (
            "audience",
            "Audience",
            "Who is the intended audience?",
            "meta",
            18,
            False,
            "high",
            "q18_audience",
            "q18",
            "dim_q18",
            ["audience"],
        ),
        (
            "implications",
            "Implications",
            "What are the broader theoretical or practical implications?",
            "meta",
            19,
            False,
            "xhigh",
            "q19_implications",
            "q19",
            "dim_q19",
            ["implications"],
        ),
        (
            "future_work",
            "Future Work",
            "What future research directions are suggested?",
            "meta",
            20,
            False,
            "xhigh",
            "q20_future_work",
            "q20",
            "dim_q20",
            ["future_work"],
        ),
        (
            "quality",
            "Quality",
            "How would you rate the overall quality? (methodology rigor, evidence strength, contribution significance)",
            "meta",
            21,
            False,
            "xhigh",
            "q21_quality",
            "q21",
            "dim_q21",
            ["quality"],
        ),
        (
            "contribution",
            "Contribution",
            "What is the explicit contribution of this work to its field?",
            "meta",
            22,
            False,
            "xhigh",
            "q22_contribution",
            "q22",
            "dim_q22",
            ["contribution"],
        ),
        (
            "source_type",
            "Source Type",
            "What type of document is this? (empirical study, review, theoretical, report, etc.)",
            "meta",
            23,
            False,
            "high",
            "q23_source_type",
            "q23",
            "dim_q23",
            [],
        ),
        (
            "other",
            "Other",
            "What else is noteworthy that the above questions don't capture?",
            "meta",
            24,
            False,
            "xhigh",
            "q24_other",
            "q24",
            "dim_q24",
            [],
        ),
        (
            "institutional_context",
            "Institutional Context",
            "What institutional or organizational context shaped this work?",
            "scholarly",
            25,
            False,
            "high",
            "q25_institutional_context",
            "q25",
            "dim_q25",
            [],
        ),
        (
            "historical_timing",
            "Historical Timing",
            "Why does this work appear now? What historical/temporal factors are relevant?",
            "scholarly",
            26,
            False,
            "xhigh",
            "q26_historical_timing",
            "q26",
            "dim_q26",
            [],
        ),
        (
            "paradigm_influence",
            "Paradigm Influence",
            "How does this work relate to dominant paradigms in its field?",
            "scholarly",
            27,
            False,
            "xhigh",
            "q27_paradigm_influence",
            "q27",
            "dim_q27",
            [],
        ),
        (
            "disciplines_bridged",
            "Disciplines Bridged",
            "What disciplines does this work bridge or draw from?",
            "scholarly",
            28,
            False,
            "high",
            "q28_disciplines_bridged",
            "q28",
            "dim_q28",
            [],
        ),
        (
            "cross_domain_insights",
            "Cross-Domain Insights",
            "What insights transfer to or from other domains?",
            "scholarly",
            29,
            False,
            "xhigh",
            "q29_cross_domain_insights",
            "q29",
            "dim_q29",
            [],
        ),
        (
            "cultural_scope",
            "Cultural Scope",
            "What cultural, geographic, or demographic scope does this cover?",
            "scholarly",
            30,
            False,
            "high",
            "q30_cultural_scope",
            "q30",
            "dim_q30",
            [],
        ),
        (
            "philosophical_assumptions",
            "Philosophical Assumptions",
            "What philosophical assumptions underlie the methodology or claims?",
            "scholarly",
            31,
            False,
            "xhigh",
            "q31_philosophical_assumptions",
            "q31",
            "dim_q31",
            [],
        ),
        (
            "deployment_gap",
            "Deployment Gap",
            "What gap exists between this research and real-world application?",
            "impact",
            32,
            False,
            "xhigh",
            "q32_deployment_gap",
            "q32",
            "dim_q32",
            [],
        ),
        (
            "infrastructure_contribution",
            "Infrastructure Contribution",
            "Does this work contribute tools, datasets, frameworks, or infrastructure?",
            "impact",
            33,
            False,
            "high",
            "q33_infrastructure_contribution",
            "q33",
            "dim_q33",
            [],
        ),
        (
            "power_dynamics",
            "Power Dynamics",
            "What power dynamics, inequities, or stakeholder tensions are relevant?",
            "impact",
            34,
            False,
            "xhigh",
            "q34_power_dynamics",
            "q34",
            "dim_q34",
            [],
        ),
        (
            "gaps_and_omissions",
            "Gaps And Omissions",
            "What important aspects does this work fail to address?",
            "impact",
            35,
            False,
            "xhigh",
            "q35_gaps_and_omissions",
            "q35",
            "dim_q35",
            [],
        ),
        (
            "dual_use_concerns",
            "Dual-Use Concerns",
            "Are there dual-use or ethical concerns with the findings or methods?",
            "impact",
            36,
            False,
            "xhigh",
            "q36_dual_use_concerns",
            "q36",
            "dim_q36",
            [],
        ),
        (
            "emergence_claims",
            "Emergence Claims",
            "Does this work describe emergent phenomena or system-level behaviors?",
            "impact",
            37,
            False,
            "xhigh",
            "q37_emergence_claims",
            "q37",
            "dim_q37",
            [],
        ),
        (
            "remaining_other",
            "Remaining Other",
            "What else is significant that no prior question has captured?",
            "impact",
            38,
            False,
            "xhigh",
            "q38_remaining_other",
            "q38",
            "dim_q38",
            [],
        ),
        (
            "network_properties",
            "Network Properties",
            "What network structures, metrics, or graph algorithms are central?",
            "impact",
            39,
            False,
            "high",
            "q39_network_properties",
            "q39",
            "dim_q39",
            [],
        ),
        (
            "policy_recommendations",
            "Policy Recommendations",
            "What specific policy recommendations or actionable guidance is proposed?",
            "impact",
            40,
            False,
            "high",
            "q40_policy_recommendations",
            "q40",
            "dim_q40",
            [],
        ),
    ]
    return [
        DimensionDefinition(
            id=dimension_id,
            label=label,
            question=question,
            section=section_id,
            order=order,
            core=core,
            reasoning_effort=reasoning_effort,
            legacy_field_name=legacy_field_name,
            legacy_short_name=legacy_short_name,
            legacy_chunk_type=legacy_chunk_type,
            roles=roles,
        )
        for (
            dimension_id,
            label,
            question,
            section_id,
            order,
            core,
            reasoning_effort,
            legacy_field_name,
            legacy_short_name,
            legacy_chunk_type,
            roles,
        ) in raw_dimensions
    ]


def build_legacy_dimension_profile() -> DimensionProfile:
    """Return the built-in legacy semantic profile."""

    sections = [
        DimensionSection(
            id="research_core",
            label="Pass 1: Research Core",
            prompt_label="Pass 1: Research Core",
            order=1,
            aliases=["pass_1", "research"],
        ),
        DimensionSection(
            id="methodology",
            label="Pass 2: Methodology",
            prompt_label="Pass 2: Methodology",
            order=2,
            aliases=["pass_2", "methods"],
        ),
        DimensionSection(
            id="context",
            label="Pass 3: Context & Discourse",
            prompt_label="Pass 3: Context & Discourse",
            order=3,
            aliases=["pass_3"],
        ),
        DimensionSection(
            id="meta",
            label="Pass 4: Meta & Audience",
            prompt_label="Pass 4: Meta & Audience",
            order=4,
            aliases=["pass_4"],
        ),
        DimensionSection(
            id="scholarly",
            label="Pass 5: Scholarly Positioning",
            prompt_label="Pass 5: Scholarly Positioning",
            order=5,
            aliases=["pass_5", "synthesis"],
        ),
        DimensionSection(
            id="impact",
            label="Pass 6: Impact, Gaps & Domain",
            prompt_label="Pass 6: Impact, Gaps & Domain",
            order=6,
            aliases=["pass_6"],
        ),
    ]
    return DimensionProfile(
        profile_id=LEGACY_PROFILE_ID,
        version=LEGACY_PROFILE_VERSION,
        title="Legacy Semantic Analysis",
        summary="Built-in 40-dimension semantic analysis profile used by existing indexes.",
        sections=sections,
        dimensions=_legacy_dimensions(),
        roles=_legacy_role_definitions(),
    )


def load_dimension_profile(path: Path | str) -> DimensionProfile:
    """Load a profile from YAML or JSON."""

    profile_path = Path(path)
    with open(profile_path, encoding="utf-8") as handle:
        if profile_path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(handle) or {}
        else:
            data = json.load(handle)
    return DimensionProfile(**data)


class DimensionRegistry:
    """Registry of known dimension profiles plus alias resolution helpers."""

    def __init__(
        self,
        profiles: dict[str, DimensionProfile] | None = None,
        active_profile_id: str = DEFAULT_DIMENSION_PROFILE,
    ):
        base_profiles = profiles or {LEGACY_PROFILE_ID: build_legacy_dimension_profile()}
        self._profiles = dict(base_profiles.items())
        self.set_active_profile(active_profile_id)

    @property
    def active_profile_id(self) -> str:
        return self._active_profile_id

    @property
    def active_profile(self) -> DimensionProfile:
        return self._profiles[self._active_profile_id]

    @property
    def profiles(self) -> dict[str, DimensionProfile]:
        return dict(self._profiles)

    def set_active_profile(self, profile_id: str) -> None:
        if profile_id not in self._profiles:
            raise KeyError(f"Unknown dimension profile: {profile_id}")
        self._active_profile_id = profile_id

    def register_profile(self, profile: DimensionProfile) -> None:
        self._profiles[profile.profile_id] = profile

    def register_profiles(self, profiles: Iterable[DimensionProfile]) -> None:
        for profile in profiles:
            self.register_profile(profile)

    def get_profile(self, profile_id: str | None = None) -> DimensionProfile:
        if profile_id is None:
            return self._profiles[self._active_profile_id]
        return self._profiles.get(profile_id, self._profiles[self._active_profile_id])

    def load_profile_paths(self, paths: Iterable[Path | str]) -> None:
        for path in paths:
            self.register_profile(load_dimension_profile(path))

    def resolve_dimension(
        self,
        identifier: str,
        profile_id: str | None = None,
    ) -> DimensionDefinition:
        profile = self.get_profile(profile_id)
        alias_map = profile.alias_map()
        if identifier in alias_map:
            return alias_map[identifier]
        raise KeyError(
            f"Unknown dimension identifier '{identifier}' for profile {profile.profile_id}"
        )

    def resolve_optional_dimension(
        self,
        identifier: str | None,
        profile_id: str | None = None,
    ) -> DimensionDefinition | None:
        if not identifier:
            return None
        try:
            return self.resolve_dimension(identifier, profile_id=profile_id)
        except KeyError:
            return None

    def resolve_role(
        self,
        role_name: str,
        profile_id: str | None = None,
    ) -> DimensionDefinition | None:
        profile = self.get_profile(profile_id)
        role = profile.role_map.get(role_name)
        lookup = role.id if role else role_name
        for dimension in profile.dimensions:
            if lookup in dimension.roles:
                return dimension
        return None

    def normalize_dimension_values(
        self,
        values: Mapping[str, Any],
        profile_id: str | None = None,
    ) -> dict[str, str | None]:
        profile = self.get_profile(profile_id)
        normalized: dict[str, str | None] = {}
        for key, value in values.items():
            dimension = self.resolve_optional_dimension(
                key,
                profile_id=profile.profile_id,
            )
            if not dimension:
                continue
            normalized[dimension.id] = normalize_dimension_value(value)
        for dimension in profile.dimensions:
            normalized.setdefault(dimension.id, None)
        return normalized

    def get_section_dimensions(
        self,
        section_id: str,
        profile_id: str | None = None,
        include_disabled: bool = False,
    ) -> list[DimensionDefinition]:
        profile = self.get_profile(profile_id)
        return profile.dimensions_for_section(
            section_id,
            include_disabled=include_disabled,
        )

    def get_sections(self, profile_id: str | None = None) -> list[DimensionSection]:
        return self.get_profile(profile_id).ordered_sections

    def get_group_names(self, profile_id: str | None = None) -> list[str]:
        return [section.id for section in self.get_sections(profile_id)]

    def get_dimension_ids(
        self,
        profile_id: str | None = None,
        include_disabled: bool = False,
    ) -> list[str]:
        profile = self.get_profile(profile_id)
        dimensions = profile.ordered_dimensions if include_disabled else profile.enabled_dimensions
        return [dimension.id for dimension in dimensions]

    def get_core_dimension_ids(self, profile_id: str | None = None) -> list[str]:
        return [dimension.id for dimension in self.get_profile(profile_id).core_dimensions]

    def get_chunk_types(
        self,
        profile_id: str | None = None,
        include_non_dimension: bool = True,
        include_disabled: bool = False,
        include_legacy_aliases: bool = True,
    ) -> list[str]:
        profile = self.get_profile(profile_id)
        dimensions = profile.ordered_dimensions if include_disabled else profile.enabled_dimensions
        chunk_types = [dimension.chunk_type for dimension in dimensions]
        if include_legacy_aliases:
            chunk_types.extend(
                dimension.legacy_chunk_type
                for dimension in dimensions
                if dimension.legacy_chunk_type
            )
        chunk_types = _dedupe_ordered(chunk_types)
        if include_non_dimension:
            return [*NON_DIMENSION_CHUNK_TYPES, *chunk_types]
        return chunk_types

    def get_dimension_groups(
        self,
        profile_id: str | None = None,
        include_disabled: bool = False,
        legacy_field_names: bool = False,
    ) -> dict[str, list[str]]:
        profile = self.get_profile(profile_id)
        groups: dict[str, list[str]] = {}
        for section in profile.ordered_sections:
            values = []
            for dimension in profile.dimensions_for_section(
                section.id,
                include_disabled=include_disabled,
            ):
                if legacy_field_names and dimension.legacy_field_name:
                    values.append(dimension.legacy_field_name)
                else:
                    values.append(dimension.id)
            groups[section.id] = values
        return groups

    def diff_profiles(
        self,
        old_profile_id: str,
        new_profile_id: str,
    ) -> DimensionProfileDiff:
        old_profile = self.get_profile(old_profile_id)
        new_profile = self.get_profile(new_profile_id)
        entries: list[DimensionProfileDiffEntry] = []

        old_dimensions = old_profile.dimension_map
        new_dimensions = new_profile.dimension_map
        all_ids = sorted(set(old_dimensions) | set(new_dimensions))

        for dimension_id in all_ids:
            old_dimension = old_dimensions.get(dimension_id)
            new_dimension = new_dimensions.get(dimension_id)

            if old_dimension and not new_dimension:
                entries.append(
                    DimensionProfileDiffEntry(
                        status="disabled",
                        dimension_id=dimension_id,
                        old_section=old_dimension.section,
                        old_question=old_dimension.question,
                    )
                )
                continue

            if new_dimension and not old_dimension:
                entries.append(
                    DimensionProfileDiffEntry(
                        status="replaced" if new_dimension.replaces else "added",
                        dimension_id=dimension_id,
                        new_section=new_dimension.section,
                        new_question=new_dimension.question,
                        replaces=list(new_dimension.replaces),
                    )
                )
                continue

            if old_dimension is None or new_dimension is None:
                continue

            if old_dimension.enabled and not new_dimension.enabled:
                entries.append(
                    DimensionProfileDiffEntry(
                        status="disabled",
                        dimension_id=dimension_id,
                        old_section=old_dimension.section,
                        new_section=new_dimension.section,
                        old_question=old_dimension.question,
                        new_question=new_dimension.question,
                    )
                )
            elif (
                old_dimension.question != new_dimension.question
                or old_dimension.section != new_dimension.section
            ):
                entries.append(
                    DimensionProfileDiffEntry(
                        status="reworded",
                        dimension_id=dimension_id,
                        old_section=old_dimension.section,
                        new_section=new_dimension.section,
                        old_question=old_dimension.question,
                        new_question=new_dimension.question,
                    )
                )

        return DimensionProfileDiff(
            from_profile_id=old_profile.profile_id,
            to_profile_id=new_profile.profile_id,
            entries=entries,
        )


_DEFAULT_REGISTRY = DimensionRegistry()


def get_default_dimension_registry() -> DimensionRegistry:
    """Return the process-wide default registry."""

    return _DEFAULT_REGISTRY


def configure_dimension_registry(
    active_profile_id: str = DEFAULT_DIMENSION_PROFILE,
    profile_paths: Iterable[Path | str] | None = None,
) -> DimensionRegistry:
    """Reset the default registry for the current process."""

    global _DEFAULT_REGISTRY
    registry = DimensionRegistry(active_profile_id=DEFAULT_DIMENSION_PROFILE)
    if profile_paths:
        registry.load_profile_paths(profile_paths)
    registry.set_active_profile(active_profile_id)
    _DEFAULT_REGISTRY = registry
    return registry


def is_dimension_payload(
    values: Mapping[str, Any],
    profile_id: str | None = None,
    registry: DimensionRegistry | None = None,
) -> bool:
    """Return True when a payload looks like semantic dimension output."""

    active_registry = registry or get_default_dimension_registry()
    resolved_profile_id = profile_id or str(values.get("profile_id") or "")

    if isinstance(values.get("dimensions"), Mapping):
        return True

    if not values.get("profile_id") and any(
        key in values for key in LEGACY_PAPER_EXTRACTION_MARKER_KEYS
    ):
        return False

    try:
        profile = active_registry.get_profile(resolved_profile_id or None)
    except KeyError:
        profile = active_registry.get_profile(DEFAULT_DIMENSION_PROFILE)

    for key in values:
        if key in EXTRACTION_METADATA_KEYS:
            continue
        if active_registry.resolve_optional_dimension(key, profile_id=profile.profile_id):
            return True
    return False


def normalize_dimension_payload(
    values: Mapping[str, Any],
    profile_id: str | None = None,
    registry: DimensionRegistry | None = None,
) -> dict[str, str | None]:
    """Normalize a raw payload into canonical dimension IDs."""

    active_registry = registry or get_default_dimension_registry()
    resolved_profile_id = profile_id or str(values.get("profile_id") or "")
    try:
        profile = active_registry.get_profile(resolved_profile_id or None)
    except KeyError:
        profile = active_registry.get_profile(DEFAULT_DIMENSION_PROFILE)

    raw_dimensions = values.get("dimensions")
    if isinstance(raw_dimensions, Mapping):
        base_values = {
            key: value for key, value in values.items() if key not in EXTRACTION_METADATA_KEYS
        }
        merged_values = {**dict(raw_dimensions), **base_values}
    else:
        merged_values = {
            key: value for key, value in values.items() if key not in EXTRACTION_METADATA_KEYS
        }

    return active_registry.normalize_dimension_values(
        merged_values,
        profile_id=profile.profile_id,
    )


def normalize_dimension_input_values(
    values: Mapping[str, Any],
    profile_id: str | None = None,
    registry: DimensionRegistry | None = None,
) -> dict[str, Any]:
    """Normalize only dimension-like fields in a raw provider payload."""

    active_registry = registry or get_default_dimension_registry()
    resolved_profile_id = profile_id or str(values.get("profile_id") or "")
    try:
        profile = active_registry.get_profile(resolved_profile_id or None)
    except KeyError:
        profile = active_registry.get_profile(DEFAULT_DIMENSION_PROFILE)

    normalized = dict(values)
    raw_dimensions = values.get("dimensions")
    if isinstance(raw_dimensions, Mapping):
        normalized["dimensions"] = {
            str(key): normalize_dimension_value(value) for key, value in raw_dimensions.items()
        }

    for key, value in values.items():
        if key in EXTRACTION_METADATA_KEYS:
            continue
        if active_registry.resolve_optional_dimension(str(key), profile_id=profile.profile_id):
            normalized[key] = normalize_dimension_value(value)

    return normalized


def unwrap_extraction_record(record: Any) -> Any:
    """Unwrap nested extraction wrappers when present."""

    if (
        isinstance(record, Mapping)
        and "extraction" in record
        and isinstance(record["extraction"], Mapping)
    ):
        return record["extraction"]
    return record


def get_profile_id_from_record(record: Any) -> str:
    """Return the profile ID encoded in a record, with legacy fallback."""

    unwrapped = unwrap_extraction_record(record)
    if hasattr(unwrapped, "profile_id"):
        return unwrapped.profile_id or DEFAULT_DIMENSION_PROFILE
    if isinstance(unwrapped, Mapping):
        return str(unwrapped.get("profile_id") or DEFAULT_DIMENSION_PROFILE)
    return DEFAULT_DIMENSION_PROFILE


def get_dimension_value(
    record: Any,
    identifier: str,
    registry: DimensionRegistry | None = None,
) -> str | None:
    """Resolve a dimension or role value from a raw mapping or model instance."""

    if record is None:
        return None

    unwrapped = unwrap_extraction_record(record)
    get_dimension = getattr(type(unwrapped), "get_dimension", None)
    if callable(get_dimension):
        value = unwrapped.get_dimension(identifier)
        if value is not None:
            return value
    get_role = getattr(type(unwrapped), "get_role", None)
    if callable(get_role):
        value = unwrapped.get_role(identifier)
        if value is not None:
            return value

    if isinstance(unwrapped, Mapping):
        local_registry = registry or get_default_dimension_registry()
        profile_id = get_profile_id_from_record(unwrapped)
        raw_dimensions = unwrapped.get("dimensions")
        if isinstance(raw_dimensions, Mapping) and identifier in raw_dimensions:
            value = raw_dimensions.get(identifier)
            if value is not None:
                return normalize_dimension_value(value)
        dimension = local_registry.resolve_optional_dimension(
            identifier,
            profile_id=profile_id,
        )
        if not dimension:
            dimension = local_registry.resolve_role(identifier, profile_id=profile_id)
        if dimension and "dimensions" in unwrapped and isinstance(unwrapped["dimensions"], Mapping):
            value = unwrapped["dimensions"].get(dimension.id)
            if value is not None:
                return normalize_dimension_value(value)
        if identifier in unwrapped and unwrapped[identifier] is not None:
            return normalize_dimension_value(unwrapped[identifier])
        if dimension:
            if (
                dimension.legacy_field_name
                and unwrapped.get(dimension.legacy_field_name) is not None
            ):
                return str(unwrapped[dimension.legacy_field_name])
            if dimension.id in unwrapped and unwrapped.get(dimension.id) is not None:
                return str(unwrapped[dimension.id])
    return None


def get_dimension_map(
    record: Any,
    registry: DimensionRegistry | None = None,
) -> dict[str, str | None]:
    """Return canonical dimension values from a raw mapping or model instance."""

    if record is None:
        return {}

    unwrapped = unwrap_extraction_record(record)
    if hasattr(unwrapped, "dimension_map"):
        return dict(unwrapped.dimension_map)

    if isinstance(unwrapped, Mapping):
        local_registry = registry or get_default_dimension_registry()
        profile_id = get_profile_id_from_record(unwrapped)
        dimensions = unwrapped.get("dimensions")
        if isinstance(dimensions, Mapping):
            return {
                str(key): None if value is None else str(value) for key, value in dimensions.items()
            }
        return local_registry.normalize_dimension_values(
            unwrapped,
            profile_id=profile_id,
        )

    return {}
