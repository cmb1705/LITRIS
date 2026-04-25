"""Heuristic extraction-intent classification and routing utilities.

This module decides which extraction route is most appropriate before full
semantic extraction. The intent is persisted alongside document-type
classification so dry-run, classify-only, and normal indexing all use the same
planned routing.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal, Protocol

from src.extraction.opendataloader_extractor import (
    OpenDataLoaderHybridConfig,
    build_hybrid_config_from_processing,
)

if TYPE_CHECKING:
    from src.config import ProcessingConfig
    from src.zotero.models import PaperMetadata


EXTRACTION_INTENT_SCHEMA_VERSION = "1.0.0"
IntentSourceTier = Literal["metadata_only", "cheap_text_pass", "stored_snapshot"]

_FORMULA_SIGNAL = re.compile(
    r"\b("
    r"equation|formula|theorem|lemma|proof|corollary|optimization|objective|"
    r"lagrangian|gradient|derivative|integral|tensor|matrix|eigenvalue|"
    r"bayesian|quantum|topology|algebra"
    r")\b",
    re.IGNORECASE,
)
_PICTURE_SIGNAL = re.compile(
    r"\b(fig(?:ure)?\.?|chart|plot|diagram|image|visualization|heatmap|"
    r"scatterplot|bar chart|flowchart|schematic)\b",
    re.IGNORECASE,
)
_OCR_SIGNAL = re.compile(
    r"\b(scanned|ocr|facsimile|handwritten|archival|typewritten)\b",
    re.IGNORECASE,
)


class ExtractionIntent(str, Enum):
    """Persisted extraction routing intent."""

    FAST = "fast"
    HYBRID_OCR = "hybrid_ocr"
    HYBRID_FORMULA = "hybrid_formula"
    HYBRID_PICTURE = "hybrid_picture"
    HYBRID_FORMULA_PICTURE = "hybrid_formula_picture"
    HYBRID_OCR_FORMULA = "hybrid_ocr_formula"
    HYBRID_OCR_PICTURE = "hybrid_ocr_picture"
    HYBRID_OCR_FORMULA_PICTURE = "hybrid_ocr_formula_picture"
    MARKER_LAST_RESORT = "marker_last_resort"


@dataclass(frozen=True)
class ExtractionIntentRecord:
    """Classification output for extraction routing."""

    intent: ExtractionIntent
    confidence: float
    reasons: list[str]
    signals: dict[str, int | float | bool | str]
    classified_at: str
    source_tier: IntentSourceTier


@dataclass(frozen=True)
class HybridProfileSpec:
    """Resolved backend profile for a planned extraction route."""

    intent: ExtractionIntent
    profile_key: str
    mode: str
    backend: str | None
    client_mode: str | None
    force_ocr: bool
    enrich_formula: bool
    enrich_picture_description: bool
    fallback_to_fast: bool

    @property
    def uses_hybrid(self) -> bool:
        """Return whether this profile requires the hybrid backend."""
        return self.mode == "hybrid"

    def build_hybrid_config(
        self,
        base_config: OpenDataLoaderHybridConfig | None,
        processing: ProcessingConfig,
    ) -> OpenDataLoaderHybridConfig | None:
        """Return a concrete hybrid config for this profile.

        The base processing config provides connection/backend settings; the
        profile contributes only the intent-specific feature flags.
        """
        if not self.uses_hybrid:
            return base_config

        template = base_config or build_hybrid_config_from_processing(
            processing,
            enabled=True,
        )
        if template is None:
            return None
        return OpenDataLoaderHybridConfig(
            enabled=True,
            backend=self.backend or template.backend,
            client_mode=self.client_mode or template.client_mode,
            server_url=template.server_url,
            timeout_ms=template.timeout_ms,
            fallback_to_fast=self.fallback_to_fast,
            autostart=template.autostart,
            host=template.host,
            port=template.port,
            startup_timeout_seconds=template.startup_timeout_seconds,
            force_ocr=self.force_ocr,
            ocr_lang=template.ocr_lang,
            enrich_formula=self.enrich_formula,
            enrich_picture_description=self.enrich_picture_description,
            picture_description_prompt=template.picture_description_prompt,
            device=template.device,
            python_executable=template.python_executable,
            managed_servers=template.managed_servers,
        )


@dataclass(frozen=True)
class ExtractionPlanItem:
    """Resolved extraction plan for a single paper."""

    paper_id: str
    document_type: str
    extractable: bool
    intent: ExtractionIntent
    intent_confidence: float
    intent_reasons: list[str]
    source_tier: IntentSourceTier
    profile: HybridProfileSpec
    fallback_policy: str


class ClassificationRecordLike(Protocol):
    """Classification record fields required for extraction planning."""

    document_type: str
    extractable: bool
    extraction_intent: str | None
    intent_confidence: float
    intent_reasons: list[str]
    intent_source_tier: str | None
    hybrid_profile_key: str | None


def classify_extraction_intent(
    paper: PaperMetadata,
    *,
    text: str | None = None,
    word_count: int | None = None,
    page_count: int | None = None,
    section_markers: int | None = None,
    source_tier: IntentSourceTier = "metadata_only",
    classified_at: str = "",
    allow_picture_enrichment: bool = False,
) -> ExtractionIntentRecord:
    """Classify extraction intent using cheap heuristics.

    The classifier is deliberately deterministic and low-cost so it can run in
    classify-only and dry-run contexts without the LLM stack.
    """

    title_bits = " ".join(
        bit
        for bit in [
            paper.title,
            paper.journal or "",
            paper.abstract or "",
            paper.doi or "",
            paper.isbn or "",
            " ".join(paper.tags),
            " ".join(paper.collections),
        ]
        if bit
    )
    normalized_title_bits = title_bits.lower()
    normalized_text = (text or "").lower()
    combined_text = " ".join(part for part in [normalized_title_bits, normalized_text] if part)

    formula_hits = _count_pattern_hits(_FORMULA_SIGNAL, combined_text)
    picture_hits = _count_pattern_hits(_PICTURE_SIGNAL, combined_text)
    ocr_hits = _count_pattern_hits(_OCR_SIGNAL, combined_text)

    words_per_page = 0.0
    if word_count is not None and page_count:
        words_per_page = word_count / max(page_count, 1)

    likely_scanned = False
    if page_count and page_count >= 4:
        likely_scanned = (word_count is not None and word_count < 500) or words_per_page < 80
    if text is not None and len(text.strip()) < 600 and (page_count or 0) >= 6:
        likely_scanned = True
    if ocr_hits > 0:
        likely_scanned = True

    needs_formula = formula_hits >= 2 or _looks_math_heavy(normalized_title_bits)
    picture_signals_detected = picture_hits >= 2 or _looks_figure_heavy(normalized_title_bits)
    needs_picture = picture_signals_detected and allow_picture_enrichment
    needs_ocr = likely_scanned

    confidence = 0.35
    reasons: list[str] = []
    if source_tier == "metadata_only":
        confidence = 0.45 if (needs_formula or needs_picture or needs_ocr) else 0.3
    elif source_tier == "cheap_text_pass":
        confidence = 0.68 if (needs_formula or needs_picture or needs_ocr) else 0.55
    elif source_tier == "stored_snapshot":
        confidence = 0.82 if (needs_formula or needs_picture or needs_ocr) else 0.7

    if needs_ocr:
        reasons.append("ocr-signals indicate scanned or image-heavy pages")
    if needs_formula:
        reasons.append("formula/theorem density suggests math enrichment")
    if needs_picture:
        reasons.append("figure/chart density suggests picture descriptions")
    elif picture_signals_detected:
        reasons.append(
            "figure/chart signals detected but automatic picture-description hybrid is disabled"
        )
    if not reasons:
        reasons.append("no strong OCR, formula, or picture signals; defaulting to fast mode")

    intent = _intent_from_flags(
        needs_ocr=needs_ocr, needs_formula=needs_formula, needs_picture=needs_picture
    )

    if sum([needs_ocr, needs_formula, needs_picture]) >= 2:
        confidence = min(0.95, confidence + 0.12)
    elif any([needs_ocr, needs_formula, needs_picture]):
        confidence = min(0.9, confidence + 0.08)

    signals: dict[str, int | float | bool | str] = {
        "formula_hits": formula_hits,
        "picture_hits": picture_hits,
        "ocr_hits": ocr_hits,
        "needs_ocr": needs_ocr,
        "needs_formula": needs_formula,
        "needs_picture": needs_picture,
        "picture_signals_detected": picture_signals_detected,
        "picture_intent_enabled": allow_picture_enrichment,
        "word_count": word_count or 0,
        "page_count": page_count or 0,
        "words_per_page": round(words_per_page, 2),
        "section_markers": section_markers or 0,
    }
    return ExtractionIntentRecord(
        intent=intent,
        confidence=round(confidence, 3),
        reasons=reasons,
        signals=signals,
        classified_at=classified_at,
        source_tier=source_tier,
    )


def resolve_hybrid_profile(
    intent: ExtractionIntent | str | None,
    processing: ProcessingConfig,
) -> HybridProfileSpec:
    """Resolve a persisted intent to a concrete backend profile."""
    resolved_intent = _coerce_intent(intent)
    base_backend = processing.opendataloader_hybrid_backend
    base_client_mode = processing.opendataloader_hybrid_client_mode
    fallback_to_fast = processing.opendataloader_hybrid_fallback

    allow_picture_enrichment = (
        processing.opendataloader_hybrid_auto_picture_intents
        or processing.opendataloader_hybrid_enrich_picture_description
    )

    if resolved_intent == ExtractionIntent.FAST:
        return HybridProfileSpec(
            intent=resolved_intent,
            profile_key="fast",
            mode="fast",
            backend=None,
            client_mode=None,
            force_ocr=False,
            enrich_formula=False,
            enrich_picture_description=False,
            fallback_to_fast=fallback_to_fast,
        )

    force_ocr = "ocr" in resolved_intent.value
    enrich_formula = "formula" in resolved_intent.value
    enrich_picture = "picture" in resolved_intent.value
    if enrich_picture and not allow_picture_enrichment:
        enrich_picture = False
    effective_intent = _intent_from_flags(
        needs_ocr=force_ocr,
        needs_formula=enrich_formula,
        needs_picture=enrich_picture,
    )
    if effective_intent == ExtractionIntent.FAST:
        return HybridProfileSpec(
            intent=effective_intent,
            profile_key="fast",
            mode="fast",
            backend=None,
            client_mode=None,
            force_ocr=False,
            enrich_formula=False,
            enrich_picture_description=False,
            fallback_to_fast=fallback_to_fast,
        )
    client_mode = "full" if (enrich_formula or enrich_picture) else base_client_mode
    profile_key = (
        f"hybrid:"
        f"backend={base_backend}:"
        f"client={client_mode}:"
        f"ocr={int(force_ocr)}:"
        f"formula={int(enrich_formula)}:"
        f"picture={int(enrich_picture)}"
    )
    return HybridProfileSpec(
        intent=effective_intent,
        profile_key=profile_key,
        mode="hybrid",
        backend=base_backend,
        client_mode=client_mode,
        force_ocr=force_ocr,
        enrich_formula=enrich_formula,
        enrich_picture_description=enrich_picture,
        fallback_to_fast=fallback_to_fast,
    )


def build_extraction_plan(
    *,
    paper_ids: list[str],
    classification_records: Mapping[str, ClassificationRecordLike],
    processing: ProcessingConfig,
) -> list[ExtractionPlanItem]:
    """Build extraction plan items from persisted classification records."""
    plan_items: list[ExtractionPlanItem] = []
    for paper_id in paper_ids:
        record = classification_records.get(paper_id)
        if record is None:
            continue
        intent = _coerce_intent(getattr(record, "extraction_intent", None))
        profile = resolve_hybrid_profile(intent, processing)
        plan_items.append(
            ExtractionPlanItem(
                paper_id=paper_id,
                document_type=record.document_type,
                extractable=record.extractable,
                intent=intent,
                intent_confidence=record.intent_confidence,
                intent_reasons=list(record.intent_reasons),
                source_tier=_coerce_source_tier(record.intent_source_tier),
                profile=profile,
                fallback_policy="runtime_escalation_allowed",
            )
        )
    return plan_items


def group_extraction_plan_by_profile(
    plan_items: list[ExtractionPlanItem],
) -> dict[str, list[ExtractionPlanItem]]:
    """Group plan items by exact resolved profile key."""
    grouped: dict[str, list[ExtractionPlanItem]] = {}
    for item in plan_items:
        grouped.setdefault(item.profile.profile_key, []).append(item)
    return grouped


def summarize_intents(
    records: Mapping[str, ClassificationRecordLike],
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Return counts by intent, by profile key, and by confidence bucket."""
    by_intent: dict[str, int] = {}
    by_profile: dict[str, int] = {}
    confidence_buckets = {"low": 0, "medium": 0, "high": 0}
    for record in records.values():
        intent = record.extraction_intent or ExtractionIntent.FAST.value
        profile_key = record.hybrid_profile_key or "fast"
        confidence = record.intent_confidence
        by_intent[intent] = by_intent.get(intent, 0) + 1
        by_profile[profile_key] = by_profile.get(profile_key, 0) + 1
        if confidence >= 0.8:
            confidence_buckets["high"] += 1
        elif confidence >= 0.5:
            confidence_buckets["medium"] += 1
        else:
            confidence_buckets["low"] += 1
    return by_intent, by_profile, confidence_buckets


def _coerce_source_tier(source_tier: str | None) -> IntentSourceTier:
    """Normalize stored source-tier strings to the supported literal set."""
    if source_tier == "cheap_text_pass":
        return "cheap_text_pass"
    if source_tier == "stored_snapshot":
        return "stored_snapshot"
    return "metadata_only"


def _count_pattern_hits(pattern: re.Pattern[str], text: str) -> int:
    """Count regex hits in the provided text."""
    if not text:
        return 0
    return len(pattern.findall(text))


def _looks_math_heavy(text: str) -> bool:
    """Return True when title/metadata strongly suggests mathematical content."""
    keywords = (
        "mathematics",
        "mathematical",
        "algebra",
        "geometry",
        "topology",
        "optimization",
        "theorem",
        "proof",
        "quantum",
        "physics",
        "statistics",
    )
    return any(keyword in text for keyword in keywords)


def _looks_figure_heavy(text: str) -> bool:
    """Return True when title/metadata suggests visual or chart-heavy content."""
    keywords = (
        "visualization",
        "imaging",
        "chart",
        "diagram",
        "plot",
        "remote sensing",
        "microscopy",
        "radiology",
        "vision",
    )
    return any(keyword in text for keyword in keywords)


def _intent_from_flags(
    *,
    needs_ocr: bool,
    needs_formula: bool,
    needs_picture: bool,
) -> ExtractionIntent:
    """Map feature flags to the persisted enum."""
    if needs_ocr and needs_formula and needs_picture:
        return ExtractionIntent.HYBRID_OCR_FORMULA_PICTURE
    if needs_ocr and needs_formula:
        return ExtractionIntent.HYBRID_OCR_FORMULA
    if needs_ocr and needs_picture:
        return ExtractionIntent.HYBRID_OCR_PICTURE
    if needs_formula and needs_picture:
        return ExtractionIntent.HYBRID_FORMULA_PICTURE
    if needs_ocr:
        return ExtractionIntent.HYBRID_OCR
    if needs_formula:
        return ExtractionIntent.HYBRID_FORMULA
    if needs_picture:
        return ExtractionIntent.HYBRID_PICTURE
    return ExtractionIntent.FAST


def _coerce_intent(intent: ExtractionIntent | str | None) -> ExtractionIntent:
    """Normalize a stored or ad hoc intent value."""
    if isinstance(intent, ExtractionIntent):
        return intent
    if isinstance(intent, str) and intent:
        try:
            return ExtractionIntent(intent)
        except ValueError:
            return ExtractionIntent.FAST
    return ExtractionIntent.FAST
