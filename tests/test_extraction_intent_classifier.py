"""Tests for extraction-intent classification and grouped routing."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from src.analysis.classification_store import ClassificationIndex, ClassificationStore
from src.analysis.extraction_intent_classifier import (
    ExtractionIntent,
    build_extraction_plan,
    classify_extraction_intent,
    group_extraction_plan_by_profile,
    resolve_hybrid_profile,
)
from src.analysis.section_extractor import SectionExtractor
from src.config import Config
from src.indexing.orchestrator import IndexOrchestrator
from src.utils.logging_config import get_logger
from src.zotero.models import PaperMetadata


def _make_config() -> Config:
    return Config(
        zotero={
            "database_path": "zotero.sqlite",
            "storage_path": "storage",
        }
    )


def _make_paper(title: str, pdf_path: Path | None = None) -> PaperMetadata:
    return PaperMetadata(
        zotero_key="TESTKEY",
        zotero_item_id=1,
        item_type="journalArticle",
        title=title,
        journal="Journal of Tests",
        tags=[],
        collections=[],
        pdf_path=pdf_path,
        source_path=pdf_path,
        date_added=datetime(2024, 1, 1),
        date_modified=datetime(2024, 1, 2),
    )


def test_metadata_only_defaults_to_fast_when_signals_are_weak():
    record = classify_extraction_intent(
        _make_paper("A plain empirical study"),
        source_tier="metadata_only",
    )

    assert record.intent == ExtractionIntent.FAST
    assert record.confidence < 0.5


def test_cheap_text_pass_detects_formula_and_picture_needs():
    paper = _make_paper("Optimization with visual diagnostics")
    text = (
        "Equation 1 introduces the objective. The theorem and proof follow. "
        "Figure 1 shows the chart. Figure 2 shows the plot."
    )
    record = classify_extraction_intent(
        paper,
        text=text,
        word_count=900,
        page_count=5,
        section_markers=4,
        source_tier="cheap_text_pass",
        allow_picture_enrichment=True,
    )

    assert record.intent == ExtractionIntent.HYBRID_FORMULA_PICTURE
    assert record.signals["needs_formula"] is True
    assert record.signals["needs_picture"] is True


def test_picture_signals_default_to_fast_when_auto_picture_disabled():
    paper = _make_paper("Visualization-heavy analysis")
    text = "Figure 1 shows the chart. Figure 2 shows the plot and diagram."
    record = classify_extraction_intent(
        paper,
        text=text,
        word_count=900,
        page_count=5,
        section_markers=4,
        source_tier="cheap_text_pass",
    )

    assert record.intent == ExtractionIntent.FAST
    assert record.signals["picture_signals_detected"] is True
    assert record.signals["needs_picture"] is False


def test_cheap_text_pass_detects_scanned_pdf_needs_ocr():
    paper = _make_paper("Historical archive scan")
    record = classify_extraction_intent(
        paper,
        text="scanned archival document",
        word_count=220,
        page_count=8,
        section_markers=0,
        source_tier="cheap_text_pass",
    )

    assert record.intent == ExtractionIntent.HYBRID_OCR
    assert record.signals["needs_ocr"] is True


def test_build_extraction_plan_groups_by_exact_profile():
    config = _make_config()
    config.processing.opendataloader_hybrid_backend = "docling-fast"
    config.processing.opendataloader_hybrid_auto_picture_intents = True

    records = {
        "p1": type(
            "Rec",
            (),
            {
                "document_type": "research_paper",
                "extractable": True,
                "extraction_intent": "fast",
                "intent_confidence": 0.4,
                "intent_reasons": [],
                "intent_source_tier": "metadata_only",
            },
        )(),
        "p2": type(
            "Rec",
            (),
            {
                "document_type": "research_paper",
                "extractable": True,
                "extraction_intent": "hybrid_formula_picture",
                "intent_confidence": 0.88,
                "intent_reasons": ["formula", "picture"],
                "intent_source_tier": "stored_snapshot",
            },
        )(),
    }

    plan_items = build_extraction_plan(
        paper_ids=["p1", "p2"],
        classification_records=records,
        processing=config.processing,
    )
    grouped = group_extraction_plan_by_profile(plan_items)

    assert sorted(grouped) == [
        "fast",
        "hybrid:backend=docling-fast:client=full:ocr=0:formula=1:picture=1",
    ]
    assert grouped["fast"][0].intent == ExtractionIntent.FAST


def test_resolve_hybrid_profile_strips_picture_when_disabled():
    config = _make_config()

    profile = resolve_hybrid_profile(
        ExtractionIntent.HYBRID_FORMULA_PICTURE,
        config.processing,
    )

    assert profile.intent == ExtractionIntent.HYBRID_FORMULA
    assert profile.profile_key.endswith("formula=1:picture=0")


def test_runtime_override_is_recorded_for_fast_to_hybrid():
    actual_profile, reason = SectionExtractor._resolve_actual_profile(
        planned_profile_key="fast",
        extraction_method="opendataloader_hybrid",
    )

    assert actual_profile == "runtime:opendataloader_hybrid"
    assert reason == "runtime escalated from fast to hybrid"


def test_runtime_override_reports_hybrid_failure_reason():
    actual_profile, reason = SectionExtractor._resolve_actual_profile(
        planned_profile_key="hybrid:backend=docling-fast:client=full:ocr=0:formula=1:picture=0",
        extraction_method="pymupdf",
        extraction_diagnostics={"opendataloader_hybrid": "cli exit code 1"},
    )

    assert actual_profile == "pymupdf"
    assert reason == "planned hybrid failed (cli exit code 1); runtime used pymupdf"


def test_runtime_override_logging_is_emitted():
    with patch("src.analysis.section_extractor.logger.info") as mock_info:
        SectionExtractor._log_extraction_override(
            paper_id="paper-1",
            planned_profile_key="fast",
            actual_profile_key="runtime:opendataloader_hybrid",
            override_reason="runtime escalated from fast to hybrid",
        )

    mock_info.assert_called_once_with(
        "Extraction routing override for %s: planned=%s actual=%s reason=%s",
        "paper-1",
        "fast",
        "runtime:opendataloader_hybrid",
        "runtime escalated from fast to hybrid",
    )


def test_orchestrator_reuses_stored_snapshot_for_intent_classification(tmp_path):
    config = _make_config()
    config.processing.opendataloader_hybrid_auto_picture_intents = True
    index_dir = tmp_path / "index"
    orchestrator = IndexOrchestrator(
        project_root=tmp_path,
        logger=get_logger("test.intent"),
        index_dir=index_dir,
        config=config,
    )
    class_store = ClassificationStore(index_dir=index_dir)
    class_index = ClassificationIndex()

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    paper = _make_paper("Optimization with charts", pdf_path=pdf_path)
    orchestrator.store.save_text_snapshot(
        paper.paper_id,
        (
            "Equation 1 defines the objective and theorem. The proof follows in detail. "
            "Figure 1 shows the chart and Figure 2 shows the plot. "
            "Additional discussion explains the optimization procedure and visual analysis. "
            "Equation 2 refines the objective while the figures summarize the results."
        ),
        metadata={
            "pdf_path": str(pdf_path),
            "pdf_size": pdf_path.stat().st_size,
            "pdf_mtime_ns": pdf_path.stat().st_mtime_ns,
            "page_count": 3,
            "section_markers": 4,
        },
    )

    with patch(
        "src.indexing.orchestrator.PDFExtractor.extract_text_with_method",
        side_effect=AssertionError("cheap extraction should not run"),
    ):
        orchestrator._update_classification_index(
            class_store=class_store,
            class_index=class_index,
            papers=[paper],
            deleted_paper_ids=[],
            config=config,
            refresh_text=False,
        )

    record = class_index.papers[paper.paper_id]
    assert record.intent_source_tier == "stored_snapshot"
    assert record.extraction_intent == "hybrid_formula_picture"
    assert record.hybrid_profile_key.endswith("formula=1:picture=1")
