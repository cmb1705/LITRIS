"""Tests for index-scoped semantic batch planning and retry flows."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.analysis.dimensions import DimensionProfile, configure_dimension_registry
from src.analysis.schemas import DimensionedExtraction, ExtractionResult
from src.config import Config
from src.indexing.semantic_jobs import (
    build_batch_manifest,
    collect_semantic_batch,
    load_batch_manifest,
    plan_semantic_batch,
    retry_semantic_papers,
    save_batch_manifest,
    submit_semantic_batch,
)
from src.indexing.structured_store import StructuredStore


def _make_config(tmp_path: Path) -> Config:
    zotero_dir = tmp_path / "zotero"
    storage_dir = zotero_dir / "storage"
    zotero_dir.mkdir(parents=True, exist_ok=True)
    storage_dir.mkdir(parents=True, exist_ok=True)
    (zotero_dir / "zotero.sqlite").touch()
    return Config(
        zotero={
            "database_path": str(zotero_dir / "zotero.sqlite"),
            "storage_path": str(storage_dir),
        }
    )


def _custom_profile() -> DimensionProfile:
    return DimensionProfile(
        profile_id="custom_semantic_v1",
        version="1.0.0",
        title="Custom Semantic Profile",
        sections=[
            {
                "id": "research_core",
                "label": "Research Core",
                "order": 1,
            }
        ],
        dimensions=[
            {
                "id": "custom_dimension",
                "label": "Custom Dimension",
                "question": "What is the custom semantic finding?",
                "section": "research_core",
                "order": 1,
                "core": True,
                "legacy_field_name": "q01_research_question",
                "legacy_short_name": "q01",
            }
        ],
    )


def _write_profile(path: Path, profile: DimensionProfile) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(profile.model_dump(mode="json"), handle, sort_keys=False)


def _paper_record(index_dir: Path, paper_id: str) -> dict:
    pdf_path = index_dir / f"{paper_id}.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    return {
        "paper_id": paper_id,
        "zotero_key": paper_id,
        "zotero_item_id": 1,
        "title": f"Paper {paper_id}",
        "publication_year": 2024,
        "item_type": "journalArticle",
        "pdf_path": str(pdf_path),
        "source_path": str(pdf_path),
        "source_media_type": "application/pdf",
        "date_added": "2026-04-16T00:00:00",
        "date_modified": "2026-04-16T00:00:00",
    }


def _seed_index(index_dir: Path, paper_ids: list[str]) -> StructuredStore:
    store = StructuredStore(index_dir)
    store.save_papers({paper_id: _paper_record(index_dir, paper_id) for paper_id in paper_ids})
    store.save_extractions({})
    return store


def _save_snapshot(store: StructuredStore, paper_id: str, text: str) -> None:
    paper = store.load_papers()[paper_id]
    pdf_path = Path(paper["pdf_path"])
    stat = pdf_path.stat()
    store.save_text_snapshot(
        paper_id,
        text,
        metadata={
            "paper_id": paper_id,
            "source": "test_fixture",
            "pdf_path": str(pdf_path),
            "pdf_size": int(stat.st_size),
            "pdf_mtime_ns": int(stat.st_mtime_ns),
        },
    )


@pytest.fixture(autouse=True)
def _reset_dimension_registry() -> None:
    configure_dimension_registry()
    yield
    configure_dimension_registry()


def test_plan_semantic_batch_respects_selection_and_profile_scope(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    profile = _custom_profile()
    profile_path = tmp_path / "custom_profile.yaml"
    _write_profile(profile_path, profile)

    store = _seed_index(tmp_path, ["paper_001", "paper_002", "paper_003", "paper_004"])
    _save_snapshot(store, "paper_001", "Matched snapshot text")
    _save_snapshot(store, "paper_002", "Eligible snapshot text")
    _save_snapshot(store, "paper_003", "Excluded snapshot text")
    store.save_extractions(
        {
            "paper_001": {
                "paper_id": "paper_001",
                "extraction": DimensionedExtraction(
                    paper_id="paper_001",
                    profile_id=profile.profile_id,
                    profile_version=profile.version,
                    profile_fingerprint=profile.fingerprint,
                    prompt_version="2.0.0",
                    extraction_model="existing-model",
                    extracted_at="2026-04-16T00:00:00",
                    dimensions={"custom_dimension": "already extracted"},
                ).to_index_dict(),
            }
        }
    )

    plan = plan_semantic_batch(
        index_dir=tmp_path,
        config=config,
        provider="openai",
        paper_ids=["paper_001", "paper_002", "paper_003", "paper_004"],
        exclude_paper_ids=["paper_003"],
        profile_reference=str(profile_path),
    )

    assert [item.paper.paper_id for item in plan.selected] == ["paper_002"]
    assert plan.skipped_existing == ["paper_001"]
    assert plan.skipped_excluded == ["paper_003"]
    assert plan.skipped_missing_text == ["paper_004"]
    assert plan.estimated_cost["total_requests"] == len(plan.pass_definitions)


def test_submit_semantic_batch_persists_index_scoped_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    profile = _custom_profile()
    profile_path = tmp_path / "custom_profile.yaml"
    _write_profile(profile_path, profile)

    store = _seed_index(tmp_path, ["paper_001"])
    _save_snapshot(store, "paper_001", "Snapshot text for batch submit")
    plan = plan_semantic_batch(
        index_dir=tmp_path,
        config=config,
        provider="anthropic",
        paper_ids=["paper_001"],
        profile_reference=str(profile_path),
    )

    class FakeClient:
        def create_batch_requests(self, papers, text_getter, pass_definitions):
            assert [paper.paper_id for paper in papers] == ["paper_001"]
            assert text_getter(papers[0]) == plan.selected[0].text
            assert pass_definitions == plan.pass_definitions
            return [SimpleNamespace(custom_id="paper_001__pass1")]

        def submit_batch(self, requests, *, persist_state=True):
            assert persist_state is False
            assert len(requests) == 1
            return "batch_test_001"

    monkeypatch.setattr(
        "src.indexing.semantic_jobs.create_batch_client",
        lambda **kwargs: FakeClient(),
    )

    manifest = submit_semantic_batch(plan, config=config)
    saved = load_batch_manifest(tmp_path, "batch_test_001")

    assert manifest["batch_id"] == "batch_test_001"
    assert saved["profile_snapshot"]["profile_id"] == profile.profile_id
    assert saved["paper_ids"] == ["paper_001"]
    assert saved["text_sources"]["paper_001"]["text_source"] == "fulltext_snapshot"
    assert len(saved["pass_definitions"]) == len(plan.pass_definitions)


def test_collect_semantic_batch_persists_semantic_results_and_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    profile = _custom_profile()
    store = _seed_index(tmp_path, ["paper_001"])
    _save_snapshot(store, "paper_001", "Snapshot text for collect")

    plan = plan_semantic_batch(
        index_dir=tmp_path,
        config=config,
        provider="openai",
        paper_ids=["paper_001"],
        profile_reference=None,
        include_existing=True,
    )
    custom_plan = plan
    custom_plan.profile = profile
    custom_plan.pass_definitions = [("Research Core", [("q01_research_question", profile.dimensions[0].question)])]
    manifest = build_batch_manifest(plan=custom_plan, batch_id="batch_collect_001")
    save_batch_manifest(tmp_path, manifest)

    class FakeCollectClient:
        def get_results(self, batch_id, *, pass_definitions, profile, prompt_version):
            assert batch_id == "batch_collect_001"
            assert profile.profile_id == "custom_semantic_v1"
            assert pass_definitions == custom_plan.pass_definitions
            assert prompt_version == "2.0.0"
            yield ExtractionResult(
                paper_id="paper_001",
                success=True,
                extraction=DimensionedExtraction(
                    paper_id="paper_001",
                    profile_id=profile.profile_id,
                    profile_version=profile.version,
                    profile_fingerprint=profile.fingerprint,
                    prompt_version=prompt_version,
                    extraction_model="gpt-test",
                    extracted_at="2026-04-16T00:00:00",
                    dimensions={"custom_dimension": "fresh semantic answer"},
                ),
                model_used="gpt-test",
                input_tokens=123,
                output_tokens=45,
            )

    monkeypatch.setattr(
        "src.indexing.semantic_jobs.create_batch_client",
        lambda **kwargs: FakeCollectClient(),
    )
    monkeypatch.setattr(
        "src.indexing.pipeline.generate_summary",
        lambda index_dir, logger: {"total_papers": 1},
    )

    summary = collect_semantic_batch(
        index_dir=tmp_path,
        batch_id="batch_collect_001",
        config=config,
    )
    reloaded = StructuredStore(tmp_path)

    assert summary["added"] == 1
    assert reloaded.load_dimension_profile()["profile_id"] == profile.profile_id
    saved = reloaded.load_extractions()["paper_001"]
    assert saved["batch_id"] == "batch_collect_001"
    assert saved["extraction"]["dimensions"]["custom_dimension"] == "fresh semantic answer"


def test_retry_semantic_papers_uses_stored_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    profile = _custom_profile()
    store = _seed_index(tmp_path, ["paper_001"])
    _save_snapshot(store, "paper_001", "Snapshot text for retry flow")

    class FakeExtractor:
        def extract_batch(self, papers, text_snapshots):
            assert [paper.paper_id for paper in papers] == ["paper_001"]
            assert text_snapshots["paper_001"]["text"] == "Snapshot text for retry flow"
            yield ExtractionResult(
                paper_id="paper_001",
                success=True,
                extraction=DimensionedExtraction(
                    paper_id="paper_001",
                    profile_id=profile.profile_id,
                    profile_version=profile.version,
                    profile_fingerprint=profile.fingerprint,
                    prompt_version="2.0.0",
                    extraction_model="retry-model",
                    extracted_at="2026-04-16T00:00:00",
                    dimensions={"custom_dimension": "retried answer"},
                ),
                timestamp=datetime(2026, 4, 16, 0, 0, 0),
                model_used="retry-model",
                duration_seconds=1.0,
            )

    monkeypatch.setattr(
        "src.indexing.pipeline.configure_extraction_runtime",
        lambda args, config, logger: (
            args.provider or "anthropic",
            args.mode or "api",
            tmp_path / "cache",
            1,
            False,
        ),
    )
    monkeypatch.setattr(
        "src.indexing.pipeline.build_section_extractor",
        lambda *args, **kwargs: FakeExtractor(),
    )
    monkeypatch.setattr(
        "src.indexing.pipeline.generate_summary",
        lambda index_dir, logger: {"total_papers": 1},
    )

    summary = retry_semantic_papers(
        index_dir=tmp_path,
        config=config,
        logger_override=None,
        paper_ids=["paper_001"],
        provider="anthropic",
        mode="api",
        model="claude-test",
        profile=profile,
    )
    reloaded = StructuredStore(tmp_path)

    assert summary["updated"] == 1
    assert reloaded.load_dimension_profile()["profile_id"] == profile.profile_id
    assert (
        reloaded.load_extractions()["paper_001"]["extraction"]["dimensions"]["custom_dimension"]
        == "retried answer"
    )
