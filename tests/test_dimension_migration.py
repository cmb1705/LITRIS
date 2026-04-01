"""Integration tests for portable dimension migration and backfill flows."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from src.analysis.dimensions import (
    LEGACY_PROFILE_ID,
    build_legacy_dimension_profile,
    configure_dimension_registry,
    get_dimension_value,
)
from src.analysis.schemas import DimensionedExtraction, ExtractionResult
from src.config import Config
from src.indexing.orchestrator import IndexManifest
from src.indexing.structured_store import StructuredStore
from src.utils.checkpoint import CheckpointManager
from src.utils.file_utils import safe_read_json, safe_write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIMENSIONS_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "dimensions.py"


def _load_dimensions_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "litris_dimensions_script",
        DIMENSIONS_SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_config(tmp_path: Path) -> Config:
    return Config(
        zotero={
            "database_path": str(tmp_path / "zotero.sqlite"),
            "storage_path": str(tmp_path / "storage"),
        }
    )


def _paper_record(index_dir: Path, paper_id: str) -> dict:
    pdf_path = index_dir / f"{paper_id}.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    return {
        "paper_id": paper_id,
        "title": f"Paper {paper_id}",
        "author_string": "Jane Doe",
        "publication_year": 2024,
        "item_type": "journalArticle",
        "collections": ["Core"],
        "pdf_path": str(pdf_path),
        "date_added": "2026-03-31T00:00:00",
        "date_modified": "2026-03-31T00:00:00",
    }


def _legacy_extraction_record(paper_id: str) -> dict:
    return {
        "paper_id": paper_id,
        "prompt_version": "2.0.0",
        "extraction_model": "legacy-model",
        "extracted_at": "2026-03-31T00:00:00",
        "q01_research_question": "What does the paper study?",
        "q02_thesis": "Legacy thesis",
        "q07_methods": "Legacy methods",
        "q08_data": "Legacy data",
        "q34_power_dynamics": "Legacy power dynamics",
        "dimension_coverage": 0.0,
        "coverage_flags": [],
    }


def _write_legacy_index(index_dir: Path, paper_id: str = "paper_001") -> StructuredStore:
    store = StructuredStore(index_dir)
    store.save_papers({paper_id: _paper_record(index_dir, paper_id)})
    safe_write_json(
        index_dir / "semantic_analyses.json",
        {
            "schema_version": "1.0",
            "generated_at": "2026-03-31T00:00:00",
            "extraction_count": 1,
            "extractions": {
                paper_id: {
                    "paper_id": paper_id,
                    "extraction": _legacy_extraction_record(paper_id),
                }
            },
        },
    )
    return store


def _write_legacy_index_many(index_dir: Path, paper_ids: list[str]) -> StructuredStore:
    store = StructuredStore(index_dir)
    store.save_papers({paper_id: _paper_record(index_dir, paper_id) for paper_id in paper_ids})
    safe_write_json(
        index_dir / "semantic_analyses.json",
        {
            "schema_version": "1.0",
            "generated_at": "2026-03-31T00:00:00",
            "extraction_count": len(paper_ids),
            "extractions": {
                paper_id: {
                    "paper_id": paper_id,
                    "extraction": _legacy_extraction_record(paper_id),
                }
                for paper_id in paper_ids
            },
        },
    )
    return store


def _write_text_snapshot(index_dir: Path, paper_id: str, text: str) -> None:
    store = StructuredStore(index_dir)
    paper = store.load_papers()[paper_id]
    pdf_path = Path(paper["pdf_path"])
    stat = pdf_path.stat()
    store.save_text_snapshot(
        paper_id,
        text,
        metadata={
            "paper_id": paper_id,
            "source": "test_fixture",
            "extraction_method": "companion",
            "pdf_path": str(pdf_path),
            "pdf_size": int(stat.st_size),
            "pdf_mtime_ns": int(stat.st_mtime_ns),
        },
    )


def _write_profile(path: Path, profile_dict: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(profile_dict, handle, sort_keys=False, allow_unicode=False)


@pytest.fixture(autouse=True)
def _reset_dimension_registry() -> None:
    configure_dimension_registry()
    yield
    configure_dimension_registry()


@pytest.fixture
def dimensions_cli() -> ModuleType:
    return _load_dimensions_cli()


def test_legacy_index_loads_builtin_profile_and_aliases(tmp_path: Path) -> None:
    store = _write_legacy_index(tmp_path)

    profile = store.load_dimension_profile()
    loaded = store.load_extractions()
    record = loaded["paper_001"]

    assert profile["profile_id"] == LEGACY_PROFILE_ID
    assert get_dimension_value(record, "q02") == "Legacy thesis"
    assert get_dimension_value(record, "q02_thesis") == "Legacy thesis"
    assert get_dimension_value(record, "thesis") == "Legacy thesis"


def test_migrate_store_rewrites_records_and_becomes_idempotent(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    dimensions_cli: ModuleType,
) -> None:
    _write_legacy_index(tmp_path)
    logger = SimpleNamespace(info=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None)

    migrate_args = argparse.Namespace(
        index_dir=tmp_path,
        dry_run=False,
        backup_dir=None,
    )
    assert dimensions_cli.migrate_store(migrate_args, logger) == 0

    migrated = safe_read_json(tmp_path / "semantic_analyses.json", default={})
    record = migrated["extractions"]["paper_001"]["extraction"]

    assert migrated["schema_version"] == dimensions_cli.MIGRATED_EXTRACTION_SCHEMA_VERSION
    assert record["dimensions"]["thesis"] == "Legacy thesis"
    assert record["dimensions"]["methods"] == "Legacy methods"
    assert record["profile_id"] == LEGACY_PROFILE_ID

    dry_run_args = argparse.Namespace(
        index_dir=tmp_path,
        dry_run=True,
        backup_dir=None,
    )
    assert dimensions_cli.migrate_store(dry_run_args, logger) == 0
    dry_run_output = capsys.readouterr().out
    assert "records_requiring_migration: 0" in dry_run_output


def test_backfill_disable_only_skips_extractor_and_updates_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dimensions_cli: ModuleType,
) -> None:
    store = _write_legacy_index(tmp_path)
    old_profile = build_legacy_dimension_profile()
    store.save_dimension_profile(old_profile.model_dump(mode="json"))
    IndexManifest(extraction={"profile_id": old_profile.profile_id}).save(
        tmp_path / "index_manifest.json"
    )

    profile_dict = old_profile.model_dump(mode="json")
    profile_dict["profile_id"] = "legacy_semantic_disable_power"
    profile_dict["version"] = "1.0.1"
    for dimension in profile_dict["dimensions"]:
        if dimension["id"] == "power_dynamics":
            dimension["enabled"] = False
    profile_path = tmp_path / "disable_power.yaml"
    _write_profile(profile_path, profile_dict)

    embedding_calls: dict[str, object] = {}

    def _fail_build_extractor(*args, **kwargs):
        raise AssertionError("disable-only backfill should not build an extractor")

    monkeypatch.setattr(dimensions_cli, "build_section_extractor", _fail_build_extractor)
    monkeypatch.setattr(dimensions_cli, "configure_extraction_runtime", _fail_build_extractor)
    monkeypatch.setattr(dimensions_cli, "generate_scoped_raptor_summaries", lambda **kwargs: {})
    monkeypatch.setattr(
        dimensions_cli,
        "run_embedding_generation",
        lambda **kwargs: embedding_calls.update(kwargs),
    )
    monkeypatch.setattr(dimensions_cli, "compute_similarity_pairs", lambda **kwargs: 0)
    monkeypatch.setattr(dimensions_cli, "generate_summary", lambda *args, **kwargs: {})

    args = argparse.Namespace(
        index_dir=tmp_path,
        dimension_profile=profile_path,
        paper=[],
        dry_run=False,
        skip_embeddings=False,
        skip_similarity=False,
        provider=None,
        mode=None,
        model=None,
        parallel=None,
        no_cache=False,
        summary_model=None,
        methodology_model=None,
    )
    logger = SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None)
    config = _make_config(tmp_path)

    assert dimensions_cli.backfill_dimensions(args, config, logger) == 0

    updated = DimensionedExtraction.from_record(
        StructuredStore(tmp_path).load_extractions()["paper_001"]
    )
    manifest, manifest_error = IndexManifest.load(tmp_path / "index_manifest.json")

    assert updated.profile_id == "legacy_semantic_disable_power"
    assert updated.dimensions["power_dynamics"] == "Legacy power dynamics"
    assert [paper.paper_id for paper in embedding_calls["papers"]] == ["paper_001"]
    assert manifest_error is None
    assert manifest is not None
    assert manifest.extraction["profile_id"] == "legacy_semantic_disable_power"


def test_backfill_reextracts_changed_section_only_and_preserves_other_dimensions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dimensions_cli: ModuleType,
) -> None:
    store = _write_legacy_index(tmp_path)
    old_profile = build_legacy_dimension_profile()
    store.save_dimension_profile(old_profile.model_dump(mode="json"))

    profile_dict = old_profile.model_dump(mode="json")
    profile_dict["profile_id"] = "legacy_semantic_methods_v2"
    profile_dict["version"] = "2.0.0"
    for dimension in profile_dict["dimensions"]:
        if dimension["id"] == "methods":
            dimension["question"] = "What updated methods and analytical techniques are used?"
    profile_path = tmp_path / "methods_v2.yaml"
    _write_profile(profile_path, profile_dict)

    target_profile = dimensions_cli.load_dimension_profile(profile_path)
    observed: dict[str, object] = {}

    class DummyExtractor:
        def extract_batch(self, papers, progress_callback=None, section_ids=None):
            observed["section_ids"] = section_ids
            for paper in papers:
                yield ExtractionResult(
                    paper_id=paper.paper_id,
                    success=True,
                    extraction=DimensionedExtraction(
                        paper_id=paper.paper_id,
                        profile_id=target_profile.profile_id,
                        profile_version=target_profile.version,
                        profile_fingerprint=target_profile.fingerprint,
                        prompt_version="2.0.0",
                        extraction_model="target-model",
                        extracted_at="2026-03-31T01:00:00",
                        dimensions={
                            "paradigm": "Updated paradigm",
                            "methods": "Updated methods",
                            "data": "Updated data",
                            "reproducibility": "Updated reproducibility",
                            "framework": "Updated framework",
                        },
                    ),
                    model_used="target-model",
                )

    monkeypatch.setattr(
        dimensions_cli,
        "configure_extraction_runtime",
        lambda *args, **kwargs: ("anthropic", "api", tmp_path, 1, False),
    )
    monkeypatch.setattr(
        dimensions_cli,
        "build_section_extractor",
        lambda **kwargs: DummyExtractor(),
    )
    monkeypatch.setattr(dimensions_cli, "generate_scoped_raptor_summaries", lambda **kwargs: {})
    monkeypatch.setattr(dimensions_cli, "run_embedding_generation", lambda **kwargs: None)
    monkeypatch.setattr(dimensions_cli, "compute_similarity_pairs", lambda **kwargs: 0)
    monkeypatch.setattr(dimensions_cli, "generate_summary", lambda *args, **kwargs: {})

    args = argparse.Namespace(
        index_dir=tmp_path,
        dimension_profile=profile_path,
        paper=[],
        dry_run=False,
        skip_embeddings=False,
        skip_similarity=False,
        provider=None,
        mode=None,
        model=None,
        parallel=None,
        no_cache=False,
        summary_model=None,
        methodology_model=None,
    )
    logger = SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None)
    config = _make_config(tmp_path)

    assert dimensions_cli.backfill_dimensions(args, config, logger) == 0

    updated = DimensionedExtraction.from_record(
        StructuredStore(tmp_path).load_extractions()["paper_001"]
    )

    assert observed["section_ids"] == ["methodology"]
    assert updated.profile_id == "legacy_semantic_methods_v2"
    assert updated.dimensions["thesis"] == "Legacy thesis"
    assert updated.dimensions["methods"] == "Updated methods"
    assert updated.dimensions["data"] == "Updated data"
    assert updated.dimensions["power_dynamics"] == "Legacy power dynamics"


def test_backfill_reuses_matching_fulltext_snapshot_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dimensions_cli: ModuleType,
) -> None:
    store = _write_legacy_index(tmp_path)
    old_profile = build_legacy_dimension_profile()
    store.save_dimension_profile(old_profile.model_dump(mode="json"))
    _write_text_snapshot(tmp_path, "paper_001", "Stored canonical text for backfill reuse.")

    profile_dict = old_profile.model_dump(mode="json")
    profile_dict["profile_id"] = "legacy_semantic_methods_v2"
    profile_dict["version"] = "2.0.0"
    for dimension in profile_dict["dimensions"]:
        if dimension["id"] == "methods":
            dimension["question"] = "What updated methods and analytical techniques are used?"
    profile_path = tmp_path / "methods_v2.yaml"
    _write_profile(profile_path, profile_dict)

    target_profile = dimensions_cli.load_dimension_profile(profile_path)
    observed: dict[str, object] = {}

    class DummyExtractor:
        def extract_batch(self, papers, progress_callback=None, section_ids=None, text_snapshots=None):
            observed["text_snapshots"] = text_snapshots
            for paper in papers:
                yield ExtractionResult(
                    paper_id=paper.paper_id,
                    success=True,
                    extraction=DimensionedExtraction(
                        paper_id=paper.paper_id,
                        profile_id=target_profile.profile_id,
                        profile_version=target_profile.version,
                        profile_fingerprint=target_profile.fingerprint,
                        prompt_version="2.0.0",
                        extraction_model="target-model",
                        extracted_at="2026-03-31T01:00:00",
                        dimensions={"methods": "Updated methods"},
                    ),
                    model_used="target-model",
                )

    monkeypatch.setattr(
        dimensions_cli,
        "configure_extraction_runtime",
        lambda *args, **kwargs: ("anthropic", "api", tmp_path, 1, False),
    )
    monkeypatch.setattr(
        dimensions_cli,
        "build_section_extractor",
        lambda **kwargs: DummyExtractor(),
    )
    monkeypatch.setattr(dimensions_cli, "generate_scoped_raptor_summaries", lambda **kwargs: {})
    monkeypatch.setattr(dimensions_cli, "run_embedding_generation", lambda **kwargs: None)
    monkeypatch.setattr(dimensions_cli, "compute_similarity_pairs", lambda **kwargs: 0)
    monkeypatch.setattr(dimensions_cli, "generate_summary", lambda *args, **kwargs: {})

    args = argparse.Namespace(
        index_dir=tmp_path,
        dimension_profile=profile_path,
        paper=[],
        dry_run=False,
        skip_embeddings=False,
        skip_similarity=False,
        provider=None,
        mode=None,
        model=None,
        parallel=None,
        no_cache=False,
        refresh_text=False,
        summary_model=None,
        methodology_model=None,
    )
    logger = SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None)
    config = _make_config(tmp_path)

    assert dimensions_cli.backfill_dimensions(args, config, logger) == 0
    assert "paper_001" in observed["text_snapshots"]


def test_backfill_refresh_text_bypasses_stored_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dimensions_cli: ModuleType,
) -> None:
    store = _write_legacy_index(tmp_path)
    old_profile = build_legacy_dimension_profile()
    store.save_dimension_profile(old_profile.model_dump(mode="json"))
    _write_text_snapshot(tmp_path, "paper_001", "Stored canonical text for backfill reuse.")

    profile_dict = old_profile.model_dump(mode="json")
    profile_dict["profile_id"] = "legacy_semantic_methods_v2"
    profile_dict["version"] = "2.0.0"
    for dimension in profile_dict["dimensions"]:
        if dimension["id"] == "methods":
            dimension["question"] = "What updated methods and analytical techniques are used?"
    profile_path = tmp_path / "methods_v2.yaml"
    _write_profile(profile_path, profile_dict)

    target_profile = dimensions_cli.load_dimension_profile(profile_path)
    observed: dict[str, object] = {}

    class DummyExtractor:
        def extract_batch(self, papers, progress_callback=None, section_ids=None, text_snapshots=None):
            observed["text_snapshots"] = text_snapshots
            for paper in papers:
                yield ExtractionResult(
                    paper_id=paper.paper_id,
                    success=True,
                    extraction=DimensionedExtraction(
                        paper_id=paper.paper_id,
                        profile_id=target_profile.profile_id,
                        profile_version=target_profile.version,
                        profile_fingerprint=target_profile.fingerprint,
                        prompt_version="2.0.0",
                        extraction_model="target-model",
                        extracted_at="2026-03-31T01:00:00",
                        dimensions={"methods": "Updated methods"},
                    ),
                    model_used="target-model",
                    text_snapshot={
                        "text": "Refreshed canonical text",
                        "source": "refresh",
                        "extraction_method": "pymupdf",
                        "is_cleaned": True,
                        "is_truncated_for_llm": False,
                        "should_persist": True,
                    },
                )

    monkeypatch.setattr(
        dimensions_cli,
        "configure_extraction_runtime",
        lambda *args, **kwargs: ("anthropic", "api", tmp_path, 1, False),
    )
    monkeypatch.setattr(
        dimensions_cli,
        "build_section_extractor",
        lambda **kwargs: DummyExtractor(),
    )
    monkeypatch.setattr(dimensions_cli, "generate_scoped_raptor_summaries", lambda **kwargs: {})
    monkeypatch.setattr(dimensions_cli, "run_embedding_generation", lambda **kwargs: None)
    monkeypatch.setattr(dimensions_cli, "compute_similarity_pairs", lambda **kwargs: 0)
    monkeypatch.setattr(dimensions_cli, "generate_summary", lambda *args, **kwargs: {})

    args = argparse.Namespace(
        index_dir=tmp_path,
        dimension_profile=profile_path,
        paper=[],
        dry_run=False,
        skip_embeddings=False,
        skip_similarity=False,
        provider=None,
        mode=None,
        model=None,
        parallel=None,
        no_cache=False,
        refresh_text=True,
        summary_model=None,
        methodology_model=None,
    )
    logger = SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None)
    config = _make_config(tmp_path)

    assert dimensions_cli.backfill_dimensions(args, config, logger) == 0
    assert observed["text_snapshots"] == {}
    snapshot = StructuredStore(tmp_path).load_text_snapshot("paper_001")
    assert snapshot is not None
    assert snapshot["text"] == "Refreshed canonical text"


def test_fulltext_only_backfill_populates_canonical_snapshots_without_embeddings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dimensions_cli: ModuleType,
) -> None:
    store = _write_legacy_index(tmp_path)
    old_profile = build_legacy_dimension_profile()
    store.save_dimension_profile(old_profile.model_dump(mode="json"))
    profile_path = tmp_path / "legacy.yaml"
    _write_profile(profile_path, old_profile.model_dump(mode="json"))

    embedding_called = False

    class DummyExtractor:
        def extract_batch(
            self,
            papers,
            progress_callback=None,
            section_ids=None,
            text_snapshots=None,
            skip_llm=False,
        ):
            assert skip_llm is True
            assert section_ids is None
            for paper in papers:
                yield ExtractionResult(
                    paper_id=paper.paper_id,
                    success=True,
                    extraction=None,
                    text_snapshot={
                        "text": "Canonical full text",
                        "source": "fulltext_only_test",
                        "extraction_method": "pymupdf",
                        "is_cleaned": True,
                        "is_truncated_for_llm": False,
                        "should_persist": True,
                    },
                )

    monkeypatch.setattr(
        dimensions_cli,
        "configure_extraction_runtime",
        lambda *args, **kwargs: ("anthropic", "cli", tmp_path, 1, False),
    )
    monkeypatch.setattr(dimensions_cli, "build_section_extractor", lambda **kwargs: DummyExtractor())
    monkeypatch.setattr(
        dimensions_cli,
        "run_embedding_generation",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("embeddings should not run")),
    )
    monkeypatch.setattr(dimensions_cli, "generate_summary", lambda *args, **kwargs: {})

    args = argparse.Namespace(
        index_dir=tmp_path,
        dimension_profile=profile_path,
        paper=[],
        dry_run=False,
        skip_embeddings=False,
        skip_similarity=False,
        provider=None,
        mode=None,
        model=None,
        parallel=None,
        no_cache=False,
        refresh_text=True,
        resume=False,
        fulltext_only=True,
        summary_model=None,
        methodology_model=None,
    )
    logger = SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None)
    config = _make_config(tmp_path)

    assert dimensions_cli.backfill_dimensions(args, config, logger) == 0
    snapshot = StructuredStore(tmp_path).load_text_snapshot("paper_001")
    assert snapshot is not None
    assert snapshot["text"] == "Canonical full text"
    assert embedding_called is False


def test_backfill_resume_reuses_staged_extractions_and_skips_completed_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dimensions_cli: ModuleType,
) -> None:
    store = _write_legacy_index_many(tmp_path, ["paper_001", "paper_002"])
    old_profile = build_legacy_dimension_profile()
    store.save_dimension_profile(old_profile.model_dump(mode="json"))

    profile_dict = old_profile.model_dump(mode="json")
    profile_dict["profile_id"] = "legacy_semantic_methods_v2"
    profile_dict["version"] = "2.0.0"
    for dimension in profile_dict["dimensions"]:
        if dimension["id"] == "methods":
            dimension["question"] = "What updated methods and analytical techniques are used?"
    profile_path = tmp_path / "methods_v2.yaml"
    _write_profile(profile_path, profile_dict)
    target_profile = dimensions_cli.load_dimension_profile(profile_path)

    checkpoint_dir = dimensions_cli._backfill_checkpoint_dir(tmp_path, target_profile)
    checkpoint_mgr = CheckpointManager(checkpoint_dir, checkpoint_id="dimension_backfill")
    request_metadata = dimensions_cli._checkpoint_request_metadata(
        target_ids=["paper_001", "paper_002"],
        old_profile=old_profile,
        target_profile=target_profile,
        changed_sections=["methodology"],
        reextract_sections=["methodology"],
        partial_scope=False,
        refresh_text=False,
        fulltext_only=False,
    )
    checkpoint_mgr.initialize(total_items=2, metadata=request_metadata)
    checkpoint_mgr.complete_item("paper_001", success=True)
    checkpoint_mgr.save()

    staged_candidate = DimensionedExtraction(
        paper_id="paper_001",
        profile_id=target_profile.profile_id,
        profile_version=target_profile.version,
        profile_fingerprint=target_profile.fingerprint,
        prompt_version="2.0.0",
        extraction_model="target-model",
        extracted_at="2026-03-31T01:00:00",
        dimensions={"methods": "Staged methods"},
    )
    staged_extraction = dimensions_cli._merge_reextracted_dimensions(
        old_extraction=DimensionedExtraction.from_record(
            StructuredStore(tmp_path).load_extractions()["paper_001"]
        ),
        new_extraction=staged_candidate,
        target_profile=target_profile,
        section_ids=["methodology"],
    )
    dimensions_cli._save_staged_backfill_extractions(
        checkpoint_dir,
        {
            "paper_001": {
                "paper_id": "paper_001",
                "extraction": staged_extraction.to_index_dict(),
                "timestamp": "2026-03-31T01:00:00",
                "model": "target-model",
                "duration": 1.0,
                "input_tokens": 0,
                "output_tokens": 0,
            }
        },
    )

    observed: dict[str, object] = {}

    class DummyExtractor:
        def extract_batch(self, papers, progress_callback=None, section_ids=None, text_snapshots=None, skip_llm=False):
            observed["paper_ids"] = [paper.paper_id for paper in papers]
            for paper in papers:
                yield ExtractionResult(
                    paper_id=paper.paper_id,
                    success=True,
                    extraction=DimensionedExtraction(
                        paper_id=paper.paper_id,
                        profile_id=target_profile.profile_id,
                        profile_version=target_profile.version,
                        profile_fingerprint=target_profile.fingerprint,
                        prompt_version="2.0.0",
                        extraction_model="target-model",
                        extracted_at="2026-03-31T01:00:00",
                        dimensions={"methods": f"Updated methods for {paper.paper_id}"},
                    ),
                    model_used="target-model",
                )

    monkeypatch.setattr(
        dimensions_cli,
        "configure_extraction_runtime",
        lambda *args, **kwargs: ("anthropic", "api", tmp_path, 1, False),
    )
    monkeypatch.setattr(dimensions_cli, "build_section_extractor", lambda **kwargs: DummyExtractor())
    monkeypatch.setattr(dimensions_cli, "generate_scoped_raptor_summaries", lambda **kwargs: {})
    monkeypatch.setattr(dimensions_cli, "run_embedding_generation", lambda **kwargs: None)
    monkeypatch.setattr(dimensions_cli, "compute_similarity_pairs", lambda **kwargs: 0)
    monkeypatch.setattr(dimensions_cli, "generate_summary", lambda *args, **kwargs: {})

    args = argparse.Namespace(
        index_dir=tmp_path,
        dimension_profile=profile_path,
        paper=[],
        dry_run=False,
        skip_embeddings=False,
        skip_similarity=False,
        provider=None,
        mode=None,
        model=None,
        parallel=None,
        no_cache=False,
        refresh_text=False,
        resume=True,
        fulltext_only=False,
        summary_model=None,
        methodology_model=None,
    )
    logger = SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None)
    config = _make_config(tmp_path)

    assert dimensions_cli.backfill_dimensions(args, config, logger) == 0
    assert observed["paper_ids"] == ["paper_002"]

    reloaded = StructuredStore(tmp_path).load_extractions()
    paper1 = DimensionedExtraction.from_record(reloaded["paper_001"])
    paper2 = DimensionedExtraction.from_record(reloaded["paper_002"])
    assert paper1.dimensions["methods"] == "Staged methods"
    assert paper2.dimensions["methods"] == "Updated methods for paper_002"


def test_missing_fulltext_manifest_rebuilds_from_snapshot_files(tmp_path: Path) -> None:
    store = _write_legacy_index(tmp_path)
    _write_text_snapshot(tmp_path, "paper_001", "Snapshot body for manifest rebuild")

    manifest_path = tmp_path / "fulltext_manifest.json"
    manifest_path.unlink()

    rebuilt_store = StructuredStore(tmp_path)
    manifest = rebuilt_store.load_fulltext_manifest()

    assert "paper_001" in manifest
    assert manifest["paper_001"]["reconstructed"] is True
    assert manifest_path.exists()


def test_partial_backfill_keeps_index_snapshot_on_old_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dimensions_cli: ModuleType,
) -> None:
    store = _write_legacy_index(tmp_path)
    old_profile = build_legacy_dimension_profile()
    store.save_dimension_profile(old_profile.model_dump(mode="json"))
    IndexManifest(extraction={"profile_id": old_profile.profile_id}).save(
        tmp_path / "index_manifest.json"
    )

    profile_dict = old_profile.model_dump(mode="json")
    profile_dict["profile_id"] = "legacy_semantic_methods_v2"
    profile_dict["version"] = "2.0.0"
    for dimension in profile_dict["dimensions"]:
        if dimension["id"] == "methods":
            dimension["question"] = "What updated methods and analytical techniques are used?"
    profile_path = tmp_path / "methods_v2.yaml"
    _write_profile(profile_path, profile_dict)

    target_profile = dimensions_cli.load_dimension_profile(profile_path)

    class DummyExtractor:
        def extract_batch(self, papers, progress_callback=None, section_ids=None):
            for paper in papers:
                yield ExtractionResult(
                    paper_id=paper.paper_id,
                    success=True,
                    extraction=DimensionedExtraction(
                        paper_id=paper.paper_id,
                        profile_id=target_profile.profile_id,
                        profile_version=target_profile.version,
                        profile_fingerprint=target_profile.fingerprint,
                        prompt_version="2.0.0",
                        extraction_model="target-model",
                        extracted_at="2026-03-31T01:00:00",
                        dimensions={
                            "paradigm": "Updated paradigm",
                            "methods": "Updated methods",
                            "data": "Updated data",
                            "reproducibility": "Updated reproducibility",
                            "framework": "Updated framework",
                        },
                    ),
                    model_used="target-model",
                )

    monkeypatch.setattr(
        dimensions_cli,
        "configure_extraction_runtime",
        lambda *args, **kwargs: ("anthropic", "api", tmp_path, 1, False),
    )
    monkeypatch.setattr(
        dimensions_cli,
        "build_section_extractor",
        lambda **kwargs: DummyExtractor(),
    )
    monkeypatch.setattr(dimensions_cli, "generate_scoped_raptor_summaries", lambda **kwargs: {})
    monkeypatch.setattr(dimensions_cli, "run_embedding_generation", lambda **kwargs: None)
    monkeypatch.setattr(dimensions_cli, "compute_similarity_pairs", lambda **kwargs: 0)
    monkeypatch.setattr(dimensions_cli, "generate_summary", lambda *args, **kwargs: {})

    args = argparse.Namespace(
        index_dir=tmp_path,
        dimension_profile=profile_path,
        paper=["paper_001"],
        dry_run=False,
        skip_embeddings=False,
        skip_similarity=False,
        provider=None,
        mode=None,
        model=None,
        parallel=None,
        no_cache=False,
        summary_model=None,
        methodology_model=None,
    )
    logger = SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None)
    config = _make_config(tmp_path)

    assert dimensions_cli.backfill_dimensions(args, config, logger) == 0

    updated = DimensionedExtraction.from_record(
        StructuredStore(tmp_path).load_extractions()["paper_001"]
    )
    manifest, manifest_error = IndexManifest.load(tmp_path / "index_manifest.json")

    assert updated.profile_id == "legacy_semantic_methods_v2"
    assert StructuredStore(tmp_path).load_dimension_profile()["profile_id"] == LEGACY_PROFILE_ID
    assert manifest_error is None
    assert manifest is not None
    assert manifest.extraction["profile_id"] == LEGACY_PROFILE_ID


def test_backfill_does_not_advance_profile_if_embedding_refresh_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dimensions_cli: ModuleType,
) -> None:
    store = _write_legacy_index(tmp_path)
    old_profile = build_legacy_dimension_profile()
    store.save_dimension_profile(old_profile.model_dump(mode="json"))
    IndexManifest(extraction={"profile_id": old_profile.profile_id}).save(
        tmp_path / "index_manifest.json"
    )

    profile_dict = old_profile.model_dump(mode="json")
    profile_dict["profile_id"] = "legacy_semantic_methods_v2"
    profile_dict["version"] = "2.0.0"
    for dimension in profile_dict["dimensions"]:
        if dimension["id"] == "methods":
            dimension["question"] = "What updated methods and analytical techniques are used?"
    profile_path = tmp_path / "methods_v2.yaml"
    _write_profile(profile_path, profile_dict)

    target_profile = dimensions_cli.load_dimension_profile(profile_path)

    class DummyExtractor:
        def extract_batch(self, papers, progress_callback=None, section_ids=None):
            for paper in papers:
                yield ExtractionResult(
                    paper_id=paper.paper_id,
                    success=True,
                    extraction=DimensionedExtraction(
                        paper_id=paper.paper_id,
                        profile_id=target_profile.profile_id,
                        profile_version=target_profile.version,
                        profile_fingerprint=target_profile.fingerprint,
                        prompt_version="2.0.0",
                        extraction_model="target-model",
                        extracted_at="2026-03-31T01:00:00",
                        dimensions={
                            "paradigm": "Updated paradigm",
                            "methods": "Updated methods",
                            "data": "Updated data",
                            "reproducibility": "Updated reproducibility",
                            "framework": "Updated framework",
                        },
                    ),
                    model_used="target-model",
                )

    monkeypatch.setattr(
        dimensions_cli,
        "configure_extraction_runtime",
        lambda *args, **kwargs: ("anthropic", "api", tmp_path, 1, False),
    )
    monkeypatch.setattr(
        dimensions_cli,
        "build_section_extractor",
        lambda **kwargs: DummyExtractor(),
    )
    monkeypatch.setattr(dimensions_cli, "generate_scoped_raptor_summaries", lambda **kwargs: {})
    monkeypatch.setattr(
        dimensions_cli,
        "run_embedding_generation",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("embedding refresh failed")),
    )
    monkeypatch.setattr(dimensions_cli, "compute_similarity_pairs", lambda **kwargs: 0)
    monkeypatch.setattr(dimensions_cli, "generate_summary", lambda *args, **kwargs: {})

    args = argparse.Namespace(
        index_dir=tmp_path,
        dimension_profile=profile_path,
        paper=[],
        dry_run=False,
        skip_embeddings=False,
        skip_similarity=False,
        provider=None,
        mode=None,
        model=None,
        parallel=None,
        no_cache=False,
        summary_model=None,
        methodology_model=None,
    )
    logger = SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None)
    config = _make_config(tmp_path)

    with pytest.raises(RuntimeError, match="embedding refresh failed"):
        dimensions_cli.backfill_dimensions(args, config, logger)

    stored = DimensionedExtraction.from_record(
        StructuredStore(tmp_path).load_extractions()["paper_001"]
    )
    manifest, manifest_error = IndexManifest.load(tmp_path / "index_manifest.json")

    assert stored.profile_id == LEGACY_PROFILE_ID
    assert StructuredStore(tmp_path).load_dimension_profile()["profile_id"] == LEGACY_PROFILE_ID
    assert manifest_error is None
    assert manifest is not None
    assert manifest.extraction["profile_id"] == LEGACY_PROFILE_ID


def test_backfill_aborts_when_any_targeted_section_pass_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dimensions_cli: ModuleType,
) -> None:
    store = _write_legacy_index(tmp_path)
    old_profile = build_legacy_dimension_profile()
    store.save_dimension_profile(old_profile.model_dump(mode="json"))

    profile_dict = old_profile.model_dump(mode="json")
    profile_dict["profile_id"] = "legacy_semantic_methods_v2"
    profile_dict["version"] = "2.0.0"
    for dimension in profile_dict["dimensions"]:
        if dimension["id"] == "methods":
            dimension["question"] = "What updated methods and analytical techniques are used?"
    profile_path = tmp_path / "methods_v2.yaml"
    _write_profile(profile_path, profile_dict)

    target_profile = dimensions_cli.load_dimension_profile(profile_path)

    class DummyExtractor:
        def extract_batch(self, papers, progress_callback=None, section_ids=None):
            for paper in papers:
                yield ExtractionResult(
                    paper_id=paper.paper_id,
                    success=True,
                    extraction=DimensionedExtraction(
                        paper_id=paper.paper_id,
                        profile_id=target_profile.profile_id,
                        profile_version=target_profile.version,
                        profile_fingerprint=target_profile.fingerprint,
                        prompt_version="2.0.0",
                        extraction_model="target-model",
                        extracted_at="2026-03-31T01:00:00",
                        dimensions={
                            "methods": "Updated methods",
                        },
                    ),
                    model_used="target-model",
                    pass_errors=["pass 2 (Pass 2: Methodology): upstream API failure"],
                )

    monkeypatch.setattr(
        dimensions_cli,
        "configure_extraction_runtime",
        lambda *args, **kwargs: ("anthropic", "api", tmp_path, 1, False),
    )
    monkeypatch.setattr(
        dimensions_cli,
        "build_section_extractor",
        lambda **kwargs: DummyExtractor(),
    )
    monkeypatch.setattr(dimensions_cli, "generate_scoped_raptor_summaries", lambda **kwargs: {})
    monkeypatch.setattr(dimensions_cli, "run_embedding_generation", lambda **kwargs: None)
    monkeypatch.setattr(dimensions_cli, "compute_similarity_pairs", lambda **kwargs: 0)
    monkeypatch.setattr(dimensions_cli, "generate_summary", lambda *args, **kwargs: {})

    args = argparse.Namespace(
        index_dir=tmp_path,
        dimension_profile=profile_path,
        paper=[],
        dry_run=False,
        skip_embeddings=False,
        skip_similarity=False,
        provider=None,
        mode=None,
        model=None,
        parallel=None,
        no_cache=False,
        summary_model=None,
        methodology_model=None,
    )
    logger = SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None)
    config = _make_config(tmp_path)

    assert dimensions_cli.backfill_dimensions(args, config, logger) == 1

    stored = DimensionedExtraction.from_record(
        StructuredStore(tmp_path).load_extractions()["paper_001"]
    )
    assert stored.profile_id == LEGACY_PROFILE_ID
    assert stored.dimensions["methods"] == "Legacy methods"


def test_suggest_dimensions_merges_semantic_and_heuristic_proposals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dimensions_cli: ModuleType,
) -> None:
    store = _write_legacy_index(tmp_path)
    papers = store.load_papers()
    papers["paper_001"]["abstract"] = (
        "Stakeholder communities coordinate around risk and governance."
    )
    papers["paper_002"] = _paper_record(tmp_path, "paper_002")
    papers["paper_002"]["abstract"] = (
        "Complex systems research studies adaptation and coordination."
    )
    store.save_papers(papers)
    store.save_extractions(
        {
            "paper_001": {
                "paper_id": "paper_001",
                "extraction": _legacy_extraction_record("paper_001"),
            },
            "paper_002": {
                "paper_id": "paper_002",
                "extraction": {
                    **_legacy_extraction_record("paper_002"),
                    "q17_field": "Complex systems",
                    "q28_disciplines_bridged": "Policy and network science",
                },
            },
        }
    )
    store.save_similarity_pairs(
        {
            "paper_001": [{"similar_paper_id": "paper_002", "similarity_score": 0.71}],
            "paper_002": [{"similar_paper_id": "paper_001", "similarity_score": 0.71}],
        }
    )

    monkeypatch.setattr(
        dimensions_cli,
        "_call_raw_llm_prompt",
        lambda **kwargs: json.dumps(
            {
                "proposals": [
                    {
                        "dimension_id": "coordination_regime",
                        "label": "Coordination Regime",
                        "question": "How does the work characterize coordination across actors or system components?",
                        "rationale": "Coordination logic recurs across the sample but is not isolated as a dimension.",
                        "suggested_section": "impact",
                        "suggested_roles": ["coordination"],
                        "confidence": 0.82,
                        "example_papers": ["paper_001", "paper_002"],
                        "bridge_value": "Links governance and systems papers through coordination structure.",
                    }
                ]
            }
        ),
    )

    args = argparse.Namespace(
        index_dir=tmp_path,
        paper=[],
        sample_size=5,
        min_hits=1,
        max_proposals=5,
        provider="openai",
        mode="api",
        model="gpt-5.4",
        heuristic_only=False,
        output=tmp_path / "dimension_proposals.json",
        config=None,
        verbose=False,
    )
    config = _make_config(tmp_path)

    assert dimensions_cli.suggest_dimensions(args, config) == 0

    payload = safe_read_json(args.output, default={})
    proposal_ids = {proposal["dimension_id"] for proposal in payload["proposals"]}

    assert "coordination_regime" in proposal_ids
    assert "stakeholders" in proposal_ids
    coordination = next(
        proposal
        for proposal in payload["proposals"]
        if proposal["dimension_id"] == "coordination_regime"
    )
    assert coordination["proposal_sources"] == ["semantic_llm"]
    assert coordination["example_papers"][0]["paper_id"] == "paper_001"
    assert payload["semantic_candidate_count"] == 1
    assert payload["heuristic_candidate_count"] >= 1


def test_suggest_dimensions_can_run_heuristic_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dimensions_cli: ModuleType,
) -> None:
    store = _write_legacy_index(tmp_path)
    papers = store.load_papers()
    papers["paper_001"]["abstract"] = "Stakeholder groups face safety risk and coordination costs."
    store.save_papers(papers)

    called = {"llm": False}

    def _unexpected_llm_call(**kwargs):
        called["llm"] = True
        raise AssertionError("heuristic-only mode should not call the LLM")

    monkeypatch.setattr(dimensions_cli, "_call_raw_llm_prompt", _unexpected_llm_call)

    args = argparse.Namespace(
        index_dir=tmp_path,
        paper=[],
        sample_size=3,
        min_hits=1,
        max_proposals=5,
        provider=None,
        mode=None,
        model=None,
        heuristic_only=True,
        output=tmp_path / "dimension_proposals.json",
        config=None,
        verbose=False,
    )
    config = _make_config(tmp_path)

    assert dimensions_cli.suggest_dimensions(args, config) == 0

    payload = safe_read_json(args.output, default={})
    assert called["llm"] is False
    assert payload["semantic_candidate_count"] == 0
    assert payload["proposal_count"] >= 1
