"""Integration tests for portable dimension migration and backfill flows."""

from __future__ import annotations

import argparse
import importlib.util
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
