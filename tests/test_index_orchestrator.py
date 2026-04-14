"""Tests for the unified index orchestrator decision logic."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace

from src.config import Config
from src.indexing.orchestrator import (
    CHUNK_SCHEMA_VERSION,
    RAPTOR_SCHEMA_VERSION,
    SIMILARITY_SCHEMA_VERSION,
    ChangeSet,
    IndexManifest,
    IndexOrchestrator,
    PaperSnapshot,
    PendingStageWork,
    detect_snapshot_changes,
)


def _make_config(tmp_path: Path) -> Config:
    zotero_db = tmp_path / "zotero.sqlite"
    zotero_db.write_text("", encoding="utf-8")
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    return Config(
        zotero={"database_path": zotero_db, "storage_path": storage_dir},
        extraction={"provider": "anthropic", "mode": "api", "model": "claude-test"},
        embeddings={
            "model": "embed-test",
            "dimension": 384,
            "backend": "sentence-transformers",
        },
        storage={"chroma_path": tmp_path / "chroma", "cache_path": tmp_path / "cache"},
    )


def _make_args(**overrides) -> argparse.Namespace:
    defaults = {
        "sync_mode": "auto",
        "rebuild_embeddings": False,
        "paper": [],
        "collection": None,
        "index_all": False,
        "provider": None,
        "mode": None,
        "model": None,
        "summary_model": None,
        "methodology_model": None,
        "new_only": False,
        "delete_only": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_snapshot(
    paper_id: str,
    zotero_key: str,
    fingerprint: str,
    extractable: bool | None = True,
    indexed: bool | None = True,
) -> PaperSnapshot:
    return PaperSnapshot(
        paper_id=paper_id,
        zotero_key=zotero_key,
        pdf_attachment_key=paper_id.split("_", 1)[1] if "_" in paper_id else None,
        date_modified="2026-03-30T00:00:00",
        pdf_path=f"/tmp/{paper_id}.pdf",
        pdf_size=100,
        pdf_mtime_ns=123,
        source_path=f"/tmp/{paper_id}.pdf",
        source_size=100,
        source_mtime_ns=123,
        source_media_type="application/pdf",
        source_fingerprint=fingerprint,
        extractable=extractable,
        indexed=indexed,
    )


def _make_ref_db(provider: str, source_path: Path) -> SimpleNamespace:
    return SimpleNamespace(provider=provider, source_path=source_path)


def _save_matching_manifest(
    orchestrator: IndexOrchestrator,
    args: argparse.Namespace,
    config: Config,
    ref_db: SimpleNamespace,
    snapshots: dict[str, PaperSnapshot],
) -> None:
    manifest = IndexManifest(
        source_scope=orchestrator._source_scope_info(ref_db, args),
        classification_policy=orchestrator._classification_policy_info(args, config),
        extraction=orchestrator._extraction_info(args, config),
        embedding=orchestrator._embedding_info(config),
        chunk_schema_version=CHUNK_SCHEMA_VERSION,
        raptor_schema_version=RAPTOR_SCHEMA_VERSION,
        similarity_schema_version=SIMILARITY_SCHEMA_VERSION,
        pending_work={
            "extraction": PendingStageWork(),
            "embeddings": PendingStageWork(),
            "raptor": PendingStageWork(),
            "similarity": PendingStageWork(),
        },
        paper_snapshots=snapshots,
    )
    manifest.save(orchestrator.manifest_path)


def test_detect_snapshot_changes_is_attachment_aware():
    previous = {
        "ITEM_att1": _make_snapshot("ITEM_att1", "ITEM", "old-1"),
        "ITEM_att2": _make_snapshot("ITEM_att2", "ITEM", "old-2"),
    }
    current = {
        "ITEM_att2": _make_snapshot("ITEM_att2", "ITEM", "new-2"),
        "ITEM_att3": _make_snapshot("ITEM_att3", "ITEM", "new-3"),
    }

    changes = detect_snapshot_changes(previous, current)

    assert changes.new_items == ["ITEM_att3"]
    assert changes.modified_items == ["ITEM_att2"]
    assert changes.deleted_items == ["ITEM_att1"]
    assert changes.unchanged_items == []


def test_plan_sync_missing_manifest_auto_chooses_full(tmp_path):
    config = _make_config(tmp_path)
    orchestrator = IndexOrchestrator(project_root=tmp_path, logger=logging.getLogger("test"))
    args = _make_args(sync_mode="auto")
    ref_db = _make_ref_db("zotero", config.get_zotero_db_path())
    current = {"P1": _make_snapshot("P1", "Z1", "fp-1")}

    plan = orchestrator.plan_sync(
        args=args,
        config=config,
        ref_db=ref_db,
        current_snapshots=current,
        existing_papers={},
        existing_extractions={},
        class_index=SimpleNamespace(papers={}),
    )

    assert plan.resolved_mode == "full"
    assert any("manifest is missing" in reason.lower() for reason in plan.reasons)


def test_plan_sync_missing_manifest_forced_update_is_invalid(tmp_path):
    config = _make_config(tmp_path)
    orchestrator = IndexOrchestrator(project_root=tmp_path, logger=logging.getLogger("test"))
    args = _make_args(sync_mode="update")
    ref_db = _make_ref_db("zotero", config.get_zotero_db_path())
    current = {"P1": _make_snapshot("P1", "Z1", "fp-1")}

    plan = orchestrator.plan_sync(
        args=args,
        config=config,
        ref_db=ref_db,
        current_snapshots=current,
        existing_papers={},
        existing_extractions={},
        class_index=SimpleNamespace(papers={}),
    )

    assert plan.resolved_mode == "invalid"
    assert any("manifest is missing" in reason.lower() for reason in plan.reasons)


def test_plan_sync_non_zotero_auto_forces_full(tmp_path):
    config = _make_config(tmp_path)
    orchestrator = IndexOrchestrator(project_root=tmp_path, logger=logging.getLogger("test"))
    args = _make_args(sync_mode="auto")
    ref_db = _make_ref_db("bibtex", tmp_path / "refs.bib")
    current = {"P1": _make_snapshot("P1", "Z1", "fp-1")}

    plan = orchestrator.plan_sync(
        args=args,
        config=config,
        ref_db=ref_db,
        current_snapshots=current,
        existing_papers={},
        existing_extractions={},
        class_index=SimpleNamespace(papers={}),
    )

    assert plan.resolved_mode == "full"
    assert any("non-zotero" in reason.lower() for reason in plan.reasons)


def test_plan_sync_embedding_fingerprint_change_forces_full(tmp_path):
    config = _make_config(tmp_path)
    orchestrator = IndexOrchestrator(project_root=tmp_path, logger=logging.getLogger("test"))
    args = _make_args(sync_mode="auto")
    ref_db = _make_ref_db("zotero", config.get_zotero_db_path())
    snapshots = {"P1": _make_snapshot("P1", "Z1", "fp-1")}
    _save_matching_manifest(orchestrator, args, config, ref_db, snapshots)

    manifest, _ = IndexManifest.load(orchestrator.manifest_path)
    assert manifest is not None
    manifest.embedding["model"] = "embed-old"
    manifest.embedding["fingerprint"] = "stale-fingerprint"
    manifest.save(orchestrator.manifest_path)

    plan = orchestrator.plan_sync(
        args=args,
        config=config,
        ref_db=ref_db,
        current_snapshots=snapshots,
        existing_papers={"P1": {"paper_id": "P1"}},
        existing_extractions={"P1": {"paper_id": "P1", "extraction": {}}},
        class_index=SimpleNamespace(papers={}),
    )

    assert plan.resolved_mode == "full"
    assert any("embedding configuration changed" in reason.lower() for reason in plan.reasons)


def test_validate_chunk_types_pages_chroma_metadata(monkeypatch, tmp_path):
    orchestrator = IndexOrchestrator(project_root=tmp_path, logger=logging.getLogger("test"))
    calls: list[tuple[int | None, int | None]] = []

    class FakeCollection:
        def get(self, include, limit=None, offset=None):
            del include
            calls.append((limit, offset))
            pages = {
                0: [{"chunk_type": "abstract"}, {"chunk_type": "dim_q01"}],
                2: [{"chunk_type": "raptor_overview"}],
                3: [],
            }
            return {"metadatas": pages.get(offset, [])}

    class FakeVectorStore:
        def __init__(self, _path):
            self.collection = FakeCollection()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("src.indexing.orchestrator.VectorStore", FakeVectorStore)

    assert orchestrator._validate_chunk_types() is None
    assert calls == [(500, 0), (500, 2), (500, 3)]


def test_plan_sync_extraction_drift_stays_incremental_in_auto_mode(tmp_path):
    config = _make_config(tmp_path)
    orchestrator = IndexOrchestrator(project_root=tmp_path, logger=logging.getLogger("test"))
    ref_db = _make_ref_db("zotero", config.get_zotero_db_path())
    snapshots = {"P1": _make_snapshot("P1", "Z1", "fp-1")}
    _save_matching_manifest(orchestrator, _make_args(sync_mode="auto"), config, ref_db, snapshots)

    plan = orchestrator.plan_sync(
        args=_make_args(
            sync_mode="auto",
            provider="openai",
            mode="cli",
            model="gpt-5.4",
        ),
        config=config,
        ref_db=ref_db,
        current_snapshots=snapshots,
        existing_papers={"P1": {"paper_id": "P1"}},
        existing_extractions={"P1": {"paper_id": "P1", "extraction": {}}},
        class_index=SimpleNamespace(papers={}),
    )

    assert plan.resolved_mode == "update"
    assert plan.noop is True
    assert not any("extraction configuration changed" in reason.lower() for reason in plan.reasons)
    assert len(plan.advisories) == 1
    assert "baseline: anthropic/api/claude-test" in plan.advisories[0]
    assert "requested: openai/cli/gpt-5.4" in plan.advisories[0]


def test_plan_sync_update_mode_allows_extraction_drift(tmp_path):
    config = _make_config(tmp_path)
    orchestrator = IndexOrchestrator(project_root=tmp_path, logger=logging.getLogger("test"))
    ref_db = _make_ref_db("zotero", config.get_zotero_db_path())
    snapshots = {"P1": _make_snapshot("P1", "Z1", "fp-1")}
    _save_matching_manifest(orchestrator, _make_args(sync_mode="auto"), config, ref_db, snapshots)

    plan = orchestrator.plan_sync(
        args=_make_args(
            sync_mode="update",
            provider="openai",
            mode="cli",
            model="gpt-5.4",
        ),
        config=config,
        ref_db=ref_db,
        current_snapshots=snapshots,
        existing_papers={"P1": {"paper_id": "P1"}},
        existing_extractions={"P1": {"paper_id": "P1", "extraction": {}}},
        class_index=SimpleNamespace(papers={}),
    )

    assert plan.resolved_mode == "update"
    assert plan.noop is True
    assert plan.advisories


def test_plan_sync_no_changes_with_matching_manifest_is_noop(tmp_path):
    config = _make_config(tmp_path)
    orchestrator = IndexOrchestrator(project_root=tmp_path, logger=logging.getLogger("test"))
    args = _make_args(sync_mode="auto")
    ref_db = _make_ref_db("zotero", config.get_zotero_db_path())
    snapshots = {"P1": _make_snapshot("P1", "Z1", "fp-1")}
    _save_matching_manifest(orchestrator, args, config, ref_db, snapshots)

    plan = orchestrator.plan_sync(
        args=args,
        config=config,
        ref_db=ref_db,
        current_snapshots=snapshots,
        existing_papers={"P1": {"paper_id": "P1"}},
        existing_extractions={},
        class_index=SimpleNamespace(papers={}),
    )

    assert plan.resolved_mode == "update"
    assert plan.noop is True


def test_describe_extraction_requirements_reports_missing_full_rebuild_items(tmp_path):
    _make_config(tmp_path)
    orchestrator = IndexOrchestrator(project_root=tmp_path, logger=logging.getLogger("test"))
    args = _make_args(sync_mode="full")
    plan = SimpleNamespace(
        resolved_mode="full",
        requires_full_extraction=False,
        forced_paper_ids=[],
        change_set=ChangeSet(),
        pending_work={
            "extraction": PendingStageWork(),
        },
    )
    desired_index_ids = {"P1"}
    current_papers_by_id = {
        "P1": SimpleNamespace(
            paper_id="P1",
            title="Missing Extraction Paper",
            zotero_key="Z1",
        )
    }

    requirements = orchestrator._describe_extraction_requirements(
        args=args,
        plan=plan,
        desired_index_ids=desired_index_ids,
        existing_extractions={},
        current_papers_by_id=current_papers_by_id,
    )

    assert requirements == [
        {
            "paper_id": "P1",
            "title": "Missing Extraction Paper",
            "zotero_key": "Z1",
            "reason": "missing stored extraction for full rebuild",
        }
    ]
    assert orchestrator._format_extraction_requirement_lines(requirements) == [
        "P1: Missing Extraction Paper (zotero_key=Z1) [missing stored extraction for full rebuild]"
    ]


def test_targeted_full_refresh_scopes_only_forced_paper_when_manifest_missing(tmp_path):
    config = _make_config(tmp_path)
    orchestrator = IndexOrchestrator(project_root=tmp_path, logger=logging.getLogger("test"))
    args = _make_args(sync_mode="full", paper=["P1"])
    plan = SimpleNamespace(
        resolved_mode="full",
        full_rebuild=True,
        requires_full_extraction=False,
        forced_paper_ids=["P1"],
        change_set=ChangeSet(new_items=["P1", "P2"], modified_items=[], deleted_items=[]),
        pending_work={
            "extraction": PendingStageWork(),
            "embeddings": PendingStageWork(),
        },
    )
    desired_index_ids = {"P1", "P2"}
    current_papers_by_id = {
        "P1": SimpleNamespace(paper_id="P1", title="Paper 1", zotero_key="Z1"),
        "P2": SimpleNamespace(paper_id="P2", title="Paper 2", zotero_key="Z2"),
    }

    assert orchestrator._classification_scope_ids(
        args=args,
        plan=plan,
        current_papers_by_id=current_papers_by_id,
        class_index=SimpleNamespace(papers={}),
        config=config,
    ) == ["P1"]
    assert orchestrator._scoped_current_ids(
        args=args,
        plan=plan,
        desired_index_ids=desired_index_ids,
    ) == {"P1"}


def test_resolved_manifest_snapshots_omits_unprocessed_new_papers(tmp_path):
    _make_config(tmp_path)
    orchestrator = IndexOrchestrator(project_root=tmp_path, logger=logging.getLogger("test"))
    previous = {"P1": _make_snapshot("P1", "Z1", "fp-1")}
    current = {
        "P1": _make_snapshot("P1", "Z1", "fp-1"),
        "P2": _make_snapshot("P2", "Z2", "fp-2"),
    }
    class_index = SimpleNamespace(
        papers={
            "P1": SimpleNamespace(extractable=True),
            "P2": SimpleNamespace(extractable=True),
        }
    )

    resolved = orchestrator._resolved_manifest_snapshots(
        previous_snapshots=previous,
        current_snapshots=current,
        class_index=class_index,
        indexed_ids={"P1"},
        preserved_current_ids={"P2"},
    )

    assert sorted(resolved) == ["P1"]
    assert resolved["P1"].indexed is True


def test_resolved_manifest_snapshots_preserves_previous_pending_snapshot(tmp_path):
    _make_config(tmp_path)
    orchestrator = IndexOrchestrator(project_root=tmp_path, logger=logging.getLogger("test"))
    previous = {"P1": _make_snapshot("P1", "Z1", "old-fp")}
    current = {"P1": _make_snapshot("P1", "Z1", "new-fp")}
    class_index = SimpleNamespace(papers={"P1": SimpleNamespace(extractable=True)})

    resolved = orchestrator._resolved_manifest_snapshots(
        previous_snapshots=previous,
        current_snapshots=current,
        class_index=class_index,
        indexed_ids=set(),
        preserved_current_ids={"P1"},
    )

    assert resolved["P1"].source_fingerprint == "old-fp"
    assert resolved["P1"].indexed is True


def test_plan_sync_pending_extraction_work_prevents_noop(tmp_path):
    config = _make_config(tmp_path)
    orchestrator = IndexOrchestrator(project_root=tmp_path, logger=logging.getLogger("test"))
    args = _make_args(sync_mode="auto")
    ref_db = _make_ref_db("zotero", config.get_zotero_db_path())
    current = {
        "P1": _make_snapshot("P1", "Z1", "fp-1"),
        "P2": _make_snapshot("P2", "Z2", "fp-2"),
    }

    manifest = IndexManifest(
        source_scope=orchestrator._source_scope_info(ref_db, args),
        classification_policy=orchestrator._classification_policy_info(args, config),
        extraction=orchestrator._extraction_info(args, config),
        embedding=orchestrator._embedding_info(config),
        chunk_schema_version=CHUNK_SCHEMA_VERSION,
        raptor_schema_version=RAPTOR_SCHEMA_VERSION,
        similarity_schema_version=SIMILARITY_SCHEMA_VERSION,
        pending_work={
            "extraction": PendingStageWork(paper_ids=["P2"]),
            "embeddings": PendingStageWork(),
            "raptor": PendingStageWork(),
            "similarity": PendingStageWork(),
        },
        paper_snapshots={"P1": current["P1"]},
    )
    manifest.save(orchestrator.manifest_path)

    plan = orchestrator.plan_sync(
        args=args,
        config=config,
        ref_db=ref_db,
        current_snapshots=current,
        existing_papers={"P1": {"paper_id": "P1"}},
        existing_extractions={"P1": {"paper_id": "P1", "extraction": {}}},
        class_index=SimpleNamespace(papers={}),
    )

    assert plan.resolved_mode == "update"
    assert plan.noop is False
    assert plan.pending_work["extraction"].paper_ids == ["P2"]
    assert orchestrator._scoped_current_ids(
        args=args,
        plan=plan,
        desired_index_ids={"P1", "P2"},
    ) == {"P2"}
    assert orchestrator._extraction_required_ids(
        args=args,
        plan=plan,
        desired_index_ids={"P1", "P2"},
        existing_extractions={"P1": {"paper_id": "P1", "extraction": {}}},
    ) == {"P2"}
    assert orchestrator._embedding_scope_ids(
        args=args,
        plan=plan,
        final_papers={"P1": {"paper_id": "P1"}, "P2": {"paper_id": "P2"}},
        final_extractions={"P1": {"paper_id": "P1"}, "P2": {"paper_id": "P2"}},
        failed_extraction_ids=set(),
    ) == {"P2"}
