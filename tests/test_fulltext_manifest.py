"""Tests for full-text manifest persistence and concurrency safety.

These tests focus on the interaction between multiple ``StructuredStore``
instances writing to the same on-disk manifest, which is the pattern used
by the orchestrator (long-lived ``self.store``) and the extraction
pipeline subroutine (short-lived local store created inside
``run_extraction``).
"""

from __future__ import annotations

from pathlib import Path

from src.indexing.structured_store import StructuredStore
from src.utils.file_utils import safe_read_json


def _read_disk_snapshots(index_dir: Path) -> dict[str, dict]:
    data = safe_read_json(index_dir / "fulltext_manifest.json", default={})
    if isinstance(data, dict) and "snapshots" in data:
        return data["snapshots"]
    return data if isinstance(data, dict) else {}


def test_flush_preserves_entries_written_by_concurrent_store(tmp_path: Path) -> None:
    """Regression: orchestrator flush must not clobber pipeline writes.

    Reproduces the bug where ``IndexOrchestrator.self.store`` loads the
    manifest into cache (via ``load_reusable_text_snapshots`` calling
    ``load_text_snapshot``), the extraction pipeline then writes new
    snapshots through its own local ``StructuredStore`` instance, and the
    orchestrator finally calls ``flush_fulltext_manifest``. Before the
    fix, the stale orchestrator cache overwrote the fresh on-disk
    manifest, dropping the pipeline's new entries.
    """

    # 1. Pre-existing entry persisted by some prior run.
    seed_store = StructuredStore(tmp_path)
    seed_store.save_text_snapshot("paper_old", "old paper text")

    # 2. Long-lived store (orchestrator's self.store) loads the manifest
    #    into cache via load_text_snapshot. After this, the cache holds
    #    only {paper_old} and the cached mtime matches the on-disk file.
    long_lived = StructuredStore(tmp_path)
    long_lived.load_text_snapshot("paper_old")
    assert long_lived._fulltext_manifest_cache is not None
    assert "paper_old" in long_lived._fulltext_manifest_cache

    # 3. Pipeline's transient store writes a new snapshot. This persists
    #    the manifest, so the on-disk file now has both entries.
    transient = StructuredStore(tmp_path)
    transient.save_text_snapshot("paper_new", "freshly extracted text")

    on_disk_after_pipeline = _read_disk_snapshots(tmp_path)
    assert "paper_old" in on_disk_after_pipeline
    assert "paper_new" in on_disk_after_pipeline

    # 4. Orchestrator flushes its (stale) cache. Without the fix this
    #    overwrites the manifest with the pre-pipeline state, dropping
    #    paper_new.
    long_lived.flush_fulltext_manifest()

    on_disk_after_flush = _read_disk_snapshots(tmp_path)
    assert "paper_old" in on_disk_after_flush, "Pre-existing entry must survive"
    assert "paper_new" in on_disk_after_flush, (
        "Pipeline-written entry must not be clobbered by stale orchestrator cache"
    )


def test_flush_with_no_external_writes_is_idempotent(tmp_path: Path) -> None:
    """Sanity check: flush after local-only writes still persists the cache."""

    store = StructuredStore(tmp_path)
    store.save_text_snapshot("paper_a", "alpha")
    store.save_text_snapshot("paper_b", "beta")

    # Force a flush against an in-sync cache.
    store.flush_fulltext_manifest()

    on_disk = _read_disk_snapshots(tmp_path)
    assert set(on_disk) == {"paper_a", "paper_b"}
