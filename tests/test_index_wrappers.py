"""Tests for script wrappers that forward to the unified orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path

import scripts.rebuild_raptor_similarity as rebuild_script
import scripts.update_index as update_script


def test_update_index_wrapper_forwards_to_orchestrator(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("version: 1.0.0\n", encoding="utf-8")

    captured = {}

    class DummyOrchestrator:
        def __init__(self, project_root, logger):
            captured["project_root"] = project_root

        def run(self, args, config):
            captured["sync_mode"] = args.sync_mode
            captured["dry_run"] = args.dry_run
            captured["detect_only"] = args.detect_only
            captured["new_only"] = args.new_only
            captured["delete_only"] = args.delete_only
            return 0

    monkeypatch.setattr(update_script, "IndexOrchestrator", DummyOrchestrator)
    monkeypatch.setattr(update_script.Config, "load", classmethod(lambda cls, path: object()))
    monkeypatch.setattr(
        update_script,
        "setup_logging",
        lambda level="INFO": logging.getLogger("update-wrapper-test"),
    )
    monkeypatch.setattr(
        update_script,
        "parse_args",
        lambda: update_script.argparse.Namespace(
            config=Path(config_path),
            detect_only=True,
            new_only=True,
            delete_only=False,
            skip_extraction=False,
            skip_embeddings=False,
            mode=None,
            summary_model=None,
            methodology_model=None,
            limit=None,
            verbose=False,
        ),
    )

    result = update_script.main()

    assert result == 0
    assert captured["sync_mode"] == "update"
    assert captured["dry_run"] is True
    assert captured["detect_only"] is True
    assert captured["new_only"] is True


def test_rebuild_raptor_similarity_wrapper_forwards_to_orchestrator(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("version: 1.0.0\n", encoding="utf-8")

    captured = {}

    class DummyOrchestrator:
        def __init__(self, project_root, logger):
            captured["project_root"] = project_root

        def refresh_derived_artifacts(
            self,
            config,
            mode,
            provider,
            model,
            force,
            skip_similarity,
            dry_run,
        ):
            captured["mode"] = mode
            captured["provider"] = provider
            captured["model"] = model
            captured["force"] = force
            captured["skip_similarity"] = skip_similarity
            captured["dry_run"] = dry_run
            return 0

    monkeypatch.setattr(rebuild_script, "IndexOrchestrator", DummyOrchestrator)
    monkeypatch.setattr(rebuild_script.Config, "load", classmethod(lambda cls, path: object()))
    monkeypatch.setattr(
        rebuild_script,
        "setup_logging",
        lambda level="INFO": logging.getLogger("raptor-wrapper-test"),
    )
    monkeypatch.setattr(
        rebuild_script,
        "parse_args",
        lambda: rebuild_script.argparse.Namespace(
            mode="llm",
            provider="google",
            model="gemini-test",
            config=Path(config_path),
            force=True,
            skip_similarity=True,
            dry_run=True,
            verbose=False,
        ),
    )

    result = rebuild_script.main()

    assert result == 0
    assert captured["mode"] == "llm"
    assert captured["provider"] == "google"
    assert captured["model"] == "gemini-test"
    assert captured["force"] is True
    assert captured["skip_similarity"] is True
    assert captured["dry_run"] is True
