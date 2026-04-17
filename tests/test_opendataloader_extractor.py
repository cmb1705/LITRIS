"""Tests for OpenDataLoader hybrid backend helpers."""

from pathlib import Path
from unittest.mock import patch

from src.config import ManagedHybridServerConfig, ProcessingConfig
from src.extraction import opendataloader_extractor
from src.extraction.opendataloader_extractor import ManagedHybridServerSpec


def test_hybrid_server_executable_prefers_current_interpreter_dir(tmp_path):
    """Prefer the launcher installed in the active virtualenv."""
    scripts_dir = tmp_path / "Scripts"
    scripts_dir.mkdir()
    local_exe = scripts_dir / "opendataloader-pdf-hybrid.exe"
    local_exe.write_text("")

    with (
        patch.object(
            opendataloader_extractor.sys,
            "executable",
            str(scripts_dir / "python.exe"),
        ),
        patch.object(opendataloader_extractor.sys, "prefix", str(tmp_path)),
        patch(
            "src.extraction.opendataloader_extractor.shutil.which",
            return_value=r"C:\global\opendataloader-pdf-hybrid.exe",
        ),
    ):
        assert opendataloader_extractor._hybrid_server_executable() == str(local_exe)


def test_hybrid_server_executable_falls_back_to_path(tmp_path):
    """Use PATH lookup when the active virtualenv has no launcher."""
    with (
        patch.object(
            opendataloader_extractor.sys,
            "executable",
            str(tmp_path / "Scripts" / "python.exe"),
        ),
        patch.object(opendataloader_extractor.sys, "prefix", str(tmp_path)),
        patch(
            "src.extraction.opendataloader_extractor.shutil.which",
            side_effect=[
                r"C:\global\opendataloader-pdf-hybrid.exe",
                None,
            ],
        ),
    ):
        assert opendataloader_extractor._hybrid_server_executable() == (
            r"C:\global\opendataloader-pdf-hybrid.exe"
        )


def test_hybrid_server_executable_for_python_prefers_given_install(tmp_path):
    """Resolve the launcher next to the configured Python install."""
    python_dir = tmp_path / "Python310"
    scripts_dir = python_dir / "Scripts"
    scripts_dir.mkdir(parents=True)
    python_exe = python_dir / "python.exe"
    python_exe.write_text("")
    local_exe = scripts_dir / "opendataloader-pdf-hybrid.exe"
    local_exe.write_text("")

    resolved = opendataloader_extractor.hybrid_server_executable_for_python(
        python_exe,
        allow_path_fallback=False,
    )

    assert resolved == str(local_exe)


def test_resolved_server_url_uses_exact_managed_pool_match():
    """Exact managed profile matches should win over the legacy URL."""
    config = opendataloader_extractor.OpenDataLoaderHybridConfig(
        enabled=True,
        server_url="http://legacy:7000",
        enrich_formula=True,
        enrich_picture_description=True,
        managed_servers=(
            ManagedHybridServerSpec(name="base", url="http://127.0.0.1:5002"),
            ManagedHybridServerSpec(
                name="formula_picture",
                url="http://127.0.0.1:5008",
                enrich_formula=True,
                enrich_picture_description=True,
            ),
        ),
    )

    assert config.resolved_server_url() == "http://127.0.0.1:5008"


def test_resolved_server_url_falls_back_to_legacy_url_when_pool_has_no_match():
    """Legacy single-server settings remain valid when the pool has no match."""
    config = opendataloader_extractor.OpenDataLoaderHybridConfig(
        enabled=True,
        server_url="legacy-host:7000",
        force_ocr=True,
        managed_servers=(ManagedHybridServerSpec(name="base", url="http://127.0.0.1:5002"),),
    )

    assert config.resolved_server_url() == "http://legacy-host:7000"


def test_build_hybrid_config_from_processing_uses_managed_pool_defaults():
    """ProcessingConfig should build the managed fixed-pool hybrid config."""
    processing = ProcessingConfig(opendataloader_hybrid_enabled=True)

    config = opendataloader_extractor.build_hybrid_config_from_processing(processing)

    assert config is not None
    assert config.device == "cuda"
    assert config.python_executable == processing.opendataloader_hybrid_python_executable
    assert len(config.managed_servers) == 8
    assert config.resolved_server_url() == "http://127.0.0.1:5002"


def test_build_hybrid_config_from_processing_preserves_legacy_single_url():
    """Explicit legacy single-server URLs should still be respected."""
    processing = ProcessingConfig(
        opendataloader_hybrid_enabled=True,
        opendataloader_hybrid_url="legacy-host:7000",
        opendataloader_hybrid_servers={},
    )

    config = opendataloader_extractor.build_hybrid_config_from_processing(processing)

    assert config is not None
    assert config.resolved_server_url() == "http://legacy-host:7000"


def test_ensure_hybrid_server_health_checks_managed_pool_without_autostart():
    """Managed pool mode should never try to spawn a local server."""
    config = opendataloader_extractor.OpenDataLoaderHybridConfig(
        enabled=True,
        force_ocr=True,
        managed_servers=(
            ManagedHybridServerSpec(
                name="ocr",
                url="http://127.0.0.1:5003",
                force_ocr=True,
            ),
        ),
    )

    with (
        patch(
            "src.extraction.opendataloader_extractor.is_hybrid_server_reachable",
            return_value=False,
        ),
        patch(
            "src.extraction.opendataloader_extractor._hybrid_server_executable"
        ) as mock_executable,
    ):
        assert opendataloader_extractor.ensure_hybrid_server(config) is None
        mock_executable.assert_not_called()


def test_build_managed_server_specs_from_processing_models():
    """Managed server configs from ProcessingConfig should become exact-match specs."""
    processing = ProcessingConfig(
        opendataloader_hybrid_servers={
            "ocr_formula": ManagedHybridServerConfig(
                url="127.0.0.1:5006",
                force_ocr=True,
                enrich_formula=True,
            )
        }
    )

    specs = opendataloader_extractor.build_managed_server_specs(
        processing.opendataloader_hybrid_servers
    )

    assert specs == (
        ManagedHybridServerSpec(
            name="ocr_formula",
            url="http://127.0.0.1:5006",
            force_ocr=True,
            enrich_formula=True,
            enrich_picture_description=False,
        ),
    )


def test_extract_with_opendataloader_logs_resolved_hybrid_endpoint(tmp_path):
    """Hybrid extraction logs should identify the managed pool endpoint and flags."""

    class DummyOpenDataLoaderModule:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] | None = None

        def convert(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            output_dir = Path(str(kwargs["output_dir"]))
            (output_dir / "paper.md").write_text("word " * 150, encoding="utf-8")

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    java_path = tmp_path / "java.exe"
    java_path.write_text("", encoding="utf-8")
    dummy_module = DummyOpenDataLoaderModule()
    hybrid_config = opendataloader_extractor.OpenDataLoaderHybridConfig(
        enabled=True,
        backend="docling-fast",
        force_ocr=True,
        enrich_formula=True,
        enrich_picture_description=True,
        fallback_to_fast=True,
        timeout_ms=2500,
        managed_servers=(
            ManagedHybridServerSpec(name="base", url="http://127.0.0.1:5002"),
            ManagedHybridServerSpec(
                name="ocr_formula_picture",
                url="http://127.0.0.1:5009",
                force_ocr=True,
                enrich_formula=True,
                enrich_picture_description=True,
            ),
        ),
    )

    with (
        patch.object(opendataloader_extractor, "OPENDATALOADER_AVAILABLE", True),
        patch.object(opendataloader_extractor, "_JAVA_PATH", str(java_path)),
        patch.object(opendataloader_extractor, "opendataloader_pdf", dummy_module),
        patch.object(opendataloader_extractor.logger, "info") as mock_info,
    ):
        text = opendataloader_extractor.extract_with_opendataloader(
            pdf_path,
            mode="hybrid",
            hybrid_config=hybrid_config,
        )

    assert text is not None
    assert dummy_module.kwargs is not None
    assert dummy_module.kwargs["hybrid_url"] == "http://127.0.0.1:5009"

    assert any(
        call.args
        and call.args[0].startswith("OpenDataLoader extraction (hybrid): %s | endpoint=%s")
        and call.args[1:] == (
            pdf_path.name,
            "managed:ocr_formula_picture",
            "http://127.0.0.1:5009",
            "docling-fast",
            "full",
            True,
            True,
            True,
            True,
            2500,
        )
        for call in mock_info.call_args_list
    )
