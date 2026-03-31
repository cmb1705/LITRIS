"""Tests for corpus-wide gap-fill provider selection."""

import importlib.util
from pathlib import Path


def _load_run_gap_fill_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_gap_fill.py"
    spec = importlib.util.spec_from_file_location("run_gap_fill_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_gap_fill = _load_run_gap_fill_module()


def test_infer_provider_from_model_detects_openai():
    """Known GPT models map back to OpenAI."""
    assert run_gap_fill.infer_provider_from_model("gpt-5.4") == "openai"


def test_infer_provider_from_model_detects_anthropic():
    """Known Claude models map back to Anthropic."""
    assert run_gap_fill.infer_provider_from_model("claude-opus-4-6") == "anthropic"


def test_resolve_gap_fill_provider_uses_opposite_for_openai_origin():
    """OpenAI-origin extractions default to Anthropic gap-fill."""
    entry = {"extraction": {"extraction_model": "gpt-5.4"}}
    assert run_gap_fill.resolve_gap_fill_provider(entry) == "anthropic"


def test_resolve_gap_fill_provider_uses_opposite_for_anthropic_origin():
    """Anthropic-origin extractions default to OpenAI gap-fill."""
    entry = {"extraction": {"extraction_model": "claude-opus-4-6"}}
    assert run_gap_fill.resolve_gap_fill_provider(entry) == "openai"


def test_resolve_gap_fill_provider_prefers_explicit_override():
    """CLI override wins over inferred opposite-provider behavior."""
    entry = {"extraction": {"extraction_model": "gpt-5.4"}}
    assert run_gap_fill.resolve_gap_fill_provider(entry, "anthropic") == "anthropic"
    assert run_gap_fill.resolve_gap_fill_provider(entry, "openai") == "openai"


def test_resolve_gap_fill_provider_falls_back_to_prior_gap_fill_provider():
    """Unknown original models can still invert a previously recorded gap-fill provider."""
    entry = {
        "extraction": {"extraction_model": "unknown-model"},
        "gap_filled_by": "openai",
    }
    assert run_gap_fill.resolve_gap_fill_provider(entry) == "anthropic"
