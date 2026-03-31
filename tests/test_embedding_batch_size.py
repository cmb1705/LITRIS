"""Tests for configurable embedding batch sizes and Ollama auto-probing."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import scripts.build_index as build_index
from src.config import EmbeddingsConfig, parse_embedding_batch_size_setting
from src.indexing.embeddings import EmbeddingChunk, EmbeddingGenerator
from src.indexing.orchestrator import IndexOrchestrator
from src.indexing.pipeline import run_embedding_generation


def test_parse_embedding_batch_size_setting_accepts_auto_and_positive_ints() -> None:
    assert parse_embedding_batch_size_setting("auto") == "auto"
    assert parse_embedding_batch_size_setting("16") == 16
    assert parse_embedding_batch_size_setting(24) == 24

    with pytest.raises(ValueError, match="positive integer or 'auto'"):
        parse_embedding_batch_size_setting("zero")

    with pytest.raises(ValueError, match="at least 1"):
        parse_embedding_batch_size_setting(0)


def test_embeddings_config_parses_batch_size_from_yaml_style_values() -> None:
    assert EmbeddingsConfig(batch_size="auto").batch_size == "auto"
    assert EmbeddingsConfig(batch_size="12").batch_size == 12


def test_build_parse_args_accepts_embedding_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_index.py", "--embedding-batch-size", "auto"],
    )
    args = build_index.parse_args()
    assert args.embedding_batch_size == "auto"

    monkeypatch.setattr(
        sys,
        "argv",
        ["build_index.py", "--embedding-batch-size", "28"],
    )
    args = build_index.parse_args()
    assert args.embedding_batch_size == 28


def test_embedding_info_ignores_batch_size_for_fingerprint(tmp_path) -> None:
    config_auto = SimpleNamespace(
        embeddings=SimpleNamespace(
            model="embed-test",
            dimension=384,
            backend="ollama",
            query_prefix=None,
            document_prefix=None,
            batch_size="auto",
        )
    )
    config_fixed = SimpleNamespace(
        embeddings=SimpleNamespace(
            model="embed-test",
            dimension=384,
            backend="ollama",
            query_prefix=None,
            document_prefix=None,
            batch_size=32,
        )
    )
    orchestrator = IndexOrchestrator(project_root=tmp_path, logger=SimpleNamespace())

    auto_info = orchestrator._embedding_info(config_auto)
    fixed_info = orchestrator._embedding_info(config_fixed)

    assert auto_info["fingerprint"] == fixed_info["fingerprint"]
    assert auto_info["batch_size"] == "auto"
    assert fixed_info["batch_size"] == 32


def test_ollama_auto_batch_probe_balances_throughput_and_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_init_ollama(self, base_url: str) -> None:
        self._ollama_client = object()
        self.model = None
        self.embedding_dim = 4096

    monkeypatch.setattr(EmbeddingGenerator, "_init_ollama", fake_init_ollama)

    generator = EmbeddingGenerator(model_name="embed-test", backend="ollama")
    probe_calls: list[int] = []
    probe_results = {
        8: (True, 1.5),
        16: (True, 2.2),
        32: (True, 4.0),
        64: (True, 8.2),
        128: (True, 18.0),
        256: (True, 34.0),
        384: (True, 52.0),
        512: (True, 80.0),
    }

    def fake_probe_timed(self, texts: list[str]):
        batch_size = len(texts)
        probe_calls.append(batch_size)
        success, duration = probe_results[batch_size]
        from src.indexing.embeddings import OllamaProbeResult

        return OllamaProbeResult(
            batch_size=batch_size,
            duration_seconds=duration,
            success=success,
        )

    monkeypatch.setattr(EmbeddingGenerator, "_probe_ollama_batch_timed", fake_probe_timed)

    texts = [f"text {i} " + ("x" * i) for i in range(1, 1025)]
    resolved = generator.resolve_batch_size("auto", texts=texts)

    assert resolved == 384
    assert 8 in probe_calls
    assert 256 in probe_calls
    assert 384 in probe_calls
    assert 512 in probe_calls


def test_non_ollama_auto_batch_size_uses_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_init_sentence_transformers(self) -> None:
        self.model = object()
        self.embedding_dim = 384

    monkeypatch.setattr(
        EmbeddingGenerator,
        "_init_sentence_transformers",
        fake_init_sentence_transformers,
    )

    generator = EmbeddingGenerator(model_name="embed-test", backend="sentence-transformers")
    assert generator.resolve_batch_size("auto", texts=["a", "b"]) == 32


def test_run_embedding_generation_uses_resolved_batch_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured: dict[str, object] = {}

    class DummyEmbeddingGenerator:
        def __init__(self, **kwargs) -> None:
            captured["init_kwargs"] = kwargs

        def create_chunks(self, paper, extraction, raptor_summaries=None):
            return [
                EmbeddingChunk(
                    paper_id=paper.paper_id,
                    chunk_id=f"{paper.paper_id}_abstract",
                    chunk_type="abstract",
                    text="sample text",
                )
            ]

        def resolve_batch_size(self, batch_size, texts):
            captured["requested_batch_size"] = batch_size
            captured["probe_text_count"] = len(texts)
            return 24

        def generate_embeddings(self, chunks, batch_size):
            captured["resolved_batch_size"] = batch_size
            for chunk in chunks:
                chunk.embedding = [0.1, 0.2]
            return chunks

    class DummyVectorStore:
        def __init__(self, path) -> None:
            captured["vector_store_path"] = path

        def clear(self) -> None:
            captured["cleared"] = True

        def add_chunks(self, chunks, batch_size=100):
            captured["add_batch_size"] = batch_size
            return len(chunks)

    monkeypatch.setattr("src.indexing.pipeline.EmbeddingGenerator", DummyEmbeddingGenerator)
    monkeypatch.setattr("src.indexing.pipeline.VectorStore", DummyVectorStore)
    monkeypatch.setattr("src.indexing.pipeline._normalize_extractions", lambda extractions: {"P1": object()})

    run_metadata: dict[str, object] = {}
    added = run_embedding_generation(
        papers=[SimpleNamespace(paper_id="P1")],
        extractions={"P1": object()},
        index_dir=tmp_path,
        embedding_model="embed-test",
        rebuild=True,
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        embedding_backend="ollama",
        embedding_batch_size="auto",
        run_metadata=run_metadata,
    )

    assert added == 1
    assert captured["requested_batch_size"] == "auto"
    assert captured["resolved_batch_size"] == 24
    assert run_metadata["embedding_batch_size_setting"] == "auto"
    assert run_metadata["embedding_batch_size_resolved"] == 24
