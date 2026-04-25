"""Tests for configurable embedding batch sizes and Ollama auto-probing."""

from __future__ import annotations

import sys
import threading
import time
from types import SimpleNamespace

import pytest

import scripts.benchmark_embeddings as benchmark_embeddings
import scripts.build_index as build_index
from src.config import EmbeddingsConfig, parse_embedding_batch_size_setting
from src.indexing.embeddings import EmbeddingChunk, EmbeddingGenerator
from src.indexing.orchestrator import IndexOrchestrator
from src.indexing.pipeline import compute_similarity_pairs, run_embedding_generation
from src.indexing.structured_store import StructuredStore
from src.query.search import SearchEngine


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
    assert EmbeddingsConfig(device="cuda").device == "cuda"
    assert EmbeddingsConfig(device="").device is None
    assert EmbeddingsConfig(ollama_concurrency="3").ollama_concurrency == 3


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
            device="cuda",
            ollama_concurrency=1,
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
            device="cpu",
            ollama_concurrency=4,
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
    assert auto_info["device"] == "cuda"
    assert fixed_info["device"] == "cpu"
    assert auto_info["ollama_concurrency"] == 1
    assert fixed_info["ollama_concurrency"] == 4


def test_ollama_auto_batch_probe_selects_highest_throughput_within_budget(
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

    assert resolved == 32
    assert 8 in probe_calls
    assert 256 in probe_calls
    assert 384 in probe_calls
    assert 512 in probe_calls


def test_sentence_transformer_init_uses_device_and_trust_remote_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class DummyTransformer:
        def __init__(self, model_name: str, device: str | None = None, **kwargs) -> None:
            captured["model_name"] = model_name
            captured["device"] = device
            captured["kwargs"] = kwargs

        def get_sentence_embedding_dimension(self) -> int:
            return 1024

    monkeypatch.setattr("src.indexing.embeddings._SentenceTransformer", DummyTransformer)

    generator = EmbeddingGenerator(
        model_name="Alibaba-NLP/gte-large-en-v1.5",
        backend="sentence-transformers",
        device="cuda",
    )

    assert generator.embedding_dim == 1024
    assert captured["model_name"] == "Alibaba-NLP/gte-large-en-v1.5"
    assert captured["device"] == "cuda"
    assert captured["kwargs"] == {"trust_remote_code": True}


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


def test_ollama_iter_embedded_batches_uses_bounded_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_init_ollama(self, base_url: str) -> None:
        self._ollama_client = object()
        self.model = None
        self.embedding_dim = 4096

    def fake_embed_with_fallback(self, texts: list[str]) -> list[list[float]]:
        nonlocal active
        nonlocal max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.02)
            return [[float(len(text))] for text in texts]
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(EmbeddingGenerator, "_init_ollama", fake_init_ollama)
    monkeypatch.setattr(EmbeddingGenerator, "_ollama_embed_with_fallback", fake_embed_with_fallback)

    generator = EmbeddingGenerator(
        model_name="embed-test",
        backend="ollama",
        ollama_concurrency=2,
    )
    chunks = [
        EmbeddingChunk(
            paper_id="P1",
            chunk_id=f"P1_chunk_{idx}",
            chunk_type="abstract",
            text=f"text-{idx}",
        )
        for idx in range(4)
    ]

    batches = list(generator.iter_embedded_batches(chunks, batch_size=1, show_progress=False))

    assert 1 < max_active <= 2
    assert [chunk.chunk_id for batch in batches for chunk in batch] == [
        "P1_chunk_0",
        "P1_chunk_1",
        "P1_chunk_2",
        "P1_chunk_3",
    ]


def test_run_embedding_generation_uses_resolved_batch_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured: dict[str, object] = {}

    class DummyEmbeddingGenerator:
        def __init__(self, **kwargs) -> None:
            captured["init_kwargs"] = kwargs
            self.backend = kwargs["backend"]
            self.model_name = kwargs["model_name"]
            self.query_prefix = kwargs.get("query_prefix")
            self.document_prefix = kwargs.get("document_prefix")

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

        def iter_embedded_batches(self, chunks, batch_size, progress_callback=None):
            captured["resolved_batch_size"] = batch_size
            for chunk in chunks:
                chunk.embedding = [0.1, 0.2]
            if progress_callback is not None:
                progress_callback(
                    {
                        "backend": "ollama",
                        "model": "embed-test",
                        "batch_size": batch_size,
                        "completed_chunks": len(chunks),
                        "total_chunks": len(chunks),
                        "remaining_chunks": 0,
                        "elapsed_seconds": 1.5,
                        "rolling_chunks_per_second": 2.0,
                        "overall_chunks_per_second": 2.0,
                        "eta_seconds": 0.0,
                        "progress_fraction": 1.0,
                    }
                )
            yield chunks

    class DummyVectorStore:
        def __init__(self, path) -> None:
            captured["vector_store_path"] = path
            self._chunk_ids: set[str] = set()

        def clear(self) -> None:
            captured["cleared"] = True

        def count(self) -> int:
            return len(self._chunk_ids)

        def get_all_chunk_ids(self) -> set[str]:
            return set(self._chunk_ids)

        def add_chunks(self, chunks, batch_size=100, log_progress=True):
            captured["add_batch_size"] = batch_size
            captured["add_log_progress"] = log_progress
            self._chunk_ids.update(chunk.chunk_id for chunk in chunks)
            return len(chunks)

        def close(self) -> None:
            pass

    monkeypatch.setattr("src.indexing.pipeline.EmbeddingGenerator", DummyEmbeddingGenerator)
    monkeypatch.setattr("src.indexing.pipeline.VectorStore", DummyVectorStore)
    monkeypatch.setattr(
        "src.indexing.pipeline._normalize_extractions", lambda extractions: {"P1": object()}
    )

    run_metadata: dict[str, object] = {}
    added = run_embedding_generation(
        papers=[SimpleNamespace(paper_id="P1")],
        extractions={"P1": object()},
        index_dir=tmp_path,
        embedding_model="embed-test",
        rebuild=True,
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        embedding_backend="ollama",
        embedding_device="cuda",
        ollama_concurrency=3,
        embedding_batch_size="auto",
        run_metadata=run_metadata,
    )

    assert added == 1
    assert captured["init_kwargs"]["device"] == "cuda"
    assert captured["init_kwargs"]["ollama_concurrency"] == 3
    assert captured["requested_batch_size"] == "auto"
    assert captured["resolved_batch_size"] == 24
    assert captured["add_log_progress"] is False
    assert run_metadata["embedding_batch_size_setting"] == "auto"
    assert run_metadata["embedding_batch_size_resolved"] == 24
    assert run_metadata["embedding_progress"]["completed_chunks"] == 1
    assert run_metadata["embedding_resume"]["status"] == "complete"


def test_run_embedding_generation_resumes_staged_rebuild(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class DummyEmbeddingGenerator:
        invocation_count = 0

        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.backend = kwargs["backend"]
            self.model_name = kwargs["model_name"]
            self.query_prefix = kwargs.get("query_prefix")
            self.document_prefix = kwargs.get("document_prefix")

        def create_chunks(self, paper, extraction, raptor_summaries=None):
            return [
                EmbeddingChunk(
                    paper_id=paper.paper_id,
                    chunk_id=f"{paper.paper_id}_chunk_{idx}",
                    chunk_type="abstract",
                    text=f"text-{idx}",
                )
                for idx in range(3)
            ]

        def resolve_batch_size(self, batch_size, texts):
            return 1

        def iter_embedded_batches(self, chunks, batch_size, progress_callback=None):
            type(self).invocation_count += 1
            for idx, chunk in enumerate(chunks, start=1):
                chunk.embedding = [float(idx), 0.5]
                if progress_callback is not None:
                    progress_callback(
                        {
                            "backend": "ollama",
                            "model": "embed-test",
                            "batch_size": batch_size,
                            "completed_chunks": idx,
                            "total_chunks": len(chunks),
                            "remaining_chunks": len(chunks) - idx,
                            "elapsed_seconds": float(idx),
                            "rolling_chunks_per_second": 1.0,
                            "overall_chunks_per_second": 1.0,
                            "eta_seconds": float(len(chunks) - idx),
                            "progress_fraction": idx / len(chunks),
                        }
                    )
                yield [chunk]
                if type(self).invocation_count == 1 and idx == 1:
                    raise RuntimeError("interrupt after first persisted batch")

    monkeypatch.setattr("src.indexing.pipeline.EmbeddingGenerator", DummyEmbeddingGenerator)
    monkeypatch.setattr("src.indexing.pipeline.EMBEDDING_STREAM_WRITE_CHUNK_COUNT", 1)
    monkeypatch.setattr(
        "src.indexing.pipeline._normalize_extractions", lambda extractions: {"P1": object()}
    )
    progress_updates: list[dict[str, object]] = []

    stage_dir = tmp_path / "chroma_staging_resume"
    with pytest.raises(RuntimeError, match="interrupt"):
        run_embedding_generation(
            papers=[SimpleNamespace(paper_id="P1")],
            extractions={"P1": object()},
            index_dir=tmp_path,
            embedding_model="embed-test",
            rebuild=True,
            logger=SimpleNamespace(info=lambda *args, **kwargs: None),
            embedding_backend="ollama",
            embedding_batch_size=1,
            vector_store_dir=stage_dir,
            run_metadata={},
        )

    resume_state_path = stage_dir / "embedding_resume_state.json"
    assert resume_state_path.exists()

    run_metadata: dict[str, object] = {}
    added = run_embedding_generation(
        papers=[SimpleNamespace(paper_id="P1")],
        extractions={"P1": object()},
        index_dir=tmp_path,
        embedding_model="embed-test",
        rebuild=True,
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        embedding_backend="ollama",
        embedding_batch_size=1,
        vector_store_dir=stage_dir,
        run_metadata=run_metadata,
        progress_callback=progress_updates.append,
    )

    assert added == 3
    assert not resume_state_path.exists()
    assert run_metadata["embedding_resume"]["resumed"] is True
    assert run_metadata["embedding_resume"]["completed_chunks"] == 3
    assert any(update["embedding_progress"]["total_chunks"] == 3 for update in progress_updates)
    assert any(
        update["embedding_progress"]["progress_fraction"] == pytest.approx(2 / 3, abs=1e-6)
        for update in progress_updates
        if update["embedding_progress"]["completed_chunks"] == 2
    )


def test_compute_similarity_pairs_uses_stored_overview_embeddings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class DummyCollection:
        def get(self, *, where=None, include=None):
            assert where == {"chunk_type": "raptor_overview"}
            assert include == ["embeddings", "metadatas"]
            return {
                "ids": ["p1_raptor_overview", "p2_raptor_overview", "p3_raptor_overview"],
                "embeddings": [
                    [1.0, 0.0],
                    [0.9, 0.1],
                    [0.0, 1.0],
                ],
                "metadatas": [
                    {"paper_id": "p1"},
                    {"paper_id": "p2"},
                    {"paper_id": "p3"},
                ],
            }

    class DummyVectorStore:
        def __init__(self, path) -> None:
            self.collection = DummyCollection()

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            return None

    monkeypatch.setattr("src.indexing.pipeline.VectorStore", DummyVectorStore)
    monkeypatch.setattr(
        "src.indexing.pipeline.EmbeddingGenerator",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("fallback should not be used")),
    )

    total_pairs = compute_similarity_pairs(
        index_dir=tmp_path,
        embedding_model="embed-test",
        top_n=1,
    )

    assert total_pairs == 3
    pairs = StructuredStore(tmp_path).load_similarity_pairs()
    assert pairs["p1"][0]["similar_paper_id"] == "p2"
    assert pairs["p2"][0]["similar_paper_id"] == "p1"
    assert pairs["p3"][0]["similar_paper_id"] == "p2"


def test_compute_similarity_pairs_reembeds_overview_texts_on_chroma_embedding_read_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured: dict[str, object] = {}

    class DummyCollection:
        def get(self, *, where=None, include=None):
            if include == ["embeddings", "metadatas"]:
                raise RuntimeError("Error executing plan: Internal error: Error finding id")
            assert where == {"chunk_type": "raptor_overview"}
            assert include == ["documents", "metadatas"]
            return {
                "ids": ["p1_raptor_overview", "p2_raptor_overview", "p3_raptor_overview"],
                "documents": ["alpha", "alphabet", "zeta"],
                "metadatas": [
                    {"paper_id": "p1"},
                    {"paper_id": "p2"},
                    {"paper_id": "p3"},
                ],
            }

    class DummyVectorStore:
        def __init__(self, path) -> None:
            self.collection = DummyCollection()

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            return None

    class DummyEmbeddingGenerator:
        def __init__(self, **kwargs) -> None:
            captured["init_kwargs"] = kwargs

        def resolve_batch_size(self, batch_size, texts):
            captured["requested_batch_size"] = batch_size
            captured["texts"] = list(texts)
            return 11

        def generate_embeddings(self, chunks, batch_size, show_progress):
            captured["resolved_batch_size"] = batch_size
            captured["show_progress"] = show_progress
            embedding_map = {
                "alpha": [1.0, 0.0],
                "alphabet": [0.95, 0.05],
                "zeta": [0.0, 1.0],
            }
            for chunk in chunks:
                chunk.embedding = embedding_map[chunk.text]
            return chunks

    monkeypatch.setattr("src.indexing.pipeline.VectorStore", DummyVectorStore)
    monkeypatch.setattr("src.indexing.pipeline.EmbeddingGenerator", DummyEmbeddingGenerator)

    total_pairs = compute_similarity_pairs(
        index_dir=tmp_path,
        embedding_model="embed-test",
        top_n=1,
        embedding_backend="ollama",
        embedding_device="cuda",
        ollama_base_url="http://localhost:11434",
        ollama_concurrency=4,
        query_prefix="query: ",
        document_prefix="doc: ",
        embedding_batch_size="auto",
    )

    assert total_pairs == 3
    assert captured["init_kwargs"] == {
        "model_name": "embed-test",
        "device": "cuda",
        "backend": "ollama",
        "ollama_base_url": "http://localhost:11434",
        "ollama_concurrency": 4,
        "query_prefix": "query: ",
        "document_prefix": "doc: ",
    }
    assert captured["requested_batch_size"] == "auto"
    assert captured["texts"] == ["alpha", "alphabet", "zeta"]
    assert captured["resolved_batch_size"] == 11
    assert captured["show_progress"] is False
    pairs = StructuredStore(tmp_path).load_similarity_pairs()
    assert pairs["p1"][0]["similar_paper_id"] == "p2"
    assert pairs["p2"][0]["similar_paper_id"] == "p1"
    assert pairs["p3"][0]["similar_paper_id"] == "p2"


def test_search_engine_passes_embedding_device(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    class DummyVectorStore:
        def __init__(self, path) -> None:
            self.path = path

        def close(self) -> None:
            return None

    class DummyEmbeddingGenerator:
        def __init__(self, **kwargs) -> None:
            captured["kwargs"] = kwargs

    monkeypatch.setattr("src.query.search.VectorStore", DummyVectorStore)
    monkeypatch.setattr("src.query.search.EmbeddingGenerator", DummyEmbeddingGenerator)

    engine = SearchEngine(
        index_dir=tmp_path,
        embedding_model="embed-test",
        embedding_backend="sentence-transformers",
        embedding_device="cuda",
        embedding_ollama_concurrency=5,
    )

    assert captured["kwargs"]["device"] == "cuda"
    assert captured["kwargs"]["ollama_concurrency"] == 5
    engine.close()


def test_benchmark_embeddings_main_strips_internal_payloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class DummyConfig:
        def __init__(self) -> None:
            self.embeddings = SimpleNamespace(
                model="embed-test",
                backend="ollama",
                device="cuda",
                ollama_base_url="http://localhost:11434",
                ollama_concurrency=2,
                query_prefix="query: ",
                document_prefix="doc: ",
                batch_size="auto",
            )

        def get_index_path(self, _project_root):
            return tmp_path

        @classmethod
        def load(cls, _path):
            return cls()

    class DummyEmbeddingGenerator:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def embed_text(self, text: str) -> list[float]:
            return [1.0, 0.0] if "alpha" in text else [0.0, 1.0]

    output_dir = tmp_path / "benchmarks"
    monkeypatch.setattr(benchmark_embeddings, "Config", DummyConfig)
    monkeypatch.setattr(
        benchmark_embeddings,
        "load_benchmark_chunks",
        lambda **kwargs: [
            EmbeddingChunk(
                paper_id="P1",
                chunk_id="P1_abstract",
                chunk_type="abstract",
                text="alpha",
                metadata={},
            )
        ],
    )
    monkeypatch.setattr(
        benchmark_embeddings,
        "default_candidates",
        lambda config: [
            benchmark_embeddings.BenchmarkCandidate(
                backend="ollama",
                model="embed-test",
                device="cuda",
                ollama_base_url=config.embeddings.ollama_base_url,
                ollama_concurrency=config.embeddings.ollama_concurrency,
                query_prefix=config.embeddings.query_prefix,
                document_prefix=config.embeddings.document_prefix,
                label="baseline",
            )
        ],
    )
    monkeypatch.setattr(
        benchmark_embeddings,
        "benchmark_candidate",
        lambda **kwargs: {
            "candidate": {
                "backend": "ollama",
                "model": "embed-test",
                "device": "cuda",
                "ollama_base_url": "http://localhost:11434",
                "ollama_concurrency": 2,
                "query_prefix": "query: ",
                "document_prefix": "doc: ",
                "label": "baseline",
            },
            "resolved_batch_size": 8,
            "chunk_count": 1,
            "wall_seconds": 0.25,
            "chunks_per_second": 4.0,
            "python_cpu_seconds": 0.1,
            "python_cpu_ratio": 0.4,
            "python_current_memory_bytes": 128,
            "python_peak_memory_bytes": 256,
            "gpu_start": None,
            "gpu_end": None,
            "_paper_ids": ["P1"],
            "_document_embeddings": [[1.0, 0.0]],
        },
    )
    monkeypatch.setattr(benchmark_embeddings, "benchmark_ollama_concurrency", lambda **kwargs: [])
    monkeypatch.setattr(benchmark_embeddings, "EmbeddingGenerator", DummyEmbeddingGenerator)
    monkeypatch.setattr(
        benchmark_embeddings,
        "parse_args",
        lambda: SimpleNamespace(
            config=None,
            index_dir=tmp_path,
            paper_limit=1,
            query_limit=1,
            top_k=1,
            embedding_batch_size=None,
            candidate=[],
            ollama_concurrency="1",
            output_dir=output_dir,
        ),
    )

    assert benchmark_embeddings.main() == 0

    report_path = next(output_dir.glob("embedding_benchmark_*.json"))
    report = report_path.read_text(encoding="utf-8")
    assert "_document_embeddings" not in report
    assert "_paper_ids" not in report


def test_load_raptor_summaries_reads_wrapped_cache(tmp_path) -> None:
    cache_path = tmp_path / "raptor_summaries.json"
    cache_path.write_text(
        """
{
  "schema_version": "1.0.0",
  "summaries": {
    "P1": {
      "paper_overview": "Overview",
      "core_contribution": "Contribution"
    }
  }
}
""".strip(),
        encoding="utf-8",
    )

    summaries = benchmark_embeddings._load_raptor_summaries(tmp_path)

    assert summaries["P1"] == benchmark_embeddings.RaptorSummaries(
        paper_id="P1",
        paper_overview="Overview",
        core_contribution="Contribution",
    )
