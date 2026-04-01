"""Resilience tests for checkpointing and vector-store writes."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.indexing.embeddings import EmbeddingChunk, EmbeddingGenerator
from src.indexing.pipeline import (
    CHECKPOINT_EXTRACTIONS_FILENAME,
    CHECKPOINT_METADATA_FILENAME,
    CHECKPOINT_PAPERS_FILENAME,
    save_checkpoint,
)
from src.indexing.vector_store import VectorStore


def test_save_checkpoint_uses_checkpoint_artifacts(tmp_path: Path) -> None:
    save_checkpoint(
        index_dir=tmp_path,
        papers=[{"paper_id": "P1", "title": "Paper 1"}],
        extractions={"P1": {"paper_id": "P1", "extraction": {}}},
        metadata={"status": "running"},
    )

    assert (tmp_path / CHECKPOINT_PAPERS_FILENAME).exists()
    assert (tmp_path / CHECKPOINT_EXTRACTIONS_FILENAME).exists()
    assert (tmp_path / CHECKPOINT_METADATA_FILENAME).exists()
    assert not (tmp_path / "papers.json").exists()
    assert not (tmp_path / "semantic_analyses.json").exists()


def test_ollama_embed_batch_splits_failed_batches() -> None:
    class DummyClient:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def embed(self, model: str, input: list[str]) -> dict[str, list[list[float]]]:
            self.calls.append(list(input))
            if len(input) > 1:
                raise RuntimeError("connection reset by peer")
            return {"embeddings": [[float(len(input[0]))]]}

    generator = EmbeddingGenerator.__new__(EmbeddingGenerator)
    generator.model_name = "dummy"
    generator._ollama_client = DummyClient()
    generator.ollama_max_retries = 1
    generator.ollama_retry_backoff_seconds = 0.0

    embeddings = generator._ollama_embed_batch(["a", "bbbb"], batch_size=2)

    assert embeddings == [[1.0], [4.0]]
    assert generator._ollama_client.calls[0] == ["a", "bbbb"]


def test_replace_papers_restores_previous_chunks_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = VectorStore(tmp_path / "chroma")
    original_chunk = EmbeddingChunk(
        paper_id="P1",
        chunk_id="P1_old",
        chunk_type="abstract",
        text="old text",
        embedding=[0.1, 0.2],
        metadata={"title": "Old"},
    )
    store.add_chunks([original_chunk])

    original_upsert = store.collection.upsert

    def flaky_upsert(*, ids, embeddings, documents, metadatas) -> None:
        if "P1_new" in ids:
            raise RuntimeError("simulated write failure")
        original_upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    monkeypatch.setattr(store.collection, "upsert", flaky_upsert)

    with pytest.raises(RuntimeError, match="simulated write failure"):
        store.replace_papers(
            chunks=[
                EmbeddingChunk(
                    paper_id="P1",
                    chunk_id="P1_new",
                    chunk_type="abstract",
                    text="new text",
                    embedding=[0.3, 0.4],
                    metadata={"title": "New"},
                )
            ],
            scope_paper_ids=["P1"],
            delete_paper_ids=[],
        )

    restored = store.collection.get(where={"paper_id": "P1"}, include=["documents"])
    assert restored["ids"] == ["P1_old"]
    assert restored["documents"] == ["old text"]


def test_replace_papers_continues_when_backup_export_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = VectorStore(tmp_path / "chroma")
    store.add_chunks(
        [
            EmbeddingChunk(
                paper_id="P1",
                chunk_id="P1_old",
                chunk_type="abstract",
                text="old text",
                embedding=[0.1, 0.2],
                metadata={"title": "Old"},
            )
        ]
    )

    monkeypatch.setattr(
        store,
        "export_paper_chunks",
        lambda paper_ids: (_ for _ in ()).throw(RuntimeError("backup export failed")),
    )

    added = store.replace_papers(
        chunks=[
            EmbeddingChunk(
                paper_id="P1",
                chunk_id="P1_new",
                chunk_type="abstract",
                text="new text",
                embedding=[0.3, 0.4],
                metadata={"title": "New"},
            )
        ],
        scope_paper_ids=["P1"],
        delete_paper_ids=[],
    )

    assert added == 1
    current = store.collection.get(where={"paper_id": "P1"}, include=["documents"])
    assert current["ids"] == ["P1_new"]
    assert current["documents"] == ["new text"]
