#!/usr/bin/env python
"""Benchmark embedding candidates on a fixed index slice without mutating the live index."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tracemalloc
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter, process_time
from typing import Any

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.raptor import RaptorSummaries
from src.analysis.schemas import SemanticAnalysis
from src.config import Config, parse_embedding_batch_size_setting
from src.indexing.embeddings import EmbeddingChunk, EmbeddingGenerator
from src.indexing.structured_store import StructuredStore
from src.utils.file_utils import safe_read_json, safe_write_json
from src.utils.logging_config import setup_logging
from src.zotero.models import PaperMetadata


@dataclass(frozen=True)
class BenchmarkCandidate:
    """One embedding backend/model candidate to benchmark."""

    backend: str
    model: str
    device: str | None = None
    ollama_base_url: str | None = None
    ollama_concurrency: int = 1
    query_prefix: str | None = None
    document_prefix: str | None = None
    label: str | None = None

    @property
    def resolved_label(self) -> str:
        """Return a stable human-readable label."""
        if self.label:
            return self.label
        device_suffix = f"-{self.device}" if self.device else ""
        return f"{self.backend}:{self.model}{device_suffix}"


def parse_candidate_spec(spec: str) -> BenchmarkCandidate:
    """Parse ``backend|model|device|label`` into a benchmark candidate."""
    parts = [part.strip() for part in spec.split("|")]
    if len(parts) < 2:
        raise ValueError(
            "candidate spec must use backend|model|device|label (device and label may be blank)"
        )
    backend = parts[0]
    model = parts[1]
    device = parts[2] if len(parts) > 2 and parts[2] else None
    label = parts[3] if len(parts) > 3 and parts[3] else None
    return BenchmarkCandidate(
        backend=backend,
        model=model,
        device=device,
        label=label,
    )


def default_candidates(config: Config) -> list[BenchmarkCandidate]:
    """Return the default baseline + comparison candidates for workstation testing."""
    baseline = BenchmarkCandidate(
        backend=config.embeddings.backend,
        model=config.embeddings.model,
        device=config.embeddings.device,
        ollama_base_url=config.embeddings.ollama_base_url,
        ollama_concurrency=config.embeddings.ollama_concurrency,
        query_prefix=config.embeddings.query_prefix,
        document_prefix=config.embeddings.document_prefix,
        label="baseline",
    )
    gte_large = BenchmarkCandidate(
        backend="sentence-transformers",
        model="Alibaba-NLP/gte-large-en-v1.5",
        device="cuda",
        label="gte-large-en-v1.5",
    )
    qwen_small = BenchmarkCandidate(
        backend="ollama",
        model="qwen3-embedding:0.6b",
        ollama_base_url=config.embeddings.ollama_base_url,
        ollama_concurrency=config.embeddings.ollama_concurrency,
        query_prefix=config.embeddings.query_prefix,
        document_prefix=config.embeddings.document_prefix,
        label="qwen3-embedding-0.6b",
    )
    return [baseline, gte_large, qwen_small]


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Override the index directory to benchmark",
    )
    parser.add_argument(
        "--paper-limit",
        type=int,
        default=50,
        help="Maximum number of indexed papers to include in the benchmark slice",
    )
    parser.add_argument(
        "--query-limit",
        type=int,
        default=20,
        help="Maximum number of chunk texts to reuse as query probes for retrieval deltas",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-k unique paper ids to compare for retrieval overlap",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=parse_embedding_batch_size_setting,
        default=None,
        metavar="N|auto",
        help="Override embedding batch size for all candidates",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate spec: backend|model|device|label",
    )
    parser.add_argument(
        "--ollama-concurrency",
        default="1,2,4",
        help="Comma-separated concurrency levels for the Ollama concurrency spike",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "out" / "experiments" / "embedding_benchmarks",
        help="Directory for JSON/CSV scorecards",
    )
    return parser.parse_args()


def _gpu_snapshot() -> dict[str, int] | None:
    """Return a best-effort GPU utilization snapshot using ``nvidia-smi``."""
    binary = shutil.which("nvidia-smi")
    if not binary:
        return None
    try:
        result = subprocess.run(
            [
                binary,
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    first_line = next((line.strip() for line in result.stdout.splitlines() if line.strip()), "")
    if not first_line:
        return None
    try:
        utilization, memory_used, memory_total = [int(part.strip()) for part in first_line.split(",")]
    except ValueError:
        return None
    return {
        "utilization_gpu_percent": utilization,
        "memory_used_mb": memory_used,
        "memory_total_mb": memory_total,
    }


def _load_raptor_summaries(index_dir: Path) -> dict[str, RaptorSummaries]:
    """Load Raptor summaries when present."""
    raw = safe_read_json(index_dir / "raptor_summaries.json", default={}) or {}
    if isinstance(raw, dict) and isinstance(raw.get("summaries"), dict):
        raw = raw["summaries"]
    summaries: dict[str, RaptorSummaries] = {}
    for paper_id, summary_data in raw.items():
        if not isinstance(summary_data, dict):
            continue
        if "paper_overview" not in summary_data and "core_contribution" not in summary_data:
            continue
        summaries[paper_id] = RaptorSummaries(
            paper_id=paper_id,
            paper_overview=str(summary_data.get("paper_overview", "")),
            core_contribution=str(summary_data.get("core_contribution", "")),
        )
    return summaries


def _normalize_extractions(extractions: dict[str, dict | SemanticAnalysis]) -> dict[str, SemanticAnalysis]:
    """Normalize extraction payloads into ``SemanticAnalysis`` models."""
    normalized: dict[str, SemanticAnalysis] = {}
    for paper_id, ext_data in extractions.items():
        if isinstance(ext_data, SemanticAnalysis):
            normalized[paper_id] = ext_data
            continue
        inner = ext_data.get("extraction", ext_data) if isinstance(ext_data, dict) else ext_data
        normalized[paper_id] = SemanticAnalysis(**inner)
    return normalized


def _paper_from_index_dict(paper_id: str, paper_data: dict[str, Any]) -> PaperMetadata:
    """Build ``PaperMetadata`` from stored index JSON without importing the orchestrator."""
    return PaperMetadata(
        paper_id=paper_id,
        zotero_key=paper_data.get("zotero_key", paper_id.split("_")[0]),
        zotero_item_id=paper_data.get("zotero_item_id", 0),
        item_type=paper_data.get("item_type", "journalArticle"),
        title=paper_data.get("title", "Unknown"),
        authors=paper_data.get("authors", []),
        publication_year=paper_data.get("publication_year"),
        publication_date=paper_data.get("publication_date"),
        journal=paper_data.get("journal"),
        doi=paper_data.get("doi"),
        abstract=paper_data.get("abstract"),
        collections=paper_data.get("collections", []),
        tags=paper_data.get("tags", []),
        pdf_path=paper_data.get("pdf_path"),
        pdf_attachment_key=paper_data.get("pdf_attachment_key"),
        source_path=paper_data.get("source_path"),
        source_attachment_key=paper_data.get("source_attachment_key"),
        source_media_type=paper_data.get("source_media_type"),
        date_added=paper_data.get("date_added") or "2020-01-01T00:00:00",
        date_modified=paper_data.get("date_modified") or "2020-01-01T00:00:00",
    )


def load_benchmark_chunks(
    *,
    index_dir: Path,
    config: Config,
    paper_limit: int,
) -> list[EmbeddingChunk]:
    """Load a stable benchmark slice from the existing index artifacts."""
    store = StructuredStore(index_dir)
    paper_dicts = store.load_papers()
    normalized_extractions = _normalize_extractions(store.load_extractions())
    raptor_summaries = _load_raptor_summaries(index_dir)
    chunker = EmbeddingGenerator(
        model_name=config.embeddings.model,
        device=config.embeddings.device,
        backend=config.embeddings.backend,
        ollama_base_url=config.embeddings.ollama_base_url,
        ollama_concurrency=config.embeddings.ollama_concurrency,
        query_prefix=config.embeddings.query_prefix,
        document_prefix=config.embeddings.document_prefix,
    )
    selected_ids = [
        paper_id for paper_id in sorted(paper_dicts) if paper_id in normalized_extractions
    ][:paper_limit]
    chunks: list[EmbeddingChunk] = []
    for paper_id in selected_ids:
        chunks.extend(
            chunker.create_chunks(
                _paper_from_index_dict(paper_id, paper_dicts[paper_id]),
                normalized_extractions[paper_id],
                raptor_summaries=raptor_summaries.get(paper_id),
            )
        )
    return chunks


def _copy_chunks(chunks: list[EmbeddingChunk]) -> list[EmbeddingChunk]:
    """Return detached chunk copies for one benchmark run."""
    return [
        EmbeddingChunk(
            paper_id=chunk.paper_id,
            chunk_id=chunk.chunk_id,
            chunk_type=chunk.chunk_type,
            text=chunk.text,
            metadata=dict(chunk.metadata),
        )
        for chunk in chunks
    ]


def _select_query_texts(chunks: list[EmbeddingChunk], query_limit: int) -> list[str]:
    """Choose stable query probes from the chunk corpus."""
    preferred = [chunk.text for chunk in chunks if chunk.chunk_type == "raptor_overview"]
    if len(preferred) < query_limit:
        preferred.extend(chunk.text for chunk in chunks if chunk.chunk_type == "abstract")
    if len(preferred) < query_limit:
        preferred.extend(chunk.text for chunk in chunks)
    return preferred[:query_limit]


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """Return row-normalized vectors for cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def _top_k_unique_paper_ids(
    query_vector: np.ndarray,
    doc_vectors: np.ndarray,
    paper_ids: list[str],
    top_k: int,
) -> list[str]:
    """Return the top-k unique paper ids ranked by cosine similarity."""
    scores = doc_vectors @ query_vector
    ranked_indices = np.argsort(scores)[::-1]
    ordered: list[str] = []
    seen: set[str] = set()
    for idx in ranked_indices:
        paper_id = paper_ids[int(idx)]
        if paper_id in seen:
            continue
        seen.add(paper_id)
        ordered.append(paper_id)
        if len(ordered) >= top_k:
            break
    return ordered


def _retrieval_overlap_score(
    *,
    query_texts: list[str],
    top_k: int,
    baseline_docs: np.ndarray,
    baseline_doc_paper_ids: list[str],
    baseline_generator: EmbeddingGenerator,
    candidate_docs: np.ndarray,
    candidate_doc_paper_ids: list[str],
    candidate_generator: EmbeddingGenerator,
) -> dict[str, float]:
    """Compare candidate top-k paper overlap against the baseline generator."""
    baseline_norm = _normalize_rows(baseline_docs)
    candidate_norm = _normalize_rows(candidate_docs)
    overlaps: list[float] = []
    for query_text in query_texts:
        baseline_query = np.asarray(baseline_generator.embed_text(query_text), dtype=np.float32)
        candidate_query = np.asarray(candidate_generator.embed_text(query_text), dtype=np.float32)
        baseline_query = baseline_query / max(np.linalg.norm(baseline_query), 1e-9)
        candidate_query = candidate_query / max(np.linalg.norm(candidate_query), 1e-9)
        baseline_hits = set(
            _top_k_unique_paper_ids(
                baseline_query,
                baseline_norm,
                baseline_doc_paper_ids,
                top_k,
            )
        )
        candidate_hits = set(
            _top_k_unique_paper_ids(
                candidate_query,
                candidate_norm,
                candidate_doc_paper_ids,
                top_k,
            )
        )
        union = baseline_hits | candidate_hits
        overlaps.append(len(baseline_hits & candidate_hits) / len(union) if union else 1.0)
    return {
        "query_count": float(len(query_texts)),
        "avg_top_k_jaccard_overlap": float(sum(overlaps) / len(overlaps)) if overlaps else 1.0,
    }


def benchmark_candidate(
    *,
    candidate: BenchmarkCandidate,
    chunks: list[EmbeddingChunk],
    batch_size_setting: int | str | None,
) -> dict[str, Any]:
    """Benchmark one embedding candidate on the supplied chunk slice."""
    candidate_chunks = _copy_chunks(chunks)
    generator = EmbeddingGenerator(
        model_name=candidate.model,
        device=candidate.device,
        backend=candidate.backend,
        ollama_base_url=candidate.ollama_base_url or "http://localhost:11434",
        ollama_concurrency=candidate.ollama_concurrency,
        query_prefix=candidate.query_prefix,
        document_prefix=candidate.document_prefix,
    )
    resolved_batch_size = generator.resolve_batch_size(
        batch_size_setting,
        texts=[chunk.text for chunk in candidate_chunks],
    )
    start_gpu = _gpu_snapshot()
    start_cpu = process_time()
    start_wall = perf_counter()
    tracemalloc.start()
    embeddings_count = 0
    for batch in generator.iter_embedded_batches(
        candidate_chunks,
        batch_size=resolved_batch_size,
        show_progress=False,
    ):
        embeddings_count += len(batch)
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    wall_seconds = perf_counter() - start_wall
    cpu_seconds = process_time() - start_cpu
    return {
        "candidate": asdict(candidate),
        "resolved_batch_size": resolved_batch_size,
        "chunk_count": embeddings_count,
        "wall_seconds": round(wall_seconds, 3),
        "chunks_per_second": round(embeddings_count / max(wall_seconds, 1e-9), 6),
        "python_cpu_seconds": round(cpu_seconds, 3),
        "python_cpu_ratio": round(cpu_seconds / max(wall_seconds, 1e-9), 6),
        "python_current_memory_bytes": int(current_mem),
        "python_peak_memory_bytes": int(peak_mem),
        "gpu_start": start_gpu,
        "gpu_end": _gpu_snapshot(),
        "_paper_ids": [chunk.paper_id for chunk in candidate_chunks],
        "_document_embeddings": [chunk.embedding for chunk in candidate_chunks],
    }


def benchmark_ollama_concurrency(
    *,
    candidate: BenchmarkCandidate,
    texts: list[str],
    batch_size_setting: int | str | None,
    concurrency_levels: list[int],
) -> list[dict[str, Any]]:
    """Run a bounded in-process Ollama concurrency experiment for one candidate."""
    if candidate.backend != "ollama" or not texts:
        return []

    results: list[dict[str, Any]] = []
    for concurrency in concurrency_levels:
        generator = EmbeddingGenerator(
            model_name=candidate.model,
            device=candidate.device,
            backend=candidate.backend,
            ollama_base_url=candidate.ollama_base_url or "http://localhost:11434",
            ollama_concurrency=concurrency,
            query_prefix=candidate.query_prefix,
            document_prefix=candidate.document_prefix,
        )
        batch_size = generator.resolve_batch_size(batch_size_setting, texts=texts)
        start_gpu = _gpu_snapshot()
        start_cpu = process_time()
        start_wall = perf_counter()
        completed = len(generator.embed_batch(texts, batch_size=batch_size))
        wall_seconds = perf_counter() - start_wall
        cpu_seconds = process_time() - start_cpu
        results.append(
            {
                "candidate_label": candidate.resolved_label,
                "ollama_concurrency": concurrency,
                "resolved_batch_size": batch_size,
                "text_count": completed,
                "wall_seconds": round(wall_seconds, 3),
                "texts_per_second": round(completed / max(wall_seconds, 1e-9), 6),
                "python_cpu_seconds": round(cpu_seconds, 3),
                "python_cpu_ratio": round(cpu_seconds / max(wall_seconds, 1e-9), 6),
                "gpu_start": start_gpu,
                "gpu_end": _gpu_snapshot(),
            }
        )
    return results


def write_scorecards(output_dir: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    """Write JSON and CSV scorecards for one benchmark report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"embedding_benchmark_{timestamp}.json"
    csv_path = output_dir / f"embedding_benchmark_{timestamp}.csv"
    safe_write_json(json_path, report)

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "backend",
                "model",
                "device",
                "ollama_concurrency",
                "resolved_batch_size",
                "chunk_count",
                "wall_seconds",
                "chunks_per_second",
                "avg_top_k_jaccard_overlap",
                "python_peak_memory_bytes",
            ],
        )
        writer.writeheader()
        for item in report["benchmarks"]:
            writer.writerow(
                {
                    "label": item["candidate"]["label"],
                    "backend": item["candidate"]["backend"],
                    "model": item["candidate"]["model"],
                    "device": item["candidate"]["device"],
                    "ollama_concurrency": item["candidate"]["ollama_concurrency"],
                    "resolved_batch_size": item["resolved_batch_size"],
                    "chunk_count": item["chunk_count"],
                    "wall_seconds": item["wall_seconds"],
                    "chunks_per_second": item["chunks_per_second"],
                    "avg_top_k_jaccard_overlap": item.get("retrieval_delta", {}).get(
                        "avg_top_k_jaccard_overlap"
                    ),
                    "python_peak_memory_bytes": item["python_peak_memory_bytes"],
                }
            )
    return json_path, csv_path


def main() -> int:
    """Run the embedding benchmark harness."""
    args = parse_args()
    logger = setup_logging(level="INFO")
    config = Config.load(args.config)
    index_dir = args.index_dir or config.get_index_path(project_root)
    if args.embedding_batch_size is not None:
        config.embeddings.batch_size = args.embedding_batch_size

    candidates = (
        [
            BenchmarkCandidate(
                backend=spec.backend,
                model=spec.model,
                device=spec.device,
                ollama_base_url=(
                    config.embeddings.ollama_base_url if spec.backend == "ollama" else None
                ),
                ollama_concurrency=config.embeddings.ollama_concurrency,
                query_prefix=config.embeddings.query_prefix if spec.backend == "ollama" else None,
                document_prefix=config.embeddings.document_prefix if spec.backend == "ollama" else None,
                label=spec.label,
            )
            for spec in (parse_candidate_spec(item) for item in args.candidate)
        ]
        if args.candidate
        else default_candidates(config)
    )

    chunks = load_benchmark_chunks(index_dir=index_dir, config=config, paper_limit=args.paper_limit)
    if not chunks:
        logger.error("No indexed chunks available for benchmarking in %s", index_dir)
        return 1

    query_texts = _select_query_texts(chunks, args.query_limit)
    benchmarks: list[dict[str, Any]] = []
    concurrency_levels = [
        int(item.strip()) for item in args.ollama_concurrency.split(",") if item.strip()
    ]
    logger.info("Benchmarking %d candidates on %d chunks", len(candidates), len(chunks))

    baseline_generator: EmbeddingGenerator | None = None
    baseline_docs: np.ndarray | None = None
    baseline_paper_ids: list[str] | None = None

    for candidate in candidates:
        result = benchmark_candidate(
            candidate=candidate,
            chunks=chunks,
            batch_size_setting=config.embeddings.batch_size,
        )
        doc_vectors = np.asarray(result.pop("_document_embeddings"), dtype=np.float32)
        paper_ids = list(result.pop("_paper_ids"))
        if baseline_generator is None:
            baseline_generator = EmbeddingGenerator(
                model_name=candidate.model,
                device=candidate.device,
                backend=candidate.backend,
                ollama_base_url=candidate.ollama_base_url or "http://localhost:11434",
                ollama_concurrency=candidate.ollama_concurrency,
                query_prefix=candidate.query_prefix,
                document_prefix=candidate.document_prefix,
            )
            baseline_docs = doc_vectors
            baseline_paper_ids = paper_ids
            result["retrieval_delta"] = {
                "query_count": float(len(query_texts)),
                "avg_top_k_jaccard_overlap": 1.0,
            }
        else:
            assert baseline_generator is not None
            assert baseline_docs is not None
            assert baseline_paper_ids is not None
            candidate_generator = EmbeddingGenerator(
                model_name=candidate.model,
                device=candidate.device,
                backend=candidate.backend,
                ollama_base_url=candidate.ollama_base_url or "http://localhost:11434",
                ollama_concurrency=candidate.ollama_concurrency,
                query_prefix=candidate.query_prefix,
                document_prefix=candidate.document_prefix,
            )
            result["retrieval_delta"] = _retrieval_overlap_score(
                query_texts=query_texts,
                top_k=args.top_k,
                baseline_docs=baseline_docs,
                baseline_doc_paper_ids=baseline_paper_ids,
                baseline_generator=baseline_generator,
                candidate_docs=doc_vectors,
                candidate_doc_paper_ids=paper_ids,
                candidate_generator=candidate_generator,
            )
        result["candidate"]["label"] = candidate.resolved_label
        result["ollama_concurrency"] = benchmark_ollama_concurrency(
            candidate=candidate,
            texts=[chunk.text for chunk in chunks[: min(len(chunks), 64)]],
            batch_size_setting=config.embeddings.batch_size,
            concurrency_levels=concurrency_levels,
        )
        benchmarks.append(result)

    report = {
        "generated_at": datetime.now().isoformat(),
        "index_dir": str(index_dir),
        "paper_limit": args.paper_limit,
        "query_limit": args.query_limit,
        "top_k": args.top_k,
        "embedding_batch_size_setting": config.embeddings.batch_size,
        "benchmarks": benchmarks,
    }
    json_path, csv_path = write_scorecards(args.output_dir, report)
    logger.info("Embedding benchmark scorecards written to %s and %s", json_path, csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
