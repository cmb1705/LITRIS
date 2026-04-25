"""Shared indexing pipeline steps used by build and maintenance scripts."""

from __future__ import annotations

import csv
import hashlib
import inspect
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.analysis.classification_store import ClassificationIndex
from src.analysis.cli_executor import ClaudeCliAuthenticator
from src.analysis.extraction_intent_classifier import HybridProfileSpec
from src.analysis.raptor import RaptorSummaries
from src.analysis.schemas import SemanticAnalysis
from src.analysis.section_extractor import SectionExtractor
from src.config import Config
from src.extraction.opendataloader_extractor import build_hybrid_config_from_processing
from src.extraction.pdf_extractor import PDFExtractor
from src.extraction.text_cleaner import TextCleaner
from src.indexing.embeddings import EmbeddingChunk, EmbeddingGenerator
from src.indexing.structured_store import StructuredStore
from src.indexing.vector_store import VectorStore
from src.utils.checkpoint import CheckpointManager
from src.utils.file_utils import safe_read_json, safe_write_json
from src.utils.run_control import PauseRequested
from src.zotero.models import PaperMetadata

CHECKPOINT_PAPERS_FILENAME = "papers.checkpoint.json"
CHECKPOINT_EXTRACTIONS_FILENAME = "semantic_analyses.checkpoint.json"
CHECKPOINT_METADATA_FILENAME = "metadata.checkpoint.json"
EMBEDDING_RESUME_STATE_FILENAME = "embedding_resume_state.json"
EMBEDDING_STREAM_WRITE_CHUNK_COUNT = 512


@dataclass
class ExtractionRunResult:
    """Structured result for an extraction stage run."""

    paper_dicts: dict[str, dict]
    extractions: dict[str, dict]
    results: list
    interrupted: bool = False
    pause_requested: bool = False
    pending_ids: list[str] = field(default_factory=list)


def _embedding_resume_state_path(chroma_dir: Path) -> Path:
    """Return the staged rebuild resume-state path."""
    return chroma_dir / EMBEDDING_RESUME_STATE_FILENAME


def _embedding_resume_key(
    embedding_gen: EmbeddingGenerator,
    chunks: list[EmbeddingChunk],
) -> str:
    """Return a stable fingerprint for one rebuild scope and embedding config."""
    hasher = hashlib.sha256()
    for value in (
        embedding_gen.backend,
        embedding_gen.model_name,
        embedding_gen.query_prefix or "",
        embedding_gen.document_prefix or "",
    ):
        hasher.update(value.encode("utf-8"))
        hasher.update(b"\0")
    for chunk in chunks:
        hasher.update(chunk.chunk_id.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(hashlib.sha256(chunk.text.encode("utf-8")).digest())
        hasher.update(b"\0")
    return hasher.hexdigest()


def _make_embedding_progress(
    *,
    embedding_gen: EmbeddingGenerator,
    batch_size: int,
    completed_chunks: int,
    total_chunks: int,
    elapsed_seconds: float = 0.0,
    rolling_chunks_per_second: float = 0.0,
    overall_chunks_per_second: float = 0.0,
    eta_seconds: float = 0.0,
) -> dict[str, Any]:
    """Build a normalized embedding progress payload."""
    progress_fraction = 1.0 if total_chunks <= 0 else min(1.0, completed_chunks / total_chunks)
    return {
        "backend": embedding_gen.backend,
        "model": embedding_gen.model_name,
        "batch_size": batch_size,
        "completed_chunks": completed_chunks,
        "total_chunks": total_chunks,
        "remaining_chunks": max(total_chunks - completed_chunks, 0),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "rolling_chunks_per_second": round(rolling_chunks_per_second, 6),
        "overall_chunks_per_second": round(overall_chunks_per_second, 6),
        "eta_seconds": round(eta_seconds, 3),
        "progress_fraction": round(progress_fraction, 6),
    }


def _update_embedding_run_metadata(
    run_metadata: dict[str, Any] | None,
    *,
    progress: dict[str, Any],
    resume_state: dict[str, Any] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Persist the latest embedding progress snapshot in-memory and via callback."""
    payload: dict[str, Any]
    if run_metadata is None:
        payload = {"embedding_progress": dict(progress)}
        if resume_state is not None:
            payload["embedding_resume"] = dict(resume_state)
    else:
        run_metadata["embedding_progress"] = dict(progress)
        if resume_state is not None:
            run_metadata["embedding_resume"] = dict(resume_state)
        payload = dict(run_metadata)
    if progress_callback is not None:
        progress_callback(payload)


def _prepare_rebuild_resume_state(
    *,
    chroma_dir: Path,
    vector_store: VectorStore,
    embedding_gen: EmbeddingGenerator,
    chunks: list[EmbeddingChunk],
    batch_size: int,
    logger,
) -> tuple[Path, dict[str, Any], list[EmbeddingChunk]]:
    """Prepare or validate staged rebuild state for resumable embedding writes."""
    state_path = _embedding_resume_state_path(chroma_dir)
    existing_state = safe_read_json(state_path, default={}) or {}
    resume_key = _embedding_resume_key(embedding_gen, chunks)
    current_chunk_ids = {chunk.chunk_id for chunk in chunks}
    existing_count = vector_store.count()
    existing_ids: set[str] = set()
    reset_reason: str | None = None

    if existing_count > 0:
        if not isinstance(existing_state, dict) or existing_state.get("resume_key") != resume_key:
            reset_reason = "resume key mismatch"
        else:
            existing_ids = vector_store.get_all_chunk_ids()
            stale_ids = existing_ids - current_chunk_ids
            if stale_ids:
                reset_reason = f"{len(stale_ids)} staged chunk(s) are outside the current rebuild scope"

    if reset_reason is not None:
        logger.info("Resetting staged embedding rebuild cache (%s)", reset_reason)
        vector_store.clear()
        existing_ids = set()
        existing_count = 0

    reused_chunks = len(existing_ids)
    remaining_chunks = [chunk for chunk in chunks if chunk.chunk_id not in existing_ids]
    resume_state = {
        "status": "running",
        "stage_dir": str(chroma_dir),
        "state_path": str(state_path),
        "backend": embedding_gen.backend,
        "model": embedding_gen.model_name,
        "batch_size": batch_size,
        "resume_key": resume_key,
        "total_chunks": len(chunks),
        "completed_chunks": reused_chunks,
        "remaining_chunks": len(remaining_chunks),
        "reused_chunks": reused_chunks,
        "resumed": reused_chunks > 0,
        "started_at": existing_state.get("started_at", datetime.now().isoformat())
        if isinstance(existing_state, dict)
        else datetime.now().isoformat(),
        "last_persisted_at": datetime.now().isoformat(),
    }
    safe_write_json(state_path, resume_state)

    if reused_chunks:
        logger.info(
            "Resuming staged rebuild from %d/%d chunks already written in %s",
            reused_chunks,
            len(chunks),
            chroma_dir,
        )
    else:
        logger.info("Starting staged rebuild in %s", chroma_dir)

    return state_path, resume_state, remaining_chunks


def save_checkpoint(
    index_dir: Path,
    papers: list[dict],
    extractions: dict[str, dict],
    metadata: dict,
) -> None:
    """Save current papers, extractions, and metadata to disk."""
    papers_data = {
        "schema_version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "paper_count": len(papers),
        "papers": papers,
    }
    safe_write_json(index_dir / CHECKPOINT_PAPERS_FILENAME, papers_data)

    extractions_data = {
        "schema_version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "extraction_count": len(extractions),
        "extractions": extractions,
    }
    safe_write_json(index_dir / CHECKPOINT_EXTRACTIONS_FILENAME, extractions_data)
    safe_write_json(index_dir / CHECKPOINT_METADATA_FILENAME, metadata)


def get_platform_unset_command() -> str:
    """Return a platform-appropriate command to unset ANTHROPIC_API_KEY."""
    if sys.platform == "win32":
        return (
            "     PowerShell: $env:ANTHROPIC_API_KEY = $null\n"
            "     CMD:        set ANTHROPIC_API_KEY="
        )
    return "     export ANTHROPIC_API_KEY="


def verify_cli_authentication() -> bool:
    """Verify CLI authentication is set up for Anthropic CLI mode."""
    authenticator = ClaudeCliAuthenticator()
    is_auth, status = authenticator.is_authenticated()

    if is_auth:
        auth_method = authenticator.get_auth_method()
        print(f"\n[OK] CLI authentication verified: {auth_method}")
        if auth_method == "api_key":
            print("\n" + "=" * 60)
            print("WARNING: API KEY BILLING DETECTED")
            print("=" * 60)
            print("\nYou have ANTHROPIC_API_KEY set, which will incur API charges.")
            print("This is NOT using your Claude Max/Pro subscription.")
            print("\nTo use your subscription (free with plan):")
            print("  1. Unset ANTHROPIC_API_KEY before running:")
            print(get_platform_unset_command())
            print("  2. Or set CLAUDE_CODE_OAUTH_TOKEN from 'claude' login")
            print("")
            try:
                response = input("Continue with API billing? [y/N]: ")
                if response.lower() not in ("y", "yes"):
                    print("\nExiting. Unset ANTHROPIC_API_KEY to use subscription.")
                    sys.exit(0)
            except (EOFError, KeyboardInterrupt):
                print("\n\nCancelled.")
                sys.exit(1)
        return True

    print("\n" + "=" * 60)
    print("CLI Mode Authentication Required")
    print("=" * 60)
    print(f"\nStatus: {status}")
    print(authenticator.get_setup_instructions())

    try:
        response = input("\nWould you like to open Claude for authentication? [y/N]: ")
        if response.lower() in ("y", "yes"):
            try:
                authenticator.trigger_interactive_login()
                print("\nAfter completing authentication in the browser:")
                print("1. Run 'claude setup-token' in the Claude terminal")
                print("2. Set the CLAUDE_CODE_OAUTH_TOKEN environment variable")
                print("3. Re-run this script")
                sys.exit(0)
            except Exception as exc:
                print(f"\nFailed to launch Claude: {exc}")
                print("Please run 'claude' manually in a terminal to authenticate.")
                sys.exit(1)
        else:
            print("\nTo use CLI mode, you must authenticate first.")
            print("Alternatively, use '--mode api' with ANTHROPIC_API_KEY set.")
            sys.exit(1)
    except (EOFError, KeyboardInterrupt):
        print("\n\nAuthentication setup cancelled.")
        sys.exit(1)


def run_extraction(
    papers: list[PaperMetadata],
    extractor: SectionExtractor,
    index_dir: Path,
    existing_papers: dict[str, dict],
    existing_extractions: dict[str, dict],
    checkpoint_mgr: CheckpointManager,
    logger,
    text_snapshots: dict[str, dict[str, object]] | None = None,
    extraction_plans: dict[str, object] | None = None,
) -> ExtractionRunResult:
    """Run extraction and checkpoint progress for the provided paper scope."""
    paper_dicts = dict(existing_papers)
    for paper in papers:
        paper_dicts[paper.paper_id] = paper.to_index_dict()
    store = StructuredStore(index_dir)

    state = checkpoint_mgr.load()
    if not state:
        checkpoint_mgr.initialize(
            total_items=len(papers),
            metadata={"started_at": datetime.now().isoformat()},
        )

    pbar = tqdm(total=len(papers), desc="Extracting", unit="paper")
    results = []
    processed_ids: list[str] = []
    interrupted = False
    pause_requested = False
    try:
        extract_batch_kwargs = {}
        signature = inspect.signature(extractor.extract_batch)
        if "text_snapshots" in signature.parameters:
            extract_batch_kwargs["text_snapshots"] = text_snapshots
        if "extraction_plans" in signature.parameters:
            extract_batch_kwargs["extraction_plans"] = extraction_plans

        for result in extractor.extract_batch(papers, **extract_batch_kwargs):
            results.append(result)
            snapshot = result.text_snapshot or {}
            snapshot_text = snapshot.get("text") if isinstance(snapshot, dict) else None
            should_persist_snapshot = (
                isinstance(snapshot_text, str)
                and bool(snapshot_text)
                and bool(snapshot.get("should_persist", True))
            )
            if should_persist_snapshot:
                metadata = {
                    key: value
                    for key, value in snapshot.items()
                    if key not in {"text", "should_persist"}
                }
                store.save_text_snapshot(
                    result.paper_id,
                    snapshot_text,
                    metadata=metadata,
                    persist_manifest=False,
                )

            if result.success and result.extraction:
                checkpoint_mgr.complete_item(result.paper_id, success=True)
                existing_extractions[result.paper_id] = {
                    "paper_id": result.paper_id,
                    "extraction": result.extraction.to_index_dict(),
                    "timestamp": result.timestamp.isoformat(),
                    "model": result.model_used,
                    "duration": result.duration_seconds,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "extraction_method": result.extraction_method,
                    "planned_extraction_intent": result.planned_extraction_intent,
                    "planned_profile_key": result.planned_profile_key,
                    "actual_profile_key": result.actual_profile_key,
                    "extraction_routing_overridden": (result.extraction_routing_overridden),
                    "extraction_override_reason": result.extraction_override_reason,
                    "extraction_diagnostics": dict(result.extraction_diagnostics),
                }
            else:
                error = Exception(result.error) if result.error else None
                checkpoint_mgr.complete_item(result.paper_id, success=False, error=error)

            if result.extraction_routing_overridden and result.extraction_override_reason:
                logger.warning(
                    "Extraction routing override for %s: planned=%s actual=%s reason=%s",
                    result.paper_id,
                    result.planned_profile_key,
                    result.actual_profile_key,
                    result.extraction_override_reason,
                )
            if result.extraction_diagnostics:
                logger.warning(
                    "Extraction diagnostics for %s: %s",
                    result.paper_id,
                    "; ".join(
                        f"{tier}={message}"
                        for tier, message in sorted(result.extraction_diagnostics.items())
                    ),
                )

            if len(results) % 10 == 0:
                store.flush_fulltext_manifest()
                checkpoint_mgr.save()
                save_checkpoint(
                    index_dir,
                    list(paper_dicts.values()),
                    existing_extractions,
                    {
                        "last_updated": datetime.now().isoformat(),
                        "paper_count": len(paper_dicts),
                        "extraction_count": len(existing_extractions),
                        **checkpoint_mgr.get_progress(),
                    },
                )
            pbar.update(1)
            processed_ids.append(result.paper_id)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user, saving checkpoint...")
        interrupted = True
        checkpoint_mgr.save()
    except PauseRequested as exc:
        logger.warning("Pause requested, saving checkpoint... (%s)", exc)
        interrupted = True
        pause_requested = True
        checkpoint_mgr.save()
    finally:
        pbar.close()
        checkpoint_mgr.save()
        store.flush_fulltext_manifest()

    processed_id_set = set(processed_ids)
    pending_ids = [paper.paper_id for paper in papers if paper.paper_id not in processed_id_set]
    return ExtractionRunResult(
        paper_dicts=paper_dicts,
        extractions=existing_extractions,
        results=results,
        interrupted=interrupted,
        pause_requested=pause_requested,
        pending_ids=pending_ids,
    )


def run_gap_fill(
    results: list,
    existing_extractions: dict[str, dict],
    papers_to_extract: list[PaperMetadata],
    primary_provider: str,
    gap_fill_provider: str | None,
    threshold: float,
    mode: str,
    config: Config,
    index_dir: Path,
    paper_dicts: dict[str, dict],
    checkpoint_mgr: CheckpointManager,
    logger,
) -> None:
    """Run secondary-provider gap filling for low-coverage extractions."""
    from src.analysis.llm_council import CouncilConfig, LLMCouncil, ProviderConfig

    low_coverage = []
    paper_lookup = {paper.paper_id: paper for paper in papers_to_extract}

    for result in results:
        if not result.success or not result.extraction:
            continue
        extraction = result.extraction
        if not isinstance(extraction, SemanticAnalysis):
            continue
        if extraction.dimension_coverage < threshold:
            paper = paper_lookup.get(result.paper_id)
            if paper:
                low_coverage.append((paper, extraction, result))

    if not low_coverage:
        logger.info(
            "Gap-fill: all papers above %.0f%% coverage, nothing to do",
            threshold * 100,
        )
        return

    logger.info(
        "Gap-fill: %d papers below %.0f%% coverage, running secondary provider",
        len(low_coverage),
        threshold * 100,
    )

    secondary = gap_fill_provider or ("anthropic" if primary_provider == "openai" else "openai")
    council_config = CouncilConfig(
        providers=[
            ProviderConfig(name=primary_provider, weight=1.2, timeout=600, mode=mode),
            ProviderConfig(name=secondary, weight=1.0, timeout=600, mode=mode),
        ],
        aggregation_strategy="quality_weighted",
    )
    council = LLMCouncil(council_config)

    pdf_extractor = PDFExtractor(
        enable_ocr=config.processing.ocr_enabled or config.processing.ocr_on_fail,
    )
    text_cleaner = TextCleaner()
    store = StructuredStore(index_dir)

    filled_total = 0
    for idx, (paper, extraction, result) in enumerate(low_coverage, 1):
        logger.info(
            "Gap-fill %d/%d: %s (coverage: %.0f%%)",
            idx,
            len(low_coverage),
            paper.paper_id,
            extraction.dimension_coverage * 100,
        )

        if not paper.pdf_path or not paper.pdf_path.exists():
            logger.warning("Gap-fill: no PDF for %s, skipping", paper.paper_id)
            continue

        try:
            snapshot = store.load_text_snapshot(paper.paper_id)
            if snapshot and _snapshot_matches_paper(snapshot, paper):
                text = str(snapshot["text"])
            else:
                raw_text = pdf_extractor.extract_text(paper.pdf_path)
                if raw_text:
                    text = text_cleaner.clean(raw_text)
                    store.save_text_snapshot(
                        paper.paper_id,
                        text,
                        metadata={
                            "paper_id": paper.paper_id,
                            "source": "gap_fill_refresh",
                            "extraction_method": "pymupdf",
                            "is_cleaned": True,
                            "is_truncated_for_llm": False,
                            "pdf_path": str(paper.pdf_path),
                            "pdf_size": int(paper.pdf_path.stat().st_size),
                            "pdf_mtime_ns": int(paper.pdf_path.stat().st_mtime_ns),
                        },
                        persist_manifest=False,
                    )
                else:
                    logger.warning("Gap-fill: empty text for %s", paper.paper_id)
                    continue
            text = text_cleaner.truncate_for_llm(text)
        except Exception as exc:
            logger.warning(
                "Gap-fill: text extraction failed for %s: %s",
                paper.paper_id,
                exc,
            )
            continue

        authors = ", ".join(
            f"{author.last_name}, {author.first_name}" for author in (paper.authors or [])
        )

        try:
            merged, gaps_filled = council.fill_gaps(
                analysis=extraction,
                paper_id=paper.paper_id,
                title=paper.title,
                authors=authors,
                year=getattr(paper, "publication_year", None),
                item_type=paper.item_type,
                text=text[:100000],
                gap_provider=ProviderConfig(
                    name=secondary,
                    weight=1.0,
                    timeout=600,
                    mode=mode,
                ),
            )
            if gaps_filled > 0:
                filled_total += gaps_filled
                logger.info(
                    "Gap-fill: %s filled %d dimensions (%.0f%% -> %.0f%%)",
                    paper.paper_id,
                    gaps_filled,
                    extraction.dimension_coverage * 100,
                    merged.dimension_coverage * 100,
                )
                existing_extractions[paper.paper_id] = {
                    "paper_id": paper.paper_id,
                    "extraction": merged.to_index_dict(),
                    "timestamp": datetime.now().isoformat(),
                    "model": result.model_used,
                    "duration": result.duration_seconds,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "gap_filled_by": secondary,
                    "gap_filled_count": gaps_filled,
                }
            else:
                logger.info(
                    "Gap-fill: %s -- secondary provider found no new content",
                    paper.paper_id,
                )
        except Exception as exc:
            logger.error("Gap-fill failed for %s: %s", paper.paper_id, exc)

    if filled_total > 0:
        store.flush_fulltext_manifest()
        save_checkpoint(
            index_dir,
            list(paper_dicts.values()),
            existing_extractions,
            {
                "last_updated": datetime.now().isoformat(),
                "paper_count": len(paper_dicts),
                "extraction_count": len(existing_extractions),
                "gap_fill_provider": secondary,
                "gap_filled_total": filled_total,
                **checkpoint_mgr.get_progress(),
            },
        )

    store.flush_fulltext_manifest()
    logger.info(
        "Gap-fill complete: %d dimensions filled across %d papers",
        filled_total,
        len(low_coverage),
    )


def write_skipped_report(
    results: list,
    paper_dicts: dict[str, dict],
    output_dir: Path,
    logger,
) -> tuple[Path | None, Path | None]:
    """Write a skipped-items report for extraction failures that imply skipping."""
    skip_prefixes = (
        "Insufficient text content",
        "Likely non-publication",
        "No PDF available",
    )

    skipped_items = []
    for result in results:
        if result.success or not result.error:
            continue
        if not any(result.error.startswith(prefix) for prefix in skip_prefixes):
            continue
        paper = paper_dicts.get(result.paper_id, {})
        skipped_items.append(
            {
                "paper_id": result.paper_id,
                "title": paper.get("title"),
                "zotero_key": paper.get("zotero_key"),
                "item_type": paper.get("item_type"),
                "publication_year": paper.get("publication_year"),
                "pdf_path": paper.get("pdf_path"),
                "reason": result.error,
                "model": result.model_used,
                "timestamp": result.timestamp.isoformat(),
            }
        )

    if not skipped_items:
        return None, None

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"skipped_items_{timestamp}.json"
    csv_path = output_dir / f"skipped_items_{timestamp}.csv"

    safe_write_json(
        json_path,
        {
            "generated_at": datetime.now().isoformat(),
            "count": len(skipped_items),
            "items": skipped_items,
        },
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "paper_id",
                "title",
                "zotero_key",
                "item_type",
                "publication_year",
                "pdf_path",
                "reason",
                "model",
                "timestamp",
            ],
        )
        writer.writeheader()
        writer.writerows(skipped_items)

    logger.info(
        "Skipped items report: %d entries (%s, %s)",
        len(skipped_items),
        json_path.name,
        csv_path.name,
    )
    return json_path, csv_path


def _normalize_extractions(
    extractions: dict[str, dict | SemanticAnalysis],
) -> dict[str, SemanticAnalysis]:
    normalized: dict[str, SemanticAnalysis] = {}
    for paper_id, ext_data in extractions.items():
        if isinstance(ext_data, SemanticAnalysis):
            normalized[paper_id] = ext_data
            continue
        inner = ext_data.get("extraction", ext_data) if isinstance(ext_data, dict) else ext_data
        normalized[paper_id] = SemanticAnalysis(**inner)
    return normalized


def run_embedding_generation(
    papers: list[PaperMetadata],
    extractions: dict[str, dict | SemanticAnalysis],
    index_dir: Path,
    embedding_model: str,
    rebuild: bool,
    logger,
    embedding_backend: str = "sentence-transformers",
    embedding_device: str | None = None,
    ollama_base_url: str = "http://localhost:11434",
    ollama_concurrency: int = 1,
    query_prefix: str | None = None,
    document_prefix: str | None = None,
    embedding_batch_size: int | str = "auto",
    raptor_summaries: dict[str, RaptorSummaries] | None = None,
    delete_paper_ids: list[str] | None = None,
    vector_store_dir: Path | None = None,
    run_metadata: dict | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Generate embeddings and populate the vector store for the scoped papers."""
    chroma_dir = vector_store_dir or (index_dir / "chroma")
    logger.info("Initializing embedding generator...")
    embedding_gen = EmbeddingGenerator(
        model_name=embedding_model,
        device=embedding_device,
        backend=embedding_backend,
        ollama_base_url=ollama_base_url,
        ollama_concurrency=ollama_concurrency,
        query_prefix=query_prefix,
        document_prefix=document_prefix,
    )

    logger.info("Initializing vector store...")
    vector_store = VectorStore(chroma_dir)

    try:
        normalized_extractions = _normalize_extractions(extractions)
        papers_with_extractions = [
            paper for paper in papers if paper.paper_id in normalized_extractions
        ]
        scope_paper_ids = sorted({paper.paper_id for paper in papers_with_extractions})
        delete_ids = sorted(set(delete_paper_ids or []))

        if not papers_with_extractions and not rebuild and not delete_ids:
            logger.warning("No papers with extractions found for embedding")
            return 0

        logger.info("Generating embeddings for %d papers...", len(papers_with_extractions))

        all_chunks = []
        for paper in tqdm(papers_with_extractions, desc="Creating chunks"):
            extraction = normalized_extractions.get(paper.paper_id)
            if not extraction:
                continue
            chunks = embedding_gen.create_chunks(
                paper,
                extraction,
                raptor_summaries=raptor_summaries.get(paper.paper_id) if raptor_summaries else None,
            )
            all_chunks.extend(chunks)

        logger.info("Created %d chunks", len(all_chunks))
        if not all_chunks and not rebuild and not delete_ids:
            return 0

        resolved_batch_size = None
        if all_chunks:
            resolved_batch_size = embedding_gen.resolve_batch_size(
                embedding_batch_size,
                texts=[chunk.text for chunk in all_chunks],
            )
            logger.info(
                "Generating embeddings with batch size %d...",
                resolved_batch_size,
            )

        if run_metadata is not None:
            run_metadata["embedding_batch_size_setting"] = embedding_batch_size
            run_metadata["embedding_batch_size_resolved"] = resolved_batch_size

        if all_chunks and resolved_batch_size is not None:
            _update_embedding_run_metadata(
                run_metadata,
                progress=_make_embedding_progress(
                    embedding_gen=embedding_gen,
                    batch_size=resolved_batch_size,
                    completed_chunks=0,
                    total_chunks=len(all_chunks),
                ),
                progress_callback=progress_callback,
            )

        if rebuild:
            if not all_chunks:
                logger.info("Clearing existing embeddings...")
                vector_store.clear()
                resume_state_path = _embedding_resume_state_path(chroma_dir)
                if resume_state_path.exists():
                    resume_state_path.unlink()
                return 0

            logger.info("Streaming embeddings to staged vector store...")
            state_path, resume_state, remaining_chunks = _prepare_rebuild_resume_state(
                chroma_dir=chroma_dir,
                vector_store=vector_store,
                embedding_gen=embedding_gen,
                chunks=all_chunks,
                batch_size=resolved_batch_size,
                logger=logger,
            )
            _update_embedding_run_metadata(
                run_metadata,
                progress=_make_embedding_progress(
                    embedding_gen=embedding_gen,
                    batch_size=resolved_batch_size,
                    completed_chunks=resume_state["completed_chunks"],
                    total_chunks=len(all_chunks),
                ),
                resume_state=resume_state,
                progress_callback=progress_callback,
            )

            buffered_chunks: list[EmbeddingChunk] = []
            persisted_chunks = 0

            def flush_buffer() -> None:
                nonlocal buffered_chunks
                nonlocal persisted_chunks
                if not buffered_chunks:
                    return
                persisted_chunks += vector_store.add_chunks(
                    buffered_chunks,
                    batch_size=100,
                    log_progress=False,
                )
                resume_state["completed_chunks"] = resume_state["reused_chunks"] + persisted_chunks
                resume_state["remaining_chunks"] = max(
                    len(all_chunks) - resume_state["completed_chunks"],
                    0,
                )
                resume_state["last_persisted_at"] = datetime.now().isoformat()
                safe_write_json(state_path, resume_state)
                buffered_chunks = []

            def handle_progress(snapshot: dict[str, Any]) -> None:
                progress = dict(snapshot)
                progress["total_chunks"] = len(all_chunks)
                progress["completed_chunks"] = resume_state["reused_chunks"] + progress["completed_chunks"]
                progress["remaining_chunks"] = max(
                    len(all_chunks) - progress["completed_chunks"],
                    0,
                )
                progress["progress_fraction"] = round(
                    progress["completed_chunks"] / len(all_chunks),
                    6,
                )
                _update_embedding_run_metadata(
                    run_metadata,
                    progress=progress,
                    resume_state=resume_state,
                    progress_callback=progress_callback,
                )

            for batch in embedding_gen.iter_embedded_batches(
                remaining_chunks,
                batch_size=resolved_batch_size,
                progress_callback=handle_progress,
            ):
                buffered_chunks.extend(batch)
                if len(buffered_chunks) >= EMBEDDING_STREAM_WRITE_CHUNK_COUNT:
                    flush_buffer()
            flush_buffer()

            added = vector_store.count()
            resume_state["completed_chunks"] = len(all_chunks)
            resume_state["remaining_chunks"] = 0
            resume_state["status"] = "complete"
            resume_state["last_persisted_at"] = datetime.now().isoformat()
            final_progress = (
                dict(run_metadata.get("embedding_progress", {}))
                if isinstance(run_metadata, dict)
                else {}
            )
            if not final_progress:
                final_progress = _make_embedding_progress(
                    embedding_gen=embedding_gen,
                    batch_size=resolved_batch_size,
                    completed_chunks=len(all_chunks),
                    total_chunks=len(all_chunks),
                )
            else:
                final_progress["completed_chunks"] = len(all_chunks)
                final_progress["remaining_chunks"] = 0
                final_progress["progress_fraction"] = 1.0
            _update_embedding_run_metadata(
                run_metadata,
                progress=final_progress,
                resume_state=resume_state,
                progress_callback=progress_callback,
            )
            if state_path.exists():
                state_path.unlink()
            logger.info("Added %d chunks to vector store", added)
            return added

        if all_chunks:
            def handle_progress(snapshot: dict[str, Any]) -> None:
                progress = dict(snapshot)
                progress["remaining_chunks"] = max(
                    len(all_chunks) - progress["completed_chunks"],
                    0,
                )
                _update_embedding_run_metadata(
                    run_metadata,
                    progress=progress,
                    progress_callback=progress_callback,
                )

            all_chunks = embedding_gen.generate_embeddings(
                all_chunks,
                batch_size=resolved_batch_size,
                progress_callback=handle_progress,
            )

        logger.info("Writing embeddings to vector store...")
        added = vector_store.replace_papers(
            chunks=all_chunks,
            scope_paper_ids=scope_paper_ids,
            delete_paper_ids=delete_ids,
            batch_size=100,
        )
        logger.info("Added %d chunks to vector store", added)
        return added
    finally:
        vector_store.close()


def generate_summary(index_dir: Path, logger) -> dict:
    """Generate and persist summary statistics."""
    logger.info("Generating summary statistics...")
    store = StructuredStore(index_dir)
    summary = store.save_summary()
    logger.info("Summary saved with %d papers", summary.get("total_papers", 0))
    return summary


def compute_similarity_pairs(
    index_dir: Path,
    embedding_model: str,
    top_n: int = 20,
    embedding_backend: str = "sentence-transformers",
    embedding_device: str | None = None,
    ollama_base_url: str = "http://localhost:11434",
    ollama_concurrency: int = 1,
    query_prefix: str | None = None,
    document_prefix: str | None = None,
    embedding_batch_size: int | str | None = None,
    logger=None,
) -> int:
    """Compute pairwise paper similarity from `raptor_overview` embeddings."""
    import numpy as np

    chroma_dir = index_dir / "chroma"
    store = StructuredStore(index_dir)
    with VectorStore(chroma_dir) as vector_store:
        try:
            results = vector_store.collection.get(
                where={"chunk_type": "raptor_overview"},
                include=["embeddings", "metadatas"],
            )
            if logger:
                logger.info("Loaded stored raptor_overview embeddings from vector store")
            paper_ids = []
            embeddings = []
            for idx, _chunk_id in enumerate(results["ids"]):
                meta = results["metadatas"][idx] if results["metadatas"] is not None else {}
                paper_id = meta.get("paper_id", "")
                embedding = (
                    results["embeddings"][idx] if results["embeddings"] is not None else None
                )
                if paper_id and embedding is not None:
                    paper_ids.append(paper_id)
                    embeddings.append(embedding)
        except Exception as exc:
            if logger:
                logger.warning(
                    "Bulk raptor_overview embedding fetch failed (%s); "
                    "re-embedding stored overview texts for similarity rebuild",
                    exc,
                )
            results = vector_store.collection.get(
                where={"chunk_type": "raptor_overview"},
                include=["documents", "metadatas"],
            )
            if not results["ids"]:
                paper_ids = []
                embeddings = []
            else:
                overview_chunks: list[EmbeddingChunk] = []
                for idx, chunk_id in enumerate(results["ids"]):
                    meta = results["metadatas"][idx] if results["metadatas"] is not None else {}
                    paper_id = meta.get("paper_id", "")
                    document = (
                        results["documents"][idx] if results["documents"] is not None else None
                    )
                    if paper_id and document:
                        overview_chunks.append(
                            EmbeddingChunk(
                                paper_id=paper_id,
                                chunk_id=chunk_id,
                                chunk_type="raptor_overview",
                                text=document,
                            )
                        )

                if overview_chunks:
                    embedding_gen = EmbeddingGenerator(
                        model_name=embedding_model,
                        device=embedding_device,
                        backend=embedding_backend,
                        ollama_base_url=ollama_base_url,
                        ollama_concurrency=ollama_concurrency,
                        query_prefix=query_prefix,
                        document_prefix=document_prefix,
                    )
                    resolved_batch_size = embedding_gen.resolve_batch_size(
                        embedding_batch_size,
                        texts=[chunk.text for chunk in overview_chunks],
                    )
                    if logger:
                        logger.info(
                            "Re-embedding %d raptor_overview chunks with batch size %d",
                            len(overview_chunks),
                            resolved_batch_size,
                        )
                    overview_chunks = embedding_gen.generate_embeddings(
                        overview_chunks,
                        batch_size=resolved_batch_size,
                        show_progress=False,
                    )
                    paper_ids = [chunk.paper_id for chunk in overview_chunks if chunk.embedding]
                    embeddings = [chunk.embedding for chunk in overview_chunks if chunk.embedding]
                else:
                    paper_ids = []
                    embeddings = []

    if not paper_ids:
        if logger:
            logger.warning("No raptor_overview chunks found for similarity computation")
        return 0

    if len(embeddings) < 2:
        if logger:
            logger.info("Fewer than 2 papers with raptor_overview -- skipping similarity")
        return 0

    if logger:
        logger.info("Computing similarity for %d papers...", len(embeddings))

    emb_matrix = np.array(embeddings)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_normalized = emb_matrix / norms
    similarity_matrix = emb_normalized @ emb_normalized.T

    pairs: dict[str, list[dict]] = {}
    for idx, source_id in enumerate(paper_ids):
        scores = similarity_matrix[idx].copy()
        scores[idx] = -1.0
        top_indices = np.argsort(scores)[::-1][:top_n]
        similar_list = []
        for similar_idx in top_indices:
            sim_score = float(scores[similar_idx])
            if sim_score <= 0:
                break
            similar_list.append(
                {
                    "similar_paper_id": paper_ids[similar_idx],
                    "similarity_score": round(sim_score, 4),
                }
            )
        if similar_list:
            pairs[source_id] = similar_list

    store.save_similarity_pairs(
        pairs,
        metadata={
            "embedding_model": embedding_model,
            "top_n_per_paper": top_n,
            "computation_method": "cosine_similarity",
        },
    )
    total_pairs = sum(len(value) for value in pairs.values())
    if logger:
        logger.info(
            "Computed %d similarity pairs for %d papers (top %d each)",
            total_pairs,
            len(pairs),
            top_n,
        )
    return total_pairs


def configure_extraction_runtime(
    args,
    config: Config,
    logger,
) -> tuple[str, str, Path, int, bool]:
    """Apply CLI overrides to extraction config and return runtime settings."""
    provider = getattr(args, "provider", None) or config.extraction.provider
    config.extraction.apply_provider(provider)
    if getattr(args, "mode", None):
        config.extraction.mode = args.mode
    if getattr(args, "model", None):
        config.extraction.model = args.model

    mode = config.extraction.mode
    use_cache = config.extraction.use_cache and not getattr(args, "no_cache", False)
    parallel_workers = (
        args.parallel if getattr(args, "parallel", None) else config.extraction.parallel_workers
    )
    cache_dir = config.get_cache_path() / "pdf_text"

    logger.info("Using LLM provider: %s", provider)
    if getattr(args, "use_subscription", False) and mode == "cli" and provider == "anthropic":
        if os.environ.get("ANTHROPIC_API_KEY"):
            logger.info("--use-subscription: Temporarily unsetting ANTHROPIC_API_KEY")
            os.environ.pop("ANTHROPIC_API_KEY", None)

    return provider, mode, cache_dir, parallel_workers, use_cache


def load_reusable_text_snapshots(
    store: StructuredStore,
    papers: list[PaperMetadata],
    *,
    refresh_text: bool = False,
) -> dict[str, dict[str, object]]:
    """Load canonical text snapshots that match the current paper source state."""

    if refresh_text:
        return {}

    snapshots: dict[str, dict[str, object]] = {}
    for paper in papers:
        snapshot = store.load_text_snapshot(paper.paper_id)
        if snapshot is None:
            continue
        if _snapshot_matches_paper(snapshot, paper):
            snapshots[paper.paper_id] = snapshot
    return snapshots


def _snapshot_matches_paper(
    snapshot: dict[str, object],
    paper: PaperMetadata,
) -> bool:
    """Return whether a stored text snapshot still matches the current paper source."""

    if paper.source_path and Path(paper.source_path).exists():
        stored_source_path = snapshot.get("source_path")
        current_source_path = str(paper.source_path)
        if stored_source_path and stored_source_path != current_source_path:
            return False

        try:
            source_stat = paper.source_path.stat()
        except OSError:
            return False

        stored_source_size = snapshot.get("source_size")
        stored_source_mtime_ns = snapshot.get("source_mtime_ns")
        if stored_source_size is None or stored_source_mtime_ns is None:
            return True

        return int(stored_source_size) == int(source_stat.st_size) and int(
            stored_source_mtime_ns
        ) == int(source_stat.st_mtime_ns)

    if not paper.pdf_path or not paper.pdf_path.exists():
        return True

    stored_path = snapshot.get("pdf_path")
    current_path = str(paper.pdf_path)
    if stored_path and stored_path != current_path:
        return False

    try:
        stat = paper.pdf_path.stat()
    except OSError:
        return False

    stored_size = snapshot.get("pdf_size")
    stored_mtime_ns = snapshot.get("pdf_mtime_ns")
    if stored_size is None or stored_mtime_ns is None:
        return True

    return int(stored_size) == int(stat.st_size) and int(stored_mtime_ns) == int(stat.st_mtime_ns)


def build_section_extractor(
    args,
    config: Config,
    cache_dir: Path,
    mode: str,
    parallel_workers: int,
    use_cache: bool,
    run_control_path: Path | None = None,
    planned_profile: HybridProfileSpec | None = None,
) -> SectionExtractor:
    """Build the configured section extractor for a run."""
    base_hybrid_config = build_hybrid_config_from_processing(config.processing)
    effective_mode = config.processing.opendataloader_mode
    effective_hybrid_config = base_hybrid_config
    if planned_profile is not None:
        effective_mode = planned_profile.mode
        effective_hybrid_config = planned_profile.build_hybrid_config(
            base_hybrid_config,
            config.processing,
        )
    return SectionExtractor(
        cache_dir=cache_dir,
        provider=config.extraction.provider,
        mode=mode,
        model=config.extraction.model,
        max_tokens=config.extraction.max_tokens,
        timeout=config.extraction.timeout,
        min_text_length=config.processing.min_text_length,
        ocr_on_fail=config.processing.ocr_on_fail,
        skip_non_publications=config.processing.skip_non_publications,
        min_publication_words=config.processing.min_publication_words,
        min_publication_pages=config.processing.min_publication_pages,
        min_section_hits=config.processing.min_section_hits,
        ocr_enabled=config.processing.ocr_enabled,
        ocr_config=config.processing.ocr_config,
        arxiv_enabled=config.processing.arxiv_enabled,
        opendataloader_enabled=config.processing.opendataloader_enabled,
        opendataloader_mode=effective_mode,
        opendataloader_hybrid_config=effective_hybrid_config,
        opendataloader_hybrid_fallback=(config.processing.opendataloader_hybrid_fallback),
        marker_enabled=config.processing.marker_enabled,
        use_cache=use_cache,
        parallel_workers=parallel_workers,
        reasoning_effort=config.extraction.reasoning_effort,
        effort=config.extraction.effort,
        run_control_path=run_control_path,
    )


def update_classification_extraction_methods(
    class_index: ClassificationIndex,
    results: list,
) -> None:
    """Copy extraction method metadata from extraction results into classification state."""
    for result in results:
        if result.success and result.extraction_method and result.paper_id in class_index.papers:
            class_index.papers[result.paper_id].extraction_method = result.extraction_method
