"""Shared indexing pipeline steps used by build and maintenance scripts."""

from __future__ import annotations

import csv
import os
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from src.analysis.classification_store import ClassificationIndex
from src.analysis.cli_executor import ClaudeCliAuthenticator
from src.analysis.raptor import RaptorSummaries
from src.analysis.schemas import SemanticAnalysis
from src.analysis.section_extractor import SectionExtractor
from src.config import Config
from src.extraction.pdf_extractor import PDFExtractor
from src.extraction.text_cleaner import TextCleaner
from src.indexing.embeddings import EmbeddingGenerator
from src.indexing.structured_store import StructuredStore
from src.indexing.vector_store import VectorStore
from src.utils.checkpoint import CheckpointManager
from src.utils.file_utils import safe_write_json
from src.zotero.models import PaperMetadata

CHECKPOINT_PAPERS_FILENAME = "papers.checkpoint.json"
CHECKPOINT_EXTRACTIONS_FILENAME = "semantic_analyses.checkpoint.json"
CHECKPOINT_METADATA_FILENAME = "metadata.checkpoint.json"


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
) -> tuple[dict[str, dict], dict[str, dict], list]:
    """Run extraction and checkpoint progress for the provided paper scope."""
    paper_dicts = dict(existing_papers)
    for paper in papers:
        paper_dicts[paper.paper_id] = paper.to_index_dict()

    state = checkpoint_mgr.load()
    if not state:
        checkpoint_mgr.initialize(
            total_items=len(papers),
            metadata={"started_at": datetime.now().isoformat()},
        )

    pbar = tqdm(total=len(papers), desc="Extracting", unit="paper")
    results = []
    try:
        for result in extractor.extract_batch(papers):
            results.append(result)
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
                }
            else:
                error = Exception(result.error) if result.error else None
                checkpoint_mgr.complete_item(result.paper_id, success=False, error=error)

            if len(results) % 10 == 0:
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
    except KeyboardInterrupt:
        logger.warning("Interrupted by user, saving checkpoint...")
        checkpoint_mgr.save()
    finally:
        pbar.close()
        checkpoint_mgr.save()

    return paper_dicts, existing_extractions, results


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

    secondary = gap_fill_provider or (
        "anthropic" if primary_provider == "openai" else "openai"
    )
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
            raw_text = pdf_extractor.extract_text(paper.pdf_path)
            if raw_text:
                text = text_cleaner.clean(raw_text)
                text = text_cleaner.truncate_for_llm(text)
            else:
                logger.warning("Gap-fill: empty text for %s", paper.paper_id)
                continue
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
                    name=secondary, weight=1.0, timeout=600, mode=mode,
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
        skipped_items.append({
            "paper_id": result.paper_id,
            "title": paper.get("title"),
            "zotero_key": paper.get("zotero_key"),
            "item_type": paper.get("item_type"),
            "publication_year": paper.get("publication_year"),
            "pdf_path": paper.get("pdf_path"),
            "reason": result.error,
            "model": result.model_used,
            "timestamp": result.timestamp.isoformat(),
        })

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
    ollama_base_url: str = "http://localhost:11434",
    query_prefix: str | None = None,
    document_prefix: str | None = None,
    raptor_summaries: dict[str, RaptorSummaries] | None = None,
    delete_paper_ids: list[str] | None = None,
    vector_store_dir: Path | None = None,
) -> int:
    """Generate embeddings and populate the vector store for the scoped papers."""
    chroma_dir = vector_store_dir or (index_dir / "chroma")
    logger.info("Initializing embedding generator...")
    embedding_gen = EmbeddingGenerator(
        model_name=embedding_model,
        backend=embedding_backend,
        ollama_base_url=ollama_base_url,
        query_prefix=query_prefix,
        document_prefix=document_prefix,
    )

    logger.info("Initializing vector store...")
    vector_store = VectorStore(chroma_dir)
    embedding_batch_size = 16 if embedding_backend == "ollama" else 32

    if rebuild:
        logger.info("Clearing existing embeddings...")
        vector_store.clear()

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

    if all_chunks:
        logger.info(
            "Generating embeddings with batch size %d...",
            embedding_batch_size,
        )
        all_chunks = embedding_gen.generate_embeddings(
            all_chunks,
            batch_size=embedding_batch_size,
        )

    logger.info("Writing embeddings to vector store...")
    if rebuild:
        added = vector_store.add_chunks(all_chunks, batch_size=100)
    else:
        added = vector_store.replace_papers(
            chunks=all_chunks,
            scope_paper_ids=scope_paper_ids,
            delete_paper_ids=delete_ids,
            batch_size=100,
        )
    logger.info("Added %d chunks to vector store", added)
    return added


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
    logger=None,
) -> int:
    """Compute pairwise paper similarity from `raptor_overview` embeddings."""
    import numpy as np

    chroma_dir = index_dir / "chroma"
    store = StructuredStore(index_dir)
    vector_store = VectorStore(chroma_dir)

    results = vector_store.collection.get(
        where={"chunk_type": "raptor_overview"},
        include=["embeddings", "metadatas"],
    )
    if not results["ids"]:
        if logger:
            logger.warning("No raptor_overview chunks found for similarity computation")
        return 0

    paper_ids = []
    embeddings = []
    for idx, _chunk_id in enumerate(results["ids"]):
        meta = results["metadatas"][idx] if results["metadatas"] is not None else {}
        paper_id = meta.get("paper_id", "")
        embedding = results["embeddings"][idx] if results["embeddings"] is not None else None
        if paper_id and embedding is not None:
            paper_ids.append(paper_id)
            embeddings.append(embedding)

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
            similar_list.append({
                "similar_paper_id": paper_ids[similar_idx],
                "similarity_score": round(sim_score, 4),
            })
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
    resolved_model = config.extraction.model
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


def build_section_extractor(
    args,
    config: Config,
    cache_dir: Path,
    mode: str,
    parallel_workers: int,
    use_cache: bool,
) -> SectionExtractor:
    """Build the configured section extractor for a run."""
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
        use_cache=use_cache,
        parallel_workers=parallel_workers,
        reasoning_effort=config.extraction.reasoning_effort,
        effort=config.extraction.effort,
    )


def update_classification_extraction_methods(
    class_index: ClassificationIndex,
    results: list,
) -> None:
    """Copy extraction method metadata from extraction results into classification state."""
    for result in results:
        if result.success and result.extraction_method and result.paper_id in class_index.papers:
            class_index.papers[result.paper_id].extraction_method = result.extraction_method
