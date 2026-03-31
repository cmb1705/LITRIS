#!/usr/bin/env python
"""Build the literature review index from a reference library."""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tqdm import tqdm

from src.analysis.classification_store import (
    ClassificationIndex,
    ClassificationStore,
)
from src.analysis.cli_executor import ClaudeCliAuthenticator
from src.analysis.schemas import SemanticAnalysis
from src.analysis.section_extractor import SectionExtractor
from src.config import Config, parse_embedding_batch_size_setting
from src.extraction.pdf_extractor import PDFExtractor
from src.extraction.text_cleaner import TextCleaner
from src.indexing.embeddings import EmbeddingGenerator
from src.indexing.orchestrator import IndexOrchestrator
from src.indexing.structured_store import StructuredStore
from src.indexing.vector_store import VectorStore
from src.references.factory import create_reference_db
from src.utils.checkpoint import CheckpointManager
from src.utils.deduplication import (
    analyze_doi_overlap,
    extract_existing_dois,
    filter_by_doi,
)
from src.utils.file_utils import safe_read_json, safe_write_json
from src.utils.logging_config import LogContext, setup_logging
from src.zotero.models import PaperMetadata


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build literature review index from your reference library"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of papers to process",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        default=None,
        help="LLM provider: anthropic (Claude), openai (GPT-5.4)",
    )
    parser.add_argument(
        "--mode",
        choices=["api", "cli", "batch_api"],
        default=None,
        help="Extraction mode: api (direct API), cli (Claude CLI or Codex CLI), batch_api (Anthropic batch)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (e.g., gpt-5.4, claude-opus-4-6)",
    )
    parser.add_argument(
        "--summary-model",
        type=str,
        default=None,
        help="Override model for summary extraction fields",
    )
    parser.add_argument(
        "--methodology-model",
        type=str,
        default=None,
        help="Override model for methodology extraction fields",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous checkpoint",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without running",
    )
    parser.add_argument(
        "--estimate-cost",
        action="store_true",
        help="Estimate cost without running extraction",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip LLM extraction (use existing extractions)",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation and vector store",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=parse_embedding_batch_size_setting,
        default=None,
        metavar="N|auto",
        help="Override embedding batch size (positive integer or auto)",
    )
    parser.add_argument(
        "--rebuild-embeddings",
        action="store_true",
        help="Deprecated alias for --sync-mode full; with --paper, refresh only the targeted papers",
    )
    parser.add_argument(
        "--sync-mode",
        choices=["auto", "full", "update"],
        default="auto",
        help="Sync strategy: auto (default), full rebuild, or update only",
    )
    parser.add_argument(
        "--explain-plan",
        action="store_true",
        help="Print the resolved sync plan before running",
    )
    parser.add_argument(
        "--skip-similarity",
        action="store_true",
        help="Skip pre-computed similarity pair generation",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry previously failed extractions",
    )
    parser.add_argument(
        "--skip-paper",
        type=str,
        action="append",
        default=[],
        metavar="PAPER_ID",
        help="Skip specific paper ID (can be used multiple times)",
    )
    parser.add_argument(
        "--paper",
        type=str,
        action="append",
        default=[],
        metavar="PAPER_ID",
        help="Process only this paper (zotero_key or paper_id, repeatable)",
    )
    parser.add_argument(
        "--show-failed",
        action="store_true",
        help="Show list of failed papers from previous run",
    )
    parser.add_argument(
        "--show-skipped",
        action="store_true",
        help="Show list of papers without PDFs (cannot be extracted)",
    )
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Reset checkpoint to start fresh",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel extraction workers (CLI mode only, default: 1)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable extraction caching (re-extract all papers)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear extraction cache before running",
    )
    parser.add_argument(
        "--use-subscription",
        action="store_true",
        help="Use Claude subscription (Max/Pro) instead of API billing",
    )
    parser.add_argument(
        "--dedupe-by-doi",
        action="store_true",
        help="Skip papers with DOIs already in index (for cross-database deduplication)",
    )
    parser.add_argument(
        "--show-doi-overlap",
        action="store_true",
        help="Analyze DOI overlap without processing (useful before switching databases)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Filter to papers in collections matching this substring",
    )
    parser.add_argument(
        "--classify-only",
        action="store_true",
        help="Run classification pre-pass only (no extraction). "
        "Builds/updates classification_index.json.",
    )
    parser.add_argument(
        "--index-all",
        action="store_true",
        help="Extract all papers regardless of classification. "
        "Default behavior filters to academic content only.",
    )
    parser.add_argument(
        "--reclassify",
        action="store_true",
        help="Force re-classification of all papers (use with --classify-only).",
    )
    parser.add_argument(
        "--gap-fill",
        action="store_true",
        help="After extraction, fill missing dimensions using a secondary provider. "
        "Only applies to papers extracted in this run. Uses the opposite provider "
        "(anthropic<->openai) in CLI mode. Use scripts/run_gap_fill.py for a "
        "corpus-wide sweep.",
    )
    parser.add_argument(
        "--gap-fill-threshold",
        type=float,
        default=0.85,
        help="Coverage threshold below which current-run gap-filling runs "
        "(default: 0.85 = 85%%).",
    )
    parser.add_argument(
        "--gap-fill-provider",
        type=str,
        default=None,
        help="Provider for gap-filling (default: auto-select opposite of primary).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="zotero",
        choices=["zotero", "bibtex", "pdffolder", "mendeley", "endnote", "paperpile"],
        help="Reference source to load papers from (default: zotero).",
    )
    parser.add_argument(
        "--source-path",
        type=str,
        default=None,
        help="Path for non-Zotero sources (BibTeX file, PDF folder, etc.).",
    )
    return parser.parse_args()


def _print_classification_report(index: ClassificationIndex) -> None:
    """Print classification summary to terminal."""
    stats = index.stats
    total = stats.get("total", 0)
    if total == 0:
        print("No papers classified.")
        return

    print("\nClassification summary:")
    for doc_type, count in sorted(
        stats.get("by_type", {}).items(), key=lambda x: -x[1]
    ):
        pct = count / total * 100
        print(f"  {doc_type:<22} {count:>5} ({pct:.0f}%)")

    ext = stats.get("extractable_count", 0)
    non_ext = stats.get("non_extractable_count", 0)
    print(f"\nExtractable: {ext}  |  Skippable: {non_ext}")


def load_checkpoint(index_dir: Path) -> set[str]:
    """Load set of already-processed paper IDs."""
    extractions_file = index_dir / "semantic_analyses.json"
    if not extractions_file.exists():
        return set()

    data = safe_read_json(extractions_file, default={})
    if isinstance(data, dict) and "extractions" in data:
        return set(data["extractions"].keys())
    return set(data.keys())


def save_checkpoint(
    index_dir: Path,
    papers: list[dict],
    extractions: dict[str, dict],
    metadata: dict,
):
    """Save current progress to disk."""
    # Save papers using structured store format
    papers_data = {
        "schema_version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "paper_count": len(papers),
        "papers": papers,
    }
    safe_write_json(index_dir / "papers.json", papers_data)

    # Save extractions
    extractions_data = {
        "schema_version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "extraction_count": len(extractions),
        "extractions": extractions,
    }
    safe_write_json(index_dir / "semantic_analyses.json", extractions_data)

    # Save metadata
    safe_write_json(index_dir / "metadata.json", metadata)


def get_platform_unset_command() -> str:
    """Get platform-appropriate command to unset ANTHROPIC_API_KEY."""
    if sys.platform == "win32":
        return (
            "     PowerShell: $env:ANTHROPIC_API_KEY = $null\n"
            "     CMD:        set ANTHROPIC_API_KEY="
        )
    return "     export ANTHROPIC_API_KEY="


def verify_cli_authentication() -> bool:
    """Verify CLI authentication is set up for CLI mode extraction.

    Returns:
        True if authenticated, False if user needs to set up.

    Raises:
        SystemExit: If user declines to set up authentication.
    """
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

    # Not authenticated - show options
    print("\n" + "=" * 60)
    print("CLI Mode Authentication Required")
    print("=" * 60)
    print(f"\nStatus: {status}")
    print(authenticator.get_setup_instructions())

    # Offer to open browser for authentication
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
            except Exception as e:
                print(f"\nFailed to launch Claude: {e}")
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
    existing_papers: dict,
    existing_extractions: dict,
    checkpoint_mgr: CheckpointManager,
    logger,
) -> tuple[dict, dict, list]:
    """Run LLM extraction on papers.

    Returns:
        Tuple of (paper_dicts, extractions, results)
    """
    # Convert papers to dicts
    paper_dicts = dict(existing_papers)
    for paper in papers:
        paper_dicts[paper.paper_id] = paper.to_index_dict()

    # Initialize or update checkpoint
    state = checkpoint_mgr.load()
    if not state:
        checkpoint_mgr.initialize(
            total_items=len(papers),
            metadata={"started_at": datetime.now().isoformat()},
        )

    # Progress bar
    pbar = tqdm(total=len(papers), desc="Extracting", unit="paper")

    # Run extraction
    results = []
    try:
        for result in extractor.extract_batch(papers):
            results.append(result)

            # Track in checkpoint
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

            # Checkpoint every 10 papers
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


def _run_gap_fill(
    results: list,
    existing_extractions: dict,
    papers_to_extract: list,
    primary_provider: str,
    gap_fill_provider: str | None,
    threshold: float,
    mode: str,
    config,
    index_dir: Path,
    paper_dicts: dict,
    checkpoint_mgr,
    logger,
) -> None:
    """Run gap-filling on low-coverage extractions using a secondary provider.

    Iterates over successful results, identifies papers below the coverage
    threshold, and runs only the extraction passes containing missing
    dimensions using a different LLM provider.

    Args:
        results: Extraction results from primary run.
        existing_extractions: Mutable dict of all extractions (updated in-place).
        papers_to_extract: List of PaperMetadata that were extracted.
        primary_provider: Primary provider name (e.g., "openai").
        gap_fill_provider: Override provider name, or None for auto-select.
        threshold: Coverage threshold (0-1) below which gap-filling runs.
        mode: Extraction mode ("cli" or "api").
        config: Config object.
        index_dir: Index directory for saving checkpoints.
        paper_dicts: Paper dict for checkpoint saving.
        checkpoint_mgr: Checkpoint manager.
        logger: Logger instance.
    """
    from src.analysis.llm_council import (
        CouncilConfig,
        LLMCouncil,
        ProviderConfig,
    )
    from src.analysis.schemas import SemanticAnalysis

    # Identify low-coverage papers
    low_coverage = []
    paper_lookup = {p.paper_id: p for p in papers_to_extract}

    for result in results:
        if not result.success or not result.extraction:
            continue
        extraction = result.extraction
        if not isinstance(extraction, SemanticAnalysis):
            continue
        coverage = extraction.dimension_coverage
        if coverage < threshold:
            paper = paper_lookup.get(result.paper_id)
            if paper:
                low_coverage.append((paper, extraction, result))

    if not low_coverage:
        logger.info("Gap-fill: all papers above %.0f%% coverage, nothing to do", threshold * 100)
        return

    logger.info(
        "Gap-fill: %d papers below %.0f%% coverage, running secondary provider",
        len(low_coverage), threshold * 100,
    )

    # Determine secondary provider
    secondary = gap_fill_provider or ("anthropic" if primary_provider == "openai" else "openai")

    council_config = CouncilConfig(
        providers=[
            ProviderConfig(name=primary_provider, weight=1.2, timeout=600, mode=mode),
            ProviderConfig(name=secondary, weight=1.0, timeout=600, mode=mode),
        ],
        aggregation_strategy="quality_weighted",
    )
    council = LLMCouncil(council_config)

    # Load PDF extractor for text retrieval
    pdf_extractor = PDFExtractor(
        enable_ocr=config.processing.ocr_enabled or config.processing.ocr_on_fail,
    )
    text_cleaner = TextCleaner()

    filled_total = 0
    for idx, (paper, extraction, result) in enumerate(low_coverage, 1):
        logger.info(
            "Gap-fill %d/%d: %s (coverage: %.0f%%)",
            idx, len(low_coverage), paper.paper_id, extraction.dimension_coverage * 100,
        )

        # Get paper text
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
            logger.warning("Gap-fill: text extraction failed for %s: %s", paper.paper_id, exc)
            continue

        authors = ", ".join(
            f"{a.last_name}, {a.first_name}" for a in (paper.authors or [])
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
                    paper.paper_id, gaps_filled,
                    extraction.dimension_coverage * 100,
                    merged.dimension_coverage * 100,
                )
                # Update extraction in-place
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
                logger.info("Gap-fill: %s -- secondary provider found no new content", paper.paper_id)

        except Exception as exc:
            logger.error("Gap-fill failed for %s: %s", paper.paper_id, exc)

    # Save updated extractions
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

    logger.info("Gap-fill complete: %d dimensions filled across %d papers", filled_total, len(low_coverage))


def write_skipped_report(
    results: list,
    paper_dicts: dict,
    output_dir: Path,
    logger,
) -> tuple[Path | None, Path | None]:
    """Write skipped items report for likely non-publications or low text."""
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

    safe_write_json(json_path, {
        "generated_at": datetime.now().isoformat(),
        "count": len(skipped_items),
        "items": skipped_items,
    })

    import csv

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
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
        f"Skipped items report: {len(skipped_items)} entries "
        f"({json_path.name}, {csv_path.name})"
    )

    return json_path, csv_path


def run_embedding_generation(
    papers: list[PaperMetadata],
    extractions: dict[str, dict],
    index_dir: Path,
    embedding_model: str,
    rebuild: bool,
    logger,
    embedding_backend: str = "sentence-transformers",
    ollama_base_url: str = "http://localhost:11434",
    document_prefix: str | None = None,
    embedding_batch_size: int | str = "auto",
) -> int:
    """Generate embeddings and populate vector store.

    Returns:
        Number of chunks added.
    """
    chroma_dir = index_dir / "chroma"

    logger.info("Initializing embedding generator...")
    embedding_gen = EmbeddingGenerator(
        model_name=embedding_model,
        backend=embedding_backend,
        ollama_base_url=ollama_base_url,
        document_prefix=document_prefix,
    )

    logger.info("Initializing vector store...")
    vector_store = VectorStore(chroma_dir)
    try:
        if rebuild:
            logger.info("Clearing existing embeddings...")
            vector_store.clear()

        # Build paper lookups - by full paper_id and by zotero_key for backward compat
        paper_lookup = {p.paper_id: p for p in papers}
        paper_by_zotero_key: dict[str, list[PaperMetadata]] = {}
        for p in papers:
            paper_by_zotero_key.setdefault(p.zotero_key, []).append(p)

        # Filter to papers with extractions
        # Handle both old-style (zotero_key) and new-style (zotero_key_attachment_key) IDs
        papers_with_extractions = []
        extraction_lookup = {}
        skipped_ambiguous = 0

        for paper_id, ext_data in extractions.items():
            paper = None
            # Try exact match first (new-style ID)
            if paper_id in paper_lookup:
                paper = paper_lookup[paper_id]
            # Fall back to zotero_key match (old-style ID) -- only safe for
            # single-attachment items. Multi-attachment items are ambiguous
            # because we cannot determine which PDF was actually extracted.
            elif "_" not in paper_id and paper_id in paper_by_zotero_key:
                candidates = paper_by_zotero_key[paper_id]
                if len(candidates) == 1:
                    paper = candidates[0]
                else:
                    skipped_ambiguous += 1
                    logger.warning(
                        f"Skipping old-style extraction '{paper_id}': "
                        f"{len(candidates)} attachments for this zotero_key. "
                        f"Re-extract with new-style IDs to resolve ambiguity."
                    )

            if paper:
                papers_with_extractions.append(paper)
                # Extract the actual extraction data
                ext = ext_data.get("extraction", ext_data)
                extraction_lookup[paper.paper_id] = SemanticAnalysis(**ext)

        if skipped_ambiguous > 0:
            logger.warning(
                f"Skipped {skipped_ambiguous} old-style extractions with ambiguous "
                f"multi-attachment zotero_keys. Re-extract these papers to fix."
            )

        if not papers_with_extractions:
            # Provide diagnostic info to help debug matching issues
            logger.warning("No papers with extractions found for embedding")
            logger.warning(f"  Papers provided: {len(papers)}")
            logger.warning(f"  Extractions available: {len(extractions)}")
            if papers and extractions:
                sample_paper_ids = [p.paper_id for p in papers[:3]]
                sample_extraction_ids = list(extractions.keys())[:3]
                logger.warning(f"  Sample paper IDs: {sample_paper_ids}")
                logger.warning(f"  Sample extraction IDs: {sample_extraction_ids}")
                logger.warning(
                    "  Check if ID formats match (old: zotero_key, new: zotero_key_attachment)"
                )
            return 0

        logger.info(f"Generating embeddings for {len(papers_with_extractions)} papers...")

        # Create chunks
        all_chunks = []
        for paper in tqdm(papers_with_extractions, desc="Creating chunks"):
            extraction = extraction_lookup.get(paper.paper_id)
            if extraction:
                chunks = embedding_gen.create_chunks(paper, extraction)
                all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} chunks")

        # Generate embeddings in batches
        resolved_batch_size = embedding_gen.resolve_batch_size(
            embedding_batch_size,
            texts=[chunk.text for chunk in all_chunks],
        )
        logger.info(f"Generating embeddings with batch size {resolved_batch_size}...")
        all_chunks = embedding_gen.generate_embeddings(
            all_chunks,
            batch_size=resolved_batch_size,
        )

        # Add to vector store
        logger.info("Adding to vector store...")
        added = vector_store.add_chunks(all_chunks, batch_size=100)

        logger.info(f"Added {added} chunks to vector store")
        return added
    finally:
        vector_store.close()


def generate_summary(index_dir: Path, logger):
    """Generate summary statistics."""
    logger.info("Generating summary statistics...")
    store = StructuredStore(index_dir)
    summary = store.save_summary()
    logger.info(f"Summary saved with {summary.get('total_papers', 0)} papers")


def compute_similarity_pairs(
    index_dir: Path,
    embedding_model: str,
    top_n: int = 20,
    logger=None,
) -> int:
    """Compute pairwise paper similarity from raptor_overview embeddings.

    Retrieves all raptor_overview chunk embeddings from ChromaDB,
    computes cosine similarity between all pairs, and stores
    the top_n most similar papers for each paper.

    Args:
        index_dir: Path to index directory.
        embedding_model: Name of embedding model (for metadata).
        top_n: Number of similar papers to store per source paper.
        logger: Logger instance.

    Returns:
        Total number of pairs computed.
    """
    import numpy as np

    chroma_dir = index_dir / "chroma"
    store = StructuredStore(index_dir)
    with VectorStore(chroma_dir) as vector_store:
        # Get all raptor_overview chunks with embeddings
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
    for i, _chunk_id in enumerate(results["ids"]):
        meta = results["metadatas"][i] if results["metadatas"] is not None else {}
        paper_id = meta.get("paper_id", "")
        embedding = results["embeddings"][i] if results["embeddings"] is not None else None
        if paper_id and embedding is not None:
            paper_ids.append(paper_id)
            embeddings.append(embedding)

    if len(embeddings) < 2:
        if logger:
            logger.info("Fewer than 2 papers with raptor_overview -- skipping similarity")
        return 0

    if logger:
        logger.info(f"Computing similarity for {len(embeddings)} papers...")

    # Compute cosine similarity matrix
    emb_matrix = np.array(embeddings)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_normalized = emb_matrix / norms
    similarity_matrix = emb_normalized @ emb_normalized.T

    # Build pairs dict
    pairs: dict[str, list[dict]] = {}
    for i, source_id in enumerate(paper_ids):
        scores = similarity_matrix[i].copy()
        scores[i] = -1.0  # exclude self

        top_indices = np.argsort(scores)[::-1][:top_n]

        similar_list = []
        for idx in top_indices:
            sim_score = float(scores[idx])
            if sim_score <= 0:
                break
            similar_list.append({
                "similar_paper_id": paper_ids[idx],
                "similarity_score": round(sim_score, 4),
            })

        if similar_list:
            pairs[source_id] = similar_list

    # Save
    store.save_similarity_pairs(
        pairs,
        metadata={
            "embedding_model": embedding_model,
            "top_n_per_paper": top_n,
            "computation_method": "cosine_similarity",
        },
    )

    total_pairs = sum(len(v) for v in pairs.values())
    if logger:
        logger.info(
            f"Computed {total_pairs} similarity pairs "
            f"for {len(pairs)} papers (top {top_n} each)"
        )
    return total_pairs


def main():
    """Main entry point."""
    args = parse_args()
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(level=log_level)

    try:
        config = Config.load(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    if args.embedding_batch_size is not None:
        config.embeddings.batch_size = args.embedding_batch_size

    orchestrator = IndexOrchestrator(project_root=project_root, logger=logger)
    try:
        return orchestrator.run(args, config)
    except ValueError as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.error(f"Build failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
