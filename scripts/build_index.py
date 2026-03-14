#!/usr/bin/env python
"""Build the literature review index from Zotero library."""

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
from src.config import Config
from src.extraction.pdf_extractor import PDFExtractor
from src.extraction.text_cleaner import TextCleaner
from src.indexing.embeddings import EmbeddingGenerator
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
        description="Build literature review index from Zotero library"
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
        help="LLM provider: anthropic (Claude), openai (GPT-5.2)",
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
        help="Model to use (e.g., gpt-5.4, claude-opus-4-5-20251101)",
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
        "--rebuild-embeddings",
        action="store_true",
        help="Rebuild embeddings even if they exist",
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

    if rebuild:
        logger.info("Clearing existing embeddings...")
        vector_store.clear()

    # Build paper lookups - by full paper_id and by zotero_key for backward compat
    paper_lookup = {p.paper_id: p for p in papers}
    paper_by_zotero_key: dict[str, PaperMetadata] = {}
    for p in papers:
        # Only store first paper per zotero_key (for old-style ID matching)
        if p.zotero_key not in paper_by_zotero_key:
            paper_by_zotero_key[p.zotero_key] = p

    # Filter to papers with extractions
    # Handle both old-style (zotero_key) and new-style (zotero_key_attachment_key) IDs
    papers_with_extractions = []
    extraction_lookup = {}

    for paper_id, ext_data in extractions.items():
        paper = None
        # Try exact match first (new-style ID)
        if paper_id in paper_lookup:
            paper = paper_lookup[paper_id]
        # Fall back to zotero_key match (old-style ID)
        elif "_" not in paper_id and paper_id in paper_by_zotero_key:
            paper = paper_by_zotero_key[paper_id]

        if paper:
            papers_with_extractions.append(paper)
            # Extract the actual extraction data
            ext = ext_data.get("extraction", ext_data)
            extraction_lookup[paper.paper_id] = SemanticAnalysis(**ext)

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
            logger.warning("  Check if ID formats match (old: zotero_key, new: zotero_key_attachment)")
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
    logger.info("Generating embeddings...")
    all_chunks = embedding_gen.generate_embeddings(all_chunks, batch_size=32)

    # Add to vector store
    logger.info("Adding to vector store...")
    added = vector_store.add_chunks(all_chunks, batch_size=100)

    logger.info(f"Added {added} chunks to vector store")
    return added


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
    vector_store = VectorStore(chroma_dir)

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

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(level=log_level)

    # Load configuration
    try:
        config = Config.load(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1

    # Setup directories
    index_dir = project_root / "data" / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = config.get_cache_path() / "pdf_text"

    # Connect to reference source
    source = args.source
    logger.info(f"Loading papers from {source} source...")
    try:
        if source == "zotero":
            ref_db = create_reference_db(
                "zotero",
                db_path=config.get_zotero_db_path(),
                storage_path=config.get_storage_path(),
            )
        elif args.source_path:
            source_kwarg_map = {
                "bibtex": "bibtex_path",
                "pdffolder": "folder_path",
                "endnote": "xml_path",
                "mendeley": "db_path",
                "paperpile": "bibtex_path",
            }
            kwarg_name = source_kwarg_map.get(source, "path")
            ref_db = create_reference_db(source, **{kwarg_name: args.source_path})
        else:
            logger.error(f"--source-path required for source '{source}'")
            return 1
    except Exception as e:
        logger.error(f"Failed to connect to {source}: {e}")
        return 1

    # Get all papers
    with LogContext(logger, f"Loading papers from {source}"):
        all_papers = list(ref_db.get_all_papers())
        if args.collection:
            if source != "zotero":
                logger.warning(
                    f"--collection filter is Zotero-specific; "
                    f"ignoring for source '{source}'"
                )
            else:
                all_papers = [p for p in all_papers if any(args.collection in c for c in p.collections)]
                logger.info(f"Collection filter '{args.collection}': {len(all_papers)} papers matched")
        # Filter to only papers with actual PDF files
        papers = [p for p in all_papers if p.pdf_path]
        papers_without_pdf = [p for p in all_papers if not p.pdf_path]
        if papers_without_pdf:
            logger.info(f"Found {len(all_papers)} papers, {len(papers_without_pdf)} without PDFs (skipped)")
        logger.info(f"Found {len(papers)} papers with PDFs")

    # Classification pre-pass
    class_store = ClassificationStore(
        index_dir=index_dir,
        min_publication_words=config.processing.min_publication_words,
        min_publication_pages=config.processing.min_publication_pages,
    )

    if args.classify_only:
        class_index = class_store.load() if not args.reclassify else ClassificationIndex()
        pdf_extractor = PDFExtractor(
            enable_ocr=config.processing.ocr_enabled or config.processing.ocr_on_fail,
        )
        text_cleaner = TextCleaner()

        for paper in tqdm(papers, desc="Classifying"):
            if not args.reclassify and paper.paper_id in class_index.papers:
                continue

            text = None
            word_count = None
            page_count = None
            section_markers = None

            if paper.pdf_path and Path(paper.pdf_path).exists():
                pdf_path = Path(paper.pdf_path)
                try:
                    raw_text, _ = pdf_extractor.extract_text_with_method(pdf_path)
                    text = text_cleaner.clean(raw_text)
                    stats = text_cleaner.get_stats(text)
                    word_count = stats.word_count
                    page_count = pdf_extractor.get_page_count(pdf_path)
                    section_markers = text_cleaner.count_section_markers(text)
                except Exception as e:
                    logger.warning(f"Text extraction failed for {paper.title}: {e}")

            record = class_store.classify_paper(
                paper, text=text, word_count=word_count,
                page_count=page_count, section_markers=section_markers,
            )
            class_index.papers[paper.paper_id] = record

        class_store.save(class_index)
        _print_classification_report(class_index)
        return 0

    # Load or build classification index for gating
    class_index = class_store.load()
    papers_needing_classification = [
        p for p in papers if p.paper_id not in class_index.papers
    ]

    if papers_needing_classification:
        if not class_index.papers:
            logger.info(
                "No classification index found. Running inline classification. "
                "Use --classify-only for a preview."
            )
        pdf_extractor_for_class = PDFExtractor(
            enable_ocr=config.processing.ocr_enabled or config.processing.ocr_on_fail,
        )
        text_cleaner_for_class = TextCleaner()

        for paper in papers_needing_classification:
            text = None
            word_count = None
            page_count = None
            section_markers = None
            if paper.pdf_path and Path(paper.pdf_path).exists():
                pdf_path = Path(paper.pdf_path)
                try:
                    raw_text, _ = pdf_extractor_for_class.extract_text_with_method(
                        pdf_path
                    )
                    cleaned = text_cleaner_for_class.clean(raw_text)
                    stats = text_cleaner_for_class.get_stats(cleaned)
                    word_count = stats.word_count
                    page_count = pdf_extractor_for_class.get_page_count(pdf_path)
                    section_markers = text_cleaner_for_class.count_section_markers(
                        cleaned
                    )
                except Exception:
                    pass
            record = class_store.classify_paper(
                paper, text=text, word_count=word_count,
                page_count=page_count, section_markers=section_markers,
            )
            class_index.papers[paper.paper_id] = record

        class_store.save(class_index)

    # Apply gating filter (default: academic-only)
    if not args.index_all:
        extractable_ids = class_store.get_extractable_ids(class_index)
        skipped = [p for p in papers if p.paper_id not in extractable_ids]
        papers = [p for p in papers if p.paper_id in extractable_ids]

        if skipped:
            non_academic = sum(
                1 for p in skipped
                if class_index.papers.get(p.paper_id)
                and class_index.papers[p.paper_id].document_type == "non_academic"
            )
            other = len(skipped) - non_academic
            logger.info(
                f"Skipping {len(skipped)} papers "
                f"({non_academic} non-academic, {other} insufficient text). "
                f"Use --index-all to include."
            )

    # Handle show-doi-overlap (analyze DOI overlap without processing)
    if args.show_doi_overlap:
        print("\nAnalyzing DOI overlap between Zotero and existing index...")
        analysis = analyze_doi_overlap(papers, index_dir)

        print(f"\n{'=' * 60}")
        print("DOI Overlap Analysis")
        print(f"{'=' * 60}")
        print("\nExisting index:")
        print(f"  Papers with DOIs: {analysis['existing_index_dois']}")
        print("\nNew Zotero database:")
        print(f"  Total papers (with PDFs): {analysis['new_papers_total']}")
        print(f"  With DOIs: {analysis['new_with_doi']}")
        print(f"  Without DOIs: {analysis['new_without_doi']}")
        print("\nOverlap analysis:")
        print(f"  Duplicates (DOI match): {analysis['duplicates_by_doi']}")
        print(f"  Genuinely new (with DOI): {analysis['genuinely_new_with_doi']}")
        print(f"  Total to process: {len(analysis['new_papers_filtered'])}")

        if analysis['duplicate_papers']:
            print("\nDuplicate papers (will be skipped with --dedupe-by-doi):")
            print("-" * 60)
            for i, p in enumerate(analysis['duplicate_papers'][:10], 1):
                print(f"  {i}. {p.title[:55]}...")
                print(f"     DOI: {p.doi}")
            if len(analysis['duplicate_papers']) > 10:
                print(f"  ... and {len(analysis['duplicate_papers']) - 10} more")

        print(f"\n{'=' * 60}")
        print("Recommendation: Use --dedupe-by-doi to skip duplicate papers")
        print(f"{'=' * 60}")
        return 0

    # Handle show-skipped (papers without PDFs)
    if args.show_skipped:
        if papers_without_pdf:
            print(f"\nPapers without PDFs ({len(papers_without_pdf)}):")
            print("-" * 70)
            for p in papers_without_pdf:
                print(f"  {p.paper_id}: {p.title[:60]}...")
                if p.authors:
                    print(f"           {p.author_string}")
            print("-" * 70)
            print(f"Total: {len(papers_without_pdf)} papers cannot be extracted (no PDF)")
        else:
            print("\nAll papers have PDFs available.")
        return 0

    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager(index_dir, checkpoint_id="extraction")

    # Handle reset checkpoint
    if args.reset_checkpoint:
        checkpoint_mgr.reset()
        logger.info("Checkpoint reset, starting fresh")

    # Handle show failed
    if args.show_failed:
        state = checkpoint_mgr.load()
        if state and state.failed_items:
            print(f"\nFailed papers ({len(state.failed_items)}):")
            for item in state.failed_items:
                print(f"  - {item.item_id}")
                print(f"    Error: {item.error_type}: {item.error_message}")
                if item.retry_count > 0:
                    print(f"    Retries: {item.retry_count}")
        else:
            print("\nNo failed papers recorded")
        return 0

    # Load existing extractions early (for filtering and dry-run accuracy)
    existing_extractions_data = safe_read_json(
        index_dir / "semantic_analyses.json", default={}
    )
    if isinstance(existing_extractions_data, dict) and "extractions" in existing_extractions_data:
        existing_extraction_ids = set(existing_extractions_data["extractions"].keys())
    elif isinstance(existing_extractions_data, dict):
        existing_extraction_ids = set(existing_extractions_data.keys())
    else:
        existing_extraction_ids = set()

    # Identify old-style extraction IDs (just zotero_key, no underscore)
    # These need special handling for backward compatibility
    old_style_extraction_ids = {eid for eid in existing_extraction_ids if "_" not in eid}

    # Track which old-style IDs have been "consumed" (matched to first attachment)
    consumed_old_style: set[str] = set()

    def is_already_extracted(paper: PaperMetadata) -> bool:
        """Check if paper is already extracted, handling both ID formats."""
        # Exact match with new-style ID (zotero_key_attachment_key)
        if paper.paper_id in existing_extraction_ids:
            return True
        # Old-style match: consume only once per zotero_key
        # This ensures first attachment matches old extraction, second gets extracted
        if paper.zotero_key in old_style_extraction_ids:
            if paper.zotero_key not in consumed_old_style:
                consumed_old_style.add(paper.zotero_key)
                return True
        return False

    # Save full papers list for embedding generation (needs all papers, not just unextracted)
    all_papers_with_pdfs = papers.copy()

    # --paper: target specific papers and bypass already-extracted filter
    if args.paper:
        target_keys = set(args.paper)
        papers = [
            p for p in all_papers_with_pdfs
            if p.paper_id in target_keys or p.zotero_key in target_keys
        ]
        if not papers:
            logger.error(f"No papers matched --paper keys: {target_keys}")
            return 1
        logger.info(f"Targeting {len(papers)} paper(s) via --paper flag")
        total_papers = len(papers)
        already_extracted_count = 0
    else:
        # Filter out already-extracted papers BEFORE applying limit
        # This way --limit N means "extract N new papers"
        total_papers = len(papers)
        already_extracted_count = len([p for p in papers if is_already_extracted(p)])
        # Reset consumed set and filter again (since we counted above)
        consumed_old_style.clear()
        papers = [p for p in papers if not is_already_extracted(p)]
        if already_extracted_count > 0:
            logger.info(f"Filtered out {already_extracted_count} already-extracted papers")

    # DOI-based deduplication (for cross-database scenarios)
    doi_duplicate_count = 0
    if args.dedupe_by_doi:
        existing_dois = extract_existing_dois(index_dir)
        if existing_dois:
            _papers_before = len(papers)
            papers, doi_duplicates = filter_by_doi(papers, existing_dois)
            doi_duplicate_count = len(doi_duplicates)
            if doi_duplicate_count > 0:
                logger.info(
                    f"DOI deduplication: {doi_duplicate_count} papers skipped "
                    f"(matching DOIs in existing index)"
                )

    # Apply limit if specified (now applies to unextracted papers only)
    if args.limit:
        papers = papers[: args.limit]
        logger.info(f"Limited to {len(papers)} new papers (from {total_papers} total, {already_extracted_count} already done)")

    # Build skip set from command line
    skip_paper_ids = set(args.skip_paper)
    if skip_paper_ids:
        logger.info(f"Will skip {len(skip_paper_ids)} papers from command line")

    # Check for resume or retry-failed
    processed_ids = set()
    papers_to_extract = papers

    if args.retry_failed and not args.skip_extraction:
        # Load checkpoint and retry failed items
        state = checkpoint_mgr.load()
        if state:
            failed_ids = set(checkpoint_mgr.get_failed_ids())
            if failed_ids:
                # Clear failed items so they can be retried
                checkpoint_mgr.clear_failed()
                papers_to_extract = [p for p in papers if p.paper_id in failed_ids]
                logger.info(f"Retrying {len(papers_to_extract)} previously failed papers")
            else:
                logger.info("No failed papers to retry")
                papers_to_extract = []
        else:
            logger.warning("No checkpoint found, nothing to retry")
            papers_to_extract = []

    elif args.resume and not args.skip_extraction:
        # Resume from checkpoint
        state = checkpoint_mgr.load()
        if state:
            processed_ids = set(state.processed_ids)
            failed_ids = set(checkpoint_mgr.get_failed_ids())
            papers_to_extract = [
                p for p in papers
                if p.paper_id not in processed_ids and p.paper_id not in failed_ids
            ]
            logger.info(
                f"Resuming: {len(processed_ids)} successful, {len(failed_ids)} failed, "
                f"{len(papers_to_extract)} remaining"
            )
        else:
            processed_ids = load_checkpoint(index_dir)  # Legacy checkpoint
            papers_to_extract = [p for p in papers if p.paper_id not in processed_ids]
            logger.info(
                f"Resuming (legacy): {len(processed_ids)} already processed, "
                f"{len(papers_to_extract)} remaining"
            )

    # Apply skip-paper filter
    if skip_paper_ids:
        papers_to_extract = [
            p for p in papers_to_extract if p.paper_id not in skip_paper_ids
        ]

    # Note: Already-extracted papers were filtered out earlier (before --limit)

    if not papers and not args.skip_extraction:
        logger.info("No new papers to process")
        return 0

    # Dry run - just show what would be processed
    if args.dry_run:
        print(f"\nTotal in library: {total_papers}")
        print(f"Already extracted: {already_extracted_count}")
        if doi_duplicate_count > 0:
            print(f"DOI duplicates skipped: {doi_duplicate_count}")
        print(f"To extract: {len(papers_to_extract)}")

        if papers_to_extract:
            print(f"\nWould process {len(papers_to_extract)} papers:")
            for i, paper in enumerate(papers_to_extract[:20], 1):
                print(f"  {i}. {paper.title[:60]}...")
            if len(papers_to_extract) > 20:
                print(f"  ... and {len(papers_to_extract) - 20} more")
        else:
            print("\nNo new papers to extract.")
        return 0

    # Initialize extractor settings
    provider = args.provider or config.extraction.provider
    mode = args.mode or config.extraction.mode
    _model = args.model or config.extraction.model

    # Override config with CLI args
    if args.provider:
        config.extraction.provider = args.provider
    if args.model:
        config.extraction.model = args.model

    logger.info(f"Using LLM provider: {provider}")

    # Handle --use-subscription flag (Anthropic only)
    if args.use_subscription and mode == "cli" and provider == "anthropic":
        if os.environ.get("ANTHROPIC_API_KEY"):
            logger.info("--use-subscription: Temporarily unsetting ANTHROPIC_API_KEY")
            os.environ.pop("ANTHROPIC_API_KEY", None)

    # Verify CLI authentication if using CLI mode
    if mode == "cli" and not args.skip_extraction:
        if provider == "anthropic":
            verify_cli_authentication()
        elif provider == "openai":
            # OpenAI uses Codex CLI - verify it's available
            import shutil
            if not shutil.which("codex"):
                print("\n" + "=" * 60)
                print("Codex CLI Required")
                print("=" * 60)
                print("\nFor OpenAI CLI mode, install Codex CLI:")
                print("  npm i -g @openai/codex")
                print("  -- or --")
                print("  brew install --cask codex")
                print("\nThen authenticate: codex login")
                return 1

    # batch_api mode uses separate script
    if mode == "batch_api" and not args.skip_extraction:
        print("\n" + "=" * 60)
        print("Batch API Mode")
        print("=" * 60)
        print("\nThe Batch API requires a separate workflow due to async processing.")
        print("Use the batch_extract.py script instead:")
        print("\n  1. Submit batch:")
        print("     python scripts/batch_extract.py submit --limit 100")
        print("\n  2. Check status:")
        print("     python scripts/batch_extract.py status <batch_id>")
        print("\n  3. Wait for completion:")
        print("     python scripts/batch_extract.py wait <batch_id>")
        print("\n  4. Collect results:")
        print("     python scripts/batch_extract.py collect <batch_id>")
        print("\n  5. Generate embeddings (back here):")
        print("     python scripts/build_index.py --skip-extraction")
        print("\nBatch API offers 50% cost savings but processes asynchronously.")
        return 0

    # Determine extraction settings
    use_cache = config.extraction.use_cache and not args.no_cache
    parallel_workers = args.parallel if args.parallel else config.extraction.parallel_workers

    # Cost estimation
    if args.estimate_cost:
        extractor = SectionExtractor(
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
        print("\nEstimating extraction cost...")
        estimate = extractor.estimate_batch_cost(papers_to_extract)
        print(f"\nPapers with PDFs: {estimate['papers_with_pdf']}")
        if estimate.get("papers_cached", 0) > 0:
            print(f"Papers cached (will skip): {estimate['papers_cached']}")
            print(f"Papers to extract: {estimate['papers_to_extract']}")
        if "average_text_length" in estimate:
            print(f"Average text length: {estimate['average_text_length']:,} chars")
            print(f"Cost per paper: ${estimate['estimated_cost_per_paper']:.4f}")
            print(f"Total estimated cost: ${estimate['estimated_total_cost']:.2f}")
            print(f"Model: {estimate['model']}")
        else:
            print(f"Note: {estimate.get('note', 'Unknown')}")
        return 0

    start_time = datetime.now()

    # Load existing data
    existing_papers_data = safe_read_json(index_dir / "papers.json", default={})
    if isinstance(existing_papers_data, dict) and "papers" in existing_papers_data:
        existing_papers = {
            p["paper_id"]: p for p in existing_papers_data["papers"] if "paper_id" in p
        }
    elif isinstance(existing_papers_data, list):
        existing_papers = {
            p["paper_id"]: p for p in existing_papers_data if "paper_id" in p
        }
    else:
        existing_papers = {}

    # Use already-loaded extraction data (loaded earlier for filtering)
    if isinstance(existing_extractions_data, dict) and "extractions" in existing_extractions_data:
        existing_extractions = existing_extractions_data["extractions"]
    elif isinstance(existing_extractions_data, dict):
        existing_extractions = existing_extractions_data
    else:
        existing_extractions = {}

    # Step 1: LLM Extraction
    results = []
    if not args.skip_extraction and papers_to_extract:
        logger.info(f"Starting extraction with mode: {mode}")
        if parallel_workers > 1:
            logger.info(f"Using {parallel_workers} parallel workers")
        if not use_cache:
            logger.info("Extraction caching disabled")

        extractor = SectionExtractor(
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

        # Clear cache if requested
        if args.clear_cache:
            cleared = extractor.clear_cache()
            logger.info(f"Cleared {cleared} cached extractions")

        paper_dicts, existing_extractions, results = run_extraction(
            papers_to_extract,
            extractor,
            index_dir,
            existing_papers,
            existing_extractions,
            checkpoint_mgr,
            logger,
        )

        # Final save after extraction
        checkpoint_mgr.save()
        save_checkpoint(
            index_dir,
            list(paper_dicts.values()),
            existing_extractions,
            {
                "last_updated": datetime.now().isoformat(),
                "paper_count": len(paper_dicts),
                "extraction_count": len(existing_extractions),
                "extraction_mode": mode,
                "model": config.extraction.model,
                **checkpoint_mgr.get_progress(),
            },
        )

        # Track extraction method in classification index
        for result in results:
            if result.success and result.extraction_method:
                if result.paper_id in class_index.papers:
                    class_index.papers[result.paper_id].extraction_method = (
                        result.extraction_method
                    )
        class_store.save(class_index)

        successful = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)
        logger.info(f"Extraction: {successful} successful, {failed} failed")
        write_skipped_report(
            results,
            paper_dicts,
            project_root / "data" / "out" / "experiments" / "skipped_items",
            logger,
        )
    else:
        # Update paper dicts from existing
        paper_dicts = dict(existing_papers)
        for paper in papers:
            if paper.paper_id not in paper_dicts:
                paper_dicts[paper.paper_id] = paper.to_index_dict()

    # Step 2: Embedding Generation
    if not args.skip_embeddings:
        try:
            # When targeting specific papers, only regenerate their embeddings
            embed_papers = papers if args.paper else all_papers_with_pdfs
            rebuild = args.rebuild_embeddings and not args.paper
            chunks_added = run_embedding_generation(
                embed_papers,
                existing_extractions,
                index_dir,
                config.embeddings.model,
                rebuild,
                logger,
                embedding_backend=config.embeddings.backend,
                ollama_base_url=config.embeddings.ollama_base_url,
                document_prefix=config.embeddings.document_prefix,
            )
            if chunks_added == 0:
                logger.warning("No embedding chunks were added - check extraction/paper matching")
            else:
                logger.info(f"Embedding generation complete: {chunks_added} chunks")
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            import traceback
            traceback.print_exc()
            logger.error("Embedding step failed but continuing with summary generation")

    # Step 2.5: Similarity Pair Computation
    if not args.skip_embeddings and not args.skip_similarity:
        try:
            pairs_count = compute_similarity_pairs(
                index_dir,
                config.embeddings.model,
                top_n=20,
                logger=logger,
            )
            if pairs_count > 0:
                logger.info(f"Similarity computation complete: {pairs_count} pairs")
            else:
                logger.warning("No similarity pairs computed")
        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            import traceback
            traceback.print_exc()
            logger.error("Similarity step failed but continuing with summary generation")

    # Step 3: Summary Generation
    try:
        generate_summary(index_dir, logger)
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")

    # Final summary
    duration = datetime.now() - start_time
    logger.info(f"\nBuild complete in {duration}")
    logger.info(f"  Total papers with PDFs: {len(paper_dicts)}")
    logger.info(f"  Total extractions: {len(existing_extractions)}")
    if papers_without_pdf:
        logger.info(f"  Skipped (no PDF): {len(papers_without_pdf)}")
    logger.info(f"  Output: {index_dir}")

    if results:
        failed = sum(1 for r in results if not r.success)
        return 0 if failed == 0 else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
