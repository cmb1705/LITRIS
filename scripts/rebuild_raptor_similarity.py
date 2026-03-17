#!/usr/bin/env python
"""Rebuild RAPTOR summaries and similarity pairs for the existing index.

Generates RAPTOR overview/core contribution text for each paper from existing
SemanticAnalysis extractions, embeds them into ChromaDB, and then computes
pairwise similarity from the raptor_overview embeddings.

Two modes:
  --mode template  (default) Synthesize RAPTOR text from q-fields without LLM.
  --mode llm       Use LLM API to generate polished summaries (slower, costs $).

Usage:
    python scripts/rebuild_raptor_similarity.py
    python scripts/rebuild_raptor_similarity.py --mode llm --provider google --model gemini-2.5-flash
    python scripts/rebuild_raptor_similarity.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.raptor import RaptorSummaries
from src.analysis.schemas import SemanticAnalysis
from src.config import Config
from src.indexing.embeddings import EmbeddingChunk, EmbeddingGenerator
from src.indexing.structured_store import StructuredStore
from src.indexing.vector_store import VectorStore
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

# Cache file for RAPTOR summaries so we can resume on failure
RAPTOR_CACHE_FILE = PROJECT_ROOT / "data" / "index" / "raptor_summaries.json"


def _synthesize_overview(analysis: SemanticAnalysis) -> str:
    """Synthesize a paper overview from q-field data without LLM.

    Creates a structured paragraph that captures the paper's essence for
    embedding-based similarity computation.
    """
    parts = []

    # Opening: what the paper investigates
    rq = analysis.q01_research_question
    thesis = analysis.q02_thesis
    if rq:
        parts.append(rq.strip().rstrip(".") + ".")
    elif thesis:
        parts.append(thesis.strip().rstrip(".") + ".")

    # Methods
    methods = analysis.q07_methods
    if methods:
        # Take first 2 sentences max to keep it concise
        sentences = methods.split(". ")
        method_text = ". ".join(sentences[:2]).strip().rstrip(".") + "."
        parts.append(method_text)

    # Contribution
    contribution = analysis.q22_contribution
    if contribution:
        sentences = contribution.split(". ")
        contrib_text = ". ".join(sentences[:2]).strip().rstrip(".") + "."
        parts.append(contrib_text)

    # Implications (brief)
    implications = analysis.q19_implications
    if implications:
        sentences = implications.split(". ")
        parts.append(sentences[0].strip().rstrip(".") + ".")

    # Field context
    field_val = analysis.q17_field
    if field_val:
        parts.append(f"Field: {field_val.strip()}.")

    overview = " ".join(parts)

    # Truncate to ~150 words if needed
    words = overview.split()
    if len(words) > 180:
        overview = " ".join(words[:170]) + "..."

    return overview


def _synthesize_core(analysis: SemanticAnalysis) -> str:
    """Synthesize a single-sentence core contribution without LLM."""
    contribution = analysis.q22_contribution
    if contribution:
        # Take first sentence
        first_sentence = contribution.split(". ")[0].strip().rstrip(".") + "."
        words = first_sentence.split()
        if len(words) > 45:
            first_sentence = " ".join(words[:40]) + "..."
        return first_sentence

    thesis = analysis.q02_thesis
    if thesis:
        first_sentence = thesis.split(". ")[0].strip().rstrip(".") + "."
        words = first_sentence.split()
        if len(words) > 45:
            first_sentence = " ".join(words[:40]) + "..."
        return first_sentence

    return f"Study in {analysis.q17_field or 'unspecified field'}."


def load_raptor_cache() -> dict[str, dict]:
    """Load cached RAPTOR summaries from disk."""
    if RAPTOR_CACHE_FILE.exists():
        with open(RAPTOR_CACHE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("summaries", {})
    return {}


def save_raptor_cache(summaries: dict[str, dict], mode: str) -> None:
    """Save RAPTOR summaries to disk cache."""
    data = {
        "schema_version": "1.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": mode,
        "count": len(summaries),
        "summaries": summaries,
    }
    RAPTOR_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RAPTOR_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Cached {len(summaries)} RAPTOR summaries to {RAPTOR_CACHE_FILE}")


def generate_all_raptor_summaries(
    paper_dicts: dict[str, dict],
    extractions: dict[str, SemanticAnalysis],
    mode: str = "template",
    provider: str = "google",
    model: str | None = None,
    force: bool = False,
) -> dict[str, RaptorSummaries]:
    """Generate RAPTOR summaries for all papers.

    Args:
        paper_dicts: Dict of paper_id -> paper dict (from papers.json).
        extractions: Dict of paper_id -> SemanticAnalysis.
        mode: "template" (no LLM) or "llm" (API calls).
        provider: LLM provider (only used in llm mode).
        model: LLM model (only used in llm mode).
        force: If True, regenerate even if cached.

    Returns:
        Dict of paper_id -> RaptorSummaries.
    """
    # Determine common paper IDs
    common_ids = set(paper_dicts.keys()) & set(extractions.keys())

    # Load cache
    cached = {} if force else load_raptor_cache()
    results: dict[str, RaptorSummaries] = {}
    to_generate: list[str] = []

    # Restore cached entries
    for paper_id, cache_entry in cached.items():
        if paper_id in common_ids:
            results[paper_id] = RaptorSummaries(
                paper_id=paper_id,
                paper_overview=cache_entry["paper_overview"],
                core_contribution=cache_entry["core_contribution"],
            )

    # Find papers that need generation
    for paper_id in common_ids:
        if paper_id not in results:
            to_generate.append(paper_id)

    logger.info(
        f"RAPTOR status: {len(results)} cached, {len(to_generate)} to generate "
        f"(mode={mode})"
    )

    if not to_generate:
        logger.info("All papers already have RAPTOR summaries cached")
        return results

    generated_count = 0
    failed_count = 0

    for i, paper_id in enumerate(to_generate):
        analysis = extractions[paper_id]

        if mode == "template":
            overview = _synthesize_overview(analysis)
            core = _synthesize_core(analysis)
            if overview and core:
                results[paper_id] = RaptorSummaries(
                    paper_id=paper_id,
                    paper_overview=overview,
                    core_contribution=core,
                )
                generated_count += 1
            else:
                failed_count += 1
        elif mode == "llm":
            # LLM mode requires PaperMetadata -- construct minimal version
            from src.zotero.models import Author, PaperMetadata

            pdata = paper_dicts[paper_id]
            try:
                authors = []
                for a in pdata.get("authors", []):
                    if isinstance(a, dict):
                        authors.append(Author(**a))
                paper_meta = PaperMetadata(
                    zotero_key=pdata.get("zotero_key", paper_id.split("_")[0]),
                    zotero_item_id=pdata.get("zotero_key", paper_id.split("_")[0]),
                    title=pdata.get("title", "Unknown"),
                    item_type=pdata.get("item_type", "journalArticle"),
                    date_added=pdata.get("date_added", "2020-01-01T00:00:00"),
                    date_modified=pdata.get("date_modified", "2020-01-01T00:00:00"),
                    authors=authors,
                )
            except Exception as e:
                logger.warning(f"Could not create PaperMetadata for {paper_id}: {e}")
                # Fall back to template
                overview = _synthesize_overview(analysis)
                core = _synthesize_core(analysis)
                if overview and core:
                    results[paper_id] = RaptorSummaries(
                        paper_id=paper_id,
                        paper_overview=overview,
                        core_contribution=core,
                    )
                continue

            if (i + 1) % 50 == 0:
                logger.info(f"RAPTOR LLM progress: {i + 1}/{len(to_generate)}")

            from src.analysis.raptor import generate_raptor_summaries

            raptor = generate_raptor_summaries(
                paper=paper_meta,
                analysis=analysis,
                provider=provider,
                model=model,
            )
            if raptor:
                results[paper_id] = raptor
                generated_count += 1
            else:
                failed_count += 1
                # Fall back to template for failed LLM calls
                overview = _synthesize_overview(analysis)
                core = _synthesize_core(analysis)
                if overview and core:
                    results[paper_id] = RaptorSummaries(
                        paper_id=paper_id,
                        paper_overview=overview,
                        core_contribution=core,
                    )

    logger.info(
        f"RAPTOR generation complete: {generated_count} generated, "
        f"{failed_count} failed, {len(results)} total"
    )

    # Save to cache
    cache_data = {
        pid: {
            "paper_overview": r.paper_overview,
            "core_contribution": r.core_contribution,
        }
        for pid, r in results.items()
    }
    save_raptor_cache(cache_data, mode)

    return results


def embed_raptor_chunks(
    raptor_summaries: dict[str, RaptorSummaries],
    paper_dicts: dict[str, dict],
    extractions: dict[str, SemanticAnalysis],
    index_dir: Path,
    config: Config,
) -> int:
    """Create RAPTOR chunks, embed them, and add to ChromaDB.

    Returns:
        Number of chunks added.
    """
    chroma_dir = index_dir / "chroma"

    embedding_gen = EmbeddingGenerator(
        model_name=config.embeddings.model,
        backend=config.embeddings.backend,
        ollama_base_url=config.embeddings.ollama_base_url,
        document_prefix=config.embeddings.document_prefix,
    )
    vector_store = VectorStore(chroma_dir)

    # First, remove existing RAPTOR chunks so we don't have stale data
    logger.info("Removing existing RAPTOR chunks from ChromaDB...")
    try:
        existing = vector_store.collection.get(
            where={"chunk_type": {"$in": ["raptor_overview", "raptor_core"]}},
            include=[],
        )
        if existing["ids"]:
            logger.info(f"Deleting {len(existing['ids'])} existing RAPTOR chunks")
            # Delete in batches to avoid ChromaDB limits
            batch_size = 5000
            for i in range(0, len(existing["ids"]), batch_size):
                batch_ids = existing["ids"][i : i + batch_size]
                vector_store.collection.delete(ids=batch_ids)
            logger.info("Existing RAPTOR chunks removed")
        else:
            logger.info("No existing RAPTOR chunks found")
    except Exception as e:
        logger.warning(f"Could not remove existing RAPTOR chunks: {e}")

    # Create RAPTOR chunks for each paper
    all_chunks: list[EmbeddingChunk] = []
    for paper_id, raptor in raptor_summaries.items():
        pdata = paper_dicts.get(paper_id)
        analysis = extractions.get(paper_id)
        if not pdata or not analysis:
            continue

        # Derive quality_rating
        quality_rating = embedding_gen._derive_quality_rating(analysis.q21_quality)

        collections = pdata.get("collections", [])
        base_metadata = {
            "title": pdata.get("title", "Unknown"),
            "authors": pdata.get("author_string", "Unknown"),
            "year": pdata.get("publication_year"),
            "collections": ",".join(collections) if collections else "",
            "item_type": pdata.get("item_type", "journalArticle"),
            "quality_rating": quality_rating,
            "dimension_coverage": analysis.dimension_coverage,
        }

        if raptor.paper_overview:
            all_chunks.append(
                EmbeddingChunk(
                    paper_id=paper_id,
                    chunk_id=f"{paper_id}_raptor_overview",
                    chunk_type="raptor_overview",
                    text=embedding_gen._truncate_text(raptor.paper_overview),
                    metadata={**base_metadata, "dimension_group": "raptor"},
                )
            )
        if raptor.core_contribution:
            all_chunks.append(
                EmbeddingChunk(
                    paper_id=paper_id,
                    chunk_id=f"{paper_id}_raptor_core",
                    chunk_type="raptor_core",
                    text=embedding_gen._truncate_text(raptor.core_contribution),
                    metadata={**base_metadata, "dimension_group": "raptor"},
                )
            )

    logger.info(f"Created {len(all_chunks)} RAPTOR chunks for {len(raptor_summaries)} papers")

    if not all_chunks:
        logger.warning("No RAPTOR chunks to embed")
        return 0

    # Generate embeddings
    logger.info("Generating embeddings for RAPTOR chunks...")
    all_chunks = embedding_gen.generate_embeddings(all_chunks, batch_size=32)

    # Add to vector store
    logger.info("Adding RAPTOR chunks to ChromaDB...")
    added = vector_store.add_chunks(all_chunks, batch_size=100)
    logger.info(f"Added {added} RAPTOR chunks to ChromaDB")

    return added


def compute_similarity(index_dir: Path, config: Config) -> int:
    """Compute similarity pairs from raptor_overview embeddings.

    Delegates to the existing compute_similarity_pairs function in build_index.

    Returns:
        Number of pairs computed.
    """
    # Import here to avoid circular imports
    from scripts.build_index import compute_similarity_pairs

    return compute_similarity_pairs(
        index_dir=index_dir,
        embedding_model=config.embeddings.model,
        top_n=20,
        logger=logger,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild RAPTOR summaries and similarity pairs"
    )
    parser.add_argument(
        "--mode",
        choices=["template", "llm"],
        default="template",
        help="Generation mode: template (no LLM, free) or llm (API calls)",
    )
    parser.add_argument(
        "--provider",
        default="google",
        help="LLM provider for llm mode (default: google)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model for llm mode (default: provider default)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if cached",
    )
    parser.add_argument(
        "--skip-similarity",
        action="store_true",
        help="Only generate RAPTOR chunks, skip similarity computation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)

    # Load config
    config = Config.load(args.config)
    index_dir = PROJECT_ROOT / "data" / "index"

    # Load papers and extractions as raw dicts
    store = StructuredStore(index_dir)
    paper_dicts = store.load_papers()
    extraction_dicts = store.load_extractions()

    logger.info(f"Loaded {len(paper_dicts)} papers, {len(extraction_dicts)} extractions")

    # Parse extractions into SemanticAnalysis objects
    extractions: dict[str, SemanticAnalysis] = {}
    for pid, edata in extraction_dicts.items():
        try:
            inner = edata.get("extraction", edata) if isinstance(edata, dict) else edata
            extractions[pid] = SemanticAnalysis(**inner)
        except Exception as e:
            logger.debug(f"Could not parse extraction {pid}: {e}")

    # Papers that have both metadata and extractions
    common_ids = set(paper_dicts.keys()) & set(extractions.keys())
    logger.info(f"Papers with both metadata and extractions: {len(common_ids)}")

    if args.dry_run:
        cached = load_raptor_cache()
        to_generate = len(common_ids) - len(set(cached.keys()) & common_ids)
        logger.info(f"DRY RUN: Would generate {to_generate} RAPTOR summaries (mode={args.mode})")
        logger.info(f"DRY RUN: Would embed ~{len(common_ids) * 2} chunks (overview + core)")
        logger.info(f"DRY RUN: Would compute similarity pairs for {len(common_ids)} papers")
        return 0

    start_time = time.time()

    # Step 1: Generate RAPTOR summaries
    logger.info("=" * 60)
    logger.info("Step 1: Generating RAPTOR summaries")
    logger.info("=" * 60)
    raptor_summaries = generate_all_raptor_summaries(
        paper_dicts=paper_dicts,
        extractions=extractions,
        mode=args.mode,
        provider=args.provider,
        model=args.model,
        force=args.force,
    )

    if not raptor_summaries:
        logger.error("No RAPTOR summaries generated -- aborting")
        return 1

    # Step 2: Embed RAPTOR chunks into ChromaDB
    logger.info("=" * 60)
    logger.info("Step 2: Embedding RAPTOR chunks into ChromaDB")
    logger.info("=" * 60)
    chunks_added = embed_raptor_chunks(
        raptor_summaries=raptor_summaries,
        paper_dicts=paper_dicts,
        extractions=extractions,
        index_dir=index_dir,
        config=config,
    )

    # Step 3: Compute similarity pairs
    if not args.skip_similarity:
        logger.info("=" * 60)
        logger.info("Step 3: Computing similarity pairs")
        logger.info("=" * 60)
        pairs_count = compute_similarity(index_dir, config)
        logger.info(f"Similarity pairs computed: {pairs_count}")
    else:
        pairs_count = 0
        logger.info("Skipping similarity computation (--skip-similarity)")

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"RAPTOR + Similarity rebuild complete in {elapsed:.1f}s")
    logger.info(f"  RAPTOR summaries: {len(raptor_summaries)}")
    logger.info(f"  Chunks embedded: {chunks_added}")
    logger.info(f"  Similarity pairs: {pairs_count}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
