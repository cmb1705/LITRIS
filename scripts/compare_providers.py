#!/usr/bin/env python
"""Compare extraction quality between Anthropic and OpenAI providers.

Runs extraction on a single paper with both providers for side-by-side comparison.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.section_extractor import SectionExtractor
from src.config import Config
from src.indexing.structured_store import StructuredStore
from src.zotero.database import ZoteroDatabase
from src.zotero.models import PaperMetadata


def get_paper_metadata(db: ZoteroDatabase, key: str) -> PaperMetadata | None:
    """Get paper metadata by Zotero key."""
    return db.get_paper_by_key(key)


def run_extraction(
    extractor: SectionExtractor,
    paper: PaperMetadata,
) -> dict:
    """Run extraction with the given extractor."""
    try:
        result, from_cache = extractor.extract_paper(paper)

        return {
            "success": result.success,
            "from_cache": from_cache,
            "model": result.model_used,
            "duration_seconds": round(result.duration_seconds, 2),
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "confidence": (
                result.extraction.extraction_confidence
                if result.success and result.extraction
                else None
            ),
            "error": result.error,
            "thesis": (
                result.extraction.thesis_statement[:200] + "..."
                if result.success
                and result.extraction
                and result.extraction.thesis_statement
                else None
            ),
            "num_findings": (
                len(result.extraction.key_findings)
                if result.success and result.extraction
                else 0
            ),
            "num_claims": (
                len(result.extraction.key_claims)
                if result.success and result.extraction
                else 0
            ),
            "keywords": (
                result.extraction.keywords[:5]
                if result.success and result.extraction
                else []
            ),
            "full_extraction": (
                result.extraction.model_dump()
                if result.success and result.extraction
                else None
            ),
        }
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e)}


def load_index_extraction(index_dir: Path, key: str) -> tuple[dict | None, str | None]:
    """Load an existing extraction from the index by Zotero key or paper_id."""
    store = StructuredStore(index_dir)
    extractions = store.load_extractions()

    # Direct lookup by key (works for old-style Zotero-key entries)
    if key in extractions:
        return extractions[key], key

    # Try to resolve zotero_key -> paper_id
    papers = store.load_papers()
    if key in papers and key in extractions:
        return extractions[key], key

    for paper_id, paper in papers.items():
        if paper.get("zotero_key") == key:
            if paper_id in extractions:
                return extractions[paper_id], paper_id
            if key in extractions:
                return extractions[key], key

    return None, None


def format_index_extraction(entry: dict) -> dict:
    """Normalize index extraction entry to match compare output shape."""
    extraction = entry.get("extraction") if isinstance(entry, dict) else None
    if not extraction and isinstance(entry, dict):
        extraction = entry

    def _safe_len(value):
        return len(value) if isinstance(value, list) else 0

    thesis = None
    if extraction and extraction.get("thesis_statement"):
        thesis = extraction["thesis_statement"][:200] + "..."

    return {
        "success": extraction is not None,
        "from_cache": True,
        "model": entry.get("model") if isinstance(entry, dict) else None,
        "duration_seconds": (
            round(entry.get("duration", entry.get("duration_seconds", 0.0)), 2)
            if isinstance(entry, dict)
            else None
        ),
        "input_tokens": entry.get("input_tokens") if isinstance(entry, dict) else None,
        "output_tokens": entry.get("output_tokens") if isinstance(entry, dict) else None,
        "confidence": extraction.get("extraction_confidence") if extraction else None,
        "error": None if extraction else "No extraction data found",
        "thesis": thesis,
        "num_findings": _safe_len(extraction.get("key_findings") if extraction else None),
        "num_claims": _safe_len(extraction.get("key_claims") if extraction else None),
        "keywords": (
            extraction.get("keywords", [])[:5] if extraction else []
        ),
        "full_extraction": extraction,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare extraction between providers")
    parser.add_argument("--key", required=True, help="Zotero item key")
    parser.add_argument(
        "--mode",
        default="cli",
        choices=["api", "cli"],
        help="Extraction mode (default: cli for subscription)",
    )
    parser.add_argument(
        "--model",
        dest="openai_model",
        default="gpt-5.2",
        help="OpenAI model to use (alias for --openai-model)",
    )
    parser.add_argument(
        "--openai-model",
        dest="openai_model",
        default="gpt-5.2",
        help="OpenAI model to use (default: gpt-5.2)",
    )
    parser.add_argument(
        "--anthropic-model",
        default="claude-opus-4-5-20251101",
        help="Anthropic model to use (default: claude-opus-4-5-20251101)",
    )
    parser.add_argument("--save", action="store_true", help="Save full extractions to files")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip cache and force fresh extraction",
    )
    parser.add_argument(
        "--anthropic-from-index",
        action="store_true",
        help="Use existing Anthropic extraction from data/index instead of running",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/index"),
        help="Index directory for existing extractions (default: data/index)",
    )
    args = parser.parse_args()

    # Load config
    config = Config.load()
    cache_dir = Path(config.storage.cache_path)

    # Get paper
    print(f"Loading paper {args.key}...")
    db = ZoteroDatabase(config.zotero.database_path, config.zotero.storage_path)
    paper = get_paper_metadata(db, args.key)

    if not paper:
        print(f"Paper not found: {args.key}")
        return 1

    if not paper.pdf_path or not paper.pdf_path.exists():
        print(f"PDF not found for paper: {args.key}")
        return 1

    print(f"Title: {paper.title}")
    print(f"PDF: {paper.pdf_path}")

    results = {}

    # Run or load Anthropic extraction
    print("\n" + "=" * 70)
    print("Running ANTHROPIC (Claude) extraction...")
    print("=" * 70)

    if args.anthropic_from_index:
        entry, entry_key = load_index_extraction(args.index_dir, args.key)
        if not entry:
            print(f"  Error: No extraction found in index for key {args.key}")
            return 1
        results["anthropic"] = format_index_extraction(entry)
        print(f"  Loaded from index: {entry_key}")
        print(f"  Success: {results['anthropic']['success']}")
        print(f"  Model: {results['anthropic'].get('model', 'N/A')}")
        print(f"  Duration: {results['anthropic'].get('duration_seconds', 'N/A')}s")
        if results["anthropic"]["success"]:
            print(f"  Confidence: {results['anthropic']['confidence']}")
    else:
        anthropic_extractor = SectionExtractor(
            cache_dir=cache_dir,
            provider="anthropic",
            mode=args.mode,
            model=args.anthropic_model,
            timeout=config.extraction.timeout,
            ocr_on_fail=config.processing.ocr_on_fail,
            skip_non_publications=config.processing.skip_non_publications,
            min_publication_words=config.processing.min_publication_words,
            min_publication_pages=config.processing.min_publication_pages,
            min_section_hits=config.processing.min_section_hits,
            use_cache=not args.no_cache,
        )
        results["anthropic"] = run_extraction(anthropic_extractor, paper)
        print(f"  Success: {results['anthropic']['success']}")
        print(f"  From cache: {results['anthropic'].get('from_cache', False)}")
        print(f"  Duration: {results['anthropic'].get('duration_seconds', 'N/A')}s")
        if results["anthropic"]["success"]:
            print(f"  Confidence: {results['anthropic']['confidence']}")
        elif results["anthropic"].get("error"):
            print(f"  Error: {results['anthropic']['error']}")

    # Run OpenAI extraction
    print("\n" + "=" * 70)
    print("Running OPENAI (GPT) extraction...")
    print("=" * 70)

    openai_extractor = SectionExtractor(
        cache_dir=cache_dir,
        provider="openai",
        mode=args.mode,
        model=args.openai_model,
        timeout=config.extraction.timeout,
        ocr_on_fail=config.processing.ocr_on_fail,
        skip_non_publications=config.processing.skip_non_publications,
        min_publication_words=config.processing.min_publication_words,
        min_publication_pages=config.processing.min_publication_pages,
        min_section_hits=config.processing.min_section_hits,
        use_cache=not args.no_cache,
    )
    results["openai"] = run_extraction(openai_extractor, paper)
    print(f"  Success: {results['openai']['success']}")
    print(f"  From cache: {results['openai'].get('from_cache', False)}")
    print(f"  Duration: {results['openai'].get('duration_seconds', 'N/A')}s")
    if results["openai"]["success"]:
        print(f"  Confidence: {results['openai']['confidence']}")
    elif results["openai"].get("error"):
        print(f"  Error: {results['openai']['error']}")

    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\nPaper: {paper.title[:70]}...")

    metrics = ["success", "duration_seconds", "confidence", "num_findings", "num_claims"]
    print(f"\n{'Metric':<20} {'Anthropic':<25} {'OpenAI':<25}")
    print("-" * 70)
    for metric in metrics:
        a_val = results["anthropic"].get(metric, "N/A")
        o_val = results["openai"].get(metric, "N/A")
        print(f"{metric:<20} {str(a_val):<25} {str(o_val):<25}")

    # Thesis comparison
    print("\n" + "-" * 70)
    print("THESIS COMPARISON")
    print("-" * 70)
    print(f"\nAnthropic thesis:\n  {results['anthropic'].get('thesis', 'N/A')}")
    print(f"\nOpenAI thesis:\n  {results['openai'].get('thesis', 'N/A')}")

    # Keywords comparison
    print("\n" + "-" * 70)
    print("KEYWORDS COMPARISON")
    print("-" * 70)
    print(f"Anthropic: {results['anthropic'].get('keywords', [])}")
    print(f"OpenAI: {results['openai'].get('keywords', [])}")

    # Save full extractions if requested
    if args.save:
        output_dir = Path("data/comparison")
        output_dir.mkdir(parents=True, exist_ok=True)

        for provider, data in results.items():
            if data.get("full_extraction"):
                output_path = output_dir / f"{args.key}_{provider}.json"
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(
                        data["full_extraction"],
                        f,
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    )
                print(f"\nSaved {provider} extraction to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
