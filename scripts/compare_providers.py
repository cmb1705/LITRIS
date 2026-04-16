#!/usr/bin/env python
"""Compare extraction quality between Anthropic and OpenAI providers.

Runs 40-dimension semantic extraction on papers with both providers for
side-by-side comparison. Supports single or batch mode.

Usage:
    # Single paper
    python scripts/compare_providers.py --key ABCD1234

    # Batch mode (5 random papers)
    python scripts/compare_providers.py --batch 5

    # Use existing Anthropic from index, only run OpenAI fresh
    python scripts/compare_providers.py --key ABCD1234 --anthropic-from-index

    # Save full JSON output
    python scripts/compare_providers.py --batch 3 --save
"""

import argparse
import json
import random
import sys
import textwrap
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.constants import DEFAULT_MODELS
from src.analysis.schemas import SemanticAnalysis
from src.analysis.section_extractor import SectionExtractor
from src.config import Config
from src.extraction.opendataloader_extractor import build_hybrid_config
from src.indexing.structured_store import StructuredStore
from src.zotero.database import ZoteroDatabase
from src.zotero.models import PaperMetadata

DIMENSION_LABELS: dict[str, str] = {
    "q01_research_question": "Research Question",
    "q02_thesis": "Central Thesis",
    "q03_key_claims": "Key Claims",
    "q04_evidence": "Evidence",
    "q05_limitations": "Limitations",
    "q06_paradigm": "Paradigm",
    "q07_methods": "Methods",
    "q08_data": "Data Sources",
    "q09_reproducibility": "Reproducibility",
    "q10_framework": "Framework",
    "q11_traditions": "Traditions",
    "q12_key_citations": "Key Citations",
    "q13_assumptions": "Assumptions",
    "q14_counterarguments": "Counterarguments",
    "q15_novelty": "Novelty",
    "q16_stance": "Stance",
    "q17_field": "Field",
    "q18_audience": "Audience",
    "q19_implications": "Implications",
    "q20_future_work": "Future Work",
    "q21_quality": "Quality",
    "q22_contribution": "Contribution",
    "q23_source_type": "Source Type",
    "q24_other": "Other (Pass 4)",
    "q25_institutional_context": "Institutional Context",
    "q26_historical_timing": "Historical Timing",
    "q27_paradigm_influence": "Paradigm Influence",
    "q28_disciplines_bridged": "Disciplines Bridged",
    "q29_cross_domain_insights": "Cross-Domain Insights",
    "q30_cultural_scope": "Cultural Scope",
    "q31_philosophical_assumptions": "Philosophical Assumptions",
    "q32_deployment_gap": "Deployment Gap",
    "q33_infrastructure_contribution": "Infrastructure Contribution",
    "q34_power_dynamics": "Power Dynamics",
    "q35_gaps_and_omissions": "Gaps & Omissions",
    "q36_dual_use_concerns": "Dual-Use Concerns",
    "q37_emergence_claims": "Emergence Claims",
    "q38_remaining_other": "Other (Pass 6)",
    "q39_network_properties": "Network Properties",
    "q40_policy_recommendations": "Policy Recommendations",
}


def get_paper_metadata(db: ZoteroDatabase, key: str) -> PaperMetadata | None:
    """Get paper metadata by Zotero key."""
    return db.get_paper_by_key(key)


def run_extraction(
    extractor: SectionExtractor,
    paper: PaperMetadata,
) -> dict:
    """Run extraction and return normalized result dict."""
    try:
        result, from_cache = extractor.extract_paper(paper)
        extraction = result.extraction

        dimensions = {}
        if result.success and extraction:
            for field_name in SemanticAnalysis.DIMENSION_FIELDS:
                dimensions[field_name] = getattr(extraction, field_name, None)

        return {
            "success": result.success,
            "from_cache": from_cache,
            "model": result.model_used,
            "duration_seconds": round(result.duration_seconds, 2),
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "dimension_coverage": (
                extraction.dimension_coverage
                if result.success and extraction
                else 0.0
            ),
            "coverage_flags": (
                extraction.coverage_flags
                if result.success and extraction
                else []
            ),
            "error": result.error,
            "dimensions": dimensions,
            "full_extraction": (
                extraction.model_dump()
                if result.success and extraction
                else None
            ),
        }
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e), "dimensions": {}}


def load_index_extraction(index_dir: Path, key: str) -> tuple[dict | None, str | None]:
    """Load an existing extraction from the index by Zotero key or paper_id."""
    store = StructuredStore(index_dir)
    extractions = store.load_extractions()

    if key in extractions:
        return extractions[key], key

    papers = store.load_papers()
    for paper_id, paper in papers.items():
        if paper.get("zotero_key") == key:
            if paper_id in extractions:
                return extractions[paper_id], paper_id

    return None, None


def format_index_extraction(entry: dict) -> dict:
    """Normalize index extraction entry to match compare output shape."""
    extraction = entry.get("extraction") if isinstance(entry, dict) else None
    if not extraction and isinstance(entry, dict):
        extraction = entry

    dimensions = {}
    if extraction:
        for field_name in SemanticAnalysis.DIMENSION_FIELDS:
            dimensions[field_name] = extraction.get(field_name)

    filled = sum(1 for v in dimensions.values() if v is not None)
    total = len(SemanticAnalysis.DIMENSION_FIELDS)

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
        "dimension_coverage": filled / total if total > 0 else 0.0,
        "coverage_flags": extraction.get("coverage_flags", []) if extraction else [],
        "error": None if extraction else "No extraction data found",
        "dimensions": dimensions,
        "full_extraction": extraction,
    }


def print_dimension_comparison(
    anthropic_dims: dict, openai_dims: dict, verbose: bool = False,
) -> dict:
    """Print dimension-by-dimension comparison and return agreement stats."""
    a_filled = sum(1 for v in anthropic_dims.values() if v is not None)
    o_filled = sum(1 for v in openai_dims.values() if v is not None)
    both_filled = sum(
        1 for k in SemanticAnalysis.DIMENSION_FIELDS
        if anthropic_dims.get(k) is not None and openai_dims.get(k) is not None
    )
    a_only = sum(
        1 for k in SemanticAnalysis.DIMENSION_FIELDS
        if anthropic_dims.get(k) is not None and openai_dims.get(k) is None
    )
    o_only = sum(
        1 for k in SemanticAnalysis.DIMENSION_FIELDS
        if anthropic_dims.get(k) is None and openai_dims.get(k) is not None
    )
    both_null = sum(
        1 for k in SemanticAnalysis.DIMENSION_FIELDS
        if anthropic_dims.get(k) is None and openai_dims.get(k) is None
    )

    total = len(SemanticAnalysis.DIMENSION_FIELDS)

    print("\n  Dimension Coverage:")
    print(f"    Anthropic filled: {a_filled}/{total}")
    print(f"    OpenAI filled:    {o_filled}/{total}")
    print(f"    Both filled:      {both_filled}")
    print(f"    Anthropic only:   {a_only}")
    print(f"    OpenAI only:      {o_only}")
    print(f"    Both null:        {both_null}")

    if verbose:
        print(f"\n  {'Dimension':<30} {'Anthropic':<8} {'OpenAI':<8}")
        print("  " + "-" * 46)
        for field_name in SemanticAnalysis.DIMENSION_FIELDS:
            label = DIMENSION_LABELS.get(field_name, field_name)
            a_val = anthropic_dims.get(field_name)
            o_val = openai_dims.get(field_name)
            a_status = "Yes" if a_val is not None else "-"
            o_status = "Yes" if o_val is not None else "-"
            marker = ""
            if a_val is not None and o_val is None:
                marker = " [A only]"
            elif a_val is None and o_val is not None:
                marker = " [O only]"
            print(f"  {label:<30} {a_status:<8} {o_status:<8}{marker}")

    # Print side-by-side for core dimensions
    core_dims = ["q01_research_question", "q02_thesis", "q03_key_claims", "q07_methods"]
    print("\n  Core Dimension Comparison (truncated):")
    print("  " + "-" * 70)
    for field_name in core_dims:
        label = DIMENSION_LABELS.get(field_name, field_name)
        a_val = anthropic_dims.get(field_name) or "(null)"
        o_val = openai_dims.get(field_name) or "(null)"
        a_trunc = textwrap.shorten(a_val, width=200, placeholder="...")
        o_trunc = textwrap.shorten(o_val, width=200, placeholder="...")
        print(f"\n  {label}:")
        print(f"    Anthropic: {a_trunc}")
        print(f"    OpenAI:    {o_trunc}")

    return {
        "a_filled": a_filled,
        "o_filled": o_filled,
        "both_filled": both_filled,
        "a_only": a_only,
        "o_only": o_only,
        "both_null": both_null,
    }


def create_extractor(
    config: Config, provider: str, mode: str, model: str, no_cache: bool,
) -> SectionExtractor:
    """Create a SectionExtractor for the given provider."""
    hybrid_config = build_hybrid_config(
        enabled=config.processing.opendataloader_hybrid_enabled,
        backend=config.processing.opendataloader_hybrid_backend,
        client_mode=config.processing.opendataloader_hybrid_client_mode,
        server_url=config.processing.opendataloader_hybrid_url,
        timeout_ms=config.processing.opendataloader_hybrid_timeout_ms,
        autostart=config.processing.opendataloader_hybrid_autostart,
        host=config.processing.opendataloader_hybrid_host,
        port=config.processing.opendataloader_hybrid_port,
        startup_timeout_seconds=(
            config.processing.opendataloader_hybrid_startup_timeout_seconds
        ),
        force_ocr=config.processing.opendataloader_hybrid_force_ocr,
        ocr_lang=config.processing.opendataloader_hybrid_ocr_lang,
        enrich_formula=config.processing.opendataloader_hybrid_enrich_formula,
        enrich_picture_description=(
            config.processing.opendataloader_hybrid_enrich_picture_description
        ),
        picture_description_prompt=(
            config.processing.opendataloader_hybrid_picture_description_prompt
        ),
        device=config.processing.opendataloader_hybrid_device,
    )
    return SectionExtractor(
        cache_dir=Path(config.storage.cache_path),
        provider=provider,
        mode=mode,
        model=model,
        timeout=config.extraction.timeout,
        ocr_on_fail=config.processing.ocr_on_fail,
        skip_non_publications=config.processing.skip_non_publications,
        min_publication_words=config.processing.min_publication_words,
        min_publication_pages=config.processing.min_publication_pages,
        min_section_hits=config.processing.min_section_hits,
        arxiv_enabled=config.processing.arxiv_enabled,
        opendataloader_enabled=config.processing.opendataloader_enabled,
        opendataloader_mode=config.processing.opendataloader_mode,
        opendataloader_hybrid_config=hybrid_config,
        opendataloader_hybrid_fallback=(
            config.processing.opendataloader_hybrid_fallback
        ),
        marker_enabled=config.processing.marker_enabled,
        use_cache=not no_cache,
    )


def select_batch_keys(db: ZoteroDatabase, count: int) -> list[str]:
    """Select random paper keys that have PDFs available."""
    all_papers = db.get_all_papers()
    papers_with_pdfs = [
        p for p in all_papers
        if p.pdf_path and p.pdf_path.exists()
    ]
    if len(papers_with_pdfs) < count:
        count = len(papers_with_pdfs)
    selected = random.sample(papers_with_pdfs, count)
    return [p.zotero_key for p in selected]


def compare_single_paper(
    key: str,
    db: ZoteroDatabase,
    config: Config,
    args: argparse.Namespace,
    paper_num: int = 1,
    total_papers: int = 1,
) -> dict | None:
    """Run comparison for a single paper. Returns results dict or None on failure."""
    paper = get_paper_metadata(db, key)
    if not paper:
        print(f"  Paper not found: {key}")
        return None
    if not paper.pdf_path or not paper.pdf_path.exists():
        print(f"  PDF not found for paper: {key}")
        return None

    print(f"\n{'=' * 70}")
    print(f"Paper {paper_num}/{total_papers}: {paper.title[:65]}")
    print(f"Key: {key}")
    print(f"{'=' * 70}")

    results = {}

    # Anthropic extraction
    if args.anthropic_from_index:
        entry, entry_key = load_index_extraction(args.index_dir, key)
        if entry:
            results["anthropic"] = format_index_extraction(entry)
            print(f"  Anthropic: loaded from index ({entry_key})")
        else:
            print("  Anthropic: not found in index, running live...")
            extractor = create_extractor(
                config, "anthropic", args.mode, args.anthropic_model, args.no_cache,
            )
            results["anthropic"] = run_extraction(extractor, paper)
    else:
        print("  Anthropic: extracting...")
        extractor = create_extractor(
            config, "anthropic", args.mode, args.anthropic_model, args.no_cache,
        )
        results["anthropic"] = run_extraction(extractor, paper)

    a = results["anthropic"]
    if a["success"]:
        print(f"    Model: {a.get('model', 'N/A')}  Duration: {a.get('duration_seconds', 'N/A')}s"
              f"  Coverage: {a.get('dimension_coverage', 0):.0%}")
    else:
        print(f"    FAILED: {a.get('error', 'Unknown')}")

    # OpenAI extraction
    print("  OpenAI: extracting...")
    extractor = create_extractor(
        config, "openai", args.mode, args.openai_model, args.no_cache,
    )
    results["openai"] = run_extraction(extractor, paper)

    o = results["openai"]
    if o["success"]:
        print(f"    Model: {o.get('model', 'N/A')}  Duration: {o.get('duration_seconds', 'N/A')}s"
              f"  Coverage: {o.get('dimension_coverage', 0):.0%}")
    else:
        print(f"    FAILED: {o.get('error', 'Unknown')}")

    # Dimension comparison
    if a["success"] and o["success"]:
        stats = print_dimension_comparison(
            a.get("dimensions", {}),
            o.get("dimensions", {}),
            verbose=args.verbose,
        )
        results["comparison_stats"] = stats

    # Save full extractions
    if args.save:
        output_dir = Path("data/comparison")
        output_dir.mkdir(parents=True, exist_ok=True)
        for provider, data in results.items():
            if provider == "comparison_stats":
                continue
            if data.get("full_extraction"):
                output_path = output_dir / f"{key}_{provider}.json"
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(
                        data["full_extraction"], f,
                        indent=2, ensure_ascii=False, default=str,
                    )
                print(f"\n  Saved {provider} -> {output_path}")

    return {
        "key": key,
        "title": paper.title,
        "anthropic": {
            k: v for k, v in a.items()
            if k not in ("dimensions", "full_extraction")
        },
        "openai": {
            k: v for k, v in o.items()
            if k not in ("dimensions", "full_extraction")
        },
        "comparison_stats": results.get("comparison_stats"),
    }


def print_batch_summary(all_results: list[dict]) -> None:
    """Print aggregate summary across all papers."""
    print(f"\n{'=' * 70}")
    print("BATCH SUMMARY")
    print(f"{'=' * 70}")

    successful = [r for r in all_results if r is not None]
    both_ok = [
        r for r in successful
        if r["anthropic"]["success"] and r["openai"]["success"]
    ]

    print(f"\nPapers compared: {len(successful)}")
    print(f"Both providers succeeded: {len(both_ok)}")

    if not both_ok:
        print("No successful comparisons to summarize.")
        return

    # Aggregate metrics
    print(f"\n{'Metric':<25} {'Anthropic':<20} {'OpenAI':<20}")
    print("-" * 65)

    a_durations = [r["anthropic"]["duration_seconds"] for r in both_ok if r["anthropic"].get("duration_seconds")]
    o_durations = [r["openai"]["duration_seconds"] for r in both_ok if r["openai"].get("duration_seconds")]

    if a_durations and o_durations:
        print(f"{'Avg duration (s)':<25} {sum(a_durations)/len(a_durations):<20.1f} {sum(o_durations)/len(o_durations):<20.1f}")

    a_coverages = [r["anthropic"]["dimension_coverage"] for r in both_ok]
    o_coverages = [r["openai"]["dimension_coverage"] for r in both_ok]
    print(f"{'Avg coverage':<25} {sum(a_coverages)/len(a_coverages):<20.1%} {sum(o_coverages)/len(o_coverages):<20.1%}")

    a_tokens = sum(r["anthropic"].get("input_tokens", 0) + r["anthropic"].get("output_tokens", 0) for r in both_ok)
    o_tokens = sum(r["openai"].get("input_tokens", 0) + r["openai"].get("output_tokens", 0) for r in both_ok)
    print(f"{'Total tokens':<25} {a_tokens:<20,} {o_tokens:<20,}")

    # Aggregate dimension stats
    stats_list = [r["comparison_stats"] for r in both_ok if r.get("comparison_stats")]
    if stats_list:
        print(f"\n{'Dimension Agreement':<25}")
        print("-" * 65)
        def avg(key: str) -> float:
            return sum(s[key] for s in stats_list) / len(stats_list)

        print(f"  {'Avg both filled:':<25} {avg('both_filled'):.1f}/40")
        print(f"  {'Avg Anthropic only:':<25} {avg('a_only'):.1f}")
        print(f"  {'Avg OpenAI only:':<25} {avg('o_only'):.1f}")
        print(f"  {'Avg both null:':<25} {avg('both_null'):.1f}")

    # Per-paper summary table
    print(f"\n{'Paper':<40} {'A Cov':<8} {'O Cov':<8} {'A Time':<8} {'O Time':<8}")
    print("-" * 72)
    for r in both_ok:
        title = r["title"][:38]
        a_cov = f"{r['anthropic']['dimension_coverage']:.0%}"
        o_cov = f"{r['openai']['dimension_coverage']:.0%}"
        a_time = f"{r['anthropic'].get('duration_seconds', 0):.1f}s"
        o_time = f"{r['openai'].get('duration_seconds', 0):.1f}s"
        print(f"  {title:<38} {a_cov:<8} {o_cov:<8} {a_time:<8} {o_time:<8}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare 40-dimension extraction between Anthropic and OpenAI",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--key", help="Zotero item key (single paper)")
    group.add_argument("--batch", type=int, help="Number of random papers to compare")
    group.add_argument("--keys", nargs="+", help="Multiple Zotero item keys")

    parser.add_argument(
        "--mode", default="cli", choices=["api", "cli"],
        help="Extraction mode (default: cli for subscription)",
    )
    parser.add_argument(
        "--openai-model", dest="openai_model", default="gpt-5.4",
        help="OpenAI model to use (default: gpt-5.4)",
    )
    parser.add_argument(
        "--anthropic-model", default=DEFAULT_MODELS["anthropic"],
        help=f"Anthropic model to use (default: {DEFAULT_MODELS['anthropic']})",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save full extractions to data/comparison/",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Skip cache and force fresh extraction",
    )
    parser.add_argument(
        "--anthropic-from-index", action="store_true",
        help="Use existing Anthropic extraction from data/index instead of running",
    )
    parser.add_argument(
        "--index-dir", type=Path, default=Path("data/index"),
        help="Index directory for existing extractions (default: data/index)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show per-dimension fill status table",
    )
    args = parser.parse_args()

    config = Config.load()
    db = ZoteroDatabase(config.zotero.database_path, config.zotero.storage_path)

    # Determine keys to compare
    if args.key:
        keys = [args.key]
    elif args.keys:
        keys = args.keys
    else:
        print(f"Selecting {args.batch} random papers with PDFs...")
        keys = select_batch_keys(db, args.batch)
        print(f"Selected keys: {keys}")

    all_results = []
    for i, key in enumerate(keys, 1):
        result = compare_single_paper(
            key, db, config, args,
            paper_num=i, total_papers=len(keys),
        )
        all_results.append(result)

    if len(keys) > 1:
        print_batch_summary(all_results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
