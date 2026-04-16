#!/usr/bin/env python
"""Compare individual provider extractions vs LLM Council consensus.

Runs each paper through individual providers (Anthropic, OpenAI) and then
through the LLM Council (all providers in parallel with consensus aggregation).
Generates a comparative report.

Usage:
    python scripts/council_comparison.py --keys S25LQ4TJ 9422KUGH ZPG44HF2 HG7NJ4FJ PWSCAC4U
    python scripts/council_comparison.py --batch 5 --diverse
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.llm_council import (
    CouncilConfig,
    CouncilResult,
    LLMCouncil,
    ProviderConfig,
)
from src.analysis.schemas import SemanticAnalysis
from src.analysis.section_extractor import SectionExtractor
from src.config import Config
from src.extraction.opendataloader_extractor import build_hybrid_config
from src.zotero.database import ZoteroDatabase
from src.zotero.models import PaperMetadata


def create_extractor(
    config: Config,
    provider: str,
    mode: str = "cli",  # noqa: PYI041 - intentionally str for flexibility
    model: str | None = None,
    effort: str | None = None,
    reasoning_effort: str | None = None,
) -> SectionExtractor:
    """Create a SectionExtractor with highest reasoning settings."""
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
        use_cache=False,  # Always fresh for comparison
        effort=effort,
        reasoning_effort=reasoning_effort,
    )


def run_individual_extraction(
    extractor: SectionExtractor,
    paper: PaperMetadata,
    provider_name: str,
) -> dict:
    """Run extraction with a single provider and return results."""
    print(f"  {provider_name}: extracting...", flush=True)
    start = time.time()
    try:
        result, _from_cache = extractor.extract_paper(paper)
        duration = time.time() - start
        extraction = result.extraction

        dimensions = {}
        if result.success and extraction:
            for field_name in SemanticAnalysis.DIMENSION_FIELDS:
                dimensions[field_name] = getattr(extraction, field_name, None)

        filled = sum(1 for v in dimensions.values() if v is not None)
        total = len(SemanticAnalysis.DIMENSION_FIELDS)

        print(
            f"    Model: {result.model_used}  Duration: {duration:.1f}s  "
            f"Coverage: {filled}/{total} ({filled/total:.0%})",
            flush=True,
        )

        return {
            "success": result.success,
            "model": result.model_used,
            "duration_seconds": round(duration, 2),
            "dimension_coverage": filled / total if total > 0 else 0.0,
            "filled_count": filled,
            "dimensions": dimensions,
            "full_extraction": (
                extraction.model_dump() if result.success and extraction else None
            ),
            "error": result.error,
        }
    except Exception as e:
        import traceback

        traceback.print_exc()
        duration = time.time() - start
        return {
            "success": False,
            "error": str(e),
            "duration_seconds": round(duration, 2),
            "dimensions": {},
        }


def run_council_extraction(
    paper: PaperMetadata,
    mode: str = "cli",
    strategy: str = "longest",
) -> dict:
    """Run LLM Council extraction and return results."""
    print(f"  Council: extracting (parallel, mode={mode}, strategy={strategy})...", flush=True)
    start = time.time()

    council_config = CouncilConfig(
        providers=[
            ProviderConfig(
                name="anthropic", weight=1.2, timeout=600, mode=mode,
            ),
            ProviderConfig(
                name="openai", weight=1.0, timeout=600, mode=mode,
            ),
        ],
        min_responses=2,
        fallback_to_single=True,
        parallel=True,
        timeout=900,  # 15 min overall
        aggregation_strategy=strategy,
    )

    council = LLMCouncil(council_config)

    # Load paper text via PDF extractor + text cleaner
    from src.extraction.pdf_extractor import PDFExtractor
    from src.extraction.text_cleaner import TextCleaner

    pdf_extractor = PDFExtractor()
    text_cleaner = TextCleaner()

    if not paper.pdf_path:
        return {
            "success": False,
            "error": "No PDF path",
            "dimensions": {},
        }
    text = pdf_extractor.extract_text(paper.pdf_path)
    if text:
        text = text_cleaner.clean(text)
        text = text_cleaner.truncate_for_llm(text)

    if not text:
        return {
            "success": False,
            "error": "Failed to load paper text",
            "dimensions": {},
        }

    # Build author string
    authors = ", ".join(
        f"{a.last_name}, {a.first_name}" for a in (paper.authors or [])
    )

    result: CouncilResult = council.extract(
        paper_id=paper.zotero_key,
        title=paper.title,
        authors=authors,
        year=getattr(paper, "year", None),
        item_type=paper.item_type,
        text=text[:100000],  # Truncate for API limits
    )

    duration = time.time() - start

    # Extract per-provider results
    provider_results = {}
    for resp in result.provider_responses:
        dims = {}
        if resp.success and resp.extraction:
            for field_name in SemanticAnalysis.DIMENSION_FIELDS:
                dims[field_name] = getattr(resp.extraction, field_name, None)
        filled = sum(1 for v in dims.values() if v is not None)
        provider_results[resp.provider] = {
            "success": resp.success,
            "duration_seconds": round(resp.duration_seconds, 2),
            "filled_count": filled,
            "coverage": filled / 40,
            "error": resp.error,
            "dimensions": dims,
        }
        status = f"{filled}/40 ({filled/40:.0%})" if resp.success else f"FAILED: {resp.error}"
        print(f"    {resp.provider}: {resp.duration_seconds:.1f}s - {status}", flush=True)

    # Consensus dimensions
    consensus_dims = {}
    consensus_filled = 0
    if result.success and result.consensus:
        for field_name in SemanticAnalysis.DIMENSION_FIELDS:
            consensus_dims[field_name] = getattr(result.consensus, field_name, None)
        consensus_filled = sum(1 for v in consensus_dims.values() if v is not None)

    print(
        f"    Consensus: {consensus_filled}/40 ({consensus_filled/40:.0%})  "
        f"Confidence: {result.consensus_confidence:.2f}  "
        f"Total: {duration:.1f}s",
        flush=True,
    )

    return {
        "success": result.success,
        "consensus_confidence": result.consensus_confidence,
        "duration_seconds": round(duration, 2),
        "consensus_filled": consensus_filled,
        "consensus_coverage": consensus_filled / 40,
        "consensus_dimensions": consensus_dims,
        "provider_results": provider_results,
        "errors": result.errors,
        "full_consensus": (
            result.consensus.model_dump() if result.success and result.consensus else None
        ),
    }


def compare_dimensions(
    individual: dict[str, dict],
    council: dict,
) -> dict:
    """Compare individual provider dimensions vs council consensus."""
    consensus_dims = council.get("consensus_dimensions", {})

    stats = {
        "council_only": [],  # Dimensions filled by council but not by CLI individuals
        "individual_only": {},  # Per-provider: dims filled by CLI but not council
        "council_longer": 0,  # Count where council response is longer
        "individual_longer": {},
    }

    for provider_name, indiv_data in individual.items():
        indiv_dims = indiv_data.get("dimensions", {})
        stats["individual_only"][provider_name] = []
        stats["individual_longer"][provider_name] = 0

        for field_name in SemanticAnalysis.DIMENSION_FIELDS:
            c_val = consensus_dims.get(field_name)
            i_val = indiv_dims.get(field_name)

            if c_val and not i_val:
                if field_name not in stats["council_only"]:
                    stats["council_only"].append(field_name)
            elif i_val and not c_val:
                stats["individual_only"][provider_name].append(field_name)

            if c_val and i_val:
                if len(c_val) > len(i_val):
                    stats["council_longer"] += 1
                else:
                    stats["individual_longer"][provider_name] += 1

    return stats


def print_paper_comparison(
    paper: PaperMetadata,
    individual: dict[str, dict],
    council: dict,
    paper_num: int,
    total: int,
) -> dict:
    """Print detailed comparison for one paper."""
    print(f"\n{'=' * 75}")
    print(f"Paper {paper_num}/{total}: {paper.title[:70]}")
    print(f"Type: {paper.item_type}  Key: {paper.zotero_key}")
    print(f"{'=' * 75}")

    # Coverage summary table
    print(f"\n  {'Source':<25} {'Coverage':<12} {'Time':<10} {'Status'}")
    print("  " + "-" * 60)

    for name, data in individual.items():
        cov = f"{data.get('filled_count', 0)}/40 ({data.get('dimension_coverage', 0):.0%})"
        t = f"{data.get('duration_seconds', 0):.1f}s"
        status = "OK" if data.get("success") else f"FAIL: {data.get('error', '')[:30]}"
        print(f"  {name + ' (CLI)':<25} {cov:<12} {t:<10} {status}")

    council_providers = council.get("provider_results", {})
    for name, data in council_providers.items():
        cov = f"{data.get('filled_count', 0)}/40 ({data.get('coverage', 0):.0%})"
        t = f"{data.get('duration_seconds', 0):.1f}s"
        status = "OK" if data.get("success") else f"FAIL: {data.get('error', '')[:30]}"
        print(f"  {name + ' (Council)':<25} {cov:<12} {t:<10} {status}")

    c_filled = council.get("consensus_filled", 0)
    c_cov = f"{c_filled}/40 ({council.get('consensus_coverage', 0):.0%})"
    c_time = f"{council.get('duration_seconds', 0):.1f}s"
    c_conf = f"conf={council.get('consensus_confidence', 0):.2f}"
    print(f"  {'COUNCIL CONSENSUS':<25} {c_cov:<12} {c_time:<10} {c_conf}")

    # Dimension-level comparison
    comp = compare_dimensions(individual, council)

    if comp["council_only"]:
        print(f"\n  Council filled but CLI individuals missed: {comp['council_only']}")
    for prov, dims in comp["individual_only"].items():
        if dims:
            print(f"  {prov} CLI filled but council missed: {dims}")

    # Core dimension text comparison
    core = ["q01_research_question", "q02_thesis", "q07_methods"]
    labels = {"q01_research_question": "Research Question", "q02_thesis": "Thesis", "q07_methods": "Methods"}

    print("\n  Core Dimension Comparison:")
    print("  " + "-" * 70)

    consensus_dims = council.get("consensus_dimensions", {})

    for field_name in core:
        label = labels[field_name]
        print(f"\n  {label}:")

        for prov_name, data in individual.items():
            val = data.get("dimensions", {}).get(field_name, "(null)")
            if val:
                import textwrap
                val = textwrap.shorten(val, width=150, placeholder="...")
            print(f"    {prov_name} CLI: {val}")

        c_val = consensus_dims.get(field_name, "(null)")
        if c_val:
            import textwrap
            c_val = textwrap.shorten(c_val, width=150, placeholder="...")
        print(f"    CONSENSUS:     {c_val}")

    return {
        "key": paper.zotero_key,
        "title": paper.title,
        "item_type": paper.item_type,
        "individual": {
            k: {kk: vv for kk, vv in v.items() if kk != "dimensions" and kk != "full_extraction"}
            for k, v in individual.items()
        },
        "council": {
            k: v for k, v in council.items()
            if k not in ("consensus_dimensions", "provider_results", "full_consensus")
        },
        "council_providers": {
            k: {kk: vv for kk, vv in v.items() if kk != "dimensions"}
            for k, v in council_providers.items()
        },
        "comparison": comp,
    }


def print_batch_summary(all_results: list[dict]) -> None:
    """Print aggregate summary."""
    print(f"\n{'=' * 75}")
    print("AGGREGATE SUMMARY: Individual CLI vs LLM Council")
    print(f"{'=' * 75}")

    print(f"\nPapers compared: {len(all_results)}")

    # Coverage comparison
    print(f"\n{'Source':<25} {'Avg Coverage':<15} {'Avg Time':<12}")
    print("-" * 52)

    # Individual providers
    providers_seen = set()
    for r in all_results:
        providers_seen.update(r.get("individual", {}).keys())

    for prov in sorted(providers_seen):
        coverages = [
            r["individual"][prov]["dimension_coverage"]
            for r in all_results
            if prov in r.get("individual", {}) and r["individual"][prov].get("success")
        ]
        times = [
            r["individual"][prov]["duration_seconds"]
            for r in all_results
            if prov in r.get("individual", {}) and r["individual"][prov].get("success")
        ]
        if coverages:
            avg_cov = sum(coverages) / len(coverages)
            avg_time = sum(times) / len(times)
            print(f"  {prov + ' (CLI)':<23} {avg_cov:<15.1%} {avg_time:<12.1f}s")

    # Council providers
    council_providers_seen = set()
    for r in all_results:
        council_providers_seen.update(r.get("council_providers", {}).keys())

    for prov in sorted(council_providers_seen):
        coverages = [
            r["council_providers"][prov]["coverage"]
            for r in all_results
            if prov in r.get("council_providers", {}) and r["council_providers"][prov].get("success")
        ]
        times = [
            r["council_providers"][prov]["duration_seconds"]
            for r in all_results
            if prov in r.get("council_providers", {}) and r["council_providers"][prov].get("success")
        ]
        if coverages:
            avg_cov = sum(coverages) / len(coverages)
            avg_time = sum(times) / len(times)
            print(f"  {prov + ' (Council)':<23} {avg_cov:<15.1%} {avg_time:<12.1f}s")

    # Council consensus
    c_coverages = [r["council"]["consensus_coverage"] for r in all_results if r["council"].get("success")]
    c_times = [r["council"]["duration_seconds"] for r in all_results if r["council"].get("success")]
    c_confs = [r["council"]["consensus_confidence"] for r in all_results if r["council"].get("success")]

    if c_coverages:
        print(f"  {'CONSENSUS':<23} {sum(c_coverages)/len(c_coverages):<15.1%} {sum(c_times)/len(c_times):<12.1f}s")
        print(f"\n  Avg consensus confidence: {sum(c_confs)/len(c_confs):.2f}")

    # Per-paper table
    print(f"\n{'Paper':<35} {'Type':<15} {'A CLI':<7} {'O CLI':<7} {'Council':<8} {'Conf':<6}")
    print("-" * 78)
    for r in all_results:
        title = r["title"][:33]
        itype = r["item_type"][:13]
        a_cov = r["individual"].get("anthropic", {}).get("dimension_coverage", 0)
        o_cov = r["individual"].get("openai", {}).get("dimension_coverage", 0)
        c_cov = r["council"].get("consensus_coverage", 0)
        conf = r["council"].get("consensus_confidence", 0)
        print(f"  {title:<33} {itype:<15} {a_cov:<7.0%} {o_cov:<7.0%} {c_cov:<8.0%} {conf:<6.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare individual extractions vs LLM Council consensus",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--keys", nargs="+", help="Zotero item keys")
    group.add_argument("--batch", type=int, help="Number of random papers")

    parser.add_argument(
        "--diverse", action="store_true",
        help="Select diverse document types (with --batch)",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save full extractions to data/comparison/council/",
    )
    parser.add_argument(
        "--mode", choices=["cli", "api"], default="cli",
        help="Extraction mode for council providers (default: cli)",
    )
    parser.add_argument(
        "--strategy", choices=["longest", "quality_weighted", "union"],
        default="quality_weighted",
        help="Aggregation strategy for council consensus (default: quality_weighted)",
    )
    args = parser.parse_args()

    config = Config.load()
    db = ZoteroDatabase(config.zotero.database_path, config.zotero.storage_path)

    # Determine keys
    if args.keys:
        keys = args.keys
    else:
        papers = list(db.get_all_papers())
        with_pdfs = [p for p in papers if p.pdf_path and p.pdf_path.exists()]

        if args.diverse:
            import random
            by_type: dict[str, list] = {}
            for p in with_pdfs:
                by_type.setdefault(p.item_type, []).append(p)

            targets = ["journalArticle", "book", "bookSection", "thesis", "conferencePaper"]
            selected = []
            for t in targets:
                if t in by_type:
                    selected.append(random.choice(by_type[t]))
            # Fill remaining with random
            while len(selected) < args.batch:
                selected.append(random.choice(with_pdfs))
            keys = [p.zotero_key for p in selected[:args.batch]]
        else:
            import random
            selected = random.sample(with_pdfs, min(args.batch, len(with_pdfs)))
            keys = [p.zotero_key for p in selected]

    print(f"Selected keys: {keys}", flush=True)

    # Create individual extractors with highest reasoning
    anthropic_extractor = create_extractor(
        config, "anthropic", mode=args.mode, effort="high",
    )
    openai_extractor = create_extractor(
        config, "openai", mode=args.mode, reasoning_effort="xhigh",
    )

    all_results = []

    for i, key in enumerate(keys, 1):
        paper = db.get_paper_by_key(key)
        if not paper:
            print(f"\nPaper not found: {key}", flush=True)
            continue
        if not paper.pdf_path or not paper.pdf_path.exists():
            print(f"\nPDF not found for {key}", flush=True)
            continue

        # Phase 1: Individual extractions (CLI mode, highest reasoning)
        individual = {}

        result_a = run_individual_extraction(anthropic_extractor, paper, "anthropic")
        individual["anthropic"] = result_a

        result_o = run_individual_extraction(openai_extractor, paper, "openai")
        individual["openai"] = result_o

        # Phase 2: Council extraction (API mode, parallel)
        council = run_council_extraction(paper, mode=args.mode, strategy=args.strategy)

        # Phase 3: Compare and print
        summary = print_paper_comparison(
            paper, individual, council,
            paper_num=i, total=len(keys),
        )
        all_results.append(summary)

        # Save if requested
        if args.save:
            output_dir = Path("data/comparison/council")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save individual extractions
            for prov_name, data in individual.items():
                if data.get("full_extraction"):
                    path = output_dir / f"{key}_{prov_name}_cli.json"
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(data["full_extraction"], f, indent=2, ensure_ascii=False, default=str)

            # Save council consensus
            if council.get("full_consensus"):
                path = output_dir / f"{key}_council_consensus.json"
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(council["full_consensus"], f, indent=2, ensure_ascii=False, default=str)

    # Batch summary
    if len(all_results) > 1:
        print_batch_summary(all_results)

    # Save summary report
    if args.save and all_results:
        output_dir = Path("data/comparison/council")
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "council_comparison_summary.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nSaved summary to {report_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
