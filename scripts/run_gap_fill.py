#!/usr/bin/env python
"""Run gap-fill across the full corpus for papers below a coverage threshold.

By default, the script auto-selects the opposite provider for each paper based
on its existing extraction model. Pass ``--provider`` to override that behavior
and force a single provider across the whole sweep.

Usage:
    python scripts/run_gap_fill.py                                 # Default 90% threshold
    python scripts/run_gap_fill.py --threshold 0.85                # Custom threshold
    python scripts/run_gap_fill.py --threshold 0.90 --limit 50     # First 50
    python scripts/run_gap_fill.py --provider anthropic --limit 50 # Force Anthropic
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.constants import (
    ANTHROPIC_MODELS,
    DEFAULT_MODELS,
    GEMINI_MODELS,
    OPENAI_MODELS,
)
from src.analysis.llm_council import (
    CouncilConfig,
    LLMCouncil,
    ProviderConfig,
    identify_gap_passes,
)
from src.analysis.schemas import SemanticAnalysis
from src.config import Config
from src.extraction.cascade import ExtractionCascade
from src.extraction.opendataloader_extractor import build_hybrid_config_from_processing
from src.extraction.pdf_extractor import PDFExtractor
from src.extraction.text_cleaner import TextCleaner
from src.utils.logging_config import setup_logging
from src.zotero.database import ZoteroDatabase

PROVIDER_OPPOSITES = {
    "anthropic": "openai",
    "openai": "anthropic",
}
MODEL_PROVIDER_MAP = {
    "anthropic": set(ANTHROPIC_MODELS) | {DEFAULT_MODELS["anthropic"]},
    "openai": set(OPENAI_MODELS) | {DEFAULT_MODELS["openai"]},
    "google": set(GEMINI_MODELS) | {DEFAULT_MODELS["google"]},
}
LOWER_MODEL_PROVIDER_MAP = {
    provider: {model.lower() for model in models} for provider, models in MODEL_PROVIDER_MAP.items()
}


def infer_provider_from_model(model_name: str | None) -> str | None:
    """Infer provider name from a stored extraction model string."""
    if not model_name:
        return None

    normalized = model_name.strip().lower()
    if not normalized:
        return None

    if normalized.startswith("synthesis:"):
        provider = normalized.split(":", 1)[1]
        return (
            provider
            if provider in LOWER_MODEL_PROVIDER_MAP or provider in PROVIDER_OPPOSITES
            else None
        )

    for provider, known_models in LOWER_MODEL_PROVIDER_MAP.items():
        if normalized in known_models:
            return provider

    if normalized.startswith("claude"):
        return "anthropic"
    if normalized.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if normalized.startswith("gemini"):
        return "google"
    if normalized.startswith(("llama", "qwen", "mistral")):
        return "ollama"
    return None


def resolve_gap_fill_provider(
    extraction_entry: dict,
    provider_override: str | None = None,
) -> str:
    """Resolve the provider to use for a corpus-wide gap-fill target.

    By default, picks the opposite of the original extraction provider for
    Anthropic/OpenAI papers. Unknown origins fall back to Anthropic so the
    script remains usable on mixed or partially migrated corpora.
    """
    if provider_override:
        return provider_override

    extraction = extraction_entry.get("extraction")
    original_model = None
    if isinstance(extraction, dict):
        original_model = extraction.get("extraction_model")
    if not isinstance(original_model, str):
        model = extraction_entry.get("model")
        original_model = model if isinstance(model, str) else None

    original_provider = infer_provider_from_model(original_model)
    if original_provider in PROVIDER_OPPOSITES:
        return PROVIDER_OPPOSITES[original_provider]

    prior_gap_provider = extraction_entry.get("gap_filled_by")
    if isinstance(prior_gap_provider, str) and prior_gap_provider in PROVIDER_OPPOSITES:
        return PROVIDER_OPPOSITES[prior_gap_provider]

    return "anthropic"


def main():
    parser = argparse.ArgumentParser(description="Gap-fill across full corpus")
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--provider",
        default=None,
        help="Optional provider override. Default: auto-select the opposite "
        "provider of each paper's existing extraction.",
    )
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(level="DEBUG" if args.verbose else "INFO")
    config = Config.load(args.config)
    config.extraction.apply_provider(args.provider)

    # Load corpus
    db = ZoteroDatabase(config.zotero.database_path, config.zotero.storage_path)
    paper_lookup = {p.paper_id: p for p in db.get_all_papers()}

    # Load extractions
    sa_path = Path("data/index/semantic_analyses.json")
    sa = json.loads(sa_path.read_text(encoding="utf-8"))
    extractions = sa.get("extractions", {})

    # Academic item types worth gap-filling
    academic_types = {
        "journalArticle",
        "book",
        "bookSection",
        "thesis",
        "conferencePaper",
        "report",
        "preprint",
    }

    # Non-academic title patterns to skip
    skip_patterns = [
        "errata",
        "youtube",
        "playlist",
        ".jpg",
        ".png",
        ".ppt",
        "activity 12",
        "sharing.jpg",
        "draft methods",
        "draft research",
        "ppt ",
        "dd form",
    ]

    # Find academic papers below threshold
    candidates = []
    for pid, entry in extractions.items():
        ext = entry.get("extraction", {})
        filled = sum(
            1
            for k, v in ext.items()
            if k.startswith("q") and len(k) > 3 and k[1:3].isdigit() and v is not None
        )
        coverage = filled / 40
        if coverage >= args.threshold or pid not in paper_lookup:
            continue

        paper = paper_lookup[pid]

        # Filter to academic works
        if paper.item_type not in academic_types:
            continue
        title_lower = paper.title.lower()
        if any(p in title_lower for p in skip_patterns):
            continue

        candidates.append((pid, filled, coverage))

    candidates.sort(key=lambda x: x[1])

    if args.limit:
        candidates = candidates[: args.limit]

    print(f"Papers below {args.threshold:.0%}: {len(candidates)}")
    if not candidates:
        print("Nothing to gap-fill.")
        return

    provider_counts = Counter(
        resolve_gap_fill_provider(extractions[pid], args.provider)
        for pid, _filled, _coverage in candidates
    )
    if args.provider:
        print(f"Provider override: {args.provider} ({len(candidates)} papers)")
    else:
        selection_summary = ", ".join(
            f"{provider}={count}" for provider, count in sorted(provider_counts.items())
        )
        print(f"Auto-selected providers: {selection_summary}")

    # Setup
    council_config = CouncilConfig(
        providers=[
            ProviderConfig(name="anthropic", weight=1.0, timeout=600, mode="cli"),
            ProviderConfig(name="openai", weight=1.0, timeout=600, mode="cli"),
        ],
        aggregation_strategy="quality_weighted",
    )
    cascade_cache = Path("data/cache/cascade_text")
    pdf_extractor = PDFExtractor(
        enable_ocr=config.processing.ocr_enabled or config.processing.ocr_on_fail,
    )
    text_cleaner = TextCleaner()
    hybrid_config = build_hybrid_config_from_processing(config.processing)
    cascade = ExtractionCascade(
        pdf_extractor=pdf_extractor,
        enable_arxiv=config.processing.arxiv_enabled,
        enable_opendataloader=config.processing.opendataloader_enabled,
        enable_marker=config.processing.marker_enabled,
        opendataloader_mode=config.processing.opendataloader_mode,
        opendataloader_hybrid=hybrid_config,
        opendataloader_hybrid_fallback=(config.processing.opendataloader_hybrid_fallback),
    )

    filled_total = 0
    processed = 0
    failed = 0
    save_lock = __import__("threading").Lock()

    def process_one(idx_and_candidate):
        """Process a single paper gap-fill (thread-safe)."""
        nonlocal filled_total, processed, failed
        idx, (pid, filled, coverage) = idx_and_candidate

        paper = paper_lookup[pid]
        try:
            gap_passes = identify_gap_passes(SemanticAnalysis(**extractions[pid]["extraction"]))
        except Exception as exc:
            print(f"  [{idx}] {pid[:20]}: parse error: {exc}")
            failed += 1
            return

        n_gaps = sum(len(fields) for fields in gap_passes.values())
        print(
            f"[{idx}/{len(candidates)}] {pid[:20]} "
            f"{filled}/40 ({coverage:.0%}) "
            f"-- {len(gap_passes)} pass(es), {n_gaps} gaps",
            flush=True,
        )

        # Get text
        cached_path = cascade_cache / f"{pid}.txt"
        if cached_path.exists():
            text = cached_path.read_text(encoding="utf-8")
        elif paper.pdf_path and paper.pdf_path.exists():
            try:
                result = cascade.extract_text(
                    paper.pdf_path,
                    doi=getattr(paper, "doi", None),
                    url=getattr(paper, "url", None),
                )
                text = result.text
                if not result.is_markdown:
                    text = text_cleaner.clean(text)
                text = text_cleaner.truncate_for_llm(text)
            except Exception as exc:
                print(f"  [{idx}] Text extraction failed: {exc}")
                failed += 1
                return
        else:
            print(f"  [{idx}] No PDF, skipping")
            failed += 1
            return

        if not text:
            print(f"  [{idx}] Empty text, skipping")
            failed += 1
            return

        authors = ", ".join(f"{a.last_name}, {a.first_name}" for a in (paper.authors or []))

        try:
            # Each thread gets its own council instance to avoid shared client state
            thread_council = LLMCouncil(council_config)
            analysis = SemanticAnalysis(**extractions[pid]["extraction"])
            gap_provider_name = resolve_gap_fill_provider(extractions[pid], args.provider)
            gap_provider = ProviderConfig(
                name=gap_provider_name,
                weight=1.0,
                timeout=600,
                mode="cli",
            )

            merged, gaps_filled = thread_council.fill_gaps(
                analysis=analysis,
                paper_id=pid,
                title=paper.title,
                authors=authors,
                year=getattr(paper, "publication_year", None),
                item_type=paper.item_type,
                text=text[:100000],
                gap_provider=gap_provider,
            )

            with save_lock:
                if gaps_filled > 0:
                    filled_total += gaps_filled
                    new_filled = sum(
                        1
                        for k, v in merged.model_dump().items()
                        if k.startswith("q") and len(k) > 3 and k[1:3].isdigit() and v is not None
                    )
                    print(f"  [{idx}] Filled {gaps_filled} gaps ({filled}/40 -> {new_filled}/40)")

                    extractions[pid]["extraction"] = merged.to_index_dict()
                    extractions[pid]["gap_filled_by"] = gap_provider_name
                    extractions[pid]["gap_filled_count"] = gaps_filled
                else:
                    print(f"  [{idx}] No new content from gap-fill")

                processed += 1

        except Exception as exc:
            print(f"  [{idx}] Gap-fill failed: {exc}")
            failed += 1

    # Run with parallel workers
    if args.parallel > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print(f"Using {args.parallel} parallel workers")
        indexed = list(enumerate(candidates, 1))
        chunk_size = args.parallel * 3

        for chunk_start in range(0, len(indexed), chunk_size):
            chunk = indexed[chunk_start : chunk_start + chunk_size]
            with ThreadPoolExecutor(max_workers=args.parallel) as executor:
                futures = {executor.submit(process_one, item): item for item in chunk}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"  Worker error: {exc}")

            # Save after each chunk
            with save_lock:
                sa["extractions"] = extractions
                sa["extraction_count"] = len(extractions)
                sa_path.write_text(
                    json.dumps(sa, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
                print(f"  [Saved: {filled_total} gaps filled, {processed} processed]")
    else:
        for idx, candidate in enumerate(candidates, 1):
            process_one((idx, candidate))

            # Save every 10 papers
            if idx % 10 == 0:
                sa["extractions"] = extractions
                sa["extraction_count"] = len(extractions)
                sa_path.write_text(
                    json.dumps(sa, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
                print(f"  [Saved checkpoint: {filled_total} gaps filled so far]")

    # Final save
    sa["extractions"] = extractions
    sa["extraction_count"] = len(extractions)
    sa["generated_at"] = datetime.now().isoformat()
    sa_path.write_text(
        json.dumps(sa, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print(f"\nDone: {processed} processed, {failed} failed, {filled_total} gaps filled")
    print("\nGenerate embeddings: python scripts/build_index.py --skip-extraction")


if __name__ == "__main__":
    sys.exit(main() or 0)
