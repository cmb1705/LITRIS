#!/usr/bin/env python
"""Run gap-fill across the full corpus for papers below a coverage threshold.

Uses Anthropic CLI to fill missing dimensions on papers already extracted
by OpenAI batch. Only runs the specific passes containing gaps.

Usage:
    python scripts/run_gap_fill.py                          # Default 90% threshold
    python scripts/run_gap_fill.py --threshold 0.85         # Custom threshold
    python scripts/run_gap_fill.py --threshold 0.90 --limit 50  # First 50
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.llm_council import (
    CouncilConfig,
    LLMCouncil,
    ProviderConfig,
    identify_gap_passes,
)
from src.analysis.schemas import SemanticAnalysis
from src.config import Config
from src.extraction.cascade import ExtractionCascade
from src.extraction.pdf_extractor import PDFExtractor
from src.extraction.text_cleaner import TextCleaner
from src.utils.logging_config import setup_logging
from src.zotero.database import ZoteroDatabase


def main():
    parser = argparse.ArgumentParser(description="Gap-fill across full corpus")
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--provider", default="anthropic")
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
        "journalArticle", "book", "bookSection", "thesis",
        "conferencePaper", "report", "preprint",
    }

    # Non-academic title patterns to skip
    skip_patterns = [
        "errata", "youtube", "playlist", ".jpg", ".png", ".ppt",
        "activity ", "sharing", "draft ", "call ", "ppt ", "d:\\",
    ]

    # Find academic papers below threshold
    candidates = []
    for pid, entry in extractions.items():
        ext = entry.get("extraction", {})
        filled = sum(
            1 for k, v in ext.items()
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
        candidates = candidates[:args.limit]

    print(f"Papers below {args.threshold:.0%}: {len(candidates)}")
    if not candidates:
        print("Nothing to gap-fill.")
        return

    # Setup
    council_config = CouncilConfig(
        providers=[
            ProviderConfig(
                name=args.provider,
                weight=1.0,
                timeout=600,
                mode="cli",
            ),
        ],
        aggregation_strategy="quality_weighted",
    )
    council = LLMCouncil(council_config)

    cascade_cache = Path("data/cache/cascade_text")
    pdf_extractor = PDFExtractor(
        enable_ocr=config.processing.ocr_enabled or config.processing.ocr_on_fail,
    )
    text_cleaner = TextCleaner()
    cascade = ExtractionCascade(
        pdf_extractor=pdf_extractor,
        enable_arxiv=config.processing.arxiv_enabled,
        enable_marker=config.processing.marker_enabled,
    )

    filled_total = 0
    processed = 0
    failed = 0

    for idx, (pid, filled, coverage) in enumerate(candidates, 1):
        paper = paper_lookup[pid]
        gap_passes = identify_gap_passes(
            SemanticAnalysis(**extractions[pid]["extraction"])
        )
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
                print(f"  Text extraction failed: {exc}")
                failed += 1
                continue
        else:
            print("  No PDF, skipping")
            failed += 1
            continue

        if not text:
            print("  Empty text, skipping")
            failed += 1
            continue

        authors = ", ".join(
            f"{a.last_name}, {a.first_name}" for a in (paper.authors or [])
        )

        try:
            analysis = SemanticAnalysis(**extractions[pid]["extraction"])
            gap_provider = ProviderConfig(
                name=args.provider, weight=1.0, timeout=600, mode="cli",
            )

            merged, gaps_filled = council.fill_gaps(
                analysis=analysis,
                paper_id=pid,
                title=paper.title,
                authors=authors,
                year=getattr(paper, "publication_year", None),
                item_type=paper.item_type,
                text=text[:100000],
                gap_provider=gap_provider,
            )

            if gaps_filled > 0:
                filled_total += gaps_filled
                new_filled = sum(
                    1 for k, v in merged.model_dump().items()
                    if k.startswith("q") and len(k) > 3 and k[1:3].isdigit() and v is not None
                )
                print(f"  Filled {gaps_filled} gaps ({filled}/40 -> {new_filled}/40)")

                extractions[pid]["extraction"] = merged.to_index_dict()
                extractions[pid]["gap_filled_by"] = args.provider
                extractions[pid]["gap_filled_count"] = gaps_filled
            else:
                print("  No new content from gap-fill")

            processed += 1

        except Exception as exc:
            print(f"  Gap-fill failed: {exc}")
            failed += 1

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
