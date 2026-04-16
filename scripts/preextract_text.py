#!/usr/bin/env python
"""Pre-extract PDF text for all papers using the full extraction cascade.

Runs arXiv HTML -> PyMuPDF -> Marker -> OCR on every paper and caches
the cleaned text to data/cache/cascade_text/{paper_id}.txt. Subsequent
batch submissions read from this cache instead of extracting on the fly.

Usage:
    python scripts/preextract_text.py                # All papers
    python scripts/preextract_text.py --limit 100    # First 100
    python scripts/preextract_text.py --stats        # Show cache stats
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.extraction.cascade import ExtractionCascade
from src.extraction.opendataloader_extractor import build_hybrid_config
from src.extraction.pdf_extractor import PDFExtractor
from src.extraction.text_cleaner import TextCleaner
from src.utils.logging_config import setup_logging
from src.zotero.database import ZoteroDatabase

CACHE_DIR = Path("data/cache/cascade_text")


def get_cached_text(paper_id: str) -> str | None:
    """Read cached cascade text for a paper."""
    path = CACHE_DIR / f"{paper_id}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def save_cached_text(paper_id: str, text: str, method: str) -> None:
    """Save cascade text to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{paper_id}.txt"
    path.write_text(text, encoding="utf-8")
    # Save metadata alongside
    meta_path = CACHE_DIR / f"{paper_id}.method"
    meta_path.write_text(method, encoding="utf-8")


def cmd_extract(args, config):
    """Pre-extract text for all papers."""
    logger = setup_logging(level="DEBUG" if args.verbose else "INFO")

    db = ZoteroDatabase(config.zotero.database_path, config.zotero.storage_path)
    papers = [p for p in db.get_all_papers() if p.pdf_path and p.pdf_path.exists()]
    logger.info(f"Found {len(papers)} papers with PDFs")

    if args.limit:
        papers = papers[:args.limit]
        logger.info(f"Limited to {len(papers)}")

    # Skip already cached
    if not args.force:
        uncached = [p for p in papers if not (CACHE_DIR / f"{p.paper_id}.txt").exists()]
        logger.info(f"Already cached: {len(papers) - len(uncached)}, to extract: {len(uncached)}")
        papers = uncached

    if not papers:
        print("All papers already cached.")
        return

    pdf_extractor = PDFExtractor(
        enable_ocr=config.processing.ocr_enabled or config.processing.ocr_on_fail,
    )
    text_cleaner = TextCleaner()
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
    cascade = ExtractionCascade(
        pdf_extractor=pdf_extractor,
        enable_arxiv=config.processing.arxiv_enabled,
        enable_opendataloader=config.processing.opendataloader_enabled,
        enable_marker=config.processing.marker_enabled,
        opendataloader_mode=config.processing.opendataloader_mode,
        opendataloader_hybrid=hybrid_config,
        opendataloader_hybrid_fallback=(
            config.processing.opendataloader_hybrid_fallback
        ),
    )

    from collections import Counter
    methods = Counter()
    failed = 0
    start = time.time()

    for i, paper in enumerate(papers, 1):
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

            save_cached_text(paper.paper_id, text, result.method)
            methods[result.method] += 1

            if i % 25 == 0 or i == len(papers):
                elapsed = time.time() - start
                rate = i / elapsed
                remaining = (len(papers) - i) / rate if rate > 0 else 0
                print(
                    f"  {i}/{len(papers)} ({i/len(papers):.0%}) "
                    f"| {result.method:<10} "
                    f"| {len(text.split()):,} words "
                    f"| {remaining/60:.0f}m remaining",
                    flush=True,
                )
        except Exception as e:
            failed += 1
            logger.warning(f"Failed {paper.paper_id}: {e}")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} minutes")
    print(f"Methods: {dict(methods)}")
    print(f"Failed: {failed}")
    print(f"Cache: {CACHE_DIR}")


def cmd_stats(config):
    """Show cache statistics."""
    if not CACHE_DIR.exists():
        print("No cache directory yet.")
        return

    txt_files = list(CACHE_DIR.glob("*.txt"))
    method_files = list(CACHE_DIR.glob("*.method"))

    total_size = sum(f.stat().st_size for f in txt_files)

    from collections import Counter
    methods = Counter()
    for f in method_files:
        methods[f.read_text(encoding="utf-8").strip()] += 1

    db = ZoteroDatabase(config.zotero.database_path, config.zotero.storage_path)
    total_papers = sum(1 for p in db.get_all_papers() if p.pdf_path and p.pdf_path.exists())

    print(f"Cached: {len(txt_files)}/{total_papers} papers ({len(txt_files)/total_papers:.0%})")
    print(f"Size: {total_size/1024/1024:.0f} MB")
    print(f"Methods: {dict(methods)}")


def main():
    parser = argparse.ArgumentParser(description="Pre-extract PDF text via cascade")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Re-extract even if cached")
    parser.add_argument("--stats", action="store_true", help="Show cache stats")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    config = Config.load(args.config)

    if args.stats:
        cmd_stats(config)
    else:
        cmd_extract(args, config)


if __name__ == "__main__":
    sys.exit(main() or 0)
