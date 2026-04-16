"""Benchmark OpenDataLoader PDF against PyMuPDF and Marker.

Compares extraction quality and speed across extractors for diverse
PDF types in the LITRIS corpus, following the methodology from
benchmark_glm_ocr.py.

Usage:
    python scripts/benchmark_opendataloader.py
    python scripts/benchmark_opendataloader.py --quick   # 5 papers only

Set `LITRIS_OPENDATALOADER_TARGETS_JSON` to point at a private JSON file
with corpus-specific benchmark targets if you want a curated sample.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.extraction.opendataloader_extractor import build_hybrid_config
from src.extraction.text_cleaner import TextCleaner
from src.zotero.database import ZoteroDatabase

BENCHMARK_TARGETS_ENV = "LITRIS_OPENDATALOADER_TARGETS_JSON"


def word_count(text: str) -> int:
    """Count words in text."""
    if not text:
        return 0
    return len(text.split())


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Benchmark a smaller sample size.",
    )
    parser.add_argument(
        "--include-hybrid",
        action="store_true",
        help="Benchmark OpenDataLoader hybrid when a backend is reachable.",
    )
    return parser.parse_args()


def extract_pymupdf(pdf_path: Path) -> tuple[str, float]:
    """Extract text using PyMuPDF. Returns (text, elapsed_seconds)."""
    import fitz

    start = time.perf_counter()
    try:
        doc = fitz.open(str(pdf_path))
        parts: list[str] = []
        for page in doc:
            parts.append(str(page.get_text()))
        doc.close()
        text = "".join(parts)
    except Exception as exc:
        text = f"ERROR: {exc}"
    elapsed = time.perf_counter() - start
    return text, elapsed


def extract_opendataloader(
    pdf_path: Path,
    mode: str = "fast",
    use_struct_tree: bool = False,
    hybrid_config=None,
) -> tuple[str, float]:
    """Extract text using OpenDataLoader PDF.

    Returns (text, elapsed_seconds).
    """
    from src.extraction import opendataloader_extractor

    if not opendataloader_extractor.is_available():
        return "UNAVAILABLE: opendataloader-pdf or Java 11+ not found", 0.0

    start = time.perf_counter()
    try:
        text = opendataloader_extractor.extract_with_opendataloader(
            pdf_path,
            mode=mode,
            use_struct_tree=use_struct_tree,
            hybrid_config=hybrid_config,
        )
        if text is None:
            text = "EMPTY: no text extracted"
    except Exception as exc:
        text = f"ERROR: {exc}"
    elapsed = time.perf_counter() - start
    return text, elapsed


def extract_marker(pdf_path: Path) -> tuple[str, float]:
    """Extract text using Marker. Returns (text, elapsed_seconds)."""
    from src.extraction import marker_extractor

    if not marker_extractor.is_available():
        return "UNAVAILABLE: marker-pdf not installed", 0.0

    start = time.perf_counter()
    try:
        text = marker_extractor.extract_with_marker(pdf_path)
        if text is None:
            text = "EMPTY: no text extracted"
    except Exception as exc:
        text = f"ERROR: {exc}"
    elapsed = time.perf_counter() - start
    return text, elapsed


def get_page_count(pdf_path: Path) -> int:
    """Get PDF page count via PyMuPDF."""
    try:
        import fitz

        doc = fitz.open(str(pdf_path))
        count = len(doc)
        doc.close()
        return count
    except Exception:
        return 0


DEFAULT_TARGET_LIMIT = 15
QUICK_TARGET_LIMIT = 5


def is_error_text(text: str) -> bool:
    """Check if text represents an extraction error/unavailability."""
    return text.startswith(("ERROR:", "UNAVAILABLE:", "EMPTY:"))


def load_targets_from_env() -> list[tuple[str, str, str, str]] | None:
    """Load corpus-specific benchmark targets from a JSON file."""
    raw_path = os.environ.get(BENCHMARK_TARGETS_ENV)
    if not raw_path:
        return None

    path = Path(raw_path).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        (
            str(item["key"]),
            str(item["description"]),
            str(item.get("category", "custom")),
            str(item.get("odl_mode", "fast")),
        )
        for item in payload
    ]


def categorize_title(title: str) -> str:
    """Infer a loose benchmark category from a paper title."""
    lowered = title.lower()
    if any(token in lowered for token in ("proceedings", "conference", "workshop")):
        return "multi_column"
    if any(token in lowered for token in ("equation", "optimization", "physics", "math")):
        return "equations"
    if any(token in lowered for token in ("book", "handbook", "guide")):
        return "book"
    return "auto"


def autodiscover_targets(
    papers_map: dict[str, object],
    limit: int,
) -> list[tuple[str, str, str, str]]:
    """Select a generic set of benchmark PDFs from the current corpus."""
    unique_papers: dict[str, object] = {}
    for paper in papers_map.values():
        zotero_key = getattr(paper, "zotero_key", None)
        if zotero_key and zotero_key not in unique_papers:
            unique_papers[zotero_key] = paper

    selected: list[tuple[str, str, str, str]] = []
    for paper in sorted(
        unique_papers.values(),
        key=lambda item: (
            getattr(item, "title", "") or "",
            getattr(item, "zotero_key", ""),
        ),
    ):
        pdf_path = getattr(paper, "pdf_path", None)
        if not pdf_path or not pdf_path.exists():
            continue

        title = (getattr(paper, "title", "") or "Untitled PDF").strip()
        selected.append(
            (
                paper.zotero_key,
                f"Auto-selected benchmark paper: {title[:60]}",
                categorize_title(title),
                "fast",
            )
        )
        if len(selected) >= limit:
            break

    return selected


def main() -> None:
    """Run the benchmark."""
    args = parse_args()

    config = Config.load()
    db = ZoteroDatabase(config.zotero.database_path, config.zotero.storage_path)
    papers_map = {}
    for p in db.get_all_papers():
        papers_map[p.zotero_key] = p
        if p.pdf_attachment_key:
            papers_map[p.pdf_attachment_key] = p

    text_cleaner = TextCleaner()
    hybrid_config = build_hybrid_config(
        enabled=(
            config.processing.opendataloader_hybrid_enabled or args.include_hybrid
        ),
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
    hybrid_ready = None
    if hybrid_config is not None:
        from src.extraction import opendataloader_extractor

        hybrid_ready = opendataloader_extractor.ensure_hybrid_server(hybrid_config)
    target_limit = QUICK_TARGET_LIMIT if args.quick else DEFAULT_TARGET_LIMIT
    targets = load_targets_from_env() or autodiscover_targets(papers_map, target_limit)
    if not targets:
        raise RuntimeError(
            "No benchmark targets found. Set "
            f"{BENCHMARK_TARGETS_ENV} to a JSON file or ensure the corpus has PDFs."
        )

    results = []

    for item_key, description, category, _odl_mode in targets:
        paper = papers_map.get(item_key)
        if paper is None:
            print(f"SKIP: {item_key} - paper not found in database")
            continue

        pdf_path = paper.pdf_path
        if not pdf_path or not pdf_path.exists():
            print(f"SKIP: {item_key} - no PDF file")
            continue

        pages = get_page_count(pdf_path)
        print(f"\n{'=' * 70}")
        print(f"Paper: {paper.title[:70]}")
        print(f"Key: {paper.zotero_key} | Category: {category} | Pages: {pages}")
        print(f"PDF: {pdf_path.name} ({pdf_path.stat().st_size // 1024} KB)")
        print(f"Description: {description}")
        print(f"{'=' * 70}")

        record = {
            "title": paper.title,
            "zotero_key": paper.zotero_key,
            "description": description,
            "category": category,
            "pdf_name": pdf_path.name,
            "page_count": pages,
            "file_size_kb": pdf_path.stat().st_size // 1024,
        }

        # 1. PyMuPDF
        print("\n  [PyMuPDF] Extracting...")
        pymupdf_text, pymupdf_time = extract_pymupdf(pdf_path)
        pymupdf_clean = (
            text_cleaner.clean(pymupdf_text)
            if not is_error_text(pymupdf_text)
            else pymupdf_text
        )
        pymupdf_wc = word_count(pymupdf_clean)
        per_page = pymupdf_time / pages if pages else 0
        print(f"  [PyMuPDF] {pymupdf_wc} words in {pymupdf_time:.2f}s ({per_page:.3f}s/page)")
        record["pymupdf"] = {
            "word_count": pymupdf_wc,
            "time_seconds": round(pymupdf_time, 3),
            "time_per_page": round(per_page, 4),
            "preview": (pymupdf_clean or "")[:200],
        }

        # 2. OpenDataLoader (fast mode)
        print("\n  [ODL-fast] Extracting...")
        odl_text, odl_time = extract_opendataloader(pdf_path, mode="fast")
        odl_clean = (
            text_cleaner.clean(odl_text, preserve_markdown=True)
            if not is_error_text(odl_text)
            else odl_text
        )
        odl_wc = word_count(odl_clean)
        per_page = odl_time / pages if pages else 0
        print(f"  [ODL-fast] {odl_wc} words in {odl_time:.2f}s ({per_page:.3f}s/page)")
        record["odl_fast"] = {
            "word_count": odl_wc,
            "time_seconds": round(odl_time, 3),
            "time_per_page": round(per_page, 4),
            "preview": (odl_clean or "")[:200],
        }

        # 3. OpenDataLoader (fast + struct_tree)
        print("\n  [ODL-tree] Extracting with use_struct_tree=True...")
        odl_tree_text, odl_tree_time = extract_opendataloader(
            pdf_path, mode="fast", use_struct_tree=True,
        )
        odl_tree_clean = (
            text_cleaner.clean(odl_tree_text, preserve_markdown=True)
            if not is_error_text(odl_tree_text)
            else odl_tree_text
        )
        odl_tree_wc = word_count(odl_tree_clean)
        per_page = odl_tree_time / pages if pages else 0
        print(f"  [ODL-tree] {odl_tree_wc} words in {odl_tree_time:.2f}s ({per_page:.3f}s/page)")
        record["odl_struct_tree"] = {
            "word_count": odl_tree_wc,
            "time_seconds": round(odl_tree_time, 3),
            "time_per_page": round(per_page, 4),
            "preview": (odl_tree_clean or "")[:200],
        }

        # 4. OpenDataLoader hybrid
        if hybrid_ready is not None:
            print("\n  [ODL-hybrid] Extracting...")
            odl_hybrid_text, odl_hybrid_time = extract_opendataloader(
                pdf_path,
                mode="hybrid",
                hybrid_config=hybrid_ready,
            )
            odl_hybrid_clean = (
                text_cleaner.clean(odl_hybrid_text, preserve_markdown=True)
                if not is_error_text(odl_hybrid_text)
                else odl_hybrid_text
            )
            odl_hybrid_wc = word_count(odl_hybrid_clean)
            per_page = odl_hybrid_time / pages if pages else 0
            print(
                f"  [ODL-hybrid] {odl_hybrid_wc} words in "
                f"{odl_hybrid_time:.2f}s ({per_page:.3f}s/page)"
            )
            record["odl_hybrid"] = {
                "word_count": odl_hybrid_wc,
                "time_seconds": round(odl_hybrid_time, 3),
                "time_per_page": round(per_page, 4),
                "preview": (odl_hybrid_clean or "")[:200],
            }

        # 5. Marker (if available)
        print("\n  [Marker] Extracting...")
        marker_text, marker_time = extract_marker(pdf_path)
        marker_clean = (
            text_cleaner.clean(marker_text, preserve_markdown=True)
            if not is_error_text(marker_text)
            else marker_text
        )
        marker_wc = word_count(marker_clean)
        per_page = marker_time / pages if pages else 0
        print(f"  [Marker] {marker_wc} words in {marker_time:.2f}s ({per_page:.3f}s/page)")
        record["marker"] = {
            "word_count": marker_wc,
            "time_seconds": round(marker_time, 3),
            "time_per_page": round(per_page, 4),
            "preview": (marker_clean or "")[:200],
        }

        # Determine best method (highest word count, excluding errors)
        methods = {
            "pymupdf": pymupdf_wc if not is_error_text(pymupdf_clean) else 0,
            "odl_fast": odl_wc if not is_error_text(odl_clean) else 0,
            "odl_struct_tree": odl_tree_wc if not is_error_text(odl_tree_clean) else 0,
            "odl_hybrid": (
                record.get("odl_hybrid", {}).get("word_count", 0)
                if not is_error_text(record.get("odl_hybrid", {}).get("preview", ""))
                else 0
            ),
            "marker": marker_wc if not is_error_text(marker_clean) else 0,
        }
        best = max(methods, key=methods.get)  # type: ignore[arg-type]
        record["best_method"] = best
        print(f"\n  >> Best: {best} ({methods[best]} words)")

        results.append(record)

    # Save results
    output_path = Path("data/logs/opendataloader_benchmark.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    # Print summary tables
    print(f"\n{'=' * 100}")
    print("WORD COUNT SUMMARY")
    print(f"{'=' * 100}")
    header = (
        f"{'Paper':<35} {'Cat':<10} {'PyMuPDF':>8} {'ODL-fast':>9} "
        f"{'ODL-tree':>9} {'ODL-hyb':>8} {'Marker':>8} {'Best':>10}"
    )
    print(header)
    print(
        f"{'-' * 35} {'-' * 10} {'-' * 8} {'-' * 9} {'-' * 9} "
        f"{'-' * 8} {'-' * 8} {'-' * 10}"
    )
    for r in results:
        title = r["title"][:33]
        cat = r["category"][:8]
        pm = r["pymupdf"]["word_count"]
        of = r["odl_fast"]["word_count"]
        ot = r["odl_struct_tree"]["word_count"]
        oh = r.get("odl_hybrid", {}).get("word_count", 0)
        mk = r["marker"]["word_count"]
        best_val = r["best_method"]
        print(
            f"{title:<35} {cat:<10} {pm:>8} {of:>9} {ot:>9} "
            f"{oh:>8} {mk:>8} {best_val:>10}"
        )

    print(f"\n{'=' * 100}")
    print("TIMING SUMMARY (seconds)")
    print(f"{'=' * 100}")
    header = (
        f"{'Paper':<35} {'Pages':>5} {'PyMuPDF':>8} {'ODL-fast':>9} "
        f"{'ODL-tree':>9} {'ODL-hyb':>8} {'Marker':>8}"
    )
    print(header)
    print(
        f"{'-' * 35} {'-' * 5} {'-' * 8} {'-' * 9} {'-' * 9} {'-' * 8} {'-' * 8}"
    )
    for r in results:
        title = r["title"][:33]
        pages = r["page_count"]
        pm = r["pymupdf"]["time_seconds"]
        of = r["odl_fast"]["time_seconds"]
        ot = r["odl_struct_tree"]["time_seconds"]
        oh = r.get("odl_hybrid", {}).get("time_seconds", 0)
        mk = r["marker"]["time_seconds"]
        print(
            f"{title:<35} {pages:>5} {pm:>8.2f} {of:>9.2f} {ot:>9.2f} "
            f"{oh:>8.2f} {mk:>8.2f}"
        )

    # Compute averages
    if results:
        avg_pm = sum(r["pymupdf"]["time_seconds"] for r in results) / len(results)
        avg_of = sum(r["odl_fast"]["time_seconds"] for r in results) / len(results)
        avg_ot = sum(r["odl_struct_tree"]["time_seconds"] for r in results) / len(results)
        avg_oh = sum(r.get("odl_hybrid", {}).get("time_seconds", 0) for r in results) / len(results)
        avg_mk = sum(r["marker"]["time_seconds"] for r in results) / len(results)
        print(
            f"{'AVERAGE':<35} {'':>5} {avg_pm:>8.2f} {avg_of:>9.2f} "
            f"{avg_ot:>9.2f} {avg_oh:>8.2f} {avg_mk:>8.2f}"
        )

        # Word count comparison
        pm_total = sum(r["pymupdf"]["word_count"] for r in results)
        of_total = sum(r["odl_fast"]["word_count"] for r in results)
        ot_total = sum(r["odl_struct_tree"]["word_count"] for r in results)
        oh_total = sum(r.get("odl_hybrid", {}).get("word_count", 0) for r in results)
        mk_total = sum(r["marker"]["word_count"] for r in results)
        print(
            f"\nTotal words: PyMuPDF={pm_total}, ODL-fast={of_total}, "
            f"ODL-tree={ot_total}, ODL-hybrid={oh_total}, Marker={mk_total}"
        )

        # Count best-method wins
        wins = {
            "pymupdf": 0,
            "odl_fast": 0,
            "odl_struct_tree": 0,
            "odl_hybrid": 0,
            "marker": 0,
        }
        for r in results:
            wins[r["best_method"]] += 1
        print(f"Best method wins: {wins}")


if __name__ == "__main__":
    main()
