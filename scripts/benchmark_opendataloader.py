"""Benchmark OpenDataLoader PDF against PyMuPDF and Marker.

Compares extraction quality and speed across extractors for diverse
PDF types in the LITRIS corpus, following the methodology from
benchmark_glm_ocr.py.

Usage:
    python scripts/benchmark_opendataloader.py
    python scripts/benchmark_opendataloader.py --quick   # 5 papers only
"""

import json
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.extraction.text_cleaner import TextCleaner
from src.zotero.database import ZoteroDatabase


def word_count(text: str) -> int:
    """Count words in text."""
    if not text:
        return 0
    return len(text.split())


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


# Test corpus: diverse papers from LITRIS
# Format: (zotero_key, description, category, odl_mode)
TARGETS = [
    # Standard digital PDF (single column)
    ("22FLI2DD", "Digitized journal article (Star & Gerson 1987)", "journal", "fast"),
    ("22GEWMB6", "Standard journal article", "journal", "fast"),
    ("22NZCYJ3", "Standard journal article", "journal", "fast"),
    # Multi-column papers (conference proceedings)
    ("233SW49D", "Conference/multi-column paper", "multi_column", "fast"),
    ("22PCD9K3", "Conference/multi-column paper", "multi_column", "fast"),
    ("24BQNWCR", "Conference/multi-column paper", "multi_column", "fast"),
    # Equation-heavy (arXiv math/physics)
    ("ITXX4K2T", "Equation-heavy paper (meta-heuristics)", "equations", "fast"),
    ("25K65MAN", "Equation-heavy paper", "equations", "fast"),
    ("26V28YAY", "Equation-heavy paper", "equations", "fast"),
    # Scanned PDFs (from OCR corpus)
    ("V535UYAH", "Scanned research paper (betweenness)", "scanned", "fast"),
    ("A9SFRQXA", "Classic scanned (Architecture of Complexity)", "scanned", "fast"),
    ("F2YZIGEL", "Scanned research (Org Structure)", "scanned", "fast"),
    # Books/long documents (100+ pages)
    ("EXSV6C6H", "Book (writing journal article, 180pp)", "book", "fast"),
    ("UUMFULSR", "Scanned book (Book ScanCenter)", "book", "fast"),
    ("HQBDYZ25", "Long document (Task-Technology Fit)", "book", "fast"),
]

QUICK_TARGETS = TARGETS[:5]


def is_error_text(text: str) -> bool:
    """Check if text represents an extraction error/unavailability."""
    return text.startswith(("ERROR:", "UNAVAILABLE:", "EMPTY:"))


def main() -> None:
    """Run the benchmark."""
    quick_mode = "--quick" in sys.argv

    config = Config.load()
    db = ZoteroDatabase(config.zotero.database_path, config.zotero.storage_path)
    papers_map = {}
    for p in db.get_all_papers():
        papers_map[p.zotero_key] = p
        if p.pdf_attachment_key:
            papers_map[p.pdf_attachment_key] = p

    text_cleaner = TextCleaner()
    targets = QUICK_TARGETS if quick_mode else TARGETS

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

        # 4. Marker (if available)
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
    header = f"{'Paper':<35} {'Cat':<10} {'PyMuPDF':>8} {'ODL-fast':>9} {'ODL-tree':>9} {'Marker':>8} {'Best':>10}"
    print(header)
    print(f"{'-' * 35} {'-' * 10} {'-' * 8} {'-' * 9} {'-' * 9} {'-' * 8} {'-' * 10}")
    for r in results:
        title = r["title"][:33]
        cat = r["category"][:8]
        pm = r["pymupdf"]["word_count"]
        of = r["odl_fast"]["word_count"]
        ot = r["odl_struct_tree"]["word_count"]
        mk = r["marker"]["word_count"]
        best_val = r["best_method"]
        print(f"{title:<35} {cat:<10} {pm:>8} {of:>9} {ot:>9} {mk:>8} {best_val:>10}")

    print(f"\n{'=' * 100}")
    print("TIMING SUMMARY (seconds)")
    print(f"{'=' * 100}")
    header = f"{'Paper':<35} {'Pages':>5} {'PyMuPDF':>8} {'ODL-fast':>9} {'ODL-tree':>9} {'Marker':>8}"
    print(header)
    print(f"{'-' * 35} {'-' * 5} {'-' * 8} {'-' * 9} {'-' * 9} {'-' * 8}")
    for r in results:
        title = r["title"][:33]
        pages = r["page_count"]
        pm = r["pymupdf"]["time_seconds"]
        of = r["odl_fast"]["time_seconds"]
        ot = r["odl_struct_tree"]["time_seconds"]
        mk = r["marker"]["time_seconds"]
        print(f"{title:<35} {pages:>5} {pm:>8.2f} {of:>9.2f} {ot:>9.2f} {mk:>8.2f}")

    # Compute averages
    if results:
        avg_pm = sum(r["pymupdf"]["time_seconds"] for r in results) / len(results)
        avg_of = sum(r["odl_fast"]["time_seconds"] for r in results) / len(results)
        avg_ot = sum(r["odl_struct_tree"]["time_seconds"] for r in results) / len(results)
        avg_mk = sum(r["marker"]["time_seconds"] for r in results) / len(results)
        print(f"{'AVERAGE':<35} {'':>5} {avg_pm:>8.2f} {avg_of:>9.2f} {avg_ot:>9.2f} {avg_mk:>8.2f}")

        # Word count comparison
        pm_total = sum(r["pymupdf"]["word_count"] for r in results)
        of_total = sum(r["odl_fast"]["word_count"] for r in results)
        ot_total = sum(r["odl_struct_tree"]["word_count"] for r in results)
        mk_total = sum(r["marker"]["word_count"] for r in results)
        print(f"\nTotal words: PyMuPDF={pm_total}, ODL-fast={of_total}, "
              f"ODL-tree={ot_total}, Marker={mk_total}")

        # Count best-method wins
        wins = {"pymupdf": 0, "odl_fast": 0, "odl_struct_tree": 0, "marker": 0}
        for r in results:
            wins[r["best_method"]] += 1
        print(f"Best method wins: {wins}")


if __name__ == "__main__":
    main()
