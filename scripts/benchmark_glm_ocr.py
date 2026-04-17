"""Benchmark GLM-OCR against PyMuPDF and Tesseract for low-text PDFs.

Compares extraction quality across three methods for papers that were
skipped from the LITRIS index due to insufficient text from PyMuPDF.

Set `LITRIS_GLM_OCR_TARGETS_JSON` to point at a private JSON file with
corpus-specific benchmark targets if you want a curated sample.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.extraction.pdf_extractor import PDFExtractor
from src.extraction.text_cleaner import TextCleaner
from src.zotero.database import ZoteroDatabase

OLLAMA = os.environ.get(
    "OLLAMA_PATH",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
)
MAX_PAGES = 3  # Only process first 3 pages per paper
OCR_TIMEOUT = 120  # seconds per page
BENCHMARK_TARGETS_ENV = "LITRIS_GLM_OCR_TARGETS_JSON"
DEFAULT_TARGET_LIMIT = 8


def word_count(text: str) -> int:
    """Count words in text."""
    if not text:
        return 0
    return len(text.split())


def load_targets_from_env() -> list[tuple[str, str]] | None:
    """Load corpus-specific benchmark targets from a JSON file.

    The JSON file should contain a list of objects with `key` and `description`
    fields. Keeping the file path in an environment variable allows private
    corpus identifiers to stay out of version control.
    """
    raw_path = os.environ.get(BENCHMARK_TARGETS_ENV)
    if not raw_path:
        return None

    path = Path(raw_path).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [(str(item["key"]), str(item["description"])) for item in payload]


def extract_pymupdf(pdf_path: Path) -> tuple[str, float]:
    """Extract text using PyMuPDF. Returns (text, elapsed_seconds)."""
    extractor = PDFExtractor()
    start = time.perf_counter()
    try:
        text = extractor.extract_text(pdf_path)
    except Exception as exc:
        text = f"ERROR: {exc}"
    elapsed = time.perf_counter() - start
    return text, elapsed


def extract_tesseract(pdf_path: Path) -> tuple[str, float]:
    """Extract text using Tesseract OCR (first MAX_PAGES only).

    Returns (text, elapsed_seconds).
    """
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError:
        return "UNAVAILABLE: pytesseract or pdf2image not installed", 0.0

    from src.extraction.ocr_handler import _find_poppler, _find_tesseract

    tess_cmd = _find_tesseract()
    if tess_cmd is None:
        return "UNAVAILABLE: Tesseract not found", 0.0
    pytesseract.pytesseract.tesseract_cmd = tess_cmd

    start = time.perf_counter()
    try:
        poppler_path = _find_poppler()
        kwargs = {"first_page": 1, "last_page": MAX_PAGES, "dpi": 300}
        if poppler_path:
            kwargs["poppler_path"] = poppler_path

        images = convert_from_path(str(pdf_path), **kwargs)
        pages = []
        for page_num, image in enumerate(images, 1):
            page_text = pytesseract.image_to_string(image, lang="eng")
            if page_text.strip():
                pages.append(f"--- Page {page_num} ---\n{page_text}")
            image.close()
        text = "\n\n".join(pages)
    except Exception as exc:
        text = f"ERROR: {exc}"
    elapsed = time.perf_counter() - start
    return text, elapsed


GLM_TARGET_WIDTH = 1000  # GLM-OCR works best with ~1000px wide images


def extract_glm_ocr(pdf_path: Path) -> tuple[str, float]:
    """Extract text using GLM-OCR via Ollama. Returns (text, elapsed_seconds).

    GLM-OCR (2.2GB model) cannot handle high-resolution images.
    Images are downscaled to ~1000px width for reliable recognition.
    """
    try:
        from pdf2image import convert_from_path
        from PIL import Image as PILImage
    except ImportError:
        return "UNAVAILABLE: pdf2image not installed", 0.0

    start = time.perf_counter()
    try:
        # Find poppler
        from src.extraction.ocr_handler import _find_poppler

        poppler_path = _find_poppler()

        kwargs = {"first_page": 1, "last_page": MAX_PAGES, "dpi": 200}
        if poppler_path:
            kwargs["poppler_path"] = poppler_path

        images = convert_from_path(str(pdf_path), **kwargs)

        all_text = []
        for i, img in enumerate(images):
            # Downscale to ~1000px width for GLM-OCR compatibility
            if img.size[0] > GLM_TARGET_WIDTH:
                scale = GLM_TARGET_WIDTH / img.size[0]
                new_size = (GLM_TARGET_WIDTH, int(img.size[1] * scale))
                img = img.resize(new_size, PILImage.LANCZOS)

            tmppath = os.path.join(tempfile.gettempdir(), f"glm_ocr_page_{i}.png")
            img.save(tmppath)
            try:
                result = subprocess.run(
                    [
                        OLLAMA,
                        "run",
                        "glm-ocr",
                        f"OCR this image and output all text: {tmppath}",
                    ],
                    capture_output=True,
                    timeout=OCR_TIMEOUT,
                    encoding="utf-8",
                    errors="replace",
                )
                page_text = result.stdout.strip()
                # Filter out empty markdown blocks and "blurry" responses
                if page_text and "cannot be read" not in page_text.lower():
                    # Strip markdown code fences that GLM-OCR sometimes adds
                    cleaned = page_text
                    while cleaned.startswith("```"):
                        # Remove opening fence (with optional language tag)
                        end = cleaned.find("\n")
                        if end >= 0:
                            cleaned = cleaned[end + 1 :]
                        else:
                            cleaned = ""
                    if cleaned.endswith("```"):
                        cleaned = cleaned[:-3]
                    cleaned = cleaned.strip()
                    if cleaned:
                        all_text.append(f"--- Page {i + 1} ---\n{cleaned}")
            except subprocess.TimeoutExpired:
                all_text.append(f"--- Page {i + 1} ---\nTIMEOUT")
            except Exception as exc:
                all_text.append(f"--- Page {i + 1} ---\nERROR: {exc}")
            finally:
                if os.path.exists(tmppath):
                    os.unlink(tmppath)
            img.close()

        text = "\n\n".join(all_text)
    except Exception as exc:
        text = f"ERROR: {exc}"

    elapsed = time.perf_counter() - start
    return text, elapsed


def autodiscover_targets(
    papers_map: dict[str, object],
    limit: int = DEFAULT_TARGET_LIMIT,
) -> list[tuple[str, str]]:
    """Select a generic set of benchmark papers from the current corpus.

    This fallback keeps the benchmark runnable in any checkout without
    hardcoding private Zotero keys into the repository.
    """
    unique_papers: dict[str, object] = {}
    for paper in papers_map.values():
        zotero_key = getattr(paper, "zotero_key", None)
        if zotero_key and zotero_key not in unique_papers:
            unique_papers[zotero_key] = paper

    selected: list[tuple[str, str]] = []
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
        selected.append((paper.zotero_key, f"Auto-selected benchmark paper: {title[:60]}"))
        if len(selected) >= limit:
            break

    return selected


def main() -> None:
    """Run the benchmark."""
    config = Config.load()
    db = ZoteroDatabase(config.zotero.database_path, config.zotero.storage_path)
    papers_map = {}
    for p in db.get_all_papers():
        papers_map[p.zotero_key] = p
        if p.pdf_attachment_key:
            papers_map[p.pdf_attachment_key] = p

    text_cleaner = TextCleaner()

    targets = load_targets_from_env() or autodiscover_targets(papers_map)
    if not targets:
        raise RuntimeError(
            "No benchmark targets found. Set "
            f"{BENCHMARK_TARGETS_ENV} to a JSON file or ensure the corpus has PDFs."
        )

    results = []

    for item_key, description in targets:
        paper = papers_map.get(item_key)
        if paper is None:
            print(f"SKIP: {item_key} - paper not found in database")
            continue

        pdf_path = paper.pdf_path
        if not pdf_path or not pdf_path.exists():
            print(f"SKIP: {item_key} - no PDF file")
            continue

        print(f"\n{'=' * 70}")
        print(f"Paper: {paper.title[:70]}")
        print(f"Key: {paper.zotero_key}/{paper.zotero_item_id}")
        print(f"PDF: {pdf_path.name}")
        print(f"Description: {description}")
        print(f"{'=' * 70}")

        record = {
            "title": paper.title,
            "zotero_key": paper.zotero_key,
            "item_id": paper.zotero_item_id,
            "description": description,
            "pdf_name": pdf_path.name,
        }

        # 1. PyMuPDF
        print("\n  [PyMuPDF] Extracting...")
        pymupdf_text, pymupdf_time = extract_pymupdf(pdf_path)
        pymupdf_clean = (
            text_cleaner.clean(pymupdf_text)
            if pymupdf_text and not pymupdf_text.startswith("ERROR")
            else pymupdf_text
        )
        pymupdf_wc = word_count(pymupdf_clean)
        print(f"  [PyMuPDF] {pymupdf_wc} words in {pymupdf_time:.2f}s")
        if pymupdf_clean:
            preview = pymupdf_clean[:200].replace("\n", " ")
            print(f"  [PyMuPDF] Preview: {preview}")
        record["pymupdf"] = {
            "word_count": pymupdf_wc,
            "time_seconds": round(pymupdf_time, 2),
            "preview": (pymupdf_clean or "")[:200],
        }

        # 2. Tesseract OCR (first 3 pages only via pdf2image)
        print("\n  [Tesseract] Extracting...")
        tess_text, tess_time = extract_tesseract(pdf_path)
        tess_clean = (
            text_cleaner.clean(tess_text)
            if tess_text and not tess_text.startswith(("ERROR", "UNAVAILABLE"))
            else tess_text
        )
        tess_wc = word_count(tess_clean)
        print(f"  [Tesseract] {tess_wc} words in {tess_time:.2f}s")
        if tess_clean:
            preview = tess_clean[:200].replace("\n", " ")
            print(f"  [Tesseract] Preview: {preview}")
        record["tesseract"] = {
            "word_count": tess_wc,
            "time_seconds": round(tess_time, 2),
            "preview": (tess_clean or "")[:200],
        }

        # 3. GLM-OCR (first 3 pages)
        print("\n  [GLM-OCR] Extracting (first 3 pages)...")
        glm_text, glm_time = extract_glm_ocr(pdf_path)
        glm_clean = (
            text_cleaner.clean(glm_text)
            if glm_text and not glm_text.startswith(("ERROR", "UNAVAILABLE"))
            else glm_text
        )
        glm_wc = word_count(glm_clean)
        print(f"  [GLM-OCR] {glm_wc} words in {glm_time:.2f}s")
        if glm_clean:
            preview = glm_clean[:200].replace("\n", " ")
            print(f"  [GLM-OCR] Preview: {preview}")
        record["glm_ocr"] = {
            "word_count": glm_wc,
            "time_seconds": round(glm_time, 2),
            "preview": (glm_clean or "")[:200],
        }

        # Quick quality assessment
        best = "none"
        if glm_wc > pymupdf_wc and glm_wc > tess_wc:
            best = "glm_ocr"
        elif tess_wc > pymupdf_wc and tess_wc > glm_wc:
            best = "tesseract"
        elif pymupdf_wc > 0:
            best = "pymupdf"
        record["best_method"] = best
        print(f"\n  >> Best: {best}")

        results.append(record)

    # Save results
    output_path = Path("data/logs/glm_ocr_benchmark.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    # Print summary table
    print(f"\n{'=' * 90}")
    print("SUMMARY")
    print(f"{'=' * 90}")
    print(f"{'Paper':<40} {'PyMuPDF':>10} {'Tesseract':>10} {'GLM-OCR':>10} {'Best':>10}")
    print(f"{'-' * 40} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
    for r in results:
        title = r["title"][:38]
        pymupdf_wc_val = r["pymupdf"]["word_count"]
        tess_wc_val = r["tesseract"]["word_count"]
        glm_wc_val = r["glm_ocr"]["word_count"]
        best_val = r["best_method"]
        print(f"{title:<40} {pymupdf_wc_val:>10} {tess_wc_val:>10} {glm_wc_val:>10} {best_val:>10}")

    print(f"\n{'Paper':<40} {'PyMuPDF(s)':>10} {'Tess(s)':>10} {'GLM(s)':>10}")
    print(f"{'-' * 40} {'-' * 10} {'-' * 10} {'-' * 10}")
    for r in results:
        title = r["title"][:38]
        print(
            f"{title:<40} {r['pymupdf']['time_seconds']:>10.1f} {r['tesseract']['time_seconds']:>10.1f} {r['glm_ocr']['time_seconds']:>10.1f}"
        )


if __name__ == "__main__":
    main()
