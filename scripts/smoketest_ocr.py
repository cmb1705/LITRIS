#!/usr/bin/env python
"""Smoketest for OCR auto-discovery."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    print("LITRIS OCR Auto-Discovery Smoketest")
    print("=" * 60)

    from src.extraction.ocr_handler import OCRHandler

    deps = OCRHandler.check_dependencies()

    print("\nDependency Status:")
    print("-" * 40)
    for key, value in deps.items():
        if isinstance(value, bool):
            status = "YES" if value else "NO"
        elif value is None:
            status = "Not found"
        else:
            status = value
        print(f"  {key}: {status}")

    print("\nOCR Available:", "YES" if OCRHandler.is_available() else "NO")

    # Summary
    print("\n" + "=" * 60)
    if OCRHandler.is_available():
        print("OCR is ready for use!")
        if deps.get("tesseract_path"):
            print(f"  Tesseract: {deps['tesseract_path']}")
        if deps.get("poppler_path"):
            print(f"  Poppler: {deps['poppler_path']}")
    else:
        print("OCR is NOT available.")
        if not deps.get("pytesseract_installed"):
            print("  - Install pytesseract: pip install pytesseract")
        if not deps.get("tesseract_available"):
            print("  - Install Tesseract OCR:")
            if sys.platform == "win32":
                print("    Windows: https://github.com/UB-Mannheim/tesseract/wiki")
            elif sys.platform == "darwin":
                print("    macOS: brew install tesseract")
            else:
                print("    Linux: sudo apt install tesseract-ocr")
        if not deps.get("pdf2image_installed"):
            print("  - Install pdf2image: pip install pdf2image")
        if not deps.get("poppler_available"):
            print("  - Install Poppler:")
            if sys.platform == "win32":
                print("    Windows: https://github.com/osber/poppler-windows")
            elif sys.platform == "darwin":
                print("    macOS: brew install poppler")
            else:
                print("    Linux: sudo apt install poppler-utils")

    return 0 if OCRHandler.is_available() else 1


if __name__ == "__main__":
    sys.exit(main())
