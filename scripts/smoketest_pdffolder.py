#!/usr/bin/env python
"""Smoketest for PDF folder reference adapter."""

import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_test_pdf(path: Path, title: str, author: str) -> None:
    """Create a minimal PDF file with metadata."""
    try:
        import pymupdf

        doc = pymupdf.open()
        doc.set_metadata({
            "title": title,
            "author": author,
            "subject": "Test abstract for smoketest",
            "keywords": "test, smoketest, pdf",
        })
        page = doc.new_page()
        page.insert_text((72, 72), f"Title: {title}\nAuthor: {author}")
        doc.save(path)
        doc.close()
    except ImportError:
        # Fallback: create empty PDF-like file
        path.write_bytes(b"%PDF-1.4\n%fake\n")


def test_filename_parsing():
    """Test filename pattern parsing."""
    print("=" * 60)
    print("Testing Filename Parsing")
    print("=" * 60)

    from src.references.pdffolder_adapter import PDFFolderReferenceDB

    # Create minimal adapter for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = PDFFolderReferenceDB(Path(tmpdir))

        test_cases = [
            ("Smith - 2020 - Machine Learning.pdf", {"authors": "Smith", "year": "2020", "title": "Machine Learning"}),
            ("Smith_2020_Machine_Learning.pdf", {"authors": "Smith", "year": "2020", "title": "Machine Learning"}),
            ("2020_Smith_Machine_Learning.pdf", {"authors": "Smith", "year": "2020", "title": "Machine Learning"}),
            ("2020 - Smith - Machine Learning.pdf", {"authors": "Smith", "year": "2020", "title": "Machine Learning"}),
            ("Just A Title.pdf", {"authors": None, "year": None, "title": "Just A Title"}),
        ]

        all_passed = True
        for filename, expected in test_cases:
            result = adapter._parse_filename(Path(filename))
            passed = all(result.get(k) == v for k, v in expected.items() if v is not None)
            status = "PASS" if passed else "FAIL"
            print(f"  {filename}: {status}")
            if not passed:
                print(f"    Expected: {expected}")
                print(f"    Got: {result}")
                all_passed = False

        return all_passed


def test_author_parsing():
    """Test author string parsing."""
    print("\n" + "=" * 60)
    print("Testing Author Parsing")
    print("=" * 60)

    from src.references.pdffolder_adapter import PDFFolderReferenceDB

    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = PDFFolderReferenceDB(Path(tmpdir))

        test_cases = [
            ("Smith, John", [("John", "Smith")]),
            ("John Smith", [("John", "Smith")]),
            ("Smith, John and Doe, Jane", [("John", "Smith"), ("Jane", "Doe")]),
            ("John Smith; Jane Doe", [("John", "Smith"), ("Jane", "Doe")]),
            ("Smith et al.", [("", "Smith")]),
            ("Smith, J. & Doe, J.", [("J.", "Smith"), ("J.", "Doe")]),
        ]

        all_passed = True
        for author_string, expected in test_cases:
            authors = adapter._parse_authors(author_string)
            result = [(a.first_name, a.last_name) for a in authors]
            passed = result == expected
            status = "PASS" if passed else "FAIL"
            print(f"  \"{author_string}\": {status}")
            if not passed:
                print(f"    Expected: {expected}")
                print(f"    Got: {result}")
                all_passed = False

        return all_passed


def test_pdf_folder_scan():
    """Test scanning a folder of PDFs."""
    print("\n" + "=" * 60)
    print("Testing PDF Folder Scan")
    print("=" * 60)

    from src.references.pdffolder_adapter import PDFFolderReferenceDB

    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)

        # Create test PDFs
        create_test_pdf(
            folder / "Smith - 2020 - Deep Learning Basics.pdf",
            title="Deep Learning Basics",
            author="John Smith",
        )
        create_test_pdf(
            folder / "Doe_2021_Neural_Networks.pdf",
            title="Neural Networks",
            author="Jane Doe",
        )

        # Create subfolder
        subfolder = folder / "conference"
        subfolder.mkdir()
        create_test_pdf(
            subfolder / "2022 - Brown - Transformers.pdf",
            title="Transformers",
            author="Bob Brown",
        )

        # Test adapter
        adapter = PDFFolderReferenceDB(folder, recursive=True)

        # Test count
        count = adapter.get_paper_count()
        print(f"  Found {count} PDFs")
        count_passed = count == 3

        # Test iteration
        papers = list(adapter.get_all_papers())
        iter_passed = len(papers) == 3
        print(f"  Iterated {len(papers)} papers")

        # Test metadata extraction
        for paper in papers:
            print(f"    - {paper.title} by {', '.join(a.full_name for a in paper.authors) or 'Unknown'}")
            if paper.collections:
                print(f"      Collection: {paper.collections[0]}")

        # Test get_paper_by_key
        key = "Smith - 2020 - Deep Learning Basics"
        paper = adapter.get_paper_by_key(key)
        key_passed = paper is not None
        print(f"  Get by key '{key}': {'Found' if paper else 'Not found'}")

        return count_passed and iter_passed and key_passed


def test_factory():
    """Test factory integration."""
    print("\n" + "=" * 60)
    print("Testing Factory Integration")
    print("=" * 60)

    from src.references.factory import create_reference_db, get_available_providers

    providers = get_available_providers()
    print(f"  Available providers: {providers}")
    pdffolder_available = "pdffolder" in providers

    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        (folder / "test.pdf").write_bytes(b"%PDF-1.4\n%test\n")

        try:
            db = create_reference_db(
                provider="pdffolder",
                folder_path=folder,
                recursive=False,
            )
            factory_works = db.provider == "pdffolder"
            print(f"  Factory created: {db.provider}")
        except Exception as e:
            print(f"  Factory error: {e}")
            factory_works = False

    return pdffolder_available and factory_works


def main():
    print("LITRIS PDF Folder Adapter Smoketest")
    print("=" * 60)

    results = {}

    results["filename_parsing"] = test_filename_parsing()
    results["author_parsing"] = test_author_parsing()
    results["folder_scan"] = test_pdf_folder_scan()
    results["factory"] = test_factory()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for test, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\nPDF folder adapter is ready!")
    else:
        print("\nSome tests failed.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
