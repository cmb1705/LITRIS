#!/usr/bin/env python
"""Smoketest for BibTeX reference adapter."""

import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Check for required dependencies early
try:
    from src.references.bibtex_adapter import BibTeXReferenceDB
    from src.references.factory import create_reference_db, get_available_providers
except ImportError as e:
    print(f"Import error: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)


SAMPLE_BIBTEX = """
@article{smith2020example,
    author = {Smith, John and Doe, Jane},
    title = {An Example Article Title},
    journal = {Journal of Examples},
    year = {2020},
    volume = {10},
    number = {2},
    pages = {100-120},
    doi = {10.1234/example.2020},
    abstract = {This is an example abstract for testing.},
    keywords = {machine learning, NLP, testing}
}

@inproceedings{jones2021conference,
    author = {Jones, Alice},
    title = {Conference Paper Example},
    booktitle = {Proceedings of Example Conference},
    year = {2021},
    pages = {50-60}
}

@book{brown2019book,
    author = {Brown, Robert},
    title = {A Book Title},
    publisher = {Example Press},
    year = {2019},
    isbn = {978-0-12345-678-9}
}

@phdthesis{wilson2022thesis,
    author = {Wilson, Emily},
    title = {A Doctoral Thesis on Research Methods},
    school = {University of Examples},
    year = {2022}
}

@misc{unknown2023misc,
    title = {A Miscellaneous Entry},
    year = {2023},
    note = {Some additional notes}
}
"""


def test_bibtex_parsing():
    """Test BibTeX file parsing."""
    print("=" * 60)
    print("Testing BibTeX Parsing")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        bib_file = Path(tmpdir) / "references.bib"
        bib_file.write_text(SAMPLE_BIBTEX, encoding="utf-8")

        adapter = BibTeXReferenceDB(bib_file)

        # Test count
        count = adapter.get_paper_count()
        print(f"  Parsed {count} entries")
        count_passed = count == 5

        # Test iteration
        papers = list(adapter.get_all_papers())
        iter_passed = len(papers) == 5
        print(f"  Iterated {len(papers)} papers")

        return count_passed and iter_passed


def test_author_parsing():
    """Test BibTeX author string parsing."""
    print("\n" + "=" * 60)
    print("Testing Author Parsing")
    print("=" * 60)

    
    with tempfile.TemporaryDirectory() as tmpdir:
        bib_file = Path(tmpdir) / "test.bib"
        bib_file.write_text("@article{test, title={Test}}", encoding="utf-8")

        adapter = BibTeXReferenceDB(bib_file)

        test_cases = [
            ("Smith, John", [("John", "Smith")]),
            ("John Smith", [("John", "Smith")]),
            ("Smith, John and Doe, Jane", [("John", "Smith"), ("Jane", "Doe")]),
            ("Smith, John and Doe, Jane and Brown, Bob", [("John", "Smith"), ("Jane", "Doe"), ("Bob", "Brown")]),
            ("van der Berg, Jan", [("Jan", "van der Berg")]),
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


def test_entry_type_mapping():
    """Test BibTeX entry type to item type mapping."""
    print("\n" + "=" * 60)
    print("Testing Entry Type Mapping")
    print("=" * 60)

    
    with tempfile.TemporaryDirectory() as tmpdir:
        bib_file = Path(tmpdir) / "references.bib"
        bib_file.write_text(SAMPLE_BIBTEX, encoding="utf-8")

        adapter = BibTeXReferenceDB(bib_file)

        expected_types = {
            "smith2020example": "journalArticle",
            "jones2021conference": "conferencePaper",
            "brown2019book": "book",
            "wilson2022thesis": "thesis",
            "unknown2023misc": "document",
        }

        all_passed = True
        for key, expected_type in expected_types.items():
            paper = adapter.get_paper_by_key(key)
            if paper:
                passed = paper.item_type == expected_type
                status = "PASS" if passed else "FAIL"
                print(f"  {key}: {paper.item_type} ({status})")
                if not passed:
                    print(f"    Expected: {expected_type}")
                    all_passed = False
            else:
                print(f"  {key}: NOT FOUND (FAIL)")
                all_passed = False

        return all_passed


def test_metadata_extraction():
    """Test metadata extraction from BibTeX entries."""
    print("\n" + "=" * 60)
    print("Testing Metadata Extraction")
    print("=" * 60)

    
    with tempfile.TemporaryDirectory() as tmpdir:
        bib_file = Path(tmpdir) / "references.bib"
        bib_file.write_text(SAMPLE_BIBTEX, encoding="utf-8")

        adapter = BibTeXReferenceDB(bib_file)
        paper = adapter.get_paper_by_key("smith2020example")

        checks = []

        # Title
        title_ok = paper.title == "An Example Article Title"
        checks.append(("title", title_ok))
        print(f"  Title: {paper.title} ({'OK' if title_ok else 'FAIL'})")

        # Year
        year_ok = paper.publication_year == 2020
        checks.append(("year", year_ok))
        print(f"  Year: {paper.publication_year} ({'OK' if year_ok else 'FAIL'})")

        # Authors
        author_ok = len(paper.authors) == 2
        checks.append(("authors", author_ok))
        print(f"  Authors: {len(paper.authors)} ({'OK' if author_ok else 'FAIL'})")

        # Journal
        journal_ok = paper.journal == "Journal of Examples"
        checks.append(("journal", journal_ok))
        print(f"  Journal: {paper.journal} ({'OK' if journal_ok else 'FAIL'})")

        # DOI
        doi_ok = paper.doi == "10.1234/example.2020"
        checks.append(("doi", doi_ok))
        print(f"  DOI: {paper.doi} ({'OK' if doi_ok else 'FAIL'})")

        # Volume/Issue
        vol_ok = paper.volume == "10" and paper.issue == "2"
        checks.append(("volume/issue", vol_ok))
        print(f"  Volume/Issue: {paper.volume}/{paper.issue} ({'OK' if vol_ok else 'FAIL'})")

        # Keywords/Tags
        tags_ok = "machine learning" in paper.tags
        checks.append(("keywords", tags_ok))
        print(f"  Keywords: {paper.tags} ({'OK' if tags_ok else 'FAIL'})")

        return all(ok for _, ok in checks)


def test_pdf_discovery():
    """Test PDF file discovery."""
    print("\n" + "=" * 60)
    print("Testing PDF Discovery")
    print("=" * 60)

    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create BibTeX file
        bib_file = tmpdir / "references.bib"
        bib_file.write_text(SAMPLE_BIBTEX, encoding="utf-8")

        # Create PDF directory with matching files
        pdf_dir = tmpdir / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "smith2020example.pdf").write_bytes(b"%PDF-1.4\n%fake\n")
        (pdf_dir / "Jones2021Conference.pdf").write_bytes(b"%PDF-1.4\n%fake\n")  # Different case

        adapter = BibTeXReferenceDB(bib_file, pdf_dir=pdf_dir)

        # Test PDF matching
        smith = adapter.get_paper_by_key("smith2020example")
        smith_has_pdf = smith.pdf_path is not None
        print(f"  smith2020example PDF: {'Found' if smith_has_pdf else 'Not found'}")

        jones = adapter.get_paper_by_key("jones2021conference")
        jones_has_pdf = jones.pdf_path is not None
        print(f"  jones2021conference PDF: {'Found' if jones_has_pdf else 'Not found'}")

        brown = adapter.get_paper_by_key("brown2019book")
        brown_no_pdf = brown.pdf_path is None
        print(f"  brown2019book PDF: {'Not found (expected)' if brown_no_pdf else 'Found (unexpected)'}")

        return smith_has_pdf and brown_no_pdf


def test_factory():
    """Test factory integration."""
    print("\n" + "=" * 60)
    print("Testing Factory Integration")
    print("=" * 60)

    
    providers = get_available_providers()
    print(f"  Available providers: {providers}")
    bibtex_available = "bibtex" in providers

    with tempfile.TemporaryDirectory() as tmpdir:
        bib_file = Path(tmpdir) / "test.bib"
        bib_file.write_text(SAMPLE_BIBTEX, encoding="utf-8")

        try:
            db = create_reference_db(
                provider="bibtex",
                bibtex_path=bib_file,
            )
            factory_works = db.provider == "bibtex"
            print(f"  Factory created: {db.provider}")
        except Exception as e:
            print(f"  Factory error: {e}")
            factory_works = False

    return bibtex_available and factory_works


def test_filter_papers():
    """Test paper filtering."""
    print("\n" + "=" * 60)
    print("Testing Paper Filtering")
    print("=" * 60)

    
    with tempfile.TemporaryDirectory() as tmpdir:
        bib_file = Path(tmpdir) / "references.bib"
        bib_file.write_text(SAMPLE_BIBTEX, encoding="utf-8")

        adapter = BibTeXReferenceDB(bib_file)

        # Filter by year
        recent = list(adapter.filter_papers(year_min=2021, has_pdf=False))
        recent_ok = len(recent) == 3  # 2021, 2022, 2023
        print(f"  Papers from 2021+: {len(recent)} ({'OK' if recent_ok else 'FAIL'})")

        old = list(adapter.filter_papers(year_max=2019, has_pdf=False))
        old_ok = len(old) == 1  # 2019
        print(f"  Papers up to 2019: {len(old)} ({'OK' if old_ok else 'FAIL'})")

        # Iterate with limit
        limited = list(adapter.iterate_papers(limit=2))
        limit_ok = len(limited) == 2
        print(f"  Limited to 2: {len(limited)} ({'OK' if limit_ok else 'FAIL'})")

        return recent_ok and old_ok and limit_ok


def main():
    print("LITRIS BibTeX Adapter Smoketest")
    print("=" * 60)

    results = {}

    results["bibtex_parsing"] = test_bibtex_parsing()
    results["author_parsing"] = test_author_parsing()
    results["type_mapping"] = test_entry_type_mapping()
    results["metadata"] = test_metadata_extraction()
    results["pdf_discovery"] = test_pdf_discovery()
    results["factory"] = test_factory()
    results["filtering"] = test_filter_papers()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for test, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\nBibTeX adapter is ready!")
    else:
        print("\nSome tests failed.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
