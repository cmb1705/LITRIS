#!/usr/bin/env python
"""Smoketest for Paperpile reference adapter."""

import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_test_bib(bib_path: Path, pdf_path: Path) -> None:
    """Create a minimal Paperpile-style BibTeX file for testing."""
    bib_content = f"""@article{{Smith2022Testing,
  author = {{Smith, Alice and Jones, Bob}},
  title = {{Test Paper Title from Paperpile}},
  journal = {{Journal of Testing}},
  year = {{2022}},
  volume = {{10}},
  number = {{3}},
  pages = {{100--110}},
  doi = {{10.1000/test}},
  abstract = {{This is a test abstract exported from Paperpile.}},
  keywords = {{testing, smoketest, validation}},
  file = {{{pdf_path}}}
}}

@book{{Johnson2020Book,
  author = {{Johnson, Carol}},
  title = {{Test Book from Paperpile}},
  publisher = {{Test Publisher}},
  year = {{2020}},
  isbn = {{978-0-00-000000-0}},
  keywords = {{books, testing}}
}}

@inproceedings{{Williams2021Conference,
  author = {{Williams, David and Brown, Emma}},
  title = {{Conference Paper Title}},
  booktitle = {{Proceedings of Test Conference}},
  year = {{2021}},
  pages = {{50--55}},
  groups = {{Project A, Important}}
}}
"""
    bib_path.write_text(bib_content, encoding="utf-8")


def test_adapter() -> bool:
    """Test basic adapter functionality."""
    print("=" * 60)
    print("Testing Paperpile Adapter")
    print("=" * 60)

    from src.references.paperpile_adapter import PaperpileReferenceDB

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%test\n")

        bib_path = tmp_path / "paperpile.bib"
        create_test_bib(bib_path, pdf_path)

        adapter = PaperpileReferenceDB(bib_path)

        count = adapter.get_paper_count()
        print(f"  Paper count: {count}")
        count_passed = count == 3

        papers = list(adapter.get_all_papers())
        print(f"  Papers loaded: {len(papers)}")
        papers_passed = len(papers) == 3

        paper = papers[0] if papers else None
        title_passed = (
            paper is not None and paper.title == "Test Paper Title from Paperpile"
        )
        print(f"  Title parsed: {'PASS' if title_passed else 'FAIL'}")

        author_passed = paper and len(paper.authors) == 2
        first_author = paper.authors[0] if paper and paper.authors else None
        author_name_passed = first_author and first_author.last_name == "Smith"
        print(f"  Authors parsed: {'PASS' if author_passed else 'FAIL'}")
        print(f"  Author name parsed: {'PASS' if author_name_passed else 'FAIL'}")

        tag_passed = paper and "testing" in paper.tags
        print(f"  Tags parsed: {'PASS' if tag_passed else 'FAIL'}")

        type_passed = paper and paper.item_type == "journalArticle"
        print(f"  Item type parsed: {'PASS' if type_passed else 'FAIL'}")

        book = next((p for p in papers if "Book" in p.title), None)
        book_type_passed = book and book.item_type == "book"
        print(f"  Book type parsed: {'PASS' if book_type_passed else 'FAIL'}")

        conf = next((p for p in papers if "Conference" in p.title), None)
        conf_type_passed = conf and conf.item_type == "conferencePaper"
        print(f"  Conference type parsed: {'PASS' if conf_type_passed else 'FAIL'}")

        conf_collections_passed = conf and len(conf.collections) > 0
        print(f"  Collections parsed: {'PASS' if conf_collections_passed else 'FAIL'}")

        by_key = adapter.get_paper_by_key("Smith2022Testing")
        key_passed = (
            by_key is not None and by_key.title == "Test Paper Title from Paperpile"
        )
        print(f"  Get by citation key: {'PASS' if key_passed else 'FAIL'}")

        by_prefixed_key = adapter.get_paper_by_key("paperpile_Smith2022Testing")
        prefixed_key_passed = (
            by_prefixed_key is not None
            and by_prefixed_key.title == "Test Paper Title from Paperpile"
        )
        print(f"  Get by prefixed key: {'PASS' if prefixed_key_passed else 'FAIL'}")

        return all(
            [
                count_passed,
                papers_passed,
                title_passed,
                author_passed,
                author_name_passed,
                tag_passed,
                type_passed,
                book_type_passed,
                conf_type_passed,
                conf_collections_passed,
                key_passed,
                prefixed_key_passed,
            ]
        )


def test_factory() -> bool:
    """Test factory integration."""
    print("\n" + "=" * 60)
    print("Testing Factory Integration")
    print("=" * 60)

    from src.references.factory import create_reference_db, get_available_providers

    providers = get_available_providers()
    print(f"  Available providers: {providers}")
    paperpile_available = "paperpile" in providers
    print(f"  Paperpile available: {'PASS' if paperpile_available else 'FAIL'}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        bib_path = tmp_path / "paperpile.bib"
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%test\n")
        create_test_bib(bib_path, pdf_path)

        try:
            db = create_reference_db(
                provider="paperpile",
                bibtex_path=bib_path,
            )
            factory_works = db.provider == "paperpile"
            print(f"  Factory created: {db.provider}")
        except Exception as e:
            print(f"  Factory error: {e}")
            factory_works = False

    return paperpile_available and factory_works


def test_sync_folder() -> bool:
    """Test sync folder PDF discovery."""
    print("\n" + "=" * 60)
    print("Testing Sync Folder Integration")
    print("=" * 60)

    from src.references.paperpile_adapter import PaperpileReferenceDB

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create sync folder structure
        sync_folder = tmp_path / "Paperpile"
        author_folder = sync_folder / "Smith"
        author_folder.mkdir(parents=True)

        # Create PDF in author folder
        pdf_path = author_folder / "Smith2022_Testing.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%test\n")

        # Create BibTeX without file field
        bib_content = """@article{Smith2022Testing,
  author = {Smith, Alice},
  title = {Test Paper},
  journal = {Test Journal},
  year = {2022}
}
"""
        bib_path = tmp_path / "paperpile.bib"
        bib_path.write_text(bib_content, encoding="utf-8")

        adapter = PaperpileReferenceDB(bib_path, sync_folder=sync_folder)
        papers = list(adapter.get_all_papers())

        paper = papers[0] if papers else None
        pdf_found = paper and paper.pdf_path is not None
        print(f"  PDF found via sync folder: {'PASS' if pdf_found else 'FAIL'}")

        return pdf_found


def main() -> int:
    print("LITRIS Paperpile Adapter Smoketest")
    print("=" * 60)

    results = {
        "adapter": test_adapter(),
        "factory": test_factory(),
        "sync_folder": test_sync_folder(),
    }

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for test, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test}: {status}")

    all_passed = all(results.values())
    print("\nPaperpile adapter is ready!" if all_passed else "\nSome tests failed.")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
