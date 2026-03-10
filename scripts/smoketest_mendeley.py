#!/usr/bin/env python
"""Smoketest for Mendeley reference adapter."""

import sqlite3
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_test_db(db_path: Path, pdf_path: Path) -> None:
    """Create a minimal Mendeley-like SQLite database for testing."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE Documents (
                id INTEGER PRIMARY KEY,
                uuid TEXT,
                title TEXT,
                abstract TEXT,
                year INTEGER,
                publication TEXT,
                type TEXT,
                doi TEXT,
                added INTEGER,
                modified INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE DocumentContributors (
                documentId INTEGER,
                firstName TEXT,
                lastName TEXT,
                position INTEGER,
                role TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE Folders (
                id INTEGER PRIMARY KEY,
                name TEXT,
                parentId INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE DocumentFolders (
                documentId INTEGER,
                folderId INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE Keywords (
                id INTEGER PRIMARY KEY,
                keyword TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE DocumentKeywords (
                documentId INTEGER,
                keywordId INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE Files (
                hash TEXT,
                localUrl TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE DocumentFiles (
                documentId INTEGER,
                fileHash TEXT
            )
            """
        )

        conn.execute(
            """
            INSERT INTO Documents
                (id, uuid, title, abstract, year, publication, type, doi, added, modified)
            VALUES
                (1, 'doc-uuid-1', 'Test Paper', 'Test abstract', 2022,
                 'Journal of Testing', 'Journal Article', '10.1000/test', 1700000000, 1700000100)
            """
        )
        conn.executemany(
            """
            INSERT INTO DocumentContributors
                (documentId, firstName, lastName, position, role)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (1, "Alice", "Smith", 1, "author"),
                (1, "Bob", "Jones", 2, "author"),
            ],
        )
        conn.executemany(
            """
            INSERT INTO Folders (id, name, parentId)
            VALUES (?, ?, ?)
            """,
            [
                (10, "Root", None),
                (11, "Subfolder", 10),
            ],
        )
        conn.execute(
            "INSERT INTO DocumentFolders (documentId, folderId) VALUES (1, 11)"
        )
        conn.execute(
            "INSERT INTO Keywords (id, keyword) VALUES (5, 'smoketest')"
        )
        conn.execute(
            "INSERT INTO DocumentKeywords (documentId, keywordId) VALUES (1, 5)"
        )
        conn.execute(
            "INSERT INTO Files (hash, localUrl) VALUES (?, ?)",
            ("filehash1", pdf_path.as_uri()),
        )
        conn.execute(
            "INSERT INTO DocumentFiles (documentId, fileHash) VALUES (1, 'filehash1')"
        )
        conn.commit()
    finally:
        conn.close()


def test_adapter() -> bool:
    """Test basic adapter functionality."""
    print("=" * 60)
    print("Testing Mendeley Adapter")
    print("=" * 60)

    from src.references.mendeley_adapter import MendeleyReferenceDB

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%test\n")

        db_path = tmp_path / "mendeley.sqlite"
        create_test_db(db_path, pdf_path)

        adapter = MendeleyReferenceDB(db_path)
        try:
            count = adapter.get_paper_count()
            print(f"  Paper count: {count}")
            count_passed = count == 1

            papers = list(adapter.get_all_papers())
            paper = papers[0] if papers else None
            print(f"  Papers loaded: {len(papers)}")
            paper_passed = paper is not None and paper.title == "Test Paper"

            author_passed = paper and len(paper.authors) == 2
            collection_passed = paper and "Root/Subfolder" in paper.collections
            tag_passed = paper and "smoketest" in paper.tags
            pdf_passed = paper and paper.pdf_path == pdf_path

            print(f"  Authors parsed: {'PASS' if author_passed else 'FAIL'}")
            print(f"  Collections parsed: {'PASS' if collection_passed else 'FAIL'}")
            print(f"  Tags parsed: {'PASS' if tag_passed else 'FAIL'}")
            print(f"  PDF path resolved: {'PASS' if pdf_passed else 'FAIL'}")

            by_key = adapter.get_paper_by_key("doc-uuid-1")
            key_passed = by_key is not None and by_key.title == "Test Paper"
            print(f"  Get by UUID: {'PASS' if key_passed else 'FAIL'}")

            return all(
                [
                    count_passed,
                    paper_passed,
                    author_passed,
                    collection_passed,
                    tag_passed,
                    pdf_passed,
                    key_passed,
                ]
            )
        finally:
            adapter.close()


def test_factory() -> bool:
    """Test factory integration."""
    print("\n" + "=" * 60)
    print("Testing Factory Integration")
    print("=" * 60)

    from src.references.factory import create_reference_db, get_available_providers

    providers = get_available_providers()
    print(f"  Available providers: {providers}")
    mendeley_available = "mendeley" in providers

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "mendeley.sqlite"
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%test\n")
        create_test_db(db_path, pdf_path)

        try:
            db = create_reference_db(
                provider="mendeley",
                db_path=db_path,
            )
            factory_works = db.provider == "mendeley"
            print(f"  Factory created: {db.provider}")
            if hasattr(db, "close"):
                db.close()
        except Exception as e:
            print(f"  Factory error: {e}")
            factory_works = False

    return mendeley_available and factory_works


def main() -> int:
    print("LITRIS Mendeley Adapter Smoketest")
    print("=" * 60)

    results = {
        "adapter": test_adapter(),
        "factory": test_factory(),
    }

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for test, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test}: {status}")

    all_passed = all(results.values())
    print("\nMendeley adapter is ready!" if all_passed else "\nSome tests failed.")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
