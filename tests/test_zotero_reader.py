"""Tests for Zotero database reader."""

import sqlite3
from datetime import datetime

import pytest

from src.zotero.database import FIELD_MAPPING, ZoteroDatabase
from src.zotero.models import Author, Collection, PaperMetadata


class TestAuthorModel:
    """Tests for Author model."""

    def test_full_name_two_fields(self):
        """Test full name with first and last name."""
        author = Author(first_name="John", last_name="Doe", order=1)
        assert author.full_name == "John Doe"

    def test_full_name_last_only(self):
        """Test full name with only last name (single-field mode)."""
        author = Author(first_name="", last_name="Aristotle", order=1)
        assert author.full_name == "Aristotle"

    def test_full_name_first_only(self):
        """Test full name with only first name."""
        author = Author(first_name="Madonna", last_name="", order=1)
        assert author.full_name == "Madonna"

    def test_full_name_empty(self):
        """Test full name when both empty."""
        author = Author(first_name="", last_name="", order=1)
        assert author.full_name == "Unknown"

    def test_default_role(self):
        """Test default role is author."""
        author = Author(first_name="John", last_name="Doe")
        assert author.role == "author"


class TestCollectionModel:
    """Tests for Collection model."""

    def test_full_path_no_parents(self):
        """Test full path for root collection."""
        coll = Collection(collection_id=1, name="My Papers")
        assert coll.full_path == "My Papers"

    def test_full_path_with_parents(self):
        """Test full path with parent collections."""
        coll = Collection(
            collection_id=3,
            name="Subsection",
            parent_path=["Research", "Topic"],
        )
        assert coll.full_path == "Research/Topic/Subsection"


class TestPaperMetadataModel:
    """Tests for PaperMetadata model."""

    @pytest.fixture
    def minimal_paper(self):
        """Create minimal valid paper metadata."""
        return PaperMetadata(
            zotero_key="ABC12345",
            zotero_item_id=100,
            item_type="journalArticle",
            title="Test Paper",
            date_added=datetime(2023, 1, 1),
            date_modified=datetime(2023, 1, 2),
        )

    def test_minimal_paper_creation(self, minimal_paper):
        """Test paper can be created with minimal fields."""
        assert minimal_paper.title == "Test Paper"
        assert minimal_paper.zotero_key == "ABC12345"
        assert minimal_paper.paper_id  # UUID generated

    def test_year_extraction_iso_date(self, minimal_paper):
        """Test year extraction from ISO date."""
        paper = PaperMetadata(
            **{**minimal_paper.model_dump(), "publication_date": "2023-05-15"}
        )
        assert paper.publication_year == 2023

    def test_year_extraction_partial_date(self, minimal_paper):
        """Test year extraction from partial date."""
        paper = PaperMetadata(
            **{**minimal_paper.model_dump(), "publication_date": "May 2022"}
        )
        assert paper.publication_year == 2022

    def test_year_extraction_year_only(self, minimal_paper):
        """Test year extraction from year-only string."""
        paper = PaperMetadata(
            **{**minimal_paper.model_dump(), "publication_date": "2021"}
        )
        assert paper.publication_year == 2021

    def test_empty_title_becomes_untitled(self):
        """Test empty title is replaced with Untitled."""
        paper = PaperMetadata(
            zotero_key="ABC12345",
            zotero_item_id=100,
            item_type="journalArticle",
            title="",
            date_added=datetime(2023, 1, 1),
            date_modified=datetime(2023, 1, 2),
        )
        assert paper.title == "Untitled"

    def test_author_string_single(self):
        """Test author string with single author."""
        paper = PaperMetadata(
            zotero_key="ABC12345",
            zotero_item_id=100,
            item_type="journalArticle",
            title="Test",
            authors=[Author(first_name="John", last_name="Doe")],
            date_added=datetime(2023, 1, 1),
            date_modified=datetime(2023, 1, 2),
        )
        assert paper.author_string == "John Doe"

    def test_author_string_two(self):
        """Test author string with two authors."""
        paper = PaperMetadata(
            zotero_key="ABC12345",
            zotero_item_id=100,
            item_type="journalArticle",
            title="Test",
            authors=[
                Author(first_name="John", last_name="Doe"),
                Author(first_name="Jane", last_name="Smith"),
            ],
            date_added=datetime(2023, 1, 1),
            date_modified=datetime(2023, 1, 2),
        )
        assert paper.author_string == "John Doe and Jane Smith"

    def test_author_string_many(self):
        """Test author string with many authors."""
        paper = PaperMetadata(
            zotero_key="ABC12345",
            zotero_item_id=100,
            item_type="journalArticle",
            title="Test",
            authors=[
                Author(first_name="John", last_name="Doe"),
                Author(first_name="Jane", last_name="Smith"),
                Author(first_name="Bob", last_name="Jones"),
            ],
            date_added=datetime(2023, 1, 1),
            date_modified=datetime(2023, 1, 2),
        )
        assert paper.author_string == "John Doe et al."

    def test_citation_key_generation(self):
        """Test citation key is generated correctly."""
        paper = PaperMetadata(
            zotero_key="ABC12345",
            zotero_item_id=100,
            item_type="journalArticle",
            title="Test",
            authors=[Author(first_name="John", last_name="Doe")],
            publication_year=2023,
            date_added=datetime(2023, 1, 1),
            date_modified=datetime(2023, 1, 2),
        )
        assert paper.citation_key == "Doe2023"

    def test_to_index_dict(self, minimal_paper):
        """Test conversion to index dictionary."""
        result = minimal_paper.to_index_dict()
        assert result["title"] == "Test Paper"
        assert result["zotero_key"] == "ABC12345"
        assert "paper_id" in result


class TestZoteroDatabase:
    """Tests for ZoteroDatabase class."""

    @pytest.fixture
    def mock_db_path(self, tmp_path):
        """Create a mock database path."""
        return tmp_path / "zotero.sqlite"

    @pytest.fixture
    def mock_storage_path(self, tmp_path):
        """Create a mock storage path."""
        storage = tmp_path / "storage"
        storage.mkdir()
        return storage

    def test_resolve_pdf_path_storage_format(self, mock_db_path, mock_storage_path):
        """Test PDF path resolution for storage format."""
        # Create mock PDF
        att_key = "ABCD1234"
        pdf_dir = mock_storage_path / att_key
        pdf_dir.mkdir()
        pdf_file = pdf_dir / "test.pdf"
        pdf_file.write_bytes(b"PDF content")

        db = ZoteroDatabase(mock_db_path, mock_storage_path)
        result = db.resolve_pdf_path(att_key, "storage:test.pdf")

        assert result == pdf_file
        assert result.exists()

    def test_resolve_pdf_path_url_attachment(self, mock_db_path, mock_storage_path):
        """Test PDF path resolution skips URL attachments."""
        db = ZoteroDatabase(mock_db_path, mock_storage_path)
        result = db.resolve_pdf_path("KEY", "http://example.com/paper.pdf")
        assert result is None

    def test_resolve_pdf_path_missing_file(self, mock_db_path, mock_storage_path):
        """Test PDF path resolution handles missing files."""
        db = ZoteroDatabase(mock_db_path, mock_storage_path)
        result = db.resolve_pdf_path("MISSING", "storage:nonexistent.pdf")
        assert result is None

    def test_resolve_pdf_path_linked_file(self, mock_db_path, mock_storage_path, tmp_path):
        """Test PDF path resolution for linked files."""
        # Create a linked file
        linked_pdf = tmp_path / "linked_paper.pdf"
        linked_pdf.write_bytes(b"PDF content")

        db = ZoteroDatabase(mock_db_path, mock_storage_path)
        result = db.resolve_pdf_path("KEY", str(linked_pdf))

        assert result == linked_pdf

    def test_resolve_pdf_path_blocks_path_traversal_dots(self, tmp_path):
        """Test that path traversal with .. is blocked."""
        mock_db_path = tmp_path / "zotero.sqlite"
        mock_db_path.touch()
        mock_storage_path = tmp_path / "storage"
        mock_storage_path.mkdir()

        db = ZoteroDatabase(mock_db_path, mock_storage_path)

        # Attempt path traversal via ..
        result = db.resolve_pdf_path("KEY", "storage:../../../etc/passwd")
        assert result is None

    def test_resolve_pdf_path_blocks_absolute_path_in_storage(self, tmp_path):
        """Test that absolute paths in storage: prefix are blocked."""
        mock_db_path = tmp_path / "zotero.sqlite"
        mock_db_path.touch()
        mock_storage_path = tmp_path / "storage"
        mock_storage_path.mkdir()

        db = ZoteroDatabase(mock_db_path, mock_storage_path)

        # Attempt absolute path injection
        result = db.resolve_pdf_path("KEY", "storage:/etc/passwd")
        assert result is None

    def test_resolve_pdf_path_blocks_traversal_outside_storage(self, tmp_path):
        """Test that resolved paths outside storage are blocked."""
        mock_db_path = tmp_path / "zotero.sqlite"
        mock_db_path.touch()
        mock_storage_path = tmp_path / "storage"
        mock_storage_path.mkdir()

        # Create a file outside storage to attempt access
        outside_file = tmp_path / "secret.txt"
        outside_file.write_text("secret data")

        db = ZoteroDatabase(mock_db_path, mock_storage_path)

        # Even if the filename looks innocent, verify resolution check
        result = db.resolve_pdf_path("KEY", "storage:..\\secret.txt")
        assert result is None

    def test_field_mapping_completeness(self):
        """Test that field mapping covers expected Zotero fields."""
        expected_fields = {
            "title",
            "abstractNote",
            "date",
            "publicationTitle",
            "volume",
            "issue",
            "pages",
            "DOI",
            "ISBN",
            "ISSN",
            "url",
        }
        assert set(FIELD_MAPPING.keys()) == expected_fields


class TestZoteroDatabaseIntegration:
    """Integration tests that require actual database (skipped by default)."""

    @pytest.fixture
    def real_config(self):
        """Load real config if available."""
        try:
            from src.config import Config
            config = Config.load()
            # Verify database exists
            db_path = config.get_zotero_db_path()
            if not db_path.exists():
                pytest.skip(f"Zotero database not found: {db_path}")
            return config
        except Exception as e:
            pytest.skip(f"Config not available for integration test: {e}")

    @pytest.mark.integration
    def test_read_only_mode(self, real_config):
        """Test that database is opened in read-only mode."""
        db = ZoteroDatabase(
            real_config.get_zotero_db_path(),
            real_config.get_storage_path(),
        )

        with db._get_connection() as conn:
            # Attempt to write should fail
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("CREATE TABLE test (id INTEGER)")

    @pytest.mark.integration
    def test_get_paper_count(self, real_config):
        """Test counting papers in database."""
        import sqlite3

        db = ZoteroDatabase(
            real_config.get_zotero_db_path(),
            real_config.get_storage_path(),
        )
        try:
            count = db.get_paper_count()
            assert count > 0
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                pytest.skip("Zotero database is locked (Zotero may be open)")
            raise
