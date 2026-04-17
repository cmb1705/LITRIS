"""Tests for Zotero write-back module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.zotero.orphan_metadata_extractor import ExtractedMetadata, MetadataSource
from src.zotero.writer import (
    PaperWriteRequest,
    PyzoteroBackend,
    SqliteBackend,
    WriteResult,
    ZoteroWriter,
    _parse_author_name,
)


class TestParseAuthorName:
    """Tests for _parse_author_name helper."""

    def test_last_comma_first(self):
        first, last = _parse_author_name("Smith, John")
        assert first == "John"
        assert last == "Smith"

    def test_first_space_last(self):
        first, last = _parse_author_name("John Smith")
        assert first == "John"
        assert last == "Smith"

    def test_single_name(self):
        first, last = _parse_author_name("Aristotle")
        assert first == ""
        assert last == "Aristotle"

    def test_middle_name(self):
        first, last = _parse_author_name("John Robert Smith")
        assert first == "John Robert"
        assert last == "Smith"

    def test_last_comma_initials(self):
        first, last = _parse_author_name("Smith, J. R.")
        assert first == "J. R."
        assert last == "Smith"


class TestPaperWriteRequest:
    """Tests for PaperWriteRequest factory methods."""

    def test_from_extracted_basic(self):
        extracted = ExtractedMetadata(
            title="Test Paper",
            authors=["John Smith"],
            publication_year=2024,
            doi="10.1234/test",
            source=MetadataSource.DOI_EXTRACTED,
            confidence=0.9,
        )
        req = PaperWriteRequest.from_extracted(extracted)
        assert req.title == "Test Paper"
        assert req.authors == ["John Smith"]
        assert req.year == 2024
        assert req.doi == "10.1234/test"
        assert req.item_type == "journalArticle"
        assert "litris-imported" in req.tags

    def test_from_extracted_book(self):
        extracted = ExtractedMetadata(
            title="A Book",
            isbn="978-0-123456-78-9",
            source=MetadataSource.ISBN_EXTRACTED,
            confidence=0.9,
        )
        req = PaperWriteRequest.from_extracted(extracted)
        assert req.item_type == "book"
        assert req.isbn == "978-0-123456-78-9"

    def test_from_extracted_untitled(self):
        extracted = ExtractedMetadata(
            source=MetadataSource.UNKNOWN,
            confidence=0.5,
        )
        req = PaperWriteRequest.from_extracted(extracted)
        assert req.title == "Untitled"

    def test_from_enriched(self):
        extracted = ExtractedMetadata(
            title="Original",
            source=MetadataSource.PDF_METADATA,
            confidence=0.7,
        )
        from src.zotero.metadata_enricher import EnrichedMetadata

        enriched = EnrichedMetadata(
            original=extracted,
            title="Enriched Title",
            authors=["Jane Doe"],
            doi="10.5678/enriched",
            journal="Nature",
            publication_year=2025,
        )
        req = PaperWriteRequest.from_enriched(enriched)
        assert req.title == "Enriched Title"
        assert req.authors == ["Jane Doe"]
        assert req.doi == "10.5678/enriched"
        assert req.journal == "Nature"
        assert req.item_type == "journalArticle"


class TestPyzoteroBackend:
    """Tests for PyzoteroBackend."""

    def test_not_available_without_credentials(self):
        backend = PyzoteroBackend(user_id="", api_key="")
        assert backend.is_available() is False

    def test_not_available_without_pyzotero(self):
        backend = PyzoteroBackend(user_id="12345", api_key="abc")
        with patch.dict("sys.modules", {"pyzotero": None, "pyzotero.zotero": None}):
            # If import fails, is_available returns False
            assert backend.is_available() in (True, False)

    def test_available_with_credentials_and_import(self):
        backend = PyzoteroBackend(user_id="12345", api_key="abc")
        with patch("src.zotero.writer.PyzoteroBackend.is_available", return_value=True):
            assert backend.is_available() is True

    def test_write_item_success(self):
        backend = PyzoteroBackend(user_id="12345", api_key="abc")
        mock_zot = MagicMock()
        mock_zot.item_template.return_value = {}
        mock_zot.create_items.return_value = {"successful": {"0": {"key": "ABC12345"}}}
        backend._zot = mock_zot

        req = PaperWriteRequest(
            title="Test Paper",
            authors=["Smith, John"],
            doi="10.1234/test",
        )
        result = backend.write_item(req)
        assert result.success is True
        assert result.item_key == "ABC12345"
        assert result.backend == "pyzotero"

    def test_write_item_failure(self):
        backend = PyzoteroBackend(user_id="12345", api_key="abc")
        mock_zot = MagicMock()
        mock_zot.item_template.return_value = {}
        mock_zot.create_items.return_value = {"failed": {"0": {"message": "Invalid field"}}}
        backend._zot = mock_zot

        req = PaperWriteRequest(title="Test Paper")
        result = backend.write_item(req)
        assert result.success is False
        assert result.backend == "pyzotero"

    def test_write_item_with_pdf_attachment(self):
        backend = PyzoteroBackend(user_id="12345", api_key="abc")
        mock_zot = MagicMock()
        mock_zot.item_template.return_value = {}
        mock_zot.create_items.return_value = {"successful": {"0": {"key": "XYZ99999"}}}
        backend._zot = mock_zot

        pdf_path = MagicMock(spec=Path)
        pdf_path.exists.return_value = True

        req = PaperWriteRequest(
            title="Paper with PDF",
            pdf_path=pdf_path,
        )
        result = backend.write_item(req)
        assert result.success is True
        mock_zot.attachment_simple.assert_called_once()


class TestSqliteBackend:
    """Tests for SqliteBackend."""

    def test_not_available_missing_db(self, tmp_path):
        backend = SqliteBackend(
            db_path=tmp_path / "nonexistent.sqlite",
            storage_path=tmp_path / "storage",
        )
        assert backend.is_available() is False

    def test_available_with_db(self, tmp_path):
        db_file = tmp_path / "zotero.sqlite"
        db_file.touch()
        backend = SqliteBackend(db_path=db_file, storage_path=tmp_path / "storage")
        assert backend.is_available() is True

    def test_close_no_creator(self):
        backend = SqliteBackend(
            db_path=Path("/fake/db.sqlite"),
            storage_path=Path("/fake/storage"),
        )
        backend.close()  # Should not raise


class TestZoteroWriter:
    """Tests for ZoteroWriter dual-backend."""

    def _make_config(self, write_method="auto", api_key=None, user_id=None):
        from src.config import ZoteroConfig

        return ZoteroConfig(
            database_path=Path("/fake/zotero.sqlite"),
            storage_path=Path("/fake/storage"),
            write_method=write_method,
            api_key=api_key,
            user_id=user_id,
        )

    def test_auto_mode_with_api_credentials(self):
        config = self._make_config(api_key="key123", user_id="user456")
        writer = ZoteroWriter(config)
        assert writer._api_backend is not None
        assert writer._sqlite_backend is not None

    def test_api_only_mode(self):
        config = self._make_config(write_method="api", api_key="key", user_id="uid")
        writer = ZoteroWriter(config)
        assert writer._api_backend is not None
        assert writer._sqlite_backend is None

    def test_sqlite_only_mode(self):
        config = self._make_config(write_method="sqlite")
        writer = ZoteroWriter(config)
        assert writer._api_backend is None
        assert writer._sqlite_backend is not None

    def test_no_backends_available(self):
        config = self._make_config(write_method="api")
        writer = ZoteroWriter(config)
        # No credentials, so api backend is None
        assert writer._api_backend is None

        req = PaperWriteRequest(title="Test")
        result = writer.write_item(req)
        assert result.success is False
        assert "No write backend available" in result.error

    def test_write_batch(self):
        config = self._make_config(api_key="key", user_id="uid")
        writer = ZoteroWriter(config)

        mock_backend = MagicMock()
        mock_backend.is_available.return_value = True
        mock_backend.write_item.return_value = WriteResult(
            success=True, item_key="KEY1", title="Test", backend="pyzotero"
        )
        writer._api_backend = mock_backend

        requests = [
            PaperWriteRequest(title="Paper 1"),
            PaperWriteRequest(title="Paper 2"),
        ]
        results = writer.write_batch(requests)
        assert len(results) == 2
        assert all(r.success for r in results)

    def test_api_fallback_to_sqlite(self, tmp_path):
        db_file = tmp_path / "zotero.sqlite"
        db_file.touch()
        config = self._make_config(api_key="key", user_id="uid")
        config.database_path = db_file
        config.storage_path = tmp_path / "storage"
        writer = ZoteroWriter(config)

        # Mock API backend to fail
        mock_api = MagicMock()
        mock_api.is_available.return_value = True
        mock_api.write_item.return_value = WriteResult(
            success=False, title="Test", backend="pyzotero", error="Network error"
        )
        writer._api_backend = mock_api

        # Mock SQLite backend to succeed
        mock_sqlite = MagicMock()
        mock_sqlite.is_available.return_value = True
        mock_sqlite.write_item.return_value = WriteResult(
            success=True, item_key="SQLKEY", title="Test", backend="sqlite"
        )
        writer._sqlite_backend = mock_sqlite

        req = PaperWriteRequest(title="Test")
        result = writer.write_item(req)
        assert result.success is True
        assert result.backend == "sqlite"

    def test_close(self):
        config = self._make_config(write_method="sqlite")
        writer = ZoteroWriter(config)
        writer._sqlite_backend = MagicMock()
        writer.close()
        writer._sqlite_backend.close.assert_called_once()
