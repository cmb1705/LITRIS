"""Mendeley reference database adapter.

Reads metadata from the Mendeley Desktop SQLite database and maps it to
LITRIS PaperMetadata objects.
"""

import hashlib
import sqlite3
from collections import defaultdict
from collections.abc import Callable, Generator
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

from src.references.base import BaseReferenceDB, ReferenceProvider
from src.utils.logging_config import get_logger
from src.zotero.models import Author, PaperMetadata

logger = get_logger(__name__)


class MendeleyReferenceDB(BaseReferenceDB):
    """Mendeley Desktop reference database adapter."""

    _DOCUMENT_TABLE_CANDIDATES = ["Documents", "Document"]
    _CONTRIBUTOR_TABLE_CANDIDATES = [
        "DocumentContributors",
        "DocumentAuthors",
        "DocumentContrib",
        "Contributors",
    ]
    _FOLDER_TABLE_CANDIDATES = ["Folders", "Collections"]
    _DOC_FOLDER_TABLE_CANDIDATES = ["DocumentFolders", "DocumentCollections"]
    _KEYWORD_TABLE_CANDIDATES = ["Keywords", "Tags"]
    _DOC_KEYWORD_TABLE_CANDIDATES = ["DocumentKeywords", "DocumentTags"]
    _FILES_TABLE_CANDIDATES = ["Files", "FilePaths"]
    _DOC_FILES_TABLE_CANDIDATES = ["DocumentFiles", "DocumentAttachments"]

    _DOC_TYPE_MAPPING = {
        "journal article": "journalArticle",
        "conference proceedings": "conferencePaper",
        "conference paper": "conferencePaper",
        "book": "book",
        "book section": "bookSection",
        "thesis": "thesis",
        "report": "report",
        "working paper": "report",
        "preprint": "preprint",
        "magazine article": "document",
        "newspaper article": "document",
        "patent": "document",
    }

    def __init__(self, db_path: Path, storage_path: Path | None = None):
        """Initialize Mendeley reference database.

        Args:
            db_path: Path to the Mendeley SQLite database.
            storage_path: Optional storage root for resolving relative file paths.
        """
        if not db_path.exists():
            raise FileNotFoundError(f"Mendeley database not found: {db_path}")

        self.db_path = db_path
        self.storage_path = storage_path
        self._conn = self._connect_readonly(db_path)
        self._conn.row_factory = sqlite3.Row

        self._tables = self._load_tables()
        self._tables_by_lower = {name.lower(): name for name in self._tables}
        self._table_columns = {
            table: self._load_columns(table) for table in self._tables
        }

        self._documents_table = self._find_table(self._DOCUMENT_TABLE_CANDIDATES)
        if not self._documents_table:
            raise ValueError("Mendeley database missing Documents table")

        self._doc_columns = self._table_columns[self._documents_table]
        self._doc_id_col = self._find_column(
            self._documents_table, ["id", "documentid", "document_id"]
        )
        self._doc_uuid_col = self._find_column(
            self._documents_table, ["uuid", "documentuuid", "document_uuid"]
        )

        if not self._doc_id_col and not self._doc_uuid_col:
            raise ValueError("Documents table missing identifier columns")

        self._relations_loaded = False
        self._authors_by_doc: dict[str, list[Author]] = {}
        self._collections_by_doc: dict[str, list[str]] = {}
        self._tags_by_doc: dict[str, list[str]] = {}
        self._pdf_by_doc: dict[str, tuple[Path, str] | None] = {}

    @property
    def provider(self) -> ReferenceProvider:
        """Return the provider identifier."""
        return "mendeley"

    @property
    def source_path(self) -> Path:
        """Return the database path."""
        return self.db_path

    def get_all_papers(
        self, progress_callback: Callable[[int, int], None] | None = None
    ) -> Generator[PaperMetadata, None, None]:
        """Get all papers from the Mendeley database.

        Memory: O(n) for relationship caches (authors, collections, tags, files).

        Args:
            progress_callback: Optional callback(current, total) for progress.

        Yields:
            PaperMetadata objects for each document.
        """
        self._load_relationships()
        total = self.get_paper_count()

        cursor = self._conn.execute(f"SELECT * FROM {self._documents_table}")
        for i, row in enumerate(cursor):
            paper = self._row_to_paper(row)
            if progress_callback:
                progress_callback(i + 1, total)
            yield paper

    def get_paper_count(self) -> int:
        """Get total count of documents.

        Returns:
            Number of documents in the database.
        """
        cursor = self._conn.execute(
            f"SELECT COUNT(*) AS count FROM {self._documents_table}"
        )
        row = cursor.fetchone()
        return int(row["count"]) if row else 0

    def get_paper_by_key(self, key: str) -> PaperMetadata | None:
        """Get a specific paper by Mendeley document ID or UUID.

        Args:
            key: Document ID (integer) or UUID (string).

        Returns:
            PaperMetadata or None if not found.
        """
        self._load_relationships()

        row = self._select_document_by_key(key)
        if not row:
            return None
        return self._row_to_paper(row)

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()

    def _connect_readonly(self, db_path: Path) -> sqlite3.Connection:
        """Open the SQLite database in read-only mode."""
        try:
            db_uri = f"file:{db_path.as_posix()}?mode=ro"
            return sqlite3.connect(db_uri, uri=True)
        except sqlite3.OperationalError:
            logger.warning("Read-only SQLite connection failed; falling back to default")
            return sqlite3.connect(db_path)

    def _load_tables(self) -> list[str]:
        """Load available tables from the SQLite database."""
        cursor = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        return [row[0] for row in cursor.fetchall()]

    def _load_columns(self, table_name: str) -> dict[str, str]:
        """Load columns for a table, keyed by lowercase name."""
        cursor = self._conn.execute(f"PRAGMA table_info({table_name})")
        columns = {}
        for row in cursor.fetchall():
            columns[row["name"].lower()] = row["name"]
        return columns

    def _find_table(self, candidates: list[str]) -> str | None:
        """Find a table name from candidate list."""
        for candidate in candidates:
            table = self._tables_by_lower.get(candidate.lower())
            if table:
                return table
        return None

    def _find_column(self, table_name: str, candidates: list[str]) -> str | None:
        """Find a column name from candidate list."""
        columns = self._table_columns.get(table_name, {})
        for candidate in candidates:
            column = columns.get(candidate.lower())
            if column:
                return column
        return None

    def _load_relationships(self) -> None:
        """Load authors, collections, tags, and file paths."""
        if self._relations_loaded:
            return

        self._authors_by_doc = self._load_authors()
        self._collections_by_doc = self._load_collections()
        self._tags_by_doc = self._load_tags()
        self._pdf_by_doc = self._load_pdf_paths()
        self._relations_loaded = True

    def _load_authors(self) -> dict[str, list[Author]]:
        """Load contributors into a document -> authors mapping."""
        table = self._find_table(self._CONTRIBUTOR_TABLE_CANDIDATES)
        if not table:
            return {}

        doc_id_col = self._find_column(table, ["documentid", "document_id"])
        first_name_col = self._find_column(table, ["firstname", "first_name", "given"])
        last_name_col = self._find_column(table, ["lastname", "last_name", "family"])
        order_col = self._find_column(table, ["position", "order", "sequence"])
        role_col = self._find_column(table, ["role", "contribution", "type"])

        if not doc_id_col or not last_name_col:
            return {}

        authors_by_doc: dict[str, list[Author]] = defaultdict(list)
        cursor = self._conn.execute(f"SELECT * FROM {table}")

        for row in cursor:
            doc_id = str(row[doc_id_col])
            order = row[order_col] if order_col else None
            if order is None or order == 0:
                order = len(authors_by_doc[doc_id]) + 1
            first_name = row[first_name_col] if first_name_col else ""
            last_name = row[last_name_col] if last_name_col else ""
            role = row[role_col] if role_col else "author"
            authors_by_doc[doc_id].append(
                Author(
                    first_name=str(first_name or ""),
                    last_name=str(last_name or ""),
                    order=int(order),
                    role=str(role or "author"),
                )
            )

        # Sort by order where available
        for doc_id, authors in authors_by_doc.items():
            authors_by_doc[doc_id] = sorted(authors, key=lambda a: a.order)

        return dict(authors_by_doc)

    def _load_collections(self) -> dict[str, list[str]]:
        """Load folder paths into a document -> collections mapping."""
        folder_table = self._find_table(self._FOLDER_TABLE_CANDIDATES)
        doc_folder_table = self._find_table(self._DOC_FOLDER_TABLE_CANDIDATES)
        if not folder_table or not doc_folder_table:
            return {}

        folder_id_col = self._find_column(folder_table, ["id", "folderid"])
        folder_name_col = self._find_column(folder_table, ["name", "title"])
        parent_id_col = self._find_column(folder_table, ["parentid", "parent_id"])
        doc_id_col = self._find_column(doc_folder_table, ["documentid", "document_id"])
        doc_folder_id_col = self._find_column(doc_folder_table, ["folderid", "collectionid"])

        if not folder_id_col or not folder_name_col or not doc_id_col or not doc_folder_id_col:
            return {}

        folders: dict[str, tuple[str, str | None]] = {}
        cursor = self._conn.execute(f"SELECT * FROM {folder_table}")
        for row in cursor:
            folder_id = str(row[folder_id_col])
            name = str(row[folder_name_col] or "").strip()
            parent_id = str(row[parent_id_col]) if parent_id_col and row[parent_id_col] else None
            folders[folder_id] = (name, parent_id)

        def build_path(folder_id: str, cache: dict[str, str]) -> str:
            if folder_id in cache:
                return cache[folder_id]
            name, parent_id = folders.get(folder_id, ("", None))
            if parent_id and parent_id in folders:
                path = f"{build_path(parent_id, cache)}/{name}" if name else build_path(parent_id, cache)
            else:
                path = name
            cache[folder_id] = path
            return path

        path_cache: dict[str, str] = {}
        collection_paths = {fid: build_path(fid, path_cache) for fid in folders}

        collections_by_doc: dict[str, list[str]] = defaultdict(list)
        cursor = self._conn.execute(f"SELECT * FROM {doc_folder_table}")
        for row in cursor:
            doc_id = str(row[doc_id_col])
            folder_id = str(row[doc_folder_id_col])
            path = collection_paths.get(folder_id)
            if path:
                collections_by_doc[doc_id].append(path)

        return dict(collections_by_doc)

    def _load_tags(self) -> dict[str, list[str]]:
        """Load keywords into a document -> tags mapping."""
        keyword_table = self._find_table(self._KEYWORD_TABLE_CANDIDATES)
        doc_keyword_table = self._find_table(self._DOC_KEYWORD_TABLE_CANDIDATES)
        if not doc_keyword_table:
            return {}

        doc_id_col = self._find_column(doc_keyword_table, ["documentid", "document_id"])
        keyword_id_col = self._find_column(doc_keyword_table, ["keywordid", "tagid"])
        keyword_value_col = self._find_column(doc_keyword_table, ["keyword", "tag"])

        if not doc_id_col:
            return {}

        keyword_lookup: dict[str, str] = {}
        if keyword_table and keyword_id_col:
            kw_id_col = self._find_column(keyword_table, ["id", "keywordid", "tagid"])
            kw_value_col = self._find_column(keyword_table, ["keyword", "tag", "name"])
            if kw_id_col and kw_value_col:
                cursor = self._conn.execute(f"SELECT * FROM {keyword_table}")
                for row in cursor:
                    keyword_lookup[str(row[kw_id_col])] = str(row[kw_value_col] or "").strip()

        tags_by_doc: dict[str, list[str]] = defaultdict(list)
        cursor = self._conn.execute(f"SELECT * FROM {doc_keyword_table}")
        for row in cursor:
            doc_id = str(row[doc_id_col])
            tag = None
            if keyword_value_col:
                tag = row[keyword_value_col]
            elif keyword_id_col:
                tag = keyword_lookup.get(str(row[keyword_id_col]))
            if tag:
                tag_str = str(tag).strip()
                if tag_str:
                    tags_by_doc[doc_id].append(tag_str)

        return dict(tags_by_doc)

    def _load_pdf_paths(self) -> dict[str, tuple[Path, str] | None]:
        """Load PDF paths into a document -> (path, attachment_key) mapping."""
        doc_files_table = self._find_table(self._DOC_FILES_TABLE_CANDIDATES)
        files_table = self._find_table(self._FILES_TABLE_CANDIDATES)

        if not doc_files_table:
            return {}

        doc_id_col = self._find_column(doc_files_table, ["documentid", "document_id"])
        file_key_col = self._find_column(doc_files_table, ["filehash", "fileid", "file_id", "hash"])
        file_path_col = self._find_column(doc_files_table, ["localurl", "filepath", "path", "url"])

        files_by_key: dict[str, tuple[Path, str]] = {}
        if files_table:
            files_key_col = self._find_column(files_table, ["hash", "filehash", "id", "fileid"])
            files_path_col = self._find_column(files_table, ["localurl", "filepath", "path", "url"])
            if files_key_col and files_path_col:
                cursor = self._conn.execute(f"SELECT * FROM {files_table}")
                for row in cursor:
                    raw_path = row[files_path_col]
                    path = self._normalize_path(raw_path)
                    if path:
                        files_by_key[str(row[files_key_col])] = (path, str(row[files_key_col]))

        pdf_by_doc: dict[str, tuple[Path, str] | None] = {}
        cursor = self._conn.execute(f"SELECT * FROM {doc_files_table}")
        for row in cursor:
            if not doc_id_col:
                continue
            doc_id = str(row[doc_id_col])
            if doc_id in pdf_by_doc:
                continue

            path = None
            attachment_key = None
            if file_path_col:
                path = self._normalize_path(row[file_path_col])
                attachment_key = str(row[file_path_col]) if path else None
            if not path and file_key_col:
                file_key = str(row[file_key_col])
                file_entry = files_by_key.get(file_key)
                if file_entry:
                    path, attachment_key = file_entry

            if path:
                pdf_by_doc[doc_id] = (path, attachment_key or doc_id)

        return pdf_by_doc

    def _select_document_by_key(self, key: str) -> sqlite3.Row | None:
        """Select a document row by numeric ID or UUID."""
        if self._doc_id_col:
            try:
                doc_id = int(key)
                cursor = self._conn.execute(
                    f"SELECT * FROM {self._documents_table} WHERE {self._doc_id_col} = ?",
                    (doc_id,),
                )
                row = cursor.fetchone()
                if row:
                    return row
            except ValueError:
                pass

        if self._doc_uuid_col:
            cursor = self._conn.execute(
                f"SELECT * FROM {self._documents_table} WHERE {self._doc_uuid_col} = ?",
                (key,),
            )
            return cursor.fetchone()

        return None

    def _row_to_paper(self, row: sqlite3.Row) -> PaperMetadata:
        """Convert a document row to PaperMetadata."""
        doc_id = str(row[self._doc_id_col]) if self._doc_id_col else None
        doc_uuid = str(row[self._doc_uuid_col]) if self._doc_uuid_col else None
        doc_key = doc_uuid or doc_id or "unknown"

        zotero_item_id = int(doc_id) if doc_id and doc_id.isdigit() else self._hash_to_int(doc_key)

        title = self._row_value(row, ["title", "documenttitle", "name"]) or "Untitled"
        abstract = self._row_value(row, ["abstract", "abstracttext"])
        year = self._row_value(row, ["year", "publicationyear"])
        publication = self._row_value(row, ["publication", "journal", "source"])
        volume = self._row_value(row, ["volume"])
        issue = self._row_value(row, ["issue", "number"])
        pages = self._row_value(row, ["pages"])
        doi = self._row_value(row, ["doi"])
        isbn = self._row_value(row, ["isbn"])
        issn = self._row_value(row, ["issn"])
        url = self._row_value(row, ["url", "link"])
        doc_type = self._row_value(row, ["type", "documenttype", "doctype"])
        added = self._row_value(row, ["added", "dateadded", "created"])
        modified = self._row_value(row, ["modified", "datemodified", "lastmodified"])

        publication_year = None
        if year is not None:
            try:
                publication_year = int(str(year))
            except ValueError:
                publication_year = None

        item_type = self._map_item_type(doc_type)
        date_added = self._parse_datetime(added)
        date_modified = self._parse_datetime(modified) or date_added

        authors = self._authors_by_doc.get(doc_id or doc_key, [])
        collections = self._collections_by_doc.get(doc_id or doc_key, [])
        tags = self._tags_by_doc.get(doc_id or doc_key, [])

        pdf_entry = self._pdf_by_doc.get(doc_id or doc_key)
        pdf_path = pdf_entry[0] if pdf_entry else None
        attachment_key = pdf_entry[1] if pdf_entry else None

        return PaperMetadata(
            zotero_key=doc_key,
            zotero_item_id=zotero_item_id,
            item_type=item_type,
            title=str(title),
            authors=authors,
            publication_year=publication_year,
            publication_date=str(publication_year) if publication_year else None,
            journal=str(publication) if publication else None,
            volume=str(volume) if volume else None,
            issue=str(issue) if issue else None,
            pages=str(pages) if pages else None,
            doi=str(doi) if doi else None,
            isbn=str(isbn) if isbn else None,
            issn=str(issn) if issn else None,
            abstract=str(abstract) if abstract else None,
            url=str(url) if url else None,
            collections=collections,
            tags=tags,
            pdf_path=pdf_path,
            pdf_attachment_key=attachment_key,
            date_added=date_added,
            date_modified=date_modified,
        )

    def _row_value(self, row: sqlite3.Row, candidates: list[str]) -> str | None:
        """Return the first available value from candidate column names."""
        for candidate in candidates:
            column = self._doc_columns.get(candidate.lower())
            if column and row[column] is not None:
                return row[column]
        return None

    def _map_item_type(self, raw_type: str | None) -> str:
        """Map Mendeley document type to Zotero-style item type."""
        if not raw_type:
            return "document"
        normalized = str(raw_type).strip().lower()
        return self._DOC_TYPE_MAPPING.get(normalized, "document")

    def _parse_datetime(self, value) -> datetime:
        """Parse timestamps from Mendeley database values."""
        if value is None:
            return datetime.now()

        if isinstance(value, (int, float)):
            timestamp = float(value)
            if timestamp > 1e12:
                timestamp /= 1000.0
            try:
                return datetime.fromtimestamp(timestamp)
            except (OSError, ValueError):
                return datetime.now()

        text = str(value).strip()
        if text.isdigit():
            return self._parse_datetime(int(text))

        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return datetime.now()

    def _normalize_path(self, value) -> Path | None:
        """Normalize a file path or file:// URL to a Path."""
        if not value:
            return None

        text = str(value)
        if text.startswith("file://"):
            parsed = urlparse(text)
            path_str = unquote(parsed.path)
            if path_str.startswith("/") and len(path_str) > 3 and path_str[2] == ":":
                path_str = path_str[1:]
            path = Path(path_str)
        else:
            path = Path(text)

        if not path.is_absolute() and self.storage_path:
            path = self.storage_path / path

        return path

    def _hash_to_int(self, value: str) -> int:
        """Hash a string to a stable 32-bit integer."""
        stable_hash = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
        return int(stable_hash, 16) % 2147483647
