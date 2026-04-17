"""Zotero SQLite database interface."""

import hashlib
import sqlite3
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.extraction.web_extractor import extract_html_attachment
from src.utils.logging_config import get_logger
from src.zotero.models import Author, Collection, PaperMetadata

logger = get_logger(__name__)

# Item types to skip (not actual papers)
SKIP_ITEM_TYPES = {"attachment", "note", "annotation"}

# Zotero field name to model attribute mapping
FIELD_MAPPING = {
    "title": "title",
    "abstractNote": "abstract",
    "date": "publication_date",
    "publicationTitle": "journal",
    "volume": "volume",
    "issue": "issue",
    "pages": "pages",
    "DOI": "doi",
    "ISBN": "isbn",
    "ISSN": "issn",
    "url": "url",
}


class ZoteroDatabase:
    """Read-only interface to Zotero SQLite database."""

    def __init__(
        self,
        db_path: Path,
        storage_path: Path,
        include_webpage_snapshots: bool = True,
    ):
        """Initialize database connection.

        Args:
            db_path: Path to zotero.sqlite file.
            storage_path: Path to Zotero storage directory.
            include_webpage_snapshots: Include local HTML attachment-backed
                webpage items as extractable sources.
        """
        self.db_path = db_path
        self.storage_path = storage_path
        self.include_webpage_snapshots = include_webpage_snapshots
        self._collection_cache: dict[int, Collection] | None = None

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a read-only database connection.

        Yields:
            SQLite connection in read-only mode.

        Raises:
            sqlite3.OperationalError: If database is locked or inaccessible.
        """
        uri = f"file:{self.db_path}?mode=ro"
        conn = None
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=30.0)
            conn.row_factory = sqlite3.Row
            yield conn
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                logger.error("Zotero database is locked. Please close Zotero and try again.")
            raise
        finally:
            if conn:
                conn.close()

    def _build_collection_cache(self, conn: sqlite3.Connection) -> dict[int, Collection]:
        """Pre-fetch all collections for hierarchy resolution.

        Args:
            conn: Database connection.

        Returns:
            Dictionary mapping collection_id to Collection.
        """
        cursor = conn.execute(
            "SELECT collectionID, collectionName, parentCollectionID FROM collections"
        )

        collections: dict[int, Collection] = {}
        for row in cursor:
            collections[row["collectionID"]] = Collection(
                collection_id=row["collectionID"],
                name=row["collectionName"],
                parent_id=row["parentCollectionID"],
            )

        # Build parent paths
        for coll in collections.values():
            path = []
            current_id = coll.parent_id
            depth = 0
            while current_id is not None and depth < 10:
                if current_id in collections:
                    parent = collections[current_id]
                    path.insert(0, parent.name)
                    current_id = parent.parent_id
                else:
                    break
                depth += 1

            coll.parent_path = path

        return collections

    def _get_items_with_attachments(
        self,
        content_types: tuple[str, ...],
        item_types: tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        """Return parent items with selected attachment content types."""
        content_placeholders = ", ".join("?" for _ in content_types)
        params: list[Any] = [*content_types, *SKIP_ITEM_TYPES]
        item_type_clause = ""
        if item_types:
            item_type_clause = f" AND it.typeName IN ({', '.join('?' for _ in item_types)})"
            params.extend(item_types)

        query = f"""
        SELECT
            parent.itemID,
            parent.key,
            parent.dateAdded,
            parent.dateModified,
            it.typeName,
            att.itemID as attachmentID,
            ia.key as attachmentKey,
            att.path as attachmentPath,
            att.contentType as attachmentContentType
        FROM items parent
        JOIN itemTypes it ON parent.itemTypeID = it.itemTypeID
        JOIN itemAttachments att ON parent.itemID = att.parentItemID
        JOIN items ia ON att.itemID = ia.itemID
        WHERE att.contentType IN ({content_placeholders})
          AND it.typeName NOT IN (?, ?, ?)
          AND parent.itemID NOT IN (SELECT itemID FROM deletedItems)
          {item_type_clause}
        ORDER BY parent.itemID
        """

        with self._get_connection() as conn:
            cursor = conn.execute(query, tuple(params))
            results = []
            for row in cursor:
                results.append(
                    {
                        "item_id": row["itemID"],
                        "key": row["key"],
                        "date_added": row["dateAdded"],
                        "date_modified": row["dateModified"],
                        "item_type": row["typeName"],
                        "attachment_id": row["attachmentID"],
                        "attachment_key": row["attachmentKey"],
                        "attachment_path": row["attachmentPath"],
                        "attachment_content_type": row["attachmentContentType"],
                    }
                )
            deduped, skipped = self._dedupe_duplicate_pdf_rows(results)
            if skipped:
                logger.info(
                    "Skipped %d duplicate attachment rows with identical parent/path pairs",
                    skipped,
                )
            return deduped

    def get_all_items_with_pdfs(self) -> list[dict[str, Any]]:
        """Get all library items that have PDF attachments.

        Returns:
            List of dicts with item info and attachment details.
        """
        return self._get_items_with_attachments(("application/pdf",))

    def get_all_items_with_webpage_snapshots(self) -> list[dict[str, Any]]:
        """Get webpage items backed by local HTML attachments."""
        if not self.include_webpage_snapshots:
            return []
        return self._get_items_with_attachments(("text/html",), item_types=("webpage",))

    def get_all_source_items(self) -> list[dict[str, Any]]:
        """Return the combined set of extractable Zotero source items."""
        pdf_items = self.get_all_items_with_pdfs()
        pdf_item_ids = {int(item["item_id"]) for item in pdf_items}
        source_items = list(pdf_items)
        for item in self.get_all_items_with_webpage_snapshots():
            if int(item["item_id"]) in pdf_item_ids:
                continue
            source_items.append(item)
        return source_items

    @staticmethod
    def _dedupe_duplicate_pdf_rows(
        rows: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int]:
        """Deduplicate exact duplicate PDF rows from Zotero attachment joins.

        Zotero can accumulate multiple attachment records that point to the same
        stored PDF under a single parent item. LITRIS treats each attachment as a
        separate paper when the parent/path pair is unique, but repeated rows for
        the exact same parent PDF should collapse to one source record.

        Args:
            rows: Attachment-backed parent rows from ``get_all_items_with_pdfs``.

        Returns:
            Tuple of ``(deduped_rows, skipped_count)``.
        """
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[int, str | None]] = set()
        skipped = 0

        for row in rows:
            identity = (int(row["item_id"]), row.get("attachment_path"))
            if identity in seen:
                skipped += 1
                continue
            seen.add(identity)
            deduped.append(row)

        return deduped, skipped

    def get_item_metadata(self, conn: sqlite3.Connection, item_id: int) -> dict[str, str]:
        """Get metadata fields for an item.

        Args:
            conn: Database connection.
            item_id: Zotero item ID.

        Returns:
            Dictionary of field_name -> value.
        """
        query = """
        SELECT f.fieldName, idv.value
        FROM itemData id
        JOIN fields f ON id.fieldID = f.fieldID
        JOIN itemDataValues idv ON id.valueID = idv.valueID
        WHERE id.itemID = ?
        """

        cursor = conn.execute(query, (item_id,))
        return {row["fieldName"]: row["value"] for row in cursor}

    def get_item_authors(self, conn: sqlite3.Connection, item_id: int) -> list[Author]:
        """Get authors for an item.

        Args:
            conn: Database connection.
            item_id: Zotero item ID.

        Returns:
            List of Author objects in order.
        """
        query = """
        SELECT
            c.firstName,
            c.lastName,
            c.fieldMode,
            ic.orderIndex,
            ct.creatorType
        FROM itemCreators ic
        JOIN creators c ON ic.creatorID = c.creatorID
        JOIN creatorTypes ct ON ic.creatorTypeID = ct.creatorTypeID
        WHERE ic.itemID = ?
        ORDER BY ic.orderIndex
        """

        cursor = conn.execute(query, (item_id,))
        authors = []
        for row in cursor:
            # fieldMode: 0 = two-field name, 1 = single-field name (in lastName)
            first_name = row["firstName"] or ""
            last_name = row["lastName"] or ""

            if row["fieldMode"] == 1:
                # Single-field name stored in lastName
                first_name = ""

            authors.append(
                Author(
                    first_name=first_name,
                    last_name=last_name,
                    order=row["orderIndex"] + 1,  # Convert to 1-based
                    role=row["creatorType"],
                )
            )

        return authors

    def get_item_collections(self, conn: sqlite3.Connection, item_id: int) -> list[str]:
        """Get collection names for an item.

        Args:
            conn: Database connection.
            item_id: Zotero item ID.

        Returns:
            List of collection full paths.
        """
        if self._collection_cache is None:
            self._collection_cache = self._build_collection_cache(conn)

        query = """
        SELECT ci.collectionID
        FROM collectionItems ci
        WHERE ci.itemID = ?
        """

        cursor = conn.execute(query, (item_id,))
        collections = []
        for row in cursor:
            coll_id = row["collectionID"]
            if coll_id in self._collection_cache:
                collections.append(self._collection_cache[coll_id].full_path)

        return collections

    def get_item_tags(self, conn: sqlite3.Connection, item_id: int) -> list[str]:
        """Get tags for an item.

        Args:
            conn: Database connection.
            item_id: Zotero item ID.

        Returns:
            List of tag names.
        """
        query = """
        SELECT t.name
        FROM itemTags it
        JOIN tags t ON it.tagID = t.tagID
        WHERE it.itemID = ?
        """

        cursor = conn.execute(query, (item_id,))
        return [row["name"] for row in cursor]

    def resolve_attachment_path(
        self, attachment_key: str, attachment_path: str | None
    ) -> Path | None:
        """Resolve the full path to a stored attachment file.

        Args:
            attachment_key: Zotero key for the attachment.
            attachment_path: Path string from database.

        Returns:
            Full Path to a local attachment or None if not found.

        Note:
            Includes path traversal protection to prevent accessing files
            outside the Zotero storage directory for storage: paths.
        """
        if not attachment_path:
            return None

        # Skip URL attachments
        if attachment_path.startswith("http"):
            logger.debug(f"Skipping URL attachment: {attachment_path[:50]}...")
            return None

        # Handle stored files (storage:{filename})
        if attachment_path.startswith("storage:"):
            filename = attachment_path[8:]  # Remove "storage:" prefix

            # Security: Validate filename doesn't contain path traversal patterns
            # Note: ".." alone is too aggressive - catches valid filenames like "paper..pdf"
            if (
                "../" in filename
                or "..\\" in filename
                or filename == ".."
                or filename.startswith("/")
                or filename.startswith("\\")
            ):
                logger.warning(f"Potential path traversal detected in attachment path: {filename}")
                return None

            file_path = self.storage_path / attachment_key / filename

            # Security: Verify resolved path is within storage directory
            try:
                resolved_path = file_path.resolve()
                storage_resolved = self.storage_path.resolve()
                if not str(resolved_path).startswith(str(storage_resolved)):
                    logger.warning(
                        f"Path traversal attempt blocked: {file_path} resolves outside storage"
                    )
                    return None
            except (OSError, ValueError) as e:
                logger.warning(f"Failed to resolve path {file_path}: {e}")
                return None

            if file_path.exists():
                return file_path
            logger.debug(f"Attachment not found at expected path: {file_path}")
            return None

        # Handle linked files (full path)
        # Note: Linked files are intentionally allowed outside storage
        # as Zotero supports linking to files anywhere on the filesystem
        linked_path = Path(attachment_path)
        if linked_path.exists():
            return linked_path

        logger.debug(f"Linked attachment not found: {attachment_path}")
        return None

    def resolve_pdf_path(self, attachment_key: str, attachment_path: str | None) -> Path | None:
        """Resolve the full path to a PDF file."""
        return self.resolve_attachment_path(attachment_key, attachment_path)

    def get_paper(self, conn: sqlite3.Connection, item_info: dict[str, Any]) -> PaperMetadata:
        """Assemble complete paper metadata.

        Args:
            conn: Database connection.
            item_info: Dict from get_all_items_with_pdfs.

        Returns:
            Complete PaperMetadata object.
        """
        item_id = item_info["item_id"]

        # Get all metadata
        metadata = self.get_item_metadata(conn, item_id)
        authors = self.get_item_authors(conn, item_id)
        collections = self.get_item_collections(conn, item_id)
        tags = self.get_item_tags(conn, item_id)

        # Resolve local source path
        source_path = self.resolve_attachment_path(
            item_info["attachment_key"], item_info["attachment_path"]
        )
        attachment_content_type = item_info.get("attachment_content_type")
        pdf_path = source_path if attachment_content_type == "application/pdf" else None
        pdf_attachment_key = (
            item_info["attachment_key"] if attachment_content_type == "application/pdf" else None
        )

        # Map Zotero fields to model fields
        model_data = {
            "zotero_key": item_info["key"],
            "zotero_item_id": item_id,
            "item_type": item_info["item_type"],
            "authors": authors,
            "collections": collections,
            "tags": tags,
            "pdf_path": pdf_path,
            "pdf_attachment_key": pdf_attachment_key,
            "source_path": source_path,
            "source_attachment_key": item_info["attachment_key"],
            "source_media_type": attachment_content_type,
            "date_added": datetime.fromisoformat(item_info["date_added"].replace(" ", "T")),
            "date_modified": datetime.fromisoformat(item_info["date_modified"].replace(" ", "T")),
        }

        # Map metadata fields
        for zotero_field, model_field in FIELD_MAPPING.items():
            if zotero_field in metadata:
                model_data[model_field] = metadata[zotero_field]

        return PaperMetadata(**model_data)

    def get_all_papers(
        self, progress_callback: Callable | None = None
    ) -> Generator[PaperMetadata, None, None]:
        """Get all locally extractable papers.

        Args:
            progress_callback: Optional callback(current, total) for progress.

        Yields:
            PaperMetadata objects for each paper.
        """
        items = self.get_all_source_items()
        total = len(items)
        logger.info(f"Found {total} items with local extractable sources")

        with self._get_connection() as conn:
            # Pre-build collection cache
            self._collection_cache = self._build_collection_cache(conn)

            for i, item_info in enumerate(items):
                try:
                    paper = self.get_paper(conn, item_info)
                    if progress_callback:
                        progress_callback(i + 1, total)
                    yield paper
                except (sqlite3.Error, KeyError, ValueError) as e:
                    # Expected errors: DB issues, missing fields, date parsing
                    logger.warning(
                        f"Failed to process item {item_info.get('item_id', 'unknown')}: "
                        f"{type(e).__name__}: {e}"
                    )
                    continue
                except ValidationError as e:
                    # Pydantic validation failed - data doesn't match schema
                    logger.warning(
                        f"Invalid data for item {item_info.get('item_id', 'unknown')}: {e}"
                    )
                    continue

    def get_paper_count(self) -> int:
        """Get total count of papers with local extractable sources.

        Returns:
            Number of papers with local source attachments.
        """
        return len(self.get_all_source_items())

    def get_paper_by_key(self, zotero_key: str) -> PaperMetadata | None:
        """Get a specific paper by Zotero key.

        Args:
            zotero_key: 8-character Zotero key.

        Returns:
            PaperMetadata or None if not found.
        """
        items = self.get_all_source_items()
        matching = [i for i in items if i["key"] == zotero_key]

        if not matching:
            return None

        with self._get_connection() as conn:
            self._collection_cache = self._build_collection_cache(conn)
            return self.get_paper(conn, matching[0])

    def load_source_text_snapshots(
        self,
        papers: list[PaperMetadata],
    ) -> dict[str, dict[str, object]]:
        """Build canonical text snapshots for HTML attachment-backed papers."""
        snapshots: dict[str, dict[str, object]] = {}
        for paper in papers:
            if paper.source_media_type != "text/html" or paper.source_path is None:
                continue
            source_path = Path(paper.source_path)
            if not source_path.exists():
                continue
            try:
                result = extract_html_attachment(source_path, url=paper.url)
            except Exception as exc:
                logger.warning(
                    "Failed to extract HTML attachment for %s (%s): %s",
                    paper.paper_id,
                    source_path,
                    exc,
                )
                continue

            stat = source_path.stat()
            snapshots[paper.paper_id] = {
                "paper_id": paper.paper_id,
                "text": result.text,
                "sha256": hashlib.sha256(result.text.encode("utf-8")).hexdigest(),
                "char_count": len(result.text),
                "word_count": result.word_count,
                "captured_at": datetime.now().isoformat(),
                "source": "html_attachment",
                "extraction_method": result.method,
                "tiers_attempted": [result.method],
                "is_markdown": False,
                "is_cleaned": True,
                "is_truncated_for_llm": False,
                "source_path": str(source_path),
                "source_size": int(stat.st_size),
                "source_mtime_ns": int(stat.st_mtime_ns),
                "source_media_type": paper.source_media_type,
                "source_attachment_key": paper.source_attachment_key,
                "pdf_path": None,
                "pdf_size": None,
                "pdf_mtime_ns": None,
                "should_persist": True,
            }
        return snapshots
