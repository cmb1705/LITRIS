"""Zotero write-back with dual API/SQLite backends.

Adds papers to a Zotero library from PDF folders or enriched metadata.
Uses pyzotero Web API when available, falls back to direct SQLite writes.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from src.utils.logging_config import get_logger
from src.zotero.metadata_enricher import EnrichedMetadata
from src.zotero.orphan_metadata_extractor import ExtractedMetadata, MetadataSource

if TYPE_CHECKING:
    from src.config import ZoteroConfig

logger = get_logger(__name__)


@dataclass
class WriteResult:
    """Result of a Zotero write operation."""

    success: bool
    item_key: str | None = None
    title: str = ""
    item_type: str = ""
    backend: str = ""
    error: str | None = None


@dataclass
class PaperWriteRequest:
    """Request to write a paper to Zotero."""

    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    isbn: str | None = None
    journal: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    abstract: str | None = None
    url: str | None = None
    issn: str | None = None
    item_type: str = "journalArticle"
    pdf_path: Path | None = None
    collection_name: str | None = None
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_enriched(cls, enriched: EnrichedMetadata) -> PaperWriteRequest:
        """Create a write request from enriched metadata.

        Args:
            enriched: Enriched metadata from the extraction pipeline.

        Returns:
            PaperWriteRequest ready to write.
        """
        # Determine item type
        item_type = "journalArticle"
        if enriched.isbn or enriched.original.isbn:
            item_type = "book"
        elif enriched.journal or enriched.issn:
            item_type = "journalArticle"

        return cls(
            title=enriched.best_title or "Untitled",
            authors=enriched.best_authors,
            year=enriched.best_year,
            doi=enriched.best_doi,
            isbn=enriched.isbn or enriched.original.isbn,
            journal=enriched.journal,
            volume=enriched.volume,
            issue=enriched.issue,
            pages=enriched.pages,
            abstract=enriched.abstract,
            url=enriched.url,
            issn=enriched.issn,
            item_type=item_type,
            pdf_path=enriched.original.pdf_path,
            tags=["litris-imported"],
        )

    @classmethod
    def from_extracted(cls, extracted: ExtractedMetadata) -> PaperWriteRequest:
        """Create a write request from basic extracted metadata.

        Args:
            extracted: Metadata extracted from PDF.

        Returns:
            PaperWriteRequest ready to write.
        """
        item_type = "journalArticle"
        if extracted.isbn:
            item_type = "book"

        return cls(
            title=extracted.title or "Untitled",
            authors=extracted.authors,
            year=extracted.publication_year,
            doi=extracted.doi,
            isbn=extracted.isbn,
            journal=extracted.journal,
            volume=extracted.volume,
            issue=extracted.issue,
            pages=extracted.pages,
            abstract=extracted.abstract,
            item_type=item_type,
            pdf_path=extracted.pdf_path,
            tags=["litris-imported"],
        )


class ZoteroBackend(ABC):
    """Abstract backend for Zotero writes."""

    @abstractmethod
    def write_item(self, request: PaperWriteRequest) -> WriteResult:
        """Write a single item to Zotero.

        Args:
            request: Paper write request.

        Returns:
            WriteResult with success/failure info.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is currently available."""


class PyzoteroBackend(ZoteroBackend):
    """Write to Zotero via pyzotero Web API.

    Requires internet access and Zotero API key.
    """

    def __init__(self, user_id: str, api_key: str):
        """Initialize pyzotero backend.

        Args:
            user_id: Zotero user ID.
            api_key: Zotero API key.
        """
        self.user_id = user_id
        self.api_key = api_key
        self._zot = None

    def _get_client(self):
        """Lazy-load pyzotero client."""
        if self._zot is None:
            try:
                from pyzotero import zotero
                self._zot = zotero.Zotero(self.user_id, "user", self.api_key)
            except ImportError as err:
                raise RuntimeError(
                    "pyzotero is not installed. "
                    "Install with: pip install pyzotero"
                ) from err
        return self._zot

    def is_available(self) -> bool:
        """Check if pyzotero is installed and credentials are set."""
        if not self.user_id or not self.api_key:
            return False
        try:
            from pyzotero import zotero  # noqa: F401
            return True
        except ImportError:
            return False

    def write_item(self, request: PaperWriteRequest) -> WriteResult:
        """Write item via pyzotero Web API."""
        try:
            zot = self._get_client()

            # Create item template
            template = zot.item_template(request.item_type)
            template["title"] = request.title

            # Add creators
            template["creators"] = []
            for author in request.authors:
                first_name, last_name = _parse_author_name(author)
                template["creators"].append({
                    "creatorType": "author",
                    "firstName": first_name,
                    "lastName": last_name,
                })

            # Add metadata fields
            if request.doi:
                template["DOI"] = request.doi
            if request.year:
                template["date"] = str(request.year)
            if request.journal and request.item_type == "journalArticle":
                template["publicationTitle"] = request.journal
            if request.volume:
                template["volume"] = request.volume
            if request.issue:
                template["issue"] = request.issue
            if request.pages:
                template["pages"] = request.pages
            if request.abstract:
                template["abstractNote"] = request.abstract[:10000]
            if request.url:
                template["url"] = request.url
            if request.isbn:
                template["ISBN"] = request.isbn
            if request.issn:
                template["ISSN"] = request.issn

            # Add tags
            template["tags"] = [{"tag": t} for t in request.tags]

            # Create the item
            resp = zot.create_items([template])

            if "successful" in resp and resp["successful"]:
                item_key = resp["successful"]["0"]["key"]

                # Attach PDF if available
                if request.pdf_path and request.pdf_path.exists():
                    try:
                        zot.attachment_simple(
                            [str(request.pdf_path)],
                            item_key,
                        )
                    except Exception as e:
                        logger.warning(
                            f"Item created but PDF attachment failed: {e}"
                        )

                # Add to collection if specified
                if request.collection_name:
                    self._add_to_collection(zot, item_key, request.collection_name)

                logger.info(
                    f"Created Zotero item via API: {item_key} - {request.title}"
                )
                return WriteResult(
                    success=True,
                    item_key=item_key,
                    title=request.title,
                    item_type=request.item_type,
                    backend="pyzotero",
                )

            # Handle failure
            error_msg = str(resp.get("failed", resp))
            return WriteResult(
                success=False,
                title=request.title,
                item_type=request.item_type,
                backend="pyzotero",
                error=error_msg,
            )

        except Exception as e:
            return WriteResult(
                success=False,
                title=request.title,
                item_type=request.item_type,
                backend="pyzotero",
                error=str(e),
            )

    def _add_to_collection(self, zot, item_key: str, collection_name: str) -> None:
        """Add an item to a collection by name, creating if needed."""
        try:
            collections = zot.collections()
            target = None
            for coll in collections:
                if coll["data"]["name"] == collection_name:
                    target = coll["key"]
                    break

            if target is None:
                # Create collection
                resp = zot.create_collections([{"name": collection_name}])
                if "successful" in resp and resp["successful"]:
                    target = resp["successful"]["0"]["key"]

            if target:
                zot.addto_collection(target, zot.item(item_key))
        except Exception as e:
            logger.warning(f"Failed to add to collection '{collection_name}': {e}")


class SqliteBackend(ZoteroBackend):
    """Write to Zotero via direct SQLite access.

    Uses the existing ParentItemCreator. Requires Zotero to be closed.
    """

    def __init__(self, db_path: Path, storage_path: Path, dry_run: bool = False):
        """Initialize SQLite backend.

        Args:
            db_path: Path to Zotero SQLite database.
            storage_path: Path to Zotero storage directory.
            dry_run: If True, preview changes without committing.
        """
        self.db_path = db_path
        self.storage_path = storage_path
        self.dry_run = dry_run
        self._creator = None

    def _get_creator(self):
        """Lazy-load ParentItemCreator."""
        if self._creator is None:
            from src.zotero.parent_item_creator import ParentItemCreator
            self._creator = ParentItemCreator(self.db_path, dry_run=self.dry_run)
        return self._creator

    def is_available(self) -> bool:
        """Check if Zotero database exists."""
        return self.db_path.exists()

    def write_item(self, request: PaperWriteRequest) -> WriteResult:
        """Write item via direct SQLite."""
        try:
            # Build an EnrichedMetadata for the ParentItemCreator
            extracted = ExtractedMetadata(
                doi=request.doi,
                isbn=request.isbn,
                title=request.title,
                authors=request.authors,
                publication_year=request.year,
                journal=request.journal,
                volume=request.volume,
                issue=request.issue,
                pages=request.pages,
                abstract=request.abstract,
                source=MetadataSource.UNKNOWN,
                confidence=0.8,
                pdf_path=request.pdf_path,
            )

            enriched = EnrichedMetadata(
                original=extracted,
                doi=request.doi,
                title=request.title,
                authors=request.authors,
                publication_year=request.year,
                journal=request.journal,
                volume=request.volume,
                issue=request.issue,
                pages=request.pages,
                abstract=request.abstract,
                url=request.url,
                issn=request.issn,
                isbn=request.isbn,
            )

            # Copy PDF to Zotero storage if provided
            if request.pdf_path and request.pdf_path.exists():
                self._copy_pdf_to_storage(request.pdf_path)

            creator = self._get_creator()

            # The ParentItemCreator needs an attachment_item_id.
            # For new PDFs (not orphans), we need to create the attachment first.
            # For now, create just the parent item metadata.
            # Full attachment linking requires the PDF to already be in Zotero storage.
            if extracted.attachment_item_id is not None:
                result = creator.create_parent_item(
                    enriched,
                    tag=request.tags[0] if request.tags else "litris-imported",
                )

                if result.success:
                    logger.info(
                        f"Created Zotero item via SQLite: {result.parent_key} - {request.title}"
                    )
                    return WriteResult(
                        success=True,
                        item_key=result.parent_key,
                        title=request.title,
                        item_type=request.item_type,
                        backend="sqlite",
                    )
                else:
                    return WriteResult(
                        success=False,
                        title=request.title,
                        item_type=request.item_type,
                        backend="sqlite",
                        error=result.error,
                    )
            else:
                # No attachment_item_id -- this is a fresh PDF, not an orphan.
                # Create a standalone item without attachment linking.
                return self._create_standalone_item(enriched, request)

        except Exception as e:
            return WriteResult(
                success=False,
                title=request.title,
                item_type=request.item_type,
                backend="sqlite",
                error=str(e),
            )

    def _create_standalone_item(
        self, enriched: EnrichedMetadata, request: PaperWriteRequest
    ) -> WriteResult:
        """Create a standalone Zotero item without an existing attachment."""
        creator = self._get_creator()
        conn = creator._get_connection()

        try:
            key = creator._generate_key()
            item_type = creator._determine_item_type(enriched)
            item_type_id = creator._get_item_type_id(item_type)

            # Create item
            item_id = creator._insert_item(conn, item_type_id, key)

            # Add metadata
            creator._insert_field(conn, item_id, "title", request.title)
            if request.doi:
                creator._insert_field(conn, item_id, "DOI", request.doi)
            if request.year:
                creator._insert_field(conn, item_id, "date", str(request.year))
            if request.journal:
                creator._insert_field(conn, item_id, "publicationTitle", request.journal)
            if request.volume:
                creator._insert_field(conn, item_id, "volume", request.volume)
            if request.issue:
                creator._insert_field(conn, item_id, "issue", request.issue)
            if request.pages:
                creator._insert_field(conn, item_id, "pages", request.pages)
            if request.abstract:
                creator._insert_field(conn, item_id, "abstractNote", request.abstract[:10000])
            if request.url:
                creator._insert_field(conn, item_id, "url", request.url)
            if request.isbn:
                creator._insert_field(conn, item_id, "ISBN", request.isbn)
            if request.issn:
                creator._insert_field(conn, item_id, "ISSN", request.issn)

            # Add authors
            for i, author in enumerate(request.authors[:20]):
                creator._insert_creator(conn, item_id, author, i)

            # Add tags
            for tag in request.tags:
                creator._add_tag(conn, item_id, tag)

            if not creator.dry_run:
                conn.commit()

            logger.info(f"Created standalone Zotero item via SQLite: {key} - {request.title}")
            return WriteResult(
                success=True,
                item_key=key,
                title=request.title,
                item_type=item_type,
                backend="sqlite",
            )

        except Exception as e:
            if not creator.dry_run:
                conn.rollback()
            return WriteResult(
                success=False,
                title=request.title,
                item_type=request.item_type,
                backend="sqlite",
                error=str(e),
            )

    def _copy_pdf_to_storage(self, pdf_path: Path) -> Path | None:
        """Copy a PDF to Zotero's storage directory.

        Args:
            pdf_path: Source PDF path.

        Returns:
            Destination path or None if copy failed.
        """
        try:
            # Create a unique storage subdirectory (mimics Zotero's key-based dirs)
            import hashlib
            hash_val = hashlib.sha256(str(pdf_path).encode()).hexdigest()[:8].upper()
            dest_dir = self.storage_path / hash_val
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / pdf_path.name
            if not dest_path.exists():
                shutil.copy2(pdf_path, dest_path)
                logger.debug(f"Copied PDF to Zotero storage: {dest_path}")
            return dest_path
        except Exception as e:
            logger.warning(f"Failed to copy PDF to Zotero storage: {e}")
            return None

    def close(self) -> None:
        """Close the SQLite connection."""
        if self._creator:
            self._creator.close()
            self._creator = None


class ZoteroWriter:
    """Dual-backend Zotero writer.

    Automatically selects the best available backend (pyzotero API or SQLite)
    based on configuration and availability.
    """

    def __init__(self, config: ZoteroConfig):
        """Initialize the writer.

        Args:
            config: Zotero configuration with write_method, api_key, user_id.
        """
        self.config = config
        self._api_backend: PyzoteroBackend | None = None
        self._sqlite_backend: SqliteBackend | None = None

        # Initialize backends based on config
        if config.write_method in ("auto", "api"):
            if config.api_key and config.user_id:
                self._api_backend = PyzoteroBackend(config.user_id, config.api_key)

        if config.write_method in ("auto", "sqlite"):
            self._sqlite_backend = SqliteBackend(
                config.database_path, config.storage_path
            )

    def write_item(self, request: PaperWriteRequest) -> WriteResult:
        """Write a paper to Zotero using the best available backend.

        Args:
            request: Paper write request.

        Returns:
            WriteResult with success/failure info.
        """
        # Try API first if available
        if self._api_backend and self._api_backend.is_available():
            result = self._api_backend.write_item(request)
            if result.success:
                return result
            logger.warning(
                f"API write failed for '{request.title}': {result.error}. "
                "Falling back to SQLite."
            )

        # Fall back to SQLite
        if self._sqlite_backend and self._sqlite_backend.is_available():
            return self._sqlite_backend.write_item(request)

        return WriteResult(
            success=False,
            title=request.title,
            item_type=request.item_type,
            backend="none",
            error="No write backend available. Configure zotero.api_key + zotero.user_id "
                  "for API writes, or ensure zotero.database_path points to a valid database.",
        )

    def write_batch(
        self,
        requests: list[PaperWriteRequest],
    ) -> list[WriteResult]:
        """Write multiple papers to Zotero.

        Args:
            requests: List of paper write requests.

        Returns:
            List of WriteResult objects.
        """
        results = []
        for req in requests:
            result = self.write_item(req)
            results.append(result)

        succeeded = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)
        logger.info(
            f"Zotero write batch complete: {succeeded} succeeded, {failed} failed"
        )
        return results

    def close(self) -> None:
        """Clean up backend resources."""
        if self._sqlite_backend:
            self._sqlite_backend.close()


def _parse_author_name(name: str) -> tuple[str, str]:
    """Parse an author name string into (first_name, last_name).

    Args:
        name: Author name in various formats.

    Returns:
        Tuple of (first_name, last_name).
    """
    if "," in name:
        parts = name.split(",", 1)
        return parts[1].strip(), parts[0].strip()
    elif " " in name:
        parts = name.rsplit(" ", 1)
        return parts[0].strip(), parts[1].strip()
    return "", name
