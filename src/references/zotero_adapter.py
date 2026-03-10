"""Zotero reference database adapter.

This adapter wraps the existing ZoteroDatabase implementation
to conform to the BaseReferenceDB interface.
"""

from collections.abc import Callable, Generator
from pathlib import Path

from src.references.base import BaseReferenceDB, ReferenceProvider
from src.zotero.database import ZoteroDatabase
from src.zotero.models import PaperMetadata


class ZoteroReferenceDB(BaseReferenceDB):
    """Zotero reference database adapter.

    Wraps the ZoteroDatabase class to provide the standard
    BaseReferenceDB interface.
    """

    def __init__(self, db_path: Path, storage_path: Path):
        """Initialize Zotero reference database.

        Args:
            db_path: Path to zotero.sqlite file.
            storage_path: Path to Zotero storage directory containing PDFs.
        """
        self.db_path = db_path
        self.storage_path = storage_path
        self._db = ZoteroDatabase(db_path, storage_path)

    @property
    def provider(self) -> ReferenceProvider:
        """Return the provider identifier."""
        return "zotero"

    @property
    def source_path(self) -> Path:
        """Return the database path."""
        return self.db_path

    def get_all_papers(
        self, progress_callback: Callable[[int, int], None] | None = None
    ) -> Generator[PaperMetadata, None, None]:
        """Get all papers from Zotero.

        Args:
            progress_callback: Optional callback(current, total) for progress.

        Yields:
            PaperMetadata objects for each paper.
        """
        yield from self._db.get_all_papers(progress_callback)

    def get_paper_count(self) -> int:
        """Get total count of papers with PDFs.

        Returns:
            Number of papers with PDF attachments.
        """
        return self._db.get_paper_count()

    def get_paper_by_key(self, key: str) -> PaperMetadata | None:
        """Get a specific paper by Zotero key.

        Args:
            key: 8-character Zotero key.

        Returns:
            PaperMetadata or None if not found.
        """
        return self._db.get_paper_by_key(key)

    def get_collections(self) -> list[str]:
        """Get all collection names.

        Returns:
            List of collection full paths.
        """
        collections = set()
        for paper in self.get_all_papers():
            collections.update(paper.collections)
        return sorted(collections)
