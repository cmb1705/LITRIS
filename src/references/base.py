"""Abstract base class for reference database interfaces."""

from abc import ABC, abstractmethod
from collections.abc import Generator
from pathlib import Path
from typing import Callable, Literal

from src.zotero.models import Author, PaperMetadata

ReferenceProvider = Literal["zotero", "bibtex", "pdffolder"]


class BaseReferenceDB(ABC):
    """Abstract base class for reference database interfaces.

    All reference manager implementations must inherit from this class
    and implement the required methods.

    The interface provides access to paper metadata and PDF files
    from various reference management systems.
    """

    @property
    @abstractmethod
    def provider(self) -> ReferenceProvider:
        """Return the provider identifier."""
        ...

    @property
    @abstractmethod
    def source_path(self) -> Path:
        """Return the primary source path (database file or directory)."""
        ...

    @abstractmethod
    def get_all_papers(
        self, progress_callback: Callable[[int, int], None] | None = None
    ) -> Generator[PaperMetadata, None, None]:
        """Get all papers from the reference database.

        Args:
            progress_callback: Optional callback(current, total) for progress.

        Yields:
            PaperMetadata objects for each paper.
        """
        ...

    @abstractmethod
    def get_paper_count(self) -> int:
        """Get total count of papers.

        Returns:
            Number of papers in the database.
        """
        ...

    @abstractmethod
    def get_paper_by_key(self, key: str) -> PaperMetadata | None:
        """Get a specific paper by its unique key.

        Args:
            key: Provider-specific unique identifier.

        Returns:
            PaperMetadata or None if not found.
        """
        ...

    def iterate_papers(
        self,
        limit: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Generator[PaperMetadata, None, None]:
        """Iterate over papers with optional limit.

        Args:
            limit: Maximum number of papers to return.
            progress_callback: Optional callback(current, total) for progress.

        Yields:
            PaperMetadata objects.
        """
        count = 0
        for paper in self.get_all_papers(progress_callback):
            if limit and count >= limit:
                return
            yield paper
            count += 1

    def filter_papers(
        self,
        collections: list[str] | None = None,
        tags: list[str] | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        has_pdf: bool = True,
    ) -> Generator[PaperMetadata, None, None]:
        """Filter papers by criteria.

        Args:
            collections: Only include papers in these collections.
            tags: Only include papers with these tags.
            year_min: Minimum publication year.
            year_max: Maximum publication year.
            has_pdf: Only include papers with PDF attachments.

        Yields:
            PaperMetadata objects matching criteria.
        """
        for paper in self.get_all_papers():
            # Filter by PDF availability
            if has_pdf and not paper.pdf_path:
                continue

            # Filter by collections
            if collections:
                paper_collections = set(paper.collections)
                if not paper_collections.intersection(collections):
                    continue

            # Filter by tags
            if tags:
                paper_tags = set(paper.tags)
                if not paper_tags.intersection(tags):
                    continue

            # Filter by year
            year = paper.publication_year
            if year_min and (year is None or year < year_min):
                continue
            if year_max and (year is None or year > year_max):
                continue

            yield paper

    @staticmethod
    def get_available_providers() -> list[str]:
        """Return list of available reference providers."""
        return ["zotero", "bibtex", "pdffolder"]

    @staticmethod
    def create_author(
        first_name: str = "",
        last_name: str = "",
        order: int = 1,
        role: str = "author",
    ) -> Author:
        """Create an Author object.

        Helper method for implementations to create Author objects.

        Args:
            first_name: Author's first name.
            last_name: Author's last name.
            order: Author order (1-based).
            role: Author role (author, editor, etc.).

        Returns:
            Author object.
        """
        return Author(
            first_name=first_name,
            last_name=last_name,
            order=order,
            role=role,
        )
