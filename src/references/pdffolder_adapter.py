"""PDF folder reference database adapter.

Provides a reference database interface for folders containing PDF files.
Extracts metadata from PDF properties and filename parsing.
"""

import hashlib
import re
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Callable

from src.references.base import BaseReferenceDB, ReferenceProvider
from src.utils.logging_config import get_logger
from src.zotero.models import Author, PaperMetadata

logger = get_logger(__name__)


class PDFFolderReferenceDB(BaseReferenceDB):
    """PDF folder reference database.

    Scans a folder for PDF files and extracts metadata from:
    1. PDF document properties (title, author, creation date)
    2. Filename parsing (author names, year, title)

    This enables LITRIS to work with papers stored as plain PDF files
    without requiring any reference management software.
    """

    # Common filename patterns for academic papers
    # Pattern: Author(s) - Year - Title.pdf
    # Pattern: Author_Year_Title.pdf
    # Pattern: Year_Author_Title.pdf
    FILENAME_PATTERNS = [
        # Author(s) - Year - Title
        re.compile(r"^(?P<authors>[^-]+)\s*-\s*(?P<year>\d{4})\s*-\s*(?P<title>.+)\.pdf$", re.IGNORECASE),
        # Author_Year_Title
        re.compile(r"^(?P<authors>[^_]+)_(?P<year>\d{4})_(?P<title>.+)\.pdf$", re.IGNORECASE),
        # Year_Author_Title
        re.compile(r"^(?P<year>\d{4})_(?P<authors>[^_]+)_(?P<title>.+)\.pdf$", re.IGNORECASE),
        # Year - Author - Title
        re.compile(r"^(?P<year>\d{4})\s*-\s*(?P<authors>[^-]+)\s*-\s*(?P<title>.+)\.pdf$", re.IGNORECASE),
        # Just title (fallback)
        re.compile(r"^(?P<title>.+)\.pdf$", re.IGNORECASE),
    ]

    def __init__(
        self,
        folder_path: Path,
        recursive: bool = True,
        extract_pdf_metadata: bool = True,
    ):
        """Initialize PDF folder reference database.

        Args:
            folder_path: Path to folder containing PDFs.
            recursive: If True, scan subfolders recursively.
            extract_pdf_metadata: If True, extract metadata from PDF properties.
        """
        self.folder_path = Path(folder_path)
        self.recursive = recursive
        self.extract_pdf_metadata = extract_pdf_metadata
        self._pdf_files: list[Path] | None = None
        self._scan_timestamp = datetime.now()

    @property
    def provider(self) -> ReferenceProvider:
        """Return the provider identifier."""
        return "pdffolder"

    @property
    def source_path(self) -> Path:
        """Return the folder path."""
        return self.folder_path

    def _scan_folder(self) -> list[Path]:
        """Scan folder for PDF files.

        Returns:
            List of PDF file paths.
        """
        if self._pdf_files is not None:
            return self._pdf_files

        if not self.folder_path.exists():
            raise FileNotFoundError(f"PDF folder not found: {self.folder_path}")

        if not self.folder_path.is_dir():
            raise ValueError(f"Path is not a directory: {self.folder_path}")

        pattern = "**/*.pdf" if self.recursive else "*.pdf"
        pdf_files = sorted(self.folder_path.glob(pattern))

        self._pdf_files = pdf_files
        logger.info(f"Found {len(pdf_files)} PDF files in {self.folder_path}")
        return pdf_files

    def _parse_filename(self, pdf_path: Path) -> dict[str, str | None]:
        """Parse metadata from filename.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            Dictionary with parsed metadata (authors, year, title).
        """
        filename = pdf_path.name
        result: dict[str, str | None] = {
            "authors": None,
            "year": None,
            "title": None,
        }

        for pattern in self.FILENAME_PATTERNS:
            match = pattern.match(filename)
            if match:
                groups = match.groupdict()
                # Strip whitespace from captured groups
                authors = groups.get("authors")
                result["authors"] = authors.strip() if authors else None
                result["year"] = groups.get("year")
                title = groups.get("title")
                result["title"] = title.strip() if title else None
                break

        # Clean up title - remove underscores, extra spaces
        if result["title"]:
            result["title"] = result["title"].replace("_", " ").strip()
            result["title"] = " ".join(result["title"].split())

        return result

    def _extract_pdf_metadata(self, pdf_path: Path) -> dict[str, str | None]:
        """Extract metadata from PDF document properties.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            Dictionary with PDF metadata.
        """
        result: dict[str, str | None] = {
            "title": None,
            "author": None,
            "creation_date": None,
            "subject": None,
            "keywords": None,
        }

        if not self.extract_pdf_metadata:
            return result

        try:
            import pymupdf

            with pymupdf.open(pdf_path) as doc:
                metadata = doc.metadata
                if metadata:
                    result["title"] = metadata.get("title") or None
                    result["author"] = metadata.get("author") or None
                    result["creation_date"] = metadata.get("creationDate") or None
                    result["subject"] = metadata.get("subject") or None
                    result["keywords"] = metadata.get("keywords") or None

        except Exception as e:
            logger.debug(f"Failed to extract PDF metadata from {pdf_path}: {e}")

        return result

    def _parse_authors(self, author_string: str | None) -> list[Author]:
        """Parse author string into Author objects.

        Handles various formats:
        - "Smith, John"
        - "John Smith"
        - "Smith, J. and Doe, J."
        - "Smith et al"
        - "Smith; Doe; Brown"

        Args:
            author_string: Author field value.

        Returns:
            List of Author objects.
        """
        if not author_string:
            return []

        authors = []

        # Remove "et al" variants
        author_string = re.sub(r"\s+et\s+al\.?", "", author_string, flags=re.IGNORECASE)

        # Split by common separators
        if " and " in author_string.lower():
            parts = re.split(r"\s+and\s+", author_string, flags=re.IGNORECASE)
        elif ";" in author_string:
            parts = author_string.split(";")
        elif "&" in author_string:
            parts = author_string.split("&")
        else:
            parts = [author_string]

        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue

            first_name = ""
            last_name = ""

            if "," in part:
                # Format: "Last, First"
                comma_parts = part.split(",", 1)
                last_name = comma_parts[0].strip()
                first_name = comma_parts[1].strip() if len(comma_parts) > 1 else ""
            else:
                # Format: "First Last" or just "Last"
                name_parts = part.split()
                if len(name_parts) >= 2:
                    last_name = name_parts[-1]
                    first_name = " ".join(name_parts[:-1])
                else:
                    last_name = part

            authors.append(Author(
                first_name=first_name,
                last_name=last_name,
                order=i + 1,
                role="author",
            ))

        return authors

    def _parse_year(self, date_string: str | None) -> int | None:
        """Extract year from various date formats.

        Args:
            date_string: Date string (could be PDF creation date or year).

        Returns:
            Year as integer or None.
        """
        if not date_string:
            return None

        # Try direct year
        if date_string.isdigit() and len(date_string) == 4:
            return int(date_string)

        # Try to find 4-digit year in string
        year_match = re.search(r"(19|20)\d{2}", date_string)
        if year_match:
            return int(year_match.group())

        return None

    def _pdf_to_paper(self, pdf_path: Path) -> PaperMetadata:
        """Convert a PDF file to PaperMetadata.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            PaperMetadata object.
        """
        # Parse filename
        filename_data = self._parse_filename(pdf_path)

        # Extract PDF metadata
        pdf_data = self._extract_pdf_metadata(pdf_path)

        # Merge metadata (PDF metadata takes precedence if available)
        title = pdf_data["title"] or filename_data["title"] or pdf_path.stem
        author_string = pdf_data["author"] or filename_data["authors"]
        year_string = filename_data["year"] or pdf_data["creation_date"]

        # Parse authors
        authors = self._parse_authors(author_string)

        # Parse year
        year = self._parse_year(year_string)

        # Generate stable ID from file path
        relative_path = pdf_path.relative_to(self.folder_path) if pdf_path.is_relative_to(self.folder_path) else pdf_path
        id_source = str(relative_path)
        stable_hash = hashlib.sha256(id_source.encode()).hexdigest()[:8]

        # Create pseudo zotero_item_id from hash
        zotero_item_id = int(stable_hash, 16) % 2147483647

        # Use relative path as key (with forward slashes for consistency)
        citation_key = str(relative_path).replace("\\", "/").replace(".pdf", "")

        # Parse keywords from PDF metadata
        tags = []
        if pdf_data["keywords"]:
            tags = [k.strip() for k in re.split(r"[,;]", pdf_data["keywords"]) if k.strip()]

        # Determine collection from subfolder
        collections = []
        if pdf_path.parent != self.folder_path:
            # Use subfolder name as collection
            rel_parent = pdf_path.parent.relative_to(self.folder_path)
            collections = [str(rel_parent).replace("\\", "/")]

        return PaperMetadata(
            paper_id=f"pdf_{stable_hash}",
            zotero_key=citation_key,
            zotero_item_id=zotero_item_id,
            item_type="document",  # Generic type for PDFs
            title=title,
            authors=authors,
            publication_year=year,
            publication_date=str(year) if year else None,
            abstract=pdf_data.get("subject"),  # Some PDFs store abstract in subject
            collections=collections,
            tags=tags,
            pdf_path=pdf_path,
            pdf_attachment_key=f"pdf_{stable_hash}",
            date_added=self._scan_timestamp,
            date_modified=datetime.fromtimestamp(pdf_path.stat().st_mtime),
        )

    def get_all_papers(
        self, progress_callback: Callable[[int, int], None] | None = None
    ) -> Generator[PaperMetadata, None, None]:
        """Get all papers from the PDF folder.

        Args:
            progress_callback: Optional callback(current, total) for progress.

        Yields:
            PaperMetadata objects for each PDF.
        """
        pdf_files = self._scan_folder()
        total = len(pdf_files)

        for i, pdf_path in enumerate(pdf_files):
            try:
                paper = self._pdf_to_paper(pdf_path)
                if progress_callback:
                    progress_callback(i + 1, total)
                yield paper
            except Exception as e:
                logger.warning(f"Failed to process {pdf_path}: {e}")
                continue

    def get_paper_count(self) -> int:
        """Get total count of PDFs.

        Returns:
            Number of PDF files in the folder.
        """
        pdf_files = self._scan_folder()
        return len(pdf_files)

    def get_paper_by_key(self, key: str) -> PaperMetadata | None:
        """Get a specific paper by its key (relative path).

        Args:
            key: Relative path to PDF (without .pdf extension).

        Returns:
            PaperMetadata or None if not found.
        """
        # Normalize key
        key = key.replace("/", "\\") if "\\" in str(self.folder_path) else key.replace("\\", "/")

        # Try with .pdf extension
        pdf_path = self.folder_path / f"{key}.pdf"
        if pdf_path.exists():
            return self._pdf_to_paper(pdf_path)

        # Try without modification
        pdf_path = self.folder_path / key
        if pdf_path.exists() and pdf_path.suffix.lower() == ".pdf":
            return self._pdf_to_paper(pdf_path)

        return None

    def reload(self) -> None:
        """Reload the PDF folder.

        Call this to pick up new files.
        """
        self._pdf_files = None
        self._scan_timestamp = datetime.now()
        self._scan_folder()
