"""Orphan PDF metadata extractor.

Extracts metadata from PDF attachments that lack parent items.
Uses multiple strategies: DOI/ISBN extraction, filename parsing,
PDF metadata, and LLM-based extraction.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pymupdf

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class MetadataSource(Enum):
    """Source of extracted metadata."""

    DOI_EXTRACTED = "doi_extracted"
    ISBN_EXTRACTED = "isbn_extracted"
    FILENAME_PARSED = "filename_parsed"
    PDF_METADATA = "pdf_metadata"
    LLM_EXTRACTED = "llm_extracted"
    UNKNOWN = "unknown"


@dataclass
class ExtractedMetadata:
    """Metadata extracted from an orphan PDF."""

    # Identifiers
    doi: str | None = None
    isbn: str | None = None

    # Core metadata
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    publication_year: int | None = None
    journal: str | None = None

    # Additional metadata
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    publisher: str | None = None
    abstract: str | None = None

    # Extraction metadata
    source: MetadataSource = MetadataSource.UNKNOWN
    confidence: float = 0.0
    extraction_notes: str = ""

    # Source file info
    pdf_path: Path | None = None
    attachment_item_id: int | None = None
    attachment_key: str | None = None


class OrphanMetadataExtractor:
    """Extracts metadata from orphan PDF attachments.

    Strategies (in order of preference):
    1. DOI extraction from PDF text -> Crossref lookup
    2. ISBN extraction from PDF text -> Open Library lookup
    3. Filename parsing for author/year/title
    4. PDF embedded metadata
    5. LLM extraction from first pages
    """

    # DOI regex pattern (matches most Crossref DOIs)
    # Reference: https://www.crossref.org/blog/dois-and-matching-regular-expressions/
    DOI_PATTERN = re.compile(
        r"\b(10\.\d{4,9}/[^\s\]\)>\"']+)",
        re.IGNORECASE,
    )

    # ISBN patterns
    ISBN_10_PATTERN = re.compile(
        r"\b(?:ISBN[-:]?\s*)?(\d{1,5}[-\s]?\d{1,7}[-\s]?\d{1,7}[-\s]?[\dX])\b",
        re.IGNORECASE,
    )
    ISBN_13_PATTERN = re.compile(
        r"\b(?:ISBN[-:]?\s*)?(97[89][-\s]?\d{1,5}[-\s]?\d{1,7}[-\s]?\d{1,7}[-\s]?\d)\b",
        re.IGNORECASE,
    )

    # Filename patterns (from pdffolder_adapter.py)
    FILENAME_PATTERNS = [
        # Author(s) - Year - Title
        re.compile(
            r"^(?P<authors>[^-]+)\s*-\s*(?P<year>\d{4})\s*-\s*(?P<title>.+)\.pdf$",
            re.IGNORECASE,
        ),
        # Author_Year_Title
        re.compile(
            r"^(?P<authors>[^_]+)_(?P<year>\d{4})_(?P<title>.+)\.pdf$",
            re.IGNORECASE,
        ),
        # Year_Author_Title
        re.compile(
            r"^(?P<year>\d{4})_(?P<authors>[^_]+)_(?P<title>.+)\.pdf$",
            re.IGNORECASE,
        ),
        # Year - Author - Title
        re.compile(
            r"^(?P<year>\d{4})\s*-\s*(?P<authors>[^-]+)\s*-\s*(?P<title>.+)\.pdf$",
            re.IGNORECASE,
        ),
        # Just title (fallback)
        re.compile(r"^(?P<title>.+)\.pdf$", re.IGNORECASE),
    ]

    def __init__(self, max_pages_for_extraction: int = 5):
        """Initialize the extractor.

        Args:
            max_pages_for_extraction: Max pages to scan for DOI/ISBN.
        """
        self.max_pages_for_extraction = max_pages_for_extraction

    def extract_text_from_pdf(self, pdf_path: Path, max_pages: int | None = None) -> str:
        """Extract text from PDF file.

        Args:
            pdf_path: Path to PDF file.
            max_pages: Maximum pages to extract (None = all).

        Returns:
            Extracted text.
        """
        if max_pages is None:
            max_pages = self.max_pages_for_extraction

        try:
            doc = pymupdf.open(pdf_path)
            pages_text = []
            for i, page in enumerate(doc):
                if max_pages and i >= max_pages:
                    break
                text = page.get_text()
                if text.strip():
                    pages_text.append(text)
            doc.close()
            return "\n\n".join(pages_text)
        except Exception as e:
            logger.warning(f"Failed to extract text from {pdf_path}: {e}")
            return ""

    def extract_doi(self, text: str) -> str | None:
        """Extract DOI from text.

        Args:
            text: Text to search.

        Returns:
            DOI string or None.
        """
        matches = self.DOI_PATTERN.findall(text)
        if not matches:
            return None

        # Clean up the DOI - remove trailing punctuation
        for match in matches:
            doi = match.rstrip(".,;:\"')")
            # Validate DOI format
            if doi.startswith("10.") and "/" in doi:
                return doi

        return None

    def extract_isbn(self, text: str) -> str | None:
        """Extract ISBN from text.

        Args:
            text: Text to search.

        Returns:
            ISBN string (normalized) or None.
        """
        # Try ISBN-13 first (more specific)
        matches = self.ISBN_13_PATTERN.findall(text)
        if matches:
            # Normalize: remove hyphens and spaces
            isbn = re.sub(r"[-\s]", "", matches[0])
            if len(isbn) == 13 and isbn.isdigit():
                return isbn

        # Try ISBN-10
        matches = self.ISBN_10_PATTERN.findall(text)
        if matches:
            isbn = re.sub(r"[-\s]", "", matches[0])
            if len(isbn) == 10:
                return isbn

        return None

    def parse_filename(self, pdf_path: Path) -> dict[str, str | None]:
        """Parse metadata from filename.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            Dictionary with authors, year, title.
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
                authors = groups.get("authors")
                result["authors"] = authors.strip() if authors else None
                result["year"] = groups.get("year")
                title = groups.get("title")
                result["title"] = title.strip() if title else None
                break

        # Clean up title
        if result["title"]:
            result["title"] = result["title"].replace("_", " ").strip()
            result["title"] = " ".join(result["title"].split())

        return result

    def extract_pdf_metadata(self, pdf_path: Path) -> dict[str, str | None]:
        """Extract embedded metadata from PDF.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            Dictionary with PDF metadata fields.
        """
        result: dict[str, str | None] = {
            "title": None,
            "author": None,
            "creation_date": None,
            "subject": None,
            "keywords": None,
        }

        try:
            doc = pymupdf.open(pdf_path)
            metadata = doc.metadata
            doc.close()

            if metadata:
                result["title"] = metadata.get("title") or None
                result["author"] = metadata.get("author") or None
                result["creation_date"] = metadata.get("creationDate") or None
                result["subject"] = metadata.get("subject") or None
                result["keywords"] = metadata.get("keywords") or None

                # Clean empty strings
                for key in result:
                    if result[key] and not result[key].strip():
                        result[key] = None

        except Exception as e:
            logger.debug(f"Failed to extract PDF metadata from {pdf_path}: {e}")

        return result

    def parse_authors(self, author_string: str | None) -> list[str]:
        """Parse author string into list of author names.

        Handles formats:
        - "Smith, John"
        - "John Smith"
        - "Smith, J. and Doe, J."
        - "Smith et al"
        - "Smith; Doe; Brown"

        Args:
            author_string: Author field value.

        Returns:
            List of author name strings.
        """
        if not author_string:
            return []

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

        authors = []
        for part in parts:
            part = part.strip()
            if part:
                authors.append(part)

        return authors

    def parse_year(self, date_string: str | None) -> int | None:
        """Extract year from various date formats.

        Args:
            date_string: Date string.

        Returns:
            Year as integer or None.
        """
        if not date_string:
            return None

        # Direct year
        if date_string.isdigit() and len(date_string) == 4:
            return int(date_string)

        # Find 4-digit year in string
        year_match = re.search(r"(19|20)\d{2}", date_string)
        if year_match:
            return int(year_match.group())

        return None

    def extract_metadata(
        self,
        pdf_path: Path,
        attachment_item_id: int | None = None,
        attachment_key: str | None = None,
    ) -> ExtractedMetadata:
        """Extract metadata from a PDF using all available strategies.

        Args:
            pdf_path: Path to PDF file.
            attachment_item_id: Zotero item ID of the attachment.
            attachment_key: Zotero key of the attachment.

        Returns:
            ExtractedMetadata with best available data.
        """
        result = ExtractedMetadata(
            pdf_path=pdf_path,
            attachment_item_id=attachment_item_id,
            attachment_key=attachment_key,
        )

        # Strategy 1: Extract text and look for DOI
        text = self.extract_text_from_pdf(pdf_path)

        if text:
            doi = self.extract_doi(text)
            if doi:
                result.doi = doi
                result.source = MetadataSource.DOI_EXTRACTED
                result.confidence = 0.9
                result.extraction_notes = f"DOI found in PDF text: {doi}"
                logger.info(f"Found DOI in {pdf_path.name}: {doi}")
                return result

            # Strategy 2: Look for ISBN
            isbn = self.extract_isbn(text)
            if isbn:
                result.isbn = isbn
                result.source = MetadataSource.ISBN_EXTRACTED
                result.confidence = 0.85
                result.extraction_notes = f"ISBN found in PDF text: {isbn}"
                logger.info(f"Found ISBN in {pdf_path.name}: {isbn}")
                return result

        # Strategy 3: Parse filename
        filename_data = self.parse_filename(pdf_path)
        if filename_data.get("title") and filename_data.get("authors"):
            result.title = filename_data["title"]
            result.authors = self.parse_authors(filename_data["authors"])
            result.publication_year = self.parse_year(filename_data.get("year"))
            result.source = MetadataSource.FILENAME_PARSED
            result.confidence = 0.6
            result.extraction_notes = "Metadata parsed from filename"
            return result

        # Strategy 4: PDF embedded metadata
        pdf_metadata = self.extract_pdf_metadata(pdf_path)
        if pdf_metadata.get("title"):
            result.title = pdf_metadata["title"]
            result.authors = self.parse_authors(pdf_metadata.get("author"))
            result.publication_year = self.parse_year(pdf_metadata.get("creation_date"))
            result.abstract = pdf_metadata.get("subject")
            result.source = MetadataSource.PDF_METADATA
            result.confidence = 0.5
            result.extraction_notes = "Metadata from PDF document properties"
            return result

        # Strategy 5: Fallback to filename title only
        if filename_data.get("title"):
            result.title = filename_data["title"]
            result.source = MetadataSource.FILENAME_PARSED
            result.confidence = 0.3
            result.extraction_notes = "Title only from filename (no structured pattern)"
            return result

        # No metadata found
        result.title = pdf_path.stem  # Use filename without extension
        result.source = MetadataSource.UNKNOWN
        result.confidence = 0.1
        result.extraction_notes = "No metadata found, using filename as title"
        return result

    def extract_with_llm(
        self,
        pdf_path: Path,
        text: str | None = None,
    ) -> ExtractedMetadata:
        """Extract metadata using LLM analysis of first pages.

        This is used as a fallback when DOI/ISBN extraction fails.

        Args:
            pdf_path: Path to PDF file.
            text: Pre-extracted text (optional).

        Returns:
            ExtractedMetadata from LLM extraction.
        """
        # Import here to avoid circular dependency
        from src.analysis.section_extractor import SectionExtractor

        if text is None:
            text = self.extract_text_from_pdf(pdf_path, max_pages=3)

        if not text or len(text) < 100:
            return ExtractedMetadata(
                pdf_path=pdf_path,
                title=pdf_path.stem,
                source=MetadataSource.UNKNOWN,
                confidence=0.1,
                extraction_notes="Insufficient text for LLM extraction",
            )

        # Use the LLM to extract metadata
        try:
            # Note: SectionExtractor imported but not used directly here
            # We use direct Anthropic API call for simpler metadata extraction
            _ = SectionExtractor  # Verify import works

            # Create a custom prompt for metadata extraction
            prompt = f"""Analyze the following text from the first pages of an academic document.
Extract the bibliographic metadata.

TEXT:
{text[:8000]}

Return a JSON object with these fields (use null if not found):
{{
    "title": "The full title of the document",
    "authors": ["List of author names"],
    "publication_year": 2024,
    "journal": "Journal or publication name",
    "abstract": "Abstract if present (first 500 chars)"
}}

Return ONLY valid JSON, no other text."""

            # Call the LLM
            import anthropic

            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse response
            import json

            # Get text from response (handle different block types)
            response_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    response_text = block.text
                    break
            data = json.loads(response_text)

            result = ExtractedMetadata(
                pdf_path=pdf_path,
                title=data.get("title"),
                authors=data.get("authors", []),
                publication_year=data.get("publication_year"),
                journal=data.get("journal"),
                abstract=data.get("abstract"),
                source=MetadataSource.LLM_EXTRACTED,
                confidence=0.7,
                extraction_notes="Metadata extracted via LLM analysis",
            )
            return result

        except Exception as e:
            logger.warning(f"LLM extraction failed for {pdf_path}: {e}")
            return ExtractedMetadata(
                pdf_path=pdf_path,
                title=pdf_path.stem,
                source=MetadataSource.UNKNOWN,
                confidence=0.1,
                extraction_notes=f"LLM extraction failed: {e}",
            )
