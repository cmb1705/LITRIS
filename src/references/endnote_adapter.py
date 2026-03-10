"""EndNote XML reference database adapter.

Provides a reference database interface for EndNote XML export files.
This enables LITRIS to work with papers exported from EndNote.

EndNote XML format reference:
- https://support.clarivate.com/Endnote/s/article/EndNote-XML-Document-Type-Definition
"""

import hashlib
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Generator
from datetime import datetime
from pathlib import Path

from src.references.base import BaseReferenceDB, ReferenceProvider
from src.utils.logging_config import get_logger
from src.zotero.models import Author, PaperMetadata

logger = get_logger(__name__)


class EndNoteReferenceDB(BaseReferenceDB):
    """EndNote XML file reference database.

    Parses EndNote XML export files and provides paper metadata through the
    standard BaseReferenceDB interface.
    """

    # EndNote reference type numbers to Zotero-style item types
    REF_TYPE_MAP = {
        # Journal Article = 17
        "17": "journalArticle",
        "journal article": "journalArticle",
        # Book = 6
        "6": "book",
        "book": "book",
        # Book Section = 5
        "5": "bookSection",
        "book section": "bookSection",
        # Conference Proceedings/Paper = 10
        "10": "conferencePaper",
        "conference proceedings": "conferencePaper",
        "conference paper": "conferencePaper",
        # Report = 27
        "27": "report",
        "report": "report",
        # Thesis = 32
        "32": "thesis",
        "thesis": "thesis",
        # Electronic Article = 43
        "43": "journalArticle",
        "electronic article": "journalArticle",
        # Web Page = 12
        "12": "webpage",
        "web page": "webpage",
        # Magazine Article = 19
        "19": "magazineArticle",
        "magazine article": "magazineArticle",
        # Newspaper Article = 23
        "23": "newspaperArticle",
        "newspaper article": "newspaperArticle",
        # Patent = 25
        "25": "patent",
        "patent": "patent",
        # Manuscript = 36
        "36": "manuscript",
        "manuscript": "manuscript",
        # Generic = 13
        "13": "document",
        "generic": "document",
    }

    def __init__(self, xml_path: Path, pdf_dir: Path | None = None):
        """Initialize EndNote reference database.

        Args:
            xml_path: Path to EndNote XML export file.
            pdf_dir: Optional directory containing PDFs. If provided,
                     attempts to match PDFs to entries.
        """
        self.xml_path = xml_path
        self.pdf_dir = pdf_dir
        self._records: list[ET.Element] | None = None
        self._parse_timestamp = datetime.now()

    @property
    def provider(self) -> ReferenceProvider:
        """Return the provider identifier."""
        return "endnote"

    @property
    def source_path(self) -> Path:
        """Return the XML file path."""
        return self.xml_path

    def _parse_xml(self) -> list[ET.Element]:
        """Parse the EndNote XML file.

        Returns:
            List of record elements.
        """
        if self._records is not None:
            return self._records

        if not self.xml_path.exists():
            raise FileNotFoundError(f"EndNote XML file not found: {self.xml_path}")

        try:
            tree = ET.parse(self.xml_path)
            root = tree.getroot()

            # Find all record elements
            # EndNote XML structure: <xml><records><record>...
            records = []

            # Try different possible paths
            for record in root.iter("record"):
                records.append(record)

            self._records = records
            logger.info(f"Parsed {len(records)} records from {self.xml_path}")
            return records

        except ET.ParseError as e:
            raise ValueError(f"Failed to parse EndNote XML: {e}") from e

    def _get_text(self, element: ET.Element | None, default: str = "") -> str:
        """Extract text from an element, handling style tags.

        EndNote XML uses <style> tags within elements for formatting.

        Args:
            element: XML element to extract text from.
            default: Default value if element is None.

        Returns:
            Extracted text content.
        """
        if element is None:
            return default

        # Collect all text including from style elements
        text_parts = []
        if element.text:
            text_parts.append(element.text)

        for child in element:
            # Get text from style elements
            if child.text:
                text_parts.append(child.text)
            if child.tail:
                text_parts.append(child.tail)

        return " ".join(text_parts).strip() or default

    def _get_nested_text(
        self, record: ET.Element, path: str, default: str = ""
    ) -> str:
        """Get text from a nested element path.

        Args:
            record: Parent record element.
            path: XPath-like path (e.g., "titles/title").
            default: Default value if not found.

        Returns:
            Text content.
        """
        element = record.find(path)
        return self._get_text(element, default)

    def _parse_authors(self, record: ET.Element) -> list[Author]:
        """Parse authors from an EndNote record.

        Args:
            record: Record element.

        Returns:
            List of Author objects.
        """
        authors = []
        contributors = record.find("contributors")
        if contributors is None:
            return authors

        # Primary authors
        authors_elem = contributors.find("authors")
        if authors_elem is not None:
            for i, author_elem in enumerate(authors_elem.findall("author")):
                author_text = self._get_text(author_elem)
                if author_text:
                    first, last = self._split_author_name(author_text)
                    authors.append(Author(
                        first_name=first,
                        last_name=last,
                        order=i + 1,
                        role="author",
                    ))

        # Secondary authors (editors)
        secondary = contributors.find("secondary-authors")
        if secondary is not None:
            for _i, author_elem in enumerate(secondary.findall("author")):
                author_text = self._get_text(author_elem)
                if author_text:
                    first, last = self._split_author_name(author_text)
                    authors.append(Author(
                        first_name=first,
                        last_name=last,
                        order=len(authors) + 1,
                        role="editor",
                    ))

        return authors

    def _split_author_name(self, name: str) -> tuple[str, str]:
        """Split author name into first and last name.

        EndNote typically uses "Last, First" format.

        Args:
            name: Full author name string.

        Returns:
            Tuple of (first_name, last_name).
        """
        if "," in name:
            parts = name.split(",", 1)
            last_name = parts[0].strip()
            first_name = parts[1].strip() if len(parts) > 1 else ""
        else:
            # Try "First Last" format
            parts = name.split()
            if len(parts) >= 2:
                last_name = parts[-1]
                first_name = " ".join(parts[:-1])
            else:
                last_name = name
                first_name = ""

        return first_name, last_name

    def _parse_year(self, record: ET.Element) -> int | None:
        """Extract publication year from record.

        Args:
            record: Record element.

        Returns:
            Year as integer or None.
        """
        # Try dates/year first
        year_text = self._get_nested_text(record, "dates/year")
        if year_text:
            try:
                # Extract first 4-digit year
                match = re.search(r"\d{4}", year_text)
                if match:
                    return int(match.group())
            except ValueError:
                pass

        # Try pub-dates
        pub_dates = record.find("dates/pub-dates")
        if pub_dates is not None:
            date_elem = pub_dates.find("date")
            if date_elem is not None:
                date_text = self._get_text(date_elem)
                match = re.search(r"\d{4}", date_text)
                if match:
                    return int(match.group())

        return None

    def _parse_keywords(self, record: ET.Element) -> list[str]:
        """Extract keywords from record.

        Args:
            record: Record element.

        Returns:
            List of keyword strings.
        """
        keywords = []
        kw_elem = record.find("keywords")
        if kw_elem is not None:
            for keyword in kw_elem.findall("keyword"):
                kw_text = self._get_text(keyword)
                if kw_text:
                    keywords.append(kw_text)
        return keywords

    def _find_pdf(self, rec_number: str, title: str) -> Path | None:
        """Find PDF file for a record.

        Args:
            rec_number: EndNote record number.
            title: Paper title for fuzzy matching.

        Returns:
            Path to PDF or None.
        """
        if not self.pdf_dir or not self.pdf_dir.exists():
            return None

        # Try exact match by record number
        exact_path = self.pdf_dir / f"{rec_number}.pdf"
        if exact_path.exists():
            return exact_path

        # Try matching by title (sanitized)
        if title:
            sanitized = re.sub(r"[^\w\s-]", "", title)[:50]
            sanitized = re.sub(r"\s+", "_", sanitized)

            for pdf_file in self.pdf_dir.glob("*.pdf"):
                if sanitized.lower() in pdf_file.stem.lower():
                    return pdf_file

        return None

    def _get_item_type(self, record: ET.Element) -> str:
        """Determine item type from record.

        Args:
            record: Record element.

        Returns:
            Zotero-style item type.
        """
        ref_type = record.find("ref-type")
        if ref_type is not None:
            # Try the name attribute first
            type_name = ref_type.get("name", "").lower()
            if type_name in self.REF_TYPE_MAP:
                return self.REF_TYPE_MAP[type_name]

            # Try the numeric value
            type_num = self._get_text(ref_type)
            if type_num in self.REF_TYPE_MAP:
                return self.REF_TYPE_MAP[type_num]

        return "document"

    def _record_to_paper(self, record: ET.Element) -> PaperMetadata:
        """Convert an EndNote record to PaperMetadata.

        Args:
            record: Parsed EndNote record element.

        Returns:
            PaperMetadata object.
        """
        # Get record number for ID
        rec_number = self._get_nested_text(record, "rec-number", "0")

        # Get basic metadata
        title = self._get_nested_text(record, "titles/title", "Untitled")
        secondary_title = self._get_nested_text(record, "titles/secondary-title")

        # Parse authors
        authors = self._parse_authors(record)

        # Get year
        year = self._parse_year(record)

        # Get journal/publication info
        periodical = self._get_nested_text(record, "periodical/full-title")
        if not periodical:
            periodical = secondary_title

        # Get other fields
        volume = self._get_nested_text(record, "volume")
        issue = self._get_nested_text(record, "number")
        pages = self._get_nested_text(record, "pages")
        abstract = self._get_nested_text(record, "abstract")

        # Get identifiers
        doi = self._get_nested_text(record, "electronic-resource-num")
        isbn = self._get_nested_text(record, "isbn")

        # Get URL
        url = ""
        urls_elem = record.find("urls/related-urls/url")
        if urls_elem is not None:
            url = self._get_text(urls_elem)

        # Get keywords
        keywords = self._parse_keywords(record)

        # Find PDF
        pdf_path = self._find_pdf(rec_number, title)

        # Generate stable ID
        id_source = f"{self.xml_path}:{rec_number}"
        stable_hash = hashlib.sha256(id_source.encode()).hexdigest()[:8]
        zotero_item_id = int(stable_hash, 16) % 2147483647

        return PaperMetadata(
            paper_id=f"endnote_{rec_number}",
            zotero_key=f"EN{rec_number}",
            zotero_item_id=zotero_item_id,
            item_type=self._get_item_type(record),
            title=title,
            authors=authors,
            publication_year=year,
            publication_date=str(year) if year else None,
            journal=periodical,
            volume=volume if volume else None,
            issue=issue if issue else None,
            pages=pages if pages else None,
            doi=doi if doi else None,
            isbn=isbn if isbn else None,
            issn=None,
            abstract=abstract if abstract else None,
            url=url if url else None,
            collections=[],
            tags=keywords,
            pdf_path=pdf_path,
            pdf_attachment_key=f"endnote_{stable_hash}" if pdf_path else None,
            date_added=self._parse_timestamp,
            date_modified=self._parse_timestamp,
        )

    def get_all_papers(
        self, progress_callback: Callable[[int, int], None] | None = None
    ) -> Generator[PaperMetadata, None, None]:
        """Get all papers from the EndNote XML file.

        Memory: O(n) records; the XML file is parsed into memory.

        Args:
            progress_callback: Optional callback(current, total) for progress.

        Yields:
            PaperMetadata objects for each record.
        """
        records = self._parse_xml()
        total = len(records)

        for i, record in enumerate(records):
            try:
                paper = self._record_to_paper(record)
                if progress_callback:
                    progress_callback(i + 1, total)
                yield paper
            except Exception as e:
                rec_num = self._get_nested_text(record, "rec-number", "unknown")
                logger.warning(f"Failed to parse record {rec_num}: {e}")
                continue

    def get_paper_count(self) -> int:
        """Get total count of records.

        Returns:
            Number of records in the XML file.
        """
        records = self._parse_xml()
        return len(records)

    def get_paper_by_key(self, key: str) -> PaperMetadata | None:
        """Get a specific paper by record number.

        Args:
            key: EndNote record number (or paper_id like "endnote_123").

        Returns:
            PaperMetadata or None if not found.
        """
        # Handle both "123" and "endnote_123" formats
        rec_number = key.replace("endnote_", "").replace("EN", "")

        records = self._parse_xml()
        for record in records:
            if self._get_nested_text(record, "rec-number") == rec_number:
                return self._record_to_paper(record)

        return None

    def reload(self) -> None:
        """Reload the XML file.

        Call this to pick up changes to the source file.
        """
        self._records = None
        self._parse_timestamp = datetime.now()
        self._parse_xml()
