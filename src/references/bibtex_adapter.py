"""BibTeX reference database adapter.

Provides a reference database interface for BibTeX (.bib) files.
This enables LITRIS to work with papers from any reference manager
that can export to BibTeX format.
"""

import hashlib
import re
from collections.abc import Callable, Generator
from datetime import datetime
from pathlib import Path

from src.references.base import BaseReferenceDB, ReferenceProvider
from src.utils.logging_config import get_logger
from src.zotero.models import Author, PaperMetadata

logger = get_logger(__name__)


class BibTeXReferenceDB(BaseReferenceDB):
    """BibTeX file reference database.

    Parses BibTeX files and provides paper metadata through the
    standard BaseReferenceDB interface.
    """

    def __init__(self, bibtex_path: Path, pdf_dir: Path | None = None):
        """Initialize BibTeX reference database.

        Args:
            bibtex_path: Path to .bib file.
            pdf_dir: Optional directory containing PDFs. If provided,
                     attempts to match PDFs to entries by citation key.
        """
        self.bibtex_path = bibtex_path
        self.pdf_dir = pdf_dir
        self._entries: list[dict] | None = None
        self._parse_timestamp = datetime.now()

    @property
    def provider(self) -> ReferenceProvider:
        """Return the provider identifier."""
        return "bibtex"

    @property
    def source_path(self) -> Path:
        """Return the BibTeX file path."""
        return self.bibtex_path

    def _parse_bibtex(self) -> list[dict]:
        """Parse the BibTeX file.

        Returns:
            List of entry dictionaries.
        """
        if self._entries is not None:
            return self._entries

        if not self.bibtex_path.exists():
            raise FileNotFoundError(f"BibTeX file not found: {self.bibtex_path}")

        entries = []
        content = self.bibtex_path.read_text(encoding="utf-8", errors="replace")

        # Simple BibTeX parser
        # Match @type{key, ... }
        entry_pattern = re.compile(
            r"@(\w+)\s*\{\s*([^,\s]+)\s*,(.+?)\n\s*\}",
            re.DOTALL,
        )

        for match in entry_pattern.finditer(content):
            entry_type = match.group(1).lower()
            citation_key = match.group(2).strip()
            fields_text = match.group(3)

            # Skip comments and strings
            if entry_type in ("comment", "string", "preamble"):
                continue

            # Parse fields
            fields = self._parse_fields(fields_text)
            fields["_type"] = entry_type
            fields["_key"] = citation_key

            entries.append(fields)

        self._entries = entries
        logger.info(f"Parsed {len(entries)} entries from {self.bibtex_path}")
        return entries

    def _parse_fields(self, fields_text: str) -> dict[str, str]:
        """Parse BibTeX field assignments.

        Args:
            fields_text: The text between @type{key, and closing }

        Returns:
            Dictionary of field name -> value.
        """
        fields = {}

        # Match field = value patterns
        # Handles: field = {value}, field = "value", field = number
        field_pattern = re.compile(
            r"(\w+)\s*=\s*(?:\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}|\"([^\"]*)\"|(\d+))",
        )

        for match in field_pattern.finditer(fields_text):
            field_name = match.group(1).lower()
            value = match.group(2) or match.group(3) or match.group(4) or ""
            # Clean up the value
            value = self._clean_latex(value.strip())
            fields[field_name] = value

        return fields

    def _clean_latex(self, text: str) -> str:
        """Remove common LaTeX formatting from text.

        Args:
            text: Text potentially containing LaTeX.

        Returns:
            Cleaned text.
        """
        if not text:
            return text

        # Remove common LaTeX commands
        text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
        text = re.sub(r"\\textit\{([^}]*)\}", r"\1", text)
        text = re.sub(r"\\emph\{([^}]*)\}", r"\1", text)
        text = re.sub(r"\\\w+\{([^}]*)\}", r"\1", text)

        # Remove escaped characters
        text = text.replace("\\&", "&")
        text = text.replace("\\%", "%")
        text = text.replace("\\$", "$")
        text = text.replace("\\_", "_")
        text = text.replace("\\#", "#")
        text = text.replace("\\~", "~")
        text = text.replace("\\^", "^")

        # Remove remaining backslashes before common characters
        text = re.sub(r"\\([a-zA-Z])", r"\1", text)

        # Clean up whitespace
        text = " ".join(text.split())

        return text

    def _parse_authors(self, author_string: str) -> list[Author]:
        """Parse BibTeX author field.

        BibTeX format: "Last1, First1 and Last2, First2 and ..."
        Also handles: "First Last and First Last"

        Args:
            author_string: The author field value.

        Returns:
            List of Author objects.
        """
        if not author_string:
            return []

        authors = []
        # Split by " and " (case insensitive)
        parts = re.split(r"\s+and\s+", author_string, flags=re.IGNORECASE)

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
                # Format: "First Middle Last" - last word is last name
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

    def _find_pdf(self, citation_key: str, entry: dict) -> Path | None:
        """Find PDF file for an entry.

        Tries multiple strategies:
        1. Check 'file' field in entry
        2. Look for {citation_key}.pdf in pdf_dir
        3. Look for partial matches

        Args:
            citation_key: The BibTeX citation key.
            entry: The entry dictionary.

        Returns:
            Path to PDF or None.
        """
        # Check file field first
        if "file" in entry:
            file_field = entry["file"]
            # Zotero/Mendeley format: "Description:path:type"
            if ":" in file_field:
                parts = file_field.split(":")
                if len(parts) >= 2:
                    pdf_path = Path(parts[1])
                    if pdf_path.exists():
                        return pdf_path
            else:
                pdf_path = Path(file_field)
                if pdf_path.exists():
                    return pdf_path

        # No pdf_dir configured
        if not self.pdf_dir or not self.pdf_dir.exists():
            return None

        # Try exact match: {key}.pdf
        exact_path = self.pdf_dir / f"{citation_key}.pdf"
        if exact_path.exists():
            return exact_path

        # Try case-insensitive match
        for pdf_file in self.pdf_dir.glob("*.pdf"):
            if pdf_file.stem.lower() == citation_key.lower():
                return pdf_file

        return None

    def _entry_to_paper(self, entry: dict) -> PaperMetadata:
        """Convert a BibTeX entry to PaperMetadata.

        Args:
            entry: Parsed BibTeX entry dictionary.

        Returns:
            PaperMetadata object.
        """
        citation_key = entry.get("_key", "unknown")
        entry_type = entry.get("_type", "article")

        # Map BibTeX entry types to Zotero-style types
        type_mapping = {
            "article": "journalArticle",
            "inproceedings": "conferencePaper",
            "conference": "conferencePaper",
            "book": "book",
            "incollection": "bookSection",
            "inbook": "bookSection",
            "phdthesis": "thesis",
            "mastersthesis": "thesis",
            "techreport": "report",
            "misc": "document",
            "unpublished": "manuscript",
        }
        item_type = type_mapping.get(entry_type, "document")

        # Parse authors
        authors = self._parse_authors(entry.get("author", ""))

        # Extract year
        year_str = entry.get("year", "")
        year = None
        if year_str:
            try:
                year = int(year_str)
            except ValueError:
                pass

        # Find PDF
        pdf_path = self._find_pdf(citation_key, entry)

        # Generate a stable ID from the BibTeX file path and citation key
        id_source = f"{self.bibtex_path}:{citation_key}"
        stable_hash = hashlib.sha256(id_source.encode()).hexdigest()[:8]

        # Create a pseudo zotero_item_id from the hash
        zotero_item_id = int(stable_hash, 16) % 2147483647  # Keep in int32 range

        return PaperMetadata(
            paper_id=f"bibtex_{citation_key}",
            zotero_key=citation_key,
            zotero_item_id=zotero_item_id,
            item_type=item_type,
            title=entry.get("title", "Untitled"),
            authors=authors,
            publication_year=year,
            publication_date=entry.get("year"),
            journal=entry.get("journal") or entry.get("booktitle"),
            volume=entry.get("volume"),
            issue=entry.get("number"),
            pages=entry.get("pages"),
            doi=entry.get("doi"),
            isbn=entry.get("isbn"),
            issn=entry.get("issn"),
            abstract=entry.get("abstract"),
            url=entry.get("url"),
            collections=[],  # BibTeX doesn't have collections
            tags=self._parse_keywords(entry.get("keywords", "")),
            pdf_path=pdf_path,
            pdf_attachment_key=f"bibtex_{stable_hash}" if pdf_path else None,
            date_added=self._parse_timestamp,
            date_modified=self._parse_timestamp,
        )

    def _parse_keywords(self, keywords_str: str) -> list[str]:
        """Parse BibTeX keywords field.

        Args:
            keywords_str: Comma or semicolon separated keywords.

        Returns:
            List of keyword strings.
        """
        if not keywords_str:
            return []

        # Split by comma or semicolon
        parts = re.split(r"[,;]", keywords_str)
        return [p.strip() for p in parts if p.strip()]

    def get_all_papers(
        self, progress_callback: Callable[[int, int], None] | None = None
    ) -> Generator[PaperMetadata, None, None]:
        """Get all papers from the BibTeX file.

        Args:
            progress_callback: Optional callback(current, total) for progress.

        Yields:
            PaperMetadata objects for each entry.
        """
        entries = self._parse_bibtex()
        total = len(entries)

        for i, entry in enumerate(entries):
            try:
                paper = self._entry_to_paper(entry)
                if progress_callback:
                    progress_callback(i + 1, total)
                yield paper
            except Exception as e:
                logger.warning(
                    f"Failed to parse entry {entry.get('_key', 'unknown')}: {e}"
                )
                continue

    def get_paper_count(self) -> int:
        """Get total count of entries.

        Returns:
            Number of entries in the BibTeX file.
        """
        entries = self._parse_bibtex()
        return len(entries)

    def get_paper_by_key(self, key: str) -> PaperMetadata | None:
        """Get a specific paper by citation key.

        Args:
            key: BibTeX citation key.

        Returns:
            PaperMetadata or None if not found.
        """
        entries = self._parse_bibtex()

        for entry in entries:
            if entry.get("_key") == key:
                return self._entry_to_paper(entry)

        return None

    def reload(self) -> None:
        """Reload the BibTeX file.

        Call this to pick up changes to the source file.
        """
        self._entries = None
        self._parse_timestamp = datetime.now()
        self._parse_bibtex()
