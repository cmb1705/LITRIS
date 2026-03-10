"""Paperpile reference database adapter.

Provides a reference database interface for Paperpile exports.
Since Paperpile does not have a public REST API, this adapter works
with BibTeX exports from Paperpile, with additional handling for
Paperpile-specific metadata fields.

To export from Paperpile:
1. Go to paperpile.com and open your library
2. Select references to export (or all)
3. Export -> BibTeX format
4. Save the .bib file

References:
- https://forum.paperpile.com/t/public-developer-api/918
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


class PaperpileReferenceDB(BaseReferenceDB):
    """Paperpile BibTeX export reference database.

    Parses BibTeX files exported from Paperpile with additional handling
    for Paperpile-specific fields and metadata.

    Note: Paperpile does not currently offer a public API. This adapter
    works with BibTeX exports, which preserve most metadata including:
    - Labels (mapped to tags)
    - Folders (mapped to collections via keywords)
    - Notes (preserved in abstract if no abstract exists)
    - PDF links (for matching with local files)
    """

    # Paperpile-specific BibTeX type mappings
    TYPE_MAP = {
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
        "online": "webpage",
        "patent": "patent",
    }

    def __init__(
        self,
        bibtex_path: Path,
        pdf_dir: Path | None = None,
        sync_folder: Path | None = None,
    ):
        """Initialize Paperpile reference database.

        Args:
            bibtex_path: Path to BibTeX file exported from Paperpile.
            pdf_dir: Optional directory containing PDFs.
            sync_folder: Path to Paperpile sync folder (Google Drive).
                        If provided, attempts to find PDFs there.
        """
        self.bibtex_path = bibtex_path
        self.pdf_dir = pdf_dir
        self.sync_folder = sync_folder
        self._entries: list[dict] | None = None
        self._parse_timestamp = datetime.now()

    @property
    def provider(self) -> ReferenceProvider:
        """Return the provider identifier."""
        return "paperpile"

    @property
    def source_path(self) -> Path:
        """Return the BibTeX file path."""
        return self.bibtex_path

    def _parse_bibtex(self) -> list[dict]:
        """Parse the Paperpile BibTeX export.

        Returns:
            List of entry dictionaries.
        """
        if self._entries is not None:
            return self._entries

        if not self.bibtex_path.exists():
            raise FileNotFoundError(f"BibTeX file not found: {self.bibtex_path}")

        entries = []
        content = self.bibtex_path.read_text(encoding="utf-8", errors="replace")

        # Parse BibTeX entries
        entry_pattern = re.compile(
            r"@(\w+)\s*\{\s*([^,\s]+)\s*,(.+?)\n\s*\}",
            re.DOTALL,
        )

        for match in entry_pattern.finditer(content):
            entry_type = match.group(1).lower()
            citation_key = match.group(2).strip()
            fields_text = match.group(3)

            if entry_type in ("comment", "string", "preamble"):
                continue

            fields = self._parse_fields(fields_text)
            fields["_type"] = entry_type
            fields["_key"] = citation_key

            entries.append(fields)

        self._entries = entries
        logger.info(f"Parsed {len(entries)} entries from Paperpile export")
        return entries

    def _parse_fields(self, fields_text: str) -> dict[str, str]:
        """Parse BibTeX field assignments.

        Args:
            fields_text: The text between @type{key, and closing }

        Returns:
            Dictionary of field name -> value.
        """
        fields = {}

        field_pattern = re.compile(
            r"(\w+)\s*=\s*(?:\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}|\"([^\"]*)\"|(\d+))",
        )

        for match in field_pattern.finditer(fields_text):
            field_name = match.group(1).lower()
            value = match.group(2) or match.group(3) or match.group(4) or ""
            value = self._clean_text(value.strip())
            fields[field_name] = value

        return fields

    def _clean_text(self, text: str) -> str:
        """Clean LaTeX formatting from text.

        Args:
            text: Text potentially containing LaTeX.

        Returns:
            Cleaned text.
        """
        if not text:
            return text

        # Remove LaTeX commands
        text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
        text = re.sub(r"\\textit\{([^}]*)\}", r"\1", text)
        text = re.sub(r"\\emph\{([^}]*)\}", r"\1", text)
        text = re.sub(r"\\\w+\{([^}]*)\}", r"\1", text)

        # Remove escaped characters
        text = text.replace("\\&", "&")
        text = text.replace("\\%", "%")
        text = text.replace("\\$", "$")
        text = text.replace("\\_", "_")

        text = " ".join(text.split())
        return text

    def _parse_authors(self, author_string: str) -> list[Author]:
        """Parse Paperpile author field.

        Args:
            author_string: The author field value.

        Returns:
            List of Author objects.
        """
        if not author_string:
            return []

        authors = []
        parts = re.split(r"\s+and\s+", author_string, flags=re.IGNORECASE)

        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue

            first_name = ""
            last_name = ""

            if "," in part:
                comma_parts = part.split(",", 1)
                last_name = comma_parts[0].strip()
                first_name = comma_parts[1].strip() if len(comma_parts) > 1 else ""
            else:
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

    def _parse_paperpile_labels(self, entry: dict) -> list[str]:
        """Extract Paperpile labels from entry.

        Paperpile stores labels in the 'keywords' or 'mendeley-tags' field.

        Args:
            entry: Entry dictionary.

        Returns:
            List of label/tag strings.
        """
        tags = []

        # Check keywords field
        keywords = entry.get("keywords", "")
        if keywords:
            parts = re.split(r"[,;]", keywords)
            tags.extend([p.strip() for p in parts if p.strip()])

        # Check mendeley-tags (Paperpile sometimes uses this)
        mendeley_tags = entry.get("mendeley-tags", "")
        if mendeley_tags:
            parts = re.split(r"[,;]", mendeley_tags)
            tags.extend([p.strip() for p in parts if p.strip()])

        return list(set(tags))  # Remove duplicates

    def _parse_paperpile_folders(self, entry: dict) -> list[str]:
        """Extract Paperpile folder information.

        Paperpile may store folder info in 'groups' or 'folder' fields.

        Args:
            entry: Entry dictionary.

        Returns:
            List of folder/collection names.
        """
        folders = []

        for field in ["groups", "folder", "folders"]:
            value = entry.get(field, "")
            if value:
                parts = re.split(r"[,;/]", value)
                folders.extend([p.strip() for p in parts if p.strip()])

        return list(set(folders))

    def _find_pdf(self, citation_key: str, entry: dict) -> Path | None:
        """Find PDF file for an entry.

        Tries multiple strategies:
        1. Check 'file' field in entry
        2. Look in Paperpile sync folder
        3. Look in pdf_dir by citation key

        Args:
            citation_key: The BibTeX citation key.
            entry: The entry dictionary.

        Returns:
            Path to PDF or None.
        """
        # Check file field (Paperpile format)
        if "file" in entry:
            file_field = entry["file"]
            # Paperpile format: "path/to/file.pdf" or "Description:path:type"
            if ":" in file_field:
                parts = file_field.split(":")
                for part in parts:
                    if part.endswith(".pdf"):
                        pdf_path = Path(part)
                        if pdf_path.exists():
                            return pdf_path
            else:
                pdf_path = Path(file_field)
                if pdf_path.exists():
                    return pdf_path

        # Check Paperpile sync folder
        if self.sync_folder and self.sync_folder.exists():
            # Paperpile organizes by first author last name
            authors = self._parse_authors(entry.get("author", ""))
            if authors:
                first_author = authors[0].last_name
                year = entry.get("year", "")

                # Try author folder
                author_folder = self.sync_folder / first_author
                if author_folder.exists():
                    for pdf_file in author_folder.glob("*.pdf"):
                        if citation_key.lower() in pdf_file.stem.lower():
                            return pdf_file
                        if year and year in pdf_file.stem:
                            return pdf_file

        # Check pdf_dir
        if self.pdf_dir and self.pdf_dir.exists():
            exact_path = self.pdf_dir / f"{citation_key}.pdf"
            if exact_path.exists():
                return exact_path

            for pdf_file in self.pdf_dir.glob("*.pdf"):
                if pdf_file.stem.lower() == citation_key.lower():
                    return pdf_file

        return None

    def _entry_to_paper(self, entry: dict) -> PaperMetadata:
        """Convert a Paperpile BibTeX entry to PaperMetadata.

        Args:
            entry: Parsed BibTeX entry dictionary.

        Returns:
            PaperMetadata object.
        """
        citation_key = entry.get("_key", "unknown")
        entry_type = entry.get("_type", "article")

        # Map entry type
        item_type = self.TYPE_MAP.get(entry_type, "document")

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

        # Get labels and folders
        tags = self._parse_paperpile_labels(entry)
        collections = self._parse_paperpile_folders(entry)

        # Find PDF
        pdf_path = self._find_pdf(citation_key, entry)

        # Generate stable ID
        id_source = f"{self.bibtex_path}:{citation_key}"
        stable_hash = hashlib.sha256(id_source.encode()).hexdigest()[:8]
        zotero_item_id = int(stable_hash, 16) % 2147483647

        # Get abstract (Paperpile may store notes here if no abstract)
        abstract = entry.get("abstract", "")
        if not abstract:
            abstract = entry.get("note", "")

        return PaperMetadata(
            paper_id=f"paperpile_{citation_key}",
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
            abstract=abstract if abstract else None,
            url=entry.get("url"),
            collections=collections,
            tags=tags,
            pdf_path=pdf_path,
            pdf_attachment_key=f"paperpile_{stable_hash}" if pdf_path else None,
            date_added=self._parse_timestamp,
            date_modified=self._parse_timestamp,
        )

    def get_all_papers(
        self, progress_callback: Callable[[int, int], None] | None = None
    ) -> Generator[PaperMetadata, None, None]:
        """Get all papers from the Paperpile export.

        Memory: O(n) entries; the BibTeX file is parsed into memory.

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
            key: Citation key (or paper_id like "paperpile_key").

        Returns:
            PaperMetadata or None if not found.
        """
        # Handle both "key" and "paperpile_key" formats
        search_key = key.replace("paperpile_", "")

        entries = self._parse_bibtex()
        for entry in entries:
            if entry.get("_key") == search_key:
                return self._entry_to_paper(entry)

        return None

    def reload(self) -> None:
        """Reload the BibTeX file.

        Call this to pick up changes to the source file.
        """
        self._entries = None
        self._parse_timestamp = datetime.now()
        self._parse_bibtex()
