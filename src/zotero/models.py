"""Pydantic models for Zotero data structures."""

import re
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, computed_field, field_validator


class Author(BaseModel):
    """Author information from Zotero."""

    first_name: str = ""
    last_name: str = ""
    order: int = 1
    role: str = "author"

    @computed_field
    @property
    def full_name(self) -> str:
        """Compute full name from first and last name."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.last_name or self.first_name or "Unknown"


class Collection(BaseModel):
    """Zotero collection with hierarchy information."""

    collection_id: int
    name: str
    parent_id: int | None = None
    parent_path: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def full_path(self) -> str:
        """Complete path including ancestors."""
        if self.parent_path:
            return "/".join(self.parent_path + [self.name])
        return self.name


class PaperMetadata(BaseModel):
    """Complete metadata for a paper from Zotero."""

    paper_id: str = ""  # Will be set from zotero_key in model_post_init
    zotero_key: str
    zotero_item_id: int
    item_type: str
    title: str
    authors: list[Author] = Field(default_factory=list)
    publication_year: int | None = None
    publication_date: str | None = None
    journal: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    doi: str | None = None
    isbn: str | None = None
    issn: str | None = None
    abstract: str | None = None
    url: str | None = None
    collections: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    pdf_path: Path | None = None
    pdf_attachment_key: str | None = None
    date_added: datetime
    date_modified: datetime
    indexed_at: datetime | None = None

    @field_validator("publication_date", mode="before")
    @classmethod
    def parse_publication_date(cls, v):
        """Clean publication date string."""
        if v is None:
            return None
        return str(v).strip() if v else None

    @field_validator("pdf_path", mode="before")
    @classmethod
    def validate_pdf_path(cls, v):
        """Convert string to Path if needed."""
        if v is None:
            return None
        if isinstance(v, str):
            return Path(v)
        return v

    @field_validator("title", mode="before")
    @classmethod
    def clean_title(cls, v):
        """Ensure title is a non-empty string."""
        if not v:
            return "Untitled"
        return str(v).strip()

    def model_post_init(self, __context):
        """Initialize computed fields after model creation."""
        # Use zotero_key as stable paper_id if not explicitly set
        if not self.paper_id:
            self.paper_id = self.zotero_key
        # Extract publication year from date if not set
        if self.publication_year is None and self.publication_date:
            self.publication_year = self._extract_year(self.publication_date)

    @staticmethod
    def _extract_year(date_str: str) -> int | None:
        """Extract year from various date formats."""
        if not date_str:
            return None

        # Try to find a 4-digit year
        year_match = re.search(r"\b(19|20)\d{2}\b", date_str)
        if year_match:
            return int(year_match.group())

        return None

    @property
    def first_author(self) -> Author | None:
        """Get the first author."""
        return self.authors[0] if self.authors else None

    @property
    def author_string(self) -> str:
        """Get formatted author string."""
        if not self.authors:
            return "Unknown"
        if len(self.authors) == 1:
            return self.authors[0].full_name
        if len(self.authors) == 2:
            return f"{self.authors[0].full_name} and {self.authors[1].full_name}"
        return f"{self.authors[0].full_name} et al."

    @property
    def citation_key(self) -> str:
        """Generate a citation key for the paper."""
        first_author = self.first_author
        author_part = first_author.last_name if first_author else "Unknown"
        year_part = str(self.publication_year) if self.publication_year else "n.d."
        # Clean author name for key
        author_part = re.sub(r"[^a-zA-Z]", "", author_part)
        return f"{author_part}{year_part}"

    def to_index_dict(self) -> dict:
        """Convert to dictionary suitable for indexing."""
        return {
            "paper_id": self.paper_id,
            "zotero_key": self.zotero_key,
            "title": self.title,
            "authors": [a.model_dump() for a in self.authors],
            "author_string": self.author_string,
            "publication_year": self.publication_year,
            "journal": self.journal,
            "doi": self.doi,
            "abstract": self.abstract,
            "collections": self.collections,
            "tags": self.tags,
            "item_type": self.item_type,
            "pdf_path": str(self.pdf_path) if self.pdf_path else None,
            "date_added": self.date_added.isoformat() if self.date_added else None,
            "indexed_at": self.indexed_at.isoformat() if self.indexed_at else None,
        }
