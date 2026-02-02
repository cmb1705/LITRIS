"""Metadata enrichment via external APIs.

Enriches extracted metadata using Crossref and Open Library APIs.
Provides DOI lookup, ISBN lookup, and title-based search.
"""

import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher

import requests

from src.utils.logging_config import get_logger
from src.zotero.orphan_metadata_extractor import ExtractedMetadata, MetadataSource

logger = get_logger(__name__)


@dataclass
class EnrichedMetadata:
    """Metadata enriched from external APIs."""

    # Original extraction
    original: ExtractedMetadata

    # Enriched fields
    doi: str | None = None
    title: str | None = None
    authors: list[str] | None = None
    publication_year: int | None = None
    journal: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    publisher: str | None = None
    abstract: str | None = None
    issn: str | None = None
    isbn: str | None = None
    url: str | None = None

    # Enrichment metadata
    enrichment_source: str | None = None
    enrichment_confidence: float = 0.0
    enrichment_notes: str = ""

    @property
    def best_title(self) -> str | None:
        """Get the best available title."""
        return self.title or self.original.title

    @property
    def best_authors(self) -> list[str]:
        """Get the best available authors."""
        return self.authors or self.original.authors or []

    @property
    def best_year(self) -> int | None:
        """Get the best available year."""
        return self.publication_year or self.original.publication_year

    @property
    def best_doi(self) -> str | None:
        """Get the best available DOI."""
        return self.doi or self.original.doi


class MetadataEnricher:
    """Enriches metadata using external APIs.

    Supports:
    - Crossref API for DOI lookup and title search
    - Open Library API for ISBN lookup
    """

    CROSSREF_API_BASE = "https://api.crossref.org"
    OPENLIBRARY_API_BASE = "https://openlibrary.org"

    # Rate limiting
    MIN_REQUEST_INTERVAL = 0.5  # seconds between requests

    def __init__(
        self,
        email: str | None = None,
        title_match_threshold: float = 0.85,
    ):
        """Initialize the enricher.

        Args:
            email: Email for Crossref polite pool (faster rate limits).
            title_match_threshold: Minimum similarity for title matching.
        """
        self.email = email
        self.title_match_threshold = title_match_threshold
        self._last_request_time = 0.0

        # Set up session with headers
        self.session = requests.Session()
        headers = {
            "User-Agent": "LITRIS/1.0 (Literature Review Indexing System; mailto:litris@example.com)",
        }
        if email:
            headers["User-Agent"] = f"LITRIS/1.0 (mailto:{email})"
        self.session.headers.update(headers)

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles.

        Args:
            title1: First title.
            title2: Second title.

        Returns:
            Similarity score 0.0 to 1.0.
        """
        # Normalize titles
        def normalize(t: str) -> str:
            t = t.lower()
            t = re.sub(r"[^\w\s]", "", t)
            t = " ".join(t.split())
            return t

        t1 = normalize(title1)
        t2 = normalize(title2)

        return SequenceMatcher(None, t1, t2).ratio()

    def lookup_doi(self, doi: str) -> dict | None:
        """Look up metadata for a DOI via Crossref.

        Args:
            doi: DOI string.

        Returns:
            Metadata dict or None if not found.
        """
        self._rate_limit()

        url = f"{self.CROSSREF_API_BASE}/works/{doi}"
        try:
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data.get("message", {})
            elif response.status_code == 404:
                logger.debug(f"DOI not found: {doi}")
                return None
            else:
                logger.warning(f"Crossref API error for DOI {doi}: {response.status_code}")
                return None
        except Exception as e:
            logger.warning(f"Failed to lookup DOI {doi}: {e}")
            return None

    def lookup_isbn(self, isbn: str) -> dict | None:
        """Look up metadata for an ISBN via Open Library.

        Args:
            isbn: ISBN string (10 or 13 digits).

        Returns:
            Metadata dict or None if not found.
        """
        self._rate_limit()

        url = f"{self.OPENLIBRARY_API_BASE}/isbn/{isbn}.json"
        try:
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                logger.debug(f"ISBN not found: {isbn}")
                return None
            else:
                logger.warning(f"Open Library API error for ISBN {isbn}: {response.status_code}")
                return None
        except Exception as e:
            logger.warning(f"Failed to lookup ISBN {isbn}: {e}")
            return None

    def search_by_title(
        self,
        title: str,
        author: str | None = None,
        year: int | None = None,
    ) -> dict | None:
        """Search Crossref for a work by title.

        Args:
            title: Title to search for.
            author: Optional author name to filter.
            year: Optional publication year to filter.

        Returns:
            Best matching metadata dict or None.
        """
        self._rate_limit()

        # Build query
        query_parts = [f'query.title="{title}"']
        if author:
            # Extract last name for search
            author_parts = author.split()
            if author_parts:
                last_name = author_parts[-1] if "," not in author else author.split(",")[0]
                query_parts.append(f"query.author={last_name}")

        query = "&".join(query_parts)
        url = f"{self.CROSSREF_API_BASE}/works?{query}&rows=5"

        try:
            response = self.session.get(url, timeout=30)
            if response.status_code != 200:
                logger.warning(f"Crossref search error: {response.status_code}")
                return None

            data = response.json()
            items = data.get("message", {}).get("items", [])

            if not items:
                return None

            # Find best match
            best_match = None
            best_score = 0.0

            for item in items:
                item_title = ""
                if item.get("title"):
                    item_title = item["title"][0] if isinstance(item["title"], list) else item["title"]

                if not item_title:
                    continue

                score = self._title_similarity(title, item_title)

                # Bonus for matching year
                if year:
                    item_year = None
                    if item.get("published-print"):
                        date_parts = item["published-print"].get("date-parts", [[]])
                        if date_parts and date_parts[0]:
                            item_year = date_parts[0][0]
                    elif item.get("published-online"):
                        date_parts = item["published-online"].get("date-parts", [[]])
                        if date_parts and date_parts[0]:
                            item_year = date_parts[0][0]

                    if item_year == year:
                        score += 0.1  # Boost for year match

                if score > best_score:
                    best_score = score
                    best_match = item

            if best_match and best_score >= self.title_match_threshold:
                logger.info(f"Found match for '{title[:50]}...' (score: {best_score:.2f})")
                return best_match
            else:
                logger.debug(f"No good match for '{title[:50]}...' (best score: {best_score:.2f})")
                return None

        except Exception as e:
            logger.warning(f"Failed to search Crossref for title: {e}")
            return None

    def _parse_crossref_metadata(self, data: dict) -> dict:
        """Parse Crossref API response into standard format.

        Args:
            data: Crossref API response.

        Returns:
            Standardized metadata dict.
        """
        result = {
            "doi": data.get("DOI"),
            "title": None,
            "authors": [],
            "year": None,
            "journal": None,
            "volume": data.get("volume"),
            "issue": data.get("issue"),
            "pages": data.get("page"),
            "publisher": data.get("publisher"),
            "abstract": None,
            "issn": None,
            "url": data.get("URL"),
        }

        # Title
        if data.get("title"):
            result["title"] = data["title"][0] if isinstance(data["title"], list) else data["title"]

        # Authors
        for author in data.get("author", []):
            name_parts = []
            if author.get("given"):
                name_parts.append(author["given"])
            if author.get("family"):
                name_parts.append(author["family"])
            if name_parts:
                result["authors"].append(" ".join(name_parts))

        # Year
        if data.get("published-print"):
            date_parts = data["published-print"].get("date-parts", [[]])
            if date_parts and date_parts[0]:
                result["year"] = date_parts[0][0]
        elif data.get("published-online"):
            date_parts = data["published-online"].get("date-parts", [[]])
            if date_parts and date_parts[0]:
                result["year"] = date_parts[0][0]
        elif data.get("created"):
            date_parts = data["created"].get("date-parts", [[]])
            if date_parts and date_parts[0]:
                result["year"] = date_parts[0][0]

        # Journal
        if data.get("container-title"):
            result["journal"] = (
                data["container-title"][0]
                if isinstance(data["container-title"], list)
                else data["container-title"]
            )

        # Abstract (may contain HTML)
        if data.get("abstract"):
            abstract = data["abstract"]
            # Strip HTML tags
            abstract = re.sub(r"<[^>]+>", "", abstract)
            result["abstract"] = abstract[:2000]  # Limit length

        # ISSN
        if data.get("ISSN"):
            result["issn"] = data["ISSN"][0] if isinstance(data["ISSN"], list) else data["ISSN"]

        return result

    def _parse_openlibrary_metadata(self, data: dict) -> dict:
        """Parse Open Library API response into standard format.

        Args:
            data: Open Library API response.

        Returns:
            Standardized metadata dict.
        """
        result = {
            "title": data.get("title"),
            "authors": [],
            "year": None,
            "publisher": None,
            "isbn": None,
        }

        # Authors (need additional API call for full names)
        # For now, just note author keys exist
        if data.get("authors"):
            for author_ref in data["authors"]:
                if isinstance(author_ref, dict) and author_ref.get("key"):
                    # Would need to fetch /authors/{key}.json
                    result["authors"].append(author_ref["key"].replace("/authors/", ""))

        # Publication year
        if data.get("publish_date"):
            year_match = re.search(r"(19|20)\d{2}", data["publish_date"])
            if year_match:
                result["year"] = int(year_match.group())

        # Publisher
        if data.get("publishers"):
            result["publisher"] = data["publishers"][0] if data["publishers"] else None

        # ISBN
        if data.get("isbn_13"):
            result["isbn"] = data["isbn_13"][0] if data["isbn_13"] else None
        elif data.get("isbn_10"):
            result["isbn"] = data["isbn_10"][0] if data["isbn_10"] else None

        return result

    def enrich(self, extracted: ExtractedMetadata) -> EnrichedMetadata:
        """Enrich extracted metadata using external APIs.

        Strategy:
        1. If DOI found, look it up in Crossref
        2. If ISBN found, look it up in Open Library
        3. Otherwise, search Crossref by title/author

        Args:
            extracted: Extracted metadata from PDF.

        Returns:
            Enriched metadata.
        """
        result = EnrichedMetadata(original=extracted)

        # Strategy 1: DOI lookup
        if extracted.doi:
            data = self.lookup_doi(extracted.doi)
            if data:
                parsed = self._parse_crossref_metadata(data)
                result.doi = parsed.get("doi")
                result.title = parsed.get("title")
                result.authors = parsed.get("authors")
                result.publication_year = parsed.get("year")
                result.journal = parsed.get("journal")
                result.volume = parsed.get("volume")
                result.issue = parsed.get("issue")
                result.pages = parsed.get("pages")
                result.publisher = parsed.get("publisher")
                result.abstract = parsed.get("abstract")
                result.issn = parsed.get("issn")
                result.url = parsed.get("url")
                result.enrichment_source = "crossref_doi"
                result.enrichment_confidence = 0.95
                result.enrichment_notes = f"Enriched via DOI lookup: {extracted.doi}"
                return result

        # Strategy 2: ISBN lookup
        if extracted.isbn:
            data = self.lookup_isbn(extracted.isbn)
            if data:
                parsed = self._parse_openlibrary_metadata(data)
                result.title = parsed.get("title")
                result.authors = parsed.get("authors")
                result.publication_year = parsed.get("year")
                result.publisher = parsed.get("publisher")
                result.isbn = parsed.get("isbn") or extracted.isbn
                result.enrichment_source = "openlibrary_isbn"
                result.enrichment_confidence = 0.85
                result.enrichment_notes = f"Enriched via ISBN lookup: {extracted.isbn}"
                return result

        # Strategy 3: Title search
        if extracted.title:
            first_author = extracted.authors[0] if extracted.authors else None
            data = self.search_by_title(
                extracted.title,
                author=first_author,
                year=extracted.publication_year,
            )
            if data:
                parsed = self._parse_crossref_metadata(data)
                result.doi = parsed.get("doi")
                result.title = parsed.get("title")
                result.authors = parsed.get("authors")
                result.publication_year = parsed.get("year")
                result.journal = parsed.get("journal")
                result.volume = parsed.get("volume")
                result.issue = parsed.get("issue")
                result.pages = parsed.get("pages")
                result.publisher = parsed.get("publisher")
                result.abstract = parsed.get("abstract")
                result.issn = parsed.get("issn")
                result.url = parsed.get("url")
                result.enrichment_source = "crossref_title_search"
                result.enrichment_confidence = 0.75
                result.enrichment_notes = f"Found via title search: {parsed.get('doi')}"
                return result

        # No enrichment possible
        result.enrichment_source = "none"
        result.enrichment_confidence = extracted.confidence
        result.enrichment_notes = "No enrichment source available"
        return result
