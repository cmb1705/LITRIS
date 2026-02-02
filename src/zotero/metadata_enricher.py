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
    - Semantic Scholar API for academic paper search
    - OpenAlex API for comprehensive academic metadata
    - Open Library API for ISBN lookup
    """

    CROSSREF_API_BASE = "https://api.crossref.org"
    OPENLIBRARY_API_BASE = "https://openlibrary.org"
    SEMANTIC_SCHOLAR_API_BASE = "https://api.semanticscholar.org/graph/v1"
    OPENALEX_API_BASE = "https://api.openalex.org"

    # Rate limiting
    MIN_REQUEST_INTERVAL = 0.5  # seconds between requests (Crossref/OpenAlex)
    SEMANTIC_SCHOLAR_INTERVAL = 1.1  # Semantic Scholar requires ~1 req/sec

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

    def _clean_title_for_search(self, title: str) -> str:
        """Clean title for better API search results.

        Removes common artifacts from filename-parsed titles:
        - Leading numbers and underscores (e.g., "1_1999")
        - Year prefixes/suffixes
        - Author name prefixes (before first dash or colon)

        Args:
            title: Raw title string.

        Returns:
            Cleaned title suitable for API search.
        """
        if not title:
            return ""

        cleaned = title

        # Remove leading number patterns like "1_", "01_", "1_1999"
        cleaned = re.sub(r"^\d+[_\s]+", "", cleaned)

        # Remove leading year patterns like "1999_", "1999 "
        cleaned = re.sub(r"^(19|20)\d{2}[_\s]+", "", cleaned)

        # Remove trailing year in parentheses or after dash
        cleaned = re.sub(r"\s*[-–]\s*(19|20)\d{2}\s*$", "", cleaned)
        cleaned = re.sub(r"\s*\((19|20)\d{2}\)\s*$", "", cleaned)

        # If title has "Author - Title" pattern, extract just the title part
        # Look for patterns like "Bowker_Star - Sorting Things Out"
        if " - " in cleaned:
            parts = cleaned.split(" - ", 1)
            if len(parts) == 2:
                # Check if first part looks like author names (short, has underscores/caps)
                first_part = parts[0].strip()
                second_part = parts[1].strip()
                # If second part is longer and more title-like, use it
                if len(second_part) > len(first_part) and len(second_part) > 10:
                    cleaned = second_part

        # Replace underscores with spaces
        cleaned = cleaned.replace("_", " ")

        # Normalize whitespace
        cleaned = " ".join(cleaned.split())

        return cleaned.strip()

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

        # Clean title for better matching
        clean_title = self._clean_title_for_search(title)
        if not clean_title:
            clean_title = title

        # Build query
        query_parts = [f'query.title="{clean_title}"']
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

            # Find best match, preferring year-matched results
            best_match = None
            best_score = 0.0
            best_year_match = None
            best_year_score = 0.0

            for item in items:
                item_title = ""
                if item.get("title"):
                    item_title = item["title"][0] if isinstance(item["title"], list) else item["title"]

                if not item_title:
                    continue

                score = self._title_similarity(clean_title, item_title)

                # Check year match
                item_year = None
                if item.get("published-print"):
                    date_parts = item["published-print"].get("date-parts", [[]])
                    if date_parts and date_parts[0]:
                        item_year = date_parts[0][0]
                elif item.get("published-online"):
                    date_parts = item["published-online"].get("date-parts", [[]])
                    if date_parts and date_parts[0]:
                        item_year = date_parts[0][0]

                # Track best year-matched result separately
                if year and item_year == year:
                    if score > best_year_score:
                        best_year_score = score
                        best_year_match = item

                if score > best_score:
                    best_score = score
                    best_match = item

            # Prefer year-matched result if it's reasonably close in score
            # This prevents matching wrong paper with same title from different year
            if best_year_match and best_year_score >= self.title_match_threshold - 0.1:
                logger.info(f"Found Crossref match for '{clean_title[:50]}...' (score: {best_year_score:.2f}, year-matched)")
                return best_year_match
            elif best_match and best_score >= self.title_match_threshold:
                logger.info(f"Found Crossref match for '{clean_title[:50]}...' (score: {best_score:.2f})")
                return best_match
            else:
                logger.debug(f"No Crossref match for '{clean_title[:50]}...' (best score: {best_score:.2f})")
                return None

        except Exception as e:
            logger.warning(f"Failed to search Crossref for title: {e}")
            return None

    def search_openalex(
        self,
        title: str,
        author: str | None = None,
        year: int | None = None,
    ) -> dict | None:
        """Search OpenAlex for a work by title.

        OpenAlex often has better coverage than Crossref for books and older works.

        Args:
            title: Title to search for.
            author: Optional author name to filter.
            year: Optional publication year to filter.

        Returns:
            Best matching metadata dict or None.
        """
        self._rate_limit()

        clean_title = self._clean_title_for_search(title)
        if not clean_title:
            clean_title = title

        # Build search query
        search_query = clean_title
        if author:
            # Extract last name
            author_parts = author.split()
            if author_parts:
                last_name = author_parts[-1] if "," not in author else author.split(",")[0]
                search_query = f"{clean_title} {last_name}"

        params = {
            "search": search_query,
            "per-page": 5,
            "mailto": self.email or "litris@example.com",
        }

        # Add year filter if available
        if year:
            params["filter"] = f"publication_year:{year}"

        url = f"{self.OPENALEX_API_BASE}/works"

        try:
            response = self.session.get(url, params=params, timeout=30)
            if response.status_code != 200:
                logger.warning(f"OpenAlex search error: {response.status_code}")
                return None

            data = response.json()
            results = data.get("results", [])

            if not results:
                return None

            # Find best match, preferring year-matched results
            best_match = None
            best_score = 0.0
            best_year_match = None
            best_year_score = 0.0

            for work in results:
                work_title = work.get("display_name", "")
                if not work_title:
                    continue

                score = self._title_similarity(clean_title, work_title)

                # Track best year-matched result separately
                if year and work.get("publication_year") == year:
                    if score > best_year_score:
                        best_year_score = score
                        best_year_match = work

                if score > best_score:
                    best_score = score
                    best_match = work

            # Prefer year-matched result if it's reasonably close in score
            if best_year_match and best_year_score >= self.title_match_threshold - 0.1:
                logger.info(f"Found OpenAlex match for '{clean_title[:50]}...' (score: {best_year_score:.2f}, year-matched)")
                return self._parse_openalex_work(best_year_match)
            elif best_match and best_score >= self.title_match_threshold:
                logger.info(f"Found OpenAlex match for '{clean_title[:50]}...' (score: {best_score:.2f})")
                return self._parse_openalex_work(best_match)
            else:
                logger.debug(f"No OpenAlex match for '{clean_title[:50]}...' (best score: {best_score:.2f})")
                return None

        except Exception as e:
            logger.warning(f"Failed to search OpenAlex for title: {e}")
            return None

    def _parse_openalex_work(self, work: dict) -> dict:
        """Parse OpenAlex work into Crossref-compatible format.

        Args:
            work: OpenAlex work dictionary.

        Returns:
            Metadata in similar format to Crossref response.
        """
        result = {
            "DOI": work.get("doi", "").replace("https://doi.org/", "") if work.get("doi") else None,
            "title": [work.get("display_name")] if work.get("display_name") else [],
            "author": [],
            "publisher": None,
            "container-title": [],
            "volume": None,
            "issue": None,
            "page": None,
            "abstract": None,
            "ISSN": None,
            "URL": work.get("id"),
        }

        # Publication year
        if work.get("publication_year"):
            result["published-print"] = {
                "date-parts": [[work["publication_year"]]]
            }

        # Authors
        for authorship in work.get("authorships", []):
            author_info = authorship.get("author", {})
            if author_info.get("display_name"):
                name_parts = author_info["display_name"].rsplit(" ", 1)
                if len(name_parts) == 2:
                    result["author"].append({
                        "given": name_parts[0],
                        "family": name_parts[1],
                    })
                else:
                    result["author"].append({"family": author_info["display_name"]})

        # Venue/Journal
        primary_location = work.get("primary_location", {})
        source = primary_location.get("source", {})
        if source.get("display_name"):
            result["container-title"] = [source["display_name"]]
            result["publisher"] = source.get("host_organization_name")

        # Biblio info
        biblio = work.get("biblio", {})
        result["volume"] = biblio.get("volume")
        result["issue"] = biblio.get("issue")
        if biblio.get("first_page"):
            result["page"] = biblio["first_page"]
            if biblio.get("last_page"):
                result["page"] = f"{biblio['first_page']}-{biblio['last_page']}"

        return result

    def search_semantic_scholar(
        self,
        title: str,
        author: str | None = None,
        year: int | None = None,
    ) -> dict | None:
        """Search Semantic Scholar for a paper by title.

        Semantic Scholar has excellent coverage of academic papers.

        Args:
            title: Title to search for.
            author: Optional author name to filter.
            year: Optional publication year to filter.

        Returns:
            Best matching metadata dict or None.
        """
        # Semantic Scholar has stricter rate limits
        elapsed = time.time() - self._last_request_time
        if elapsed < self.SEMANTIC_SCHOLAR_INTERVAL:
            time.sleep(self.SEMANTIC_SCHOLAR_INTERVAL - elapsed)
        self._last_request_time = time.time()

        clean_title = self._clean_title_for_search(title)
        if not clean_title:
            clean_title = title

        # Build search query
        search_query = clean_title
        if author:
            author_parts = author.split()
            if author_parts:
                last_name = author_parts[-1] if "," not in author else author.split(",")[0]
                search_query = f"{clean_title} {last_name}"

        params = {
            "query": search_query,
            "limit": 5,
            "fields": "paperId,externalIds,title,authors,year,venue,abstract,publicationVenue",
        }

        # Add year filter
        if year:
            params["year"] = str(year)

        url = f"{self.SEMANTIC_SCHOLAR_API_BASE}/paper/search"

        try:
            response = self.session.get(url, params=params, timeout=30)
            if response.status_code != 200:
                logger.warning(f"Semantic Scholar search error: {response.status_code}")
                return None

            data = response.json()
            papers = data.get("data", [])

            if not papers:
                return None

            # Find best match, preferring year-matched results
            best_match = None
            best_score = 0.0
            best_year_match = None
            best_year_score = 0.0

            for paper in papers:
                paper_title = paper.get("title", "")
                if not paper_title:
                    continue

                score = self._title_similarity(clean_title, paper_title)

                # Track best year-matched result separately
                if year and paper.get("year") == year:
                    if score > best_year_score:
                        best_year_score = score
                        best_year_match = paper

                if score > best_score:
                    best_score = score
                    best_match = paper

            # Prefer year-matched result if it's reasonably close in score
            if best_year_match and best_year_score >= self.title_match_threshold - 0.1:
                logger.info(f"Found Semantic Scholar match for '{clean_title[:50]}...' (score: {best_year_score:.2f}, year-matched)")
                return self._parse_semantic_scholar_paper(best_year_match)
            elif best_match and best_score >= self.title_match_threshold:
                logger.info(f"Found Semantic Scholar match for '{clean_title[:50]}...' (score: {best_score:.2f})")
                return self._parse_semantic_scholar_paper(best_match)
            else:
                logger.debug(f"No Semantic Scholar match for '{clean_title[:50]}...' (best score: {best_score:.2f})")
                return None

        except Exception as e:
            logger.warning(f"Failed to search Semantic Scholar: {e}")
            return None

    def _parse_semantic_scholar_paper(self, paper: dict) -> dict:
        """Parse Semantic Scholar paper into Crossref-compatible format.

        Args:
            paper: Semantic Scholar paper dictionary.

        Returns:
            Metadata in similar format to Crossref response.
        """
        result = {
            "DOI": None,
            "title": [paper.get("title")] if paper.get("title") else [],
            "author": [],
            "publisher": None,
            "container-title": [],
            "volume": None,
            "issue": None,
            "page": None,
            "abstract": paper.get("abstract"),
            "ISSN": None,
            "URL": f"https://www.semanticscholar.org/paper/{paper.get('paperId')}",
        }

        # DOI from external IDs
        external_ids = paper.get("externalIds", {})
        if external_ids.get("DOI"):
            result["DOI"] = external_ids["DOI"]

        # Publication year
        if paper.get("year"):
            result["published-print"] = {
                "date-parts": [[paper["year"]]]
            }

        # Authors
        for author in paper.get("authors", []):
            if author.get("name"):
                name_parts = author["name"].rsplit(" ", 1)
                if len(name_parts) == 2:
                    result["author"].append({
                        "given": name_parts[0],
                        "family": name_parts[1],
                    })
                else:
                    result["author"].append({"family": author["name"]})

        # Venue/Journal
        if paper.get("venue"):
            result["container-title"] = [paper["venue"]]
        elif paper.get("publicationVenue"):
            venue = paper["publicationVenue"]
            if venue.get("name"):
                result["container-title"] = [venue["name"]]

        return result

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

        # Strategy 3: Title search (try multiple APIs)
        if extracted.title:
            first_author = extracted.authors[0] if extracted.authors else None

            # 3a: Try Crossref title search
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
                result.enrichment_notes = f"Found via Crossref title search: {parsed.get('doi')}"
                return result

            # 3b: Try OpenAlex search
            data = self.search_openalex(
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
                result.enrichment_source = "openalex_title_search"
                result.enrichment_confidence = 0.70
                result.enrichment_notes = f"Found via OpenAlex title search: {parsed.get('doi')}"
                return result

            # 3c: Try Semantic Scholar search
            data = self.search_semantic_scholar(
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
                result.enrichment_source = "semantic_scholar_title_search"
                result.enrichment_confidence = 0.70
                result.enrichment_notes = f"Found via Semantic Scholar: {parsed.get('doi')}"
                return result

        # No enrichment possible
        result.enrichment_source = "none"
        result.enrichment_confidence = extracted.confidence
        result.enrichment_notes = "No enrichment source available"
        return result
