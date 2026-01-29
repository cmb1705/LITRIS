#!/usr/bin/env python
"""Launch the local web UI for LITRIS search using Streamlit."""

from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
from collections.abc import Iterable
from datetime import datetime
from html import escape
from pathlib import Path
from typing import cast

import streamlit as st
import streamlit.components.v1 as components

project_root = Path(__file__).resolve().parent.parent

# Optional: Citation network visualization
PYVIS_AVAILABLE = False
try:
    import networkx as nx
    from pyvis.network import Network

    PYVIS_AVAILABLE = True
except ImportError:
    pass
sys.path.insert(0, str(project_root))

from src.config import Config
from src.indexing.embeddings import CHUNK_TYPES, ChunkType
from src.query.federated import FederatedResult, FederatedSearchEngine
from src.query.retrieval import (
    OutputFormat,
    format_paper_detail,
    format_results,
    format_summary,
    save_results,
)
from src.query.search import EnrichedResult, SearchEngine

DEFAULT_TOP_K = 10
MAX_TOP_K = 50
SORT_OPTIONS = {
    "Relevance": "relevance",
    "Year: Newest first": "year_desc",
    "Year: Oldest first": "year_asc",
    "Title: A-Z": "title_asc",
    "Title: Z-A": "title_desc",
}

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Spectral:wght@400;600&display=swap');

:root {
  --bg: #f6f1ea;
  --bg-dark: #e8dfd2;
  --ink: #1f1b17;
  --muted: #4a4742;
  --accent: #c26a3b;
  --accent-soft: #f1d8c6;
  --card: rgba(255, 255, 255, 0.78);
  --stroke: rgba(74, 71, 66, 0.16);
  --shadow: 0 18px 40px rgba(28, 24, 19, 0.12);
}

html, body, [class*="css"]  {
  font-family: "Spectral", serif;
  color: var(--ink);
}

h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
  font-family: "Space Grotesk", sans-serif;
  letter-spacing: -0.02em;
}

div[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1200px 400px at 20% 0%, rgba(194, 106, 59, 0.15), transparent),
    radial-gradient(900px 300px at 90% 10%, rgba(62, 96, 83, 0.12), transparent),
    linear-gradient(180deg, var(--bg) 0%, var(--bg-dark) 100%);
}

div[data-testid="stSidebar"] {
  background: rgba(246, 241, 234, 0.85);
  border-right: 1px solid var(--stroke);
}

.app-hero {
  background: rgba(255, 255, 255, 0.62);
  border: 1px solid var(--stroke);
  border-radius: 24px;
  padding: 24px 28px;
  box-shadow: var(--shadow);
  margin-bottom: 24px;
  animation: fadeIn 0.6s ease both;
}

.hero-eyebrow {
  text-transform: uppercase;
  font-size: 12px;
  letter-spacing: 0.18em;
  color: var(--muted);
  font-family: "Space Grotesk", sans-serif;
}

.hero-title {
  font-size: 34px;
  margin: 8px 0 10px 0;
  color: var(--ink);
}

.hero-subtitle {
  font-size: 16px;
  color: var(--muted);
  margin: 0;
}

.result-card {
  background: var(--card);
  border: 1px solid var(--stroke);
  border-radius: 20px;
  padding: 18px 20px;
  box-shadow: var(--shadow);
  margin-bottom: 16px;
  animation: rise 0.45s ease both;
}

.result-title {
  font-family: "Space Grotesk", sans-serif;
  font-size: 18px;
  margin-bottom: 6px;
}

.result-meta {
  color: var(--muted);
  font-size: 13px;
  margin-bottom: 8px;
}

.result-tags {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 8px;
}

.result-tag {
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  background: var(--accent-soft);
  color: var(--ink);
  border: 1px solid rgba(194, 106, 59, 0.25);
}

.matched-text {
  color: var(--muted);
  font-size: 14px;
  line-height: 1.5;
}

.detail-panel {
  background: rgba(255, 255, 255, 0.78);
  border: 1px solid var(--stroke);
  border-radius: 24px;
  padding: 20px 22px;
  box-shadow: var(--shadow);
  animation: fadeIn 0.4s ease both;
}

.detail-title {
  font-family: "Space Grotesk", sans-serif;
  font-size: 18px;
  margin-bottom: 8px;
}

button[kind="primary"] {
  background: var(--accent) !important;
  border: none !important;
}

button[kind="primary"] p {
  color: #fff !important;
}

@keyframes rise {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 900px) {
  .hero-title { font-size: 26px; }
  .result-card { padding: 16px; }
}
</style>
"""

STYLE_DARK = """
<style>
:root {
  --bg: #1a1a1a;
  --bg-dark: #121212;
  --ink: #e8e8e8;
  --muted: #a0a0a0;
  --accent: #e08050;
  --accent-soft: #3a2820;
  --card: rgba(40, 40, 40, 0.85);
  --stroke: rgba(255, 255, 255, 0.12);
  --shadow: 0 18px 40px rgba(0, 0, 0, 0.3);
}

div[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1200px 400px at 20% 0%, rgba(224, 128, 80, 0.08), transparent),
    radial-gradient(900px 300px at 90% 10%, rgba(80, 140, 120, 0.06), transparent),
    linear-gradient(180deg, var(--bg) 0%, var(--bg-dark) 100%);
}

div[data-testid="stSidebar"] {
  background: rgba(30, 30, 30, 0.95);
  border-right: 1px solid var(--stroke);
}

.app-hero {
  background: rgba(40, 40, 40, 0.75);
}

.detail-panel {
  background: rgba(40, 40, 40, 0.85);
}

.result-tag {
  background: var(--accent-soft);
  color: var(--ink);
  border: 1px solid rgba(224, 128, 80, 0.3);
}

mark {
  background: #665030 !important;
  color: #fff !important;
}
</style>
"""

KEYBOARD_SHORTCUTS_JS = """
<script>
(function() {
    // Avoid duplicate listeners
    if (window.litrisKeyboardInit) return;
    window.litrisKeyboardInit = true;

    let focusedCardIndex = 0;
    let lastCardCount = 0;

    function getResultCards() {
        return document.querySelectorAll('.result-card');
    }

    function resetNavIfCardsChanged() {
        const cards = getResultCards();
        if (cards.length !== lastCardCount) {
            focusedCardIndex = 0;
            lastCardCount = cards.length;
            // Clear any existing highlights
            cards.forEach(card => card.style.outline = 'none');
        }
    }

    function highlightCard(index) {
        const cards = getResultCards();
        cards.forEach((card, i) => {
            card.style.outline = i === index ? '2px solid var(--accent)' : 'none';
        });
        if (cards[index]) {
            cards[index].scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }

    // Reset navigation when result cards change (new search)
    const observer = new MutationObserver(resetNavIfCardsChanged);
    observer.observe(document.body, { childList: true, subtree: true });

    document.addEventListener('keydown', function(e) {
        // Skip if typing in input/textarea
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
            if (e.key === 'Escape') {
                e.target.blur();
            }
            return;
        }

        // Check if cards changed before processing navigation
        resetNavIfCardsChanged();
        const cards = getResultCards();

        switch(e.key) {
            case '/':
                // Focus search input
                e.preventDefault();
                const searchInput = document.querySelector('input[type="text"]');
                if (searchInput) searchInput.focus();
                break;
            case 'j':
                // Next result
                if (cards.length > 0) {
                    focusedCardIndex = Math.min(focusedCardIndex + 1, cards.length - 1);
                    highlightCard(focusedCardIndex);
                }
                break;
            case 'k':
                // Previous result
                if (cards.length > 0) {
                    focusedCardIndex = Math.max(focusedCardIndex - 1, 0);
                    highlightCard(focusedCardIndex);
                }
                break;
            case 'Escape':
                // Clear highlight
                cards.forEach(card => card.style.outline = 'none');
                focusedCardIndex = 0;
                break;
        }
    });
})();
</script>
"""


def inject_styles(dark_mode: bool = False) -> None:
    """Inject global CSS styles into the Streamlit app.

    Args:
        dark_mode: If True, apply dark theme overrides.
    """
    st.markdown(STYLE, unsafe_allow_html=True)
    if dark_mode:
        st.markdown(STYLE_DARK, unsafe_allow_html=True)
    # Inject keyboard shortcuts JavaScript
    st.markdown(KEYBOARD_SHORTCUTS_JS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_engine(
    config_path: str | None,
    use_federated: bool = False,
) -> tuple[SearchEngine | FederatedSearchEngine, Config, Path, Path]:
    """Load config and initialize the search engine.

    Args:
        config_path: Optional path to config.yaml.
        use_federated: Whether to use federated search engine.

    Returns:
        Tuple of (SearchEngine or FederatedSearchEngine, Config, index_dir, results_dir).
    """
    config = Config.load(config_path)
    index_dir = project_root / "data" / "index"
    results_dir = project_root / "data" / "query_results"
    chroma_dir = config.get_chroma_path()

    if not index_dir.exists():
        raise FileNotFoundError(
            f"Index directory not found: {index_dir}\n"
            "Run scripts/build_index.py first to create the index."
        )

    # Use federated search if enabled in config or requested
    if use_federated or config.federated.enabled:
        engine = FederatedSearchEngine(
            primary_index_dir=index_dir,
            config=config.federated,
            embedding_model=config.embeddings.model,
        )
    else:
        engine = SearchEngine(
            index_dir=index_dir,
            chroma_dir=chroma_dir,
            embedding_model=config.embeddings.model,
        )
    return engine, config, index_dir, results_dir


def load_filter_options(engine: SearchEngine) -> dict[str, object]:
    """Load filter options from the index.

    Args:
        engine: Initialized SearchEngine.

    Returns:
        Dictionary with collections, item types, and year range.
    """
    if "filter_options" not in st.session_state:
        st.session_state.filter_options = {
            "collections": sorted(engine.get_collections()),
            "item_types": sorted(engine.get_item_types()),
            "year_range": engine.get_year_range(),
        }
    return st.session_state.filter_options


def execute_search(
    engine: SearchEngine,
    query: str,
    top_k: int,
    chunk_types: list[ChunkType] | None,
    year_min: int | None,
    year_max: int | None,
    collections: list[str] | None,
    item_types: list[str] | None,
    include_extraction: bool,
    deduplicate_papers: bool,
) -> list[EnrichedResult]:
    """Run a semantic search with optional filters."""
    return engine.search(
        query=query,
        top_k=top_k,
        chunk_types=chunk_types,
        year_min=year_min,
        year_max=year_max,
        collections=collections,
        item_types=item_types,
        include_paper_data=True,
        include_extraction=include_extraction,
        deduplicate_papers=deduplicate_papers,
    )


def metadata_results_to_enriched(
    papers: list[dict],
    match_label: str,
) -> list[EnrichedResult]:
    """Convert metadata-only search results into enriched results."""
    enriched = []
    for paper in papers:
        year_value = paper.get("publication_year")
        year = int(year_value) if year_value and str(year_value).isdigit() else None
        authors = paper.get("author_string", "") or ""
        if not authors and isinstance(paper.get("authors"), list):
            author_names = []
            for author in paper["authors"]:
                first = str(author.get("first_name") or "").strip()
                last = str(author.get("last_name") or "").strip()
                full = str(author.get("full_name") or "").strip()
                name = f"{first} {last}".strip() if first or last else full
                if name:
                    author_names.append(name)
            authors = ", ".join(author_names)
        if not authors:
            authors = "Unknown"

        enriched.append(
            EnrichedResult(
                paper_id=paper.get("paper_id", ""),
                title=paper.get("title", "Unknown"),
                authors=authors,
                year=year,
                collections=paper.get("collections", []) or [],
                item_type=paper.get("item_type", "") or "",
                chunk_type="metadata",
                matched_text=match_label,
                score=1.0,
                paper_data=paper,
                extraction_data={},
            )
        )
    return enriched


def execute_metadata_search(
    engine: SearchEngine,
    title_contains: str | None,
    author_contains: str | None,
    year_min: int | None,
    year_max: int | None,
    collections: list[str] | None,
    item_types: list[str] | None,
    top_k: int,
    match_label: str,
) -> list[EnrichedResult]:
    """Run a metadata-only search and normalize results."""
    collection_filter = collections[0] if collections and len(collections) == 1 else None
    item_type_filter = item_types[0] if item_types and len(item_types) == 1 else None

    results = engine.search_by_metadata(
        title_contains=title_contains,
        author_contains=author_contains,
        year_min=year_min,
        year_max=year_max,
        collection=collection_filter,
        item_type=item_type_filter,
    )

    if collections and collection_filter is None:
        results = [
            paper
            for paper in results
            if any(c in (paper.get("collections", []) or []) for c in collections)
        ]
    if item_types and item_type_filter is None:
        results = [
            paper
            for paper in results
            if paper.get("item_type") in item_types
        ]

    return metadata_results_to_enriched(results[:top_k], match_label)


def normalize_chunk_filter(selected: Iterable[str]) -> list[ChunkType] | None:
    """Normalize chunk type selections into a search filter."""
    selected_list = [c for c in selected if c in CHUNK_TYPES]
    if not selected_list or len(selected_list) == len(CHUNK_TYPES):
        return None
    return cast(list[ChunkType], selected_list)


def sort_results(results: list[EnrichedResult], sort_key: str) -> list[EnrichedResult]:
    """Sort results list based on selected key."""
    if sort_key == "year_desc":
        return sorted(results, key=lambda r: (r.year is None, -(r.year or 0)))
    if sort_key == "year_asc":
        return sorted(results, key=lambda r: (r.year is None, r.year or 0))
    if sort_key == "title_asc":
        return sorted(results, key=lambda r: (not (r.title or ""), (r.title or "").lower()))
    if sort_key == "title_desc":
        return sorted(
            results,
            key=lambda r: (not (r.title or ""), (r.title or "").lower()),
            reverse=True,
        )
    return results


def resolve_detail_markdown(
    engine: SearchEngine,
    result: EnrichedResult,
    detail_extraction: bool,
) -> str:
    """Build the detail panel markdown for a selected result."""
    paper_data = result.paper_data or {}
    extraction = result.extraction_data or None

    if detail_extraction:
        if not extraction:
            combined = engine.get_paper(result.paper_id) or {}
            paper_data = combined.get("paper", paper_data)
            extraction = combined.get("extraction")
    elif not paper_data:
        combined = engine.get_paper(result.paper_id) or {}
        paper_data = combined.get("paper", {})

    return format_paper_detail(paper_data, extraction if detail_extraction else None)


def save_export(
    results: list[EnrichedResult],
    query: str,
    export_format: OutputFormat,
    results_dir: Path,
    include_extraction: bool,
) -> Path:
    """Save results to disk and return the output path."""
    return save_results(
        results=results,
        query=query,
        output_dir=results_dir,
        output_format=export_format,
        filename_prefix="search",
        include_extraction=include_extraction,
    )


def sanitize_csv_field(field: str) -> str:
    """Sanitize field for CSV export to prevent formula injection."""
    if not field:
        return ""
    field = str(field)
    if field and field[0] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + field
    return field


def sanitize_filename_slug(value: str, max_length: int = 30) -> str:
    """Sanitize user-provided string for safe filename slugs."""
    import re

    if not value:
        return "search"

    slug = value.strip().replace(" ", "-")
    slug = re.sub(r"[^A-Za-z0-9-]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug)
    slug = slug.strip("-")
    if not slug:
        slug = "search"
    return slug[:max_length]


def is_safe_http_url(value: str) -> bool:
    """Return True for http(s) URLs with a network location."""
    if not value:
        return False
    from urllib.parse import urlparse

    parsed = urlparse(value.strip())
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def normalize_doi_url(doi: str) -> str | None:
    """Normalize DOI to a safe URL or return None if invalid."""
    import re
    from urllib.parse import urlparse

    if not doi:
        return None

    text = doi.strip()
    lower = text.lower()
    if lower.startswith("doi:"):
        text = text[4:].strip()
        lower = text.lower()

    if lower.startswith(("http://", "https://")):
        if not is_safe_http_url(text):
            return None
        parsed = urlparse(text)
        if parsed.netloc.lower() not in ("doi.org", "dx.doi.org"):
            return None
        return text

    if lower.startswith(("doi.org/", "dx.doi.org/")):
        candidate = f"https://{text}"
        return candidate if is_safe_http_url(candidate) else None

    if not re.match(r"^10\\.\\d{4,9}/[-._;()/:A-Za-z0-9]+$", text):
        return None

    return f"https://doi.org/{text}"


def highlight_query_terms(text: str, query: str) -> str:
    """Highlight query terms in text with HTML spans.

    Args:
        text: The text to highlight (will be HTML escaped).
        query: The search query to extract terms from.

    Returns:
        HTML-safe string with matched terms wrapped in highlight spans.
    """
    import re

    if not text or not query:
        return escape(text) if text else ""

    # Tokenize query into words (3+ chars to avoid highlighting common words)
    query_terms = [term.lower() for term in re.findall(r"\b\w{3,}\b", query)]
    if not query_terms:
        return escape(text)

    # Build a pattern that matches any of the query terms (case-insensitive)
    # Sort by length descending to match longer terms first
    query_terms = sorted(set(query_terms), key=len, reverse=True)
    pattern = r"\b(" + "|".join(re.escape(term) for term in query_terms) + r")\b"

    # Split text into parts, preserving matched terms
    parts = []
    last_end = 0
    for match in re.finditer(pattern, text, re.IGNORECASE):
        # Add text before match (escaped)
        if match.start() > last_end:
            parts.append(escape(text[last_end : match.start()]))
        # Add highlighted match
        parts.append(f'<mark style="background:#ffe066;padding:0 2px;">{escape(match.group())}</mark>')
        last_end = match.end()
    # Add remaining text
    if last_end < len(text):
        parts.append(escape(text[last_end:]))

    return "".join(parts)


def escape_bibtex(text: str) -> str:
    """Escape special BibTeX characters."""
    if not text:
        return ""
    replacements = [
        ("\\", "\\textbackslash{}"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("%", "\\%"),
        ("_", "\\_"),
        ("^", "\\^{}"),
        ("&", "\\&"),
        ("#", "\\#"),
        ("$", "\\$"),
        ("~", "\\textasciitilde{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def escape_markdown(text: str) -> str:
    """Escape markdown special characters to prevent injection.

    Prevents user-controlled text from being interpreted as markdown syntax,
    which could allow link injection or formatting exploits.

    Args:
        text: Text to escape.

    Returns:
        Text with markdown special characters escaped.
    """
    if not text:
        return ""
    # Escape characters that could be interpreted as markdown
    # Order matters: backslash first to avoid double-escaping
    replacements = [
        ("\\", "\\\\"),  # Backslash
        ("`", "\\`"),  # Code
        ("*", "\\*"),  # Bold/italic
        ("_", "\\_"),  # Bold/italic
        ("{", "\\{"),  # Curly braces
        ("}", "\\}"),
        ("[", "\\["),  # Links
        ("]", "\\]"),
        ("(", "\\("),  # Link URLs
        (")", "\\)"),
        ("#", "\\#"),  # Headers
        ("+", "\\+"),  # Lists
        ("-", "\\-"),  # Lists
        (".", "\\."),  # Numbered lists (only at line start, but safer to escape)
        ("!", "\\!"),  # Images
        ("|", "\\|"),  # Tables
        (">", "\\>"),  # Blockquotes
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def format_author_list(paper: dict) -> list[str]:
    """Format author list from structured metadata."""
    authors = paper.get("authors")
    if isinstance(authors, list) and authors:
        formatted = []
        for author in authors:
            first = str(author.get("first_name") or "").strip()
            last = str(author.get("last_name") or "").strip()
            full = str(author.get("full_name") or "").strip()
            if first and last:
                name = f"{last}, {first}"
            else:
                name = last or first or full
            if name:
                formatted.append(name)
        if formatted:
            return formatted

    author_string = str(paper.get("author_string") or "").strip()
    return [author_string] if author_string else []


def format_citation(paper: dict, result: EnrichedResult, style: str) -> str:
    """Format a citation string for the selected paper."""
    title = paper.get("title") or result.title or "Untitled"
    year = paper.get("publication_year") or result.year or "n.d."
    journal = paper.get("journal") or ""
    volume = paper.get("volume") or ""
    issue = paper.get("issue") or ""
    pages = paper.get("pages") or ""
    doi = paper.get("doi") or ""
    url = paper.get("url") or ""

    authors = format_author_list(paper)
    if not authors and result.authors:
        authors = [result.authors]

    if style == "APA":
        if not authors:
            author_text = "Unknown"
        elif len(authors) == 1:
            author_text = authors[0]
        elif len(authors) == 2:
            author_text = f"{authors[0]} & {authors[1]}"
        else:
            author_text = ", ".join(authors[:-1]) + f", & {authors[-1]}"
        parts = [f"{author_text} ({year}).", f"{title}."]
        if journal:
            journal_bits = [journal]
            if volume:
                journal_bits.append(volume)
            if issue:
                journal_bits[-1] = f"{journal_bits[-1]}({issue})"
            if pages:
                journal_bits.append(pages)
            parts.append(" ".join(journal_bits) + ".")
        if doi:
            parts.append(f"https://doi.org/{doi}")
        elif url:
            parts.append(url)
        return " ".join(parts)

    if style == "MLA":
        if not authors:
            author_text = "Unknown"
        elif len(authors) == 1:
            author_text = authors[0]
        elif len(authors) == 2:
            author_text = f"{authors[0]} and {authors[1]}"
        else:
            author_text = f"{authors[0]}, et al."
        parts = [f'{author_text}. "{title}."']
        if journal:
            parts.append(journal)
        if volume:
            parts.append(f"vol. {volume}")
        if issue:
            parts.append(f"no. {issue}")
        if year:
            parts.append(str(year))
        if pages:
            parts.append(f"pp. {pages}")
        if doi:
            parts.append(doi)
        return ", ".join(parts) + "."

    if style == "Chicago":
        if not authors:
            author_text = "Unknown"
        elif len(authors) == 1:
            author_text = authors[0]
        elif len(authors) == 2:
            author_text = f"{authors[0]} and {authors[1]}"
        else:
            author_text = ", ".join(authors[:-1]) + f", and {authors[-1]}"
        parts = [f'{author_text}. "{title}."']
        if journal:
            parts.append(journal)
        if volume:
            vol_issue = volume
            if issue:
                vol_issue = f"{vol_issue}, no. {issue}"
            parts.append(vol_issue)
        if year:
            parts.append(f"({year})")
        if pages:
            parts.append(f": {pages}")
        citation = " ".join(parts).replace(" )", ")").strip()
        if doi:
            citation += f". https://doi.org/{doi}."
        return citation

    if style == "BibTeX":
        author_text = " and ".join(authors) if authors else "Unknown"
        entry_type = "article" if result.item_type == "journalArticle" else "misc"
        cite_key = "citation"
        if authors:
            cite_key = authors[0].split(",")[0].replace(" ", "")
        cite_key = "".join(c for c in f"{cite_key}{year}" if c.isalnum())
        lines = [f"@{entry_type}{{{cite_key},"]
        lines.append(f"  title = {{{escape_bibtex(title)}}},")
        if author_text:
            lines.append(f"  author = {{{escape_bibtex(author_text)}}},")
        if year:
            lines.append(f"  year = {{{year}}},")
        if journal:
            lines.append(f"  journal = {{{escape_bibtex(journal)}}},")
        if volume:
            lines.append(f"  volume = {{{volume}}},")
        if issue:
            lines.append(f"  number = {{{issue}}},")
        if pages:
            lines.append(f"  pages = {{{pages}}},")
        if doi:
            lines.append(f"  doi = {{{doi}}},")
        lines.append("}")
        return "\n".join(lines)

    return title


def results_to_csv(results: list[EnrichedResult]) -> str:
    """Convert search results to CSV format."""
    buffer = io.StringIO()
    fieldnames = [
        "rank",
        "score",
        "paper_id",
        "title",
        "authors",
        "year",
        "journal",
        "item_type",
        "chunk_type",
        "collections",
        "doi",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()

    for i, result in enumerate(results, 1):
        paper = result.paper_data or {}
        writer.writerow({
            "rank": i,
            "score": round(result.score, 4),
            "paper_id": sanitize_csv_field(result.paper_id),
            "title": sanitize_csv_field(result.title),
            "authors": sanitize_csv_field(result.authors),
            "year": result.year or "",
            "journal": sanitize_csv_field(paper.get("journal", "")),
            "item_type": sanitize_csv_field(result.item_type),
            "chunk_type": sanitize_csv_field(result.chunk_type),
            "collections": sanitize_csv_field("; ".join(result.collections)),
            "doi": sanitize_csv_field(paper.get("doi", "")),
        })

    return buffer.getvalue()


def results_to_bibtex(results: list[EnrichedResult]) -> str:
    """Convert search results to BibTeX format."""
    entries = []

    bibtex_type_map = {
        "journalArticle": "article",
        "book": "book",
        "bookSection": "inbook",
        "conferencePaper": "inproceedings",
        "thesis": "phdthesis",
        "report": "techreport",
    }

    for result in results:
        paper = result.paper_data or {}
        item_type = result.item_type or "misc"
        bibtex_type = bibtex_type_map.get(item_type, "misc")

        # Build author list (prefer structured authors when available)
        authors_value = result.authors or ""
        author_part = "Unknown"
        authors = paper.get("authors")
        if isinstance(authors, list) and authors:
            formatted_authors = []
            for author in authors:
                first = str(author.get("first_name") or "").strip()
                last = str(author.get("last_name") or "").strip()
                full = str(author.get("full_name") or "").strip()
                if first and last:
                    name = f"{last}, {first}"
                else:
                    name = last or first or full
                if not name:
                    continue
                formatted_authors.append(name)
                if author_part == "Unknown":
                    author_part = (last or first or full).replace(",", " ").split()[-1] if (last or first or full) else "Unknown"
            if formatted_authors:
                authors_value = " and ".join(formatted_authors)
        if author_part == "Unknown" and authors_value:
            candidate = authors_value.split("and")[0].strip()
            if "," in candidate:
                author_part = candidate.split(",")[0].strip() or "Unknown"
            else:
                author_part = candidate.split()[-1] if candidate else "Unknown"

        year_part = result.year or "nd"
        cite_key = f"{author_part}{year_part}".replace(" ", "")
        cite_key = "".join(c for c in cite_key if c.isalnum())

        lines = [f"@{bibtex_type}{{{cite_key},"]
        lines.append(f'  title = {{{escape_bibtex(result.title)}}},')

        if authors_value:
            lines.append(f'  author = {{{escape_bibtex(authors_value)}}},')
        if result.year:
            lines.append(f"  year = {{{result.year}}},")
        if journal := paper.get("journal"):
            field = "booktitle" if bibtex_type == "inproceedings" else "journal"
            lines.append(f"  {field} = {{{escape_bibtex(journal)}}},")
        if doi := paper.get("doi"):
            lines.append(f"  doi = {{{doi}}},")
        if abstract := paper.get("abstract"):
            lines.append(f"  abstract = {{{escape_bibtex(abstract[:500])}}},")

        lines.append("}")
        entries.append("\n".join(lines))

    return "\n\n".join(entries)


def find_similar_papers(
    engine: SearchEngine,
    paper_id: str,
    top_k: int = 5,
) -> list[EnrichedResult]:
    """Find papers similar to the given paper."""
    return engine.search_similar_papers(paper_id=paper_id, top_k=top_k)


def build_similarity_network(
    center_paper: EnrichedResult,
    similar_papers: list[EnrichedResult],
    dark_mode: bool = False,
) -> str | None:
    """Build an interactive similarity network visualization.

    Args:
        center_paper: The focal paper at the center of the network.
        similar_papers: List of similar papers to display as connected nodes.
        dark_mode: If True, use dark theme colors.

    Returns:
        HTML string of the network visualization, or None if pyvis unavailable.
    """
    if not PYVIS_AVAILABLE:
        return None

    # Create NetworkX graph
    G = nx.Graph()

    # Add center node
    center_label = center_paper.title[:40] + "..." if len(center_paper.title) > 40 else center_paper.title
    center_year = f" ({center_paper.year})" if center_paper.year else ""
    G.add_node(
        center_paper.paper_id,
        label=center_label,
        title=f"{center_paper.title}{center_year}\n{center_paper.authors}",
        size=30,
        color="#e08050",  # Accent color
        font={"color": "#ffffff" if dark_mode else "#1f1b17"},
    )

    # Add similar papers as connected nodes
    for sim in similar_papers:
        sim_label = sim.title[:30] + "..." if len(sim.title) > 30 else sim.title
        sim_year = f" ({sim.year})" if sim.year else ""
        # Size based on similarity score
        node_size = 15 + int(sim.score * 15)
        G.add_node(
            sim.paper_id,
            label=sim_label,
            title=f"{sim.title}{sim_year}\n{sim.authors}\nSimilarity: {sim.score:.3f}",
            size=node_size,
            color="#508c78" if dark_mode else "#4a8c6a",
            font={"color": "#e8e8e8" if dark_mode else "#1f1b17"},
        )
        # Edge weight based on similarity
        G.add_edge(
            center_paper.paper_id,
            sim.paper_id,
            value=sim.score,
            title=f"Similarity: {sim.score:.3f}",
        )

    # Create PyVis network
    net = Network(
        height="400px",
        width="100%",
        bgcolor="#1a1a1a" if dark_mode else "#f6f1ea",
        font_color="#e8e8e8" if dark_mode else "#1f1b17",
        directed=False,
    )

    # Configure physics for nice layout
    net.set_options("""
    {
        "physics": {
            "barnesHut": {
                "gravitationalConstant": -3000,
                "centralGravity": 0.3,
                "springLength": 150,
                "springConstant": 0.05,
                "damping": 0.09
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100
        }
    }
    """)

    # Convert from NetworkX
    net.from_nx(G)

    # Generate HTML
    return net.generate_html()


def render_similarity_network(
    engine: SearchEngine,
    paper: EnrichedResult,
    num_similar: int,
    dark_mode: bool,
) -> None:
    """Render the similarity network visualization in Streamlit.

    Args:
        engine: SearchEngine instance for finding similar papers.
        paper: The focal paper to build the network around.
        num_similar: Number of similar papers to include.
        dark_mode: If True, use dark theme colors.
    """
    if not PYVIS_AVAILABLE:
        st.warning(
            "Network visualization requires pyvis. "
            "Install with: pip install pyvis networkx"
        )
        return

    with st.spinner("Building similarity network..."):
        similar = find_similar_papers(engine, paper.paper_id, num_similar)

    if not similar:
        st.info("No similar papers found to build network.")
        return

    html = build_similarity_network(paper, similar, dark_mode)
    if html:
        components.html(html, height=420, scrolling=False)
        st.caption(
            f"Network shows {len(similar)} papers similar to the focused paper. "
            "Hover over nodes for details. Drag to rearrange."
        )
    else:
        st.error("Failed to generate network visualization.")


def check_index_exists(config_path: str | None = None) -> tuple[bool, Path, bool]:
    """Check if the index directory exists and has required files.

    Args:
        config_path: Optional path to config.yaml to derive storage paths.

    Returns:
        Tuple of (index_exists, index_dir, has_embeddings).
        index_exists is True if papers.json exists.
        has_embeddings is True if chroma directory exists.
    """
    # Try to load config to get storage paths
    index_dir = project_root / "data" / "index"
    chroma_dir: Path | None = None

    if config_path:
        try:
            config = Config.load(config_path)
            chroma_dir = config.get_chroma_path()
        except Exception:
            pass  # Fall back to default paths

    if chroma_dir is None:
        chroma_dir = index_dir / "chroma"

    if not index_dir.exists():
        return False, index_dir, False

    # Only papers.json is strictly required
    if not (index_dir / "papers.json").exists():
        return False, index_dir, False

    # Check if embeddings are available (optional)
    has_embeddings = chroma_dir.exists()

    return True, index_dir, has_embeddings


def render_no_index_message() -> None:
    """Render a message when no index exists."""
    st.warning("No index found. Build the index to start searching.")
    st.markdown("""
### Getting Started

1. **Configure your reference source** in `config.yaml`
2. **Run the build script**:
   ```bash
   python scripts/build_index.py
   ```
3. **Refresh this page** after the build completes

See the [documentation](docs/usage.md) for detailed setup instructions.
    """)


def render_header() -> None:
    """Render the application header."""
    st.markdown(
        """
        <div class="app-hero">
          <div class="hero-eyebrow">LITRIS Search Workbench</div>
          <div class="hero-title">Semantic Library Explorer</div>
          <p class="hero-subtitle">
            Search your indexed papers, refine with filters, and export reports for review.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_index_summary(engine: SearchEngine, full_view: bool = False) -> None:
    """Render index summary stats.

    Args:
        engine: SearchEngine instance.
        full_view: If True, render full summary. Otherwise compact sidebar view.
    """
    if "summary" not in st.session_state:
        st.session_state.summary = engine.get_summary()

    summary = st.session_state.summary

    if full_view:
        # Full summary view for main content area
        st.markdown(format_summary(summary))
        return

    # Compact sidebar view
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Papers", summary.get("total_papers", 0))
    with col2:
        st.metric("Extractions", summary.get("total_extractions", 0))

    generated_at = summary.get("generated_at")
    if generated_at:
        st.caption(f"Summary: {generated_at}")
    if summary.get("vector_store"):
        st.caption(
            f"Chunks: {summary['vector_store'].get('total_chunks', 0)}"
        )

    # Top collections preview
    by_collection = summary.get("papers_by_collection", {})
    if by_collection:
        top_3 = list(by_collection.items())[:3]
        st.caption("Top collections: " + ", ".join(f"{c}({n})" for c, n in top_3))


def render_build_controls(config_path: str | None = None) -> None:
    """Render index build/rebuild controls in an expander.

    Args:
        config_path: Optional config file path to pass to build_index.py.
    """
    import shutil

    with st.expander("Index Build Controls", expanded=False):
        st.caption("Build or rebuild the literature index from your reference source.")

        # Configurable timeout (in hours)
        timeout_hours = st.slider(
            "Build timeout (hours)",
            min_value=1,
            max_value=12,
            value=2,
            help="Maximum time to wait for build/rebuild to complete. Increase for large libraries.",
        )
        timeout_seconds = timeout_hours * 3600

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Build Index", use_container_width=True):
                st.session_state["confirm_build"] = True
        with col2:
            if st.button("Rebuild Index", use_container_width=True):
                st.session_state["confirm_rebuild"] = True

        # Build command with optional config
        build_cmd = [sys.executable, str(project_root / "scripts" / "build_index.py")]
        if config_path:
            build_cmd.extend(["--config", config_path])

        if st.session_state.get("confirm_build"):
            st.warning("This will build the index. Proceed?")
            if st.button("Confirm Build"):
                with st.spinner("Building index... This may take a while."):
                    try:
                        result = subprocess.run(
                            build_cmd,
                            capture_output=True,
                            text=True,
                            cwd=str(project_root),
                            timeout=timeout_seconds,
                        )
                        if result.returncode == 0:
                            st.success("Index built successfully!")
                            st.code(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
                            # Clear caches
                            st.cache_resource.clear()
                            st.session_state.pop("filter_options", None)
                            st.session_state.pop("summary", None)
                        else:
                            st.error("Build failed.")
                            st.code(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
                    except subprocess.TimeoutExpired:
                        st.error(f"Build timed out after {timeout_hours} hour(s).")
                    except Exception as e:
                        st.error(f"Build error: {e}")
                st.session_state.pop("confirm_build", None)
            if st.button("Cancel", key="cancel_build"):
                st.session_state.pop("confirm_build", None)

        if st.session_state.get("confirm_rebuild"):
            st.error("This will DELETE and rebuild the entire index. All existing data will be lost!")
            if st.button("Confirm Rebuild (Destructive)"):
                with st.spinner("Rebuilding index... This may take a while."):
                    try:
                        # Delete existing index directory first
                        index_dir = project_root / "data" / "index"
                        if index_dir.exists():
                            shutil.rmtree(index_dir)
                            st.info("Cleared existing index directory.")

                        # Rebuild with reset checkpoint and embeddings
                        rebuild_cmd = build_cmd + ["--reset-checkpoint", "--rebuild-embeddings"]
                        result = subprocess.run(
                            rebuild_cmd,
                            capture_output=True,
                            text=True,
                            cwd=str(project_root),
                            timeout=timeout_seconds,
                        )
                        if result.returncode == 0:
                            st.success("Index rebuilt successfully!")
                            st.code(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
                            st.cache_resource.clear()
                            st.session_state.pop("filter_options", None)
                            st.session_state.pop("summary", None)
                        else:
                            st.error("Rebuild failed.")
                            st.code(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
                    except subprocess.TimeoutExpired:
                        st.error(f"Rebuild timed out after {timeout_hours} hour(s).")
                    except Exception as e:
                        st.error(f"Rebuild error: {e}")
                st.session_state.pop("confirm_rebuild", None)
            if st.button("Cancel", key="cancel_rebuild"):
                st.session_state.pop("confirm_rebuild", None)


def render_active_filters(
    chunk_types: list[str],
    collections: list[str],
    item_types: list[str],
    use_year_filter: bool,
    year_range: tuple[int, int],
    year_bounds: tuple[int, int],
    deduplicate: bool,
    metadata_only: bool = False,
) -> bool:
    """Render active filter chips and reset button above results.

    Args:
        chunk_types: Selected chunk types.
        collections: Selected collections.
        item_types: Selected item types.
        use_year_filter: Whether year filter is active.
        year_range: Selected (min, max) year range.
        year_bounds: Full available (min, max) year range.
        deduplicate: Whether deduplication is enabled.
        metadata_only: Whether metadata-only search is active.

    Returns:
        True if reset was requested, False otherwise.
    """
    active_filters = []

    # Check if chunk types are restricted (not all selected) - skip in metadata-only mode
    if not metadata_only and chunk_types and len(chunk_types) < len(CHUNK_TYPES):
        active_filters.append(f"Chunks: {', '.join(chunk_types)}")

    # Check collections
    if collections:
        if len(collections) <= 2:
            active_filters.append(f"Collections: {', '.join(collections)}")
        else:
            active_filters.append(f"Collections: {len(collections)} selected")

    # Check item types
    if item_types:
        if len(item_types) <= 2:
            active_filters.append(f"Types: {', '.join(item_types)}")
        else:
            active_filters.append(f"Types: {len(item_types)} selected")

    # Check year filter
    if use_year_filter and year_range != year_bounds:
        active_filters.append(f"Years: {year_range[0]}-{year_range[1]}")

    # Check deduplicate (only show if disabled, since default is True)
    if not deduplicate:
        active_filters.append("Duplicates: shown")

    if not active_filters:
        return False

    # Render as styled chips
    chips_html = " ".join(
        f'<span class="result-tag" style="margin-right: 6px;">{escape(f)}</span>'
        for f in active_filters
    )
    st.markdown(
        f'<div style="margin-bottom: 12px;"><strong>Active filters:</strong> {chips_html}</div>',
        unsafe_allow_html=True,
    )

    # Reset button
    if st.button("Clear all filters", use_container_width=False, type="secondary"):
        return True

    return False


def main() -> None:
    """Main UI entry point."""
    st.set_page_config(
        page_title="LITRIS Search Workbench",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Dark mode toggle (must be before inject_styles)
    dark_mode = st.sidebar.toggle(
        "Dark mode",
        value=st.session_state.get("dark_mode", False),
        key="dark_mode",
    )
    inject_styles(dark_mode=dark_mode)
    render_header()

    # Check if index exists before trying to load
    config_path = st.sidebar.text_input(
        "Config path (optional)",
        value="",
        help="Leave blank to auto-detect config.yaml.",
    )
    index_exists, index_dir, has_embeddings = check_index_exists(config_path or None)

    # Federated search toggle
    use_federated = st.sidebar.checkbox(
        "Federated search",
        value=st.session_state.get("use_federated", False),
        key="use_federated",
        help="Search across multiple configured indexes simultaneously.",
    )

    # Show build controls in sidebar regardless of index state
    with st.sidebar:
        render_build_controls(config_path=config_path or None)

    if not index_exists:
        render_no_index_message()
        st.stop()

    try:
        engine, _, _, results_dir = load_engine(config_path or None, use_federated)
    except FileNotFoundError as exc:
        render_no_index_message()
        st.caption(f"Details: {exc}")
        st.stop()
    except Exception as exc:  # noqa: BLE001 - user-facing error display
        st.error(f"Failed to load search engine: {exc}")
        st.info("Try rebuilding the index using the controls in the sidebar.")
        st.stop()

    st.session_state.setdefault("search_error", None)

    # Show federated index info when enabled
    if use_federated and hasattr(engine, "get_index_info"):
        with st.sidebar:
            with st.expander("Federated Indexes", expanded=False):
                info = engine.get_index_info()
                st.caption(f"Primary: {info['primary']['path']}")
                st.caption(f"Strategy: {info['merge_strategy']}")
                if info.get("indexes"):
                    for idx in info["indexes"]:
                        status = "active" if idx["loaded"] else "inactive"
                        st.caption(f"- {idx['label']} ({status}, w={idx['weight']})")
                else:
                    st.caption("No federated indexes configured")

    with st.sidebar:
        st.subheader("Filters")
        filter_options = load_filter_options(engine)

        # Handle reset request from active filters display
        if st.session_state.get("reset_filters"):
            st.session_state.pop("reset_filters", None)
            st.session_state.pop("filter_chunk_types", None)
            st.session_state.pop("filter_collections", None)
            st.session_state.pop("filter_item_types", None)
            st.session_state.pop("filter_use_year", None)
            st.session_state.pop("filter_year_range", None)
            st.session_state.pop("filter_deduplicate", None)
            st.rerun()

        # Only show chunk type filter for semantic search (not metadata-only)
        if not st.session_state.get("metadata_only", False):
            chunk_types = st.multiselect(
                "Chunk types",
                options=CHUNK_TYPES,
                default=st.session_state.get("filter_chunk_types", CHUNK_TYPES),
                key="filter_chunk_types",
            )
        else:
            chunk_types = CHUNK_TYPES  # Use all types, but don't show filter
        sort_label = st.selectbox(
            "Sort results",
            options=list(SORT_OPTIONS.keys()),
            index=list(SORT_OPTIONS.keys()).index(
                st.session_state.get("sort_label", "Relevance")
            ),
            key="sort_label",
        )
        sort_key = SORT_OPTIONS.get(sort_label, "relevance")

        # Collection filter with search and counts
        coll_search = st.text_input(
            "Search collections",
            value="",
            placeholder="Type to filter...",
            key="coll_search",
        )
        all_collections = filter_options["collections"]
        selected_colls = list(st.session_state.get("filter_collections", []))
        summary = st.session_state.get("summary", {})
        coll_counts = summary.get("papers_by_collection", {})

        # Filter collections by search and add counts
        if coll_search:
            search_lower = coll_search.lower()
            filtered_colls = [c for c in all_collections if search_lower in c.lower()]
        else:
            filtered_colls = all_collections

        hidden_selected = [c for c in selected_colls if c not in filtered_colls]
        if hidden_selected:
            st.caption(f"{len(hidden_selected)} selected collection(s) hidden by search filter.")

        # Include selected collections even if filtered out
        colls_for_options = list(dict.fromkeys(selected_colls + filtered_colls))

        # Format options with counts
        coll_options_display = []
        for c in colls_for_options:
            count = coll_counts.get(c, 0)
            coll_options_display.append(f"{c} ({count})" if count else c)

        # Map display names back to actual collection names
        display_to_name = {
            f"{c} ({coll_counts.get(c, 0)})" if coll_counts.get(c) else c: c
            for c in colls_for_options
        }

        selected_display = st.multiselect(
            "Collections",
            options=coll_options_display,
            default=[
                f"{c} ({coll_counts.get(c, 0)})" if coll_counts.get(c) else c
                for c in selected_colls
                if c in colls_for_options
            ],
        )
        collections = [display_to_name.get(d, d) for d in selected_display]
        st.session_state["filter_collections"] = collections
        item_types = st.multiselect(
            "Item types",
            options=filter_options["item_types"],
            default=st.session_state.get("filter_item_types", []),
            key="filter_item_types",
        )

        year_min, year_max = filter_options["year_range"]
        year_filter_available = year_min is not None and year_max is not None
        selected_year_min = year_min or 1900
        selected_year_max = year_max or 2100
        year_bounds = (selected_year_min, selected_year_max)

        use_year_filter = st.checkbox(
            "Filter by year",
            value=st.session_state.get("filter_use_year", year_filter_available),
            disabled=not year_filter_available,
            key="filter_use_year",
        )
        year_range = st.slider(
            "Year range",
            min_value=selected_year_min,
            max_value=selected_year_max,
            value=st.session_state.get("filter_year_range", year_bounds),
            disabled=not use_year_filter or not year_filter_available,
            key="filter_year_range",
        )

        top_k = st.slider(
            "Results",
            min_value=3,
            max_value=MAX_TOP_K,
            value=DEFAULT_TOP_K,
        )
        include_extraction = st.checkbox(
            "Include extraction in results list",
            value=False,
        )
        deduplicate_papers = st.checkbox(
            "Deduplicate by paper",
            value=st.session_state.get("filter_deduplicate", True),
            key="filter_deduplicate",
        )

        with st.expander("Index summary", expanded=False):
            render_index_summary(engine, full_view=False)

        if st.button("Refresh index metadata", use_container_width=True):
            st.session_state.pop("filter_options", None)
            st.session_state.pop("summary", None)
            st.rerun()

    # Main content tabs
    search_tab, summary_tab = st.tabs(["Search", "Index Summary"])

    with summary_tab:
        st.subheader("Full Index Summary")
        render_index_summary(engine, full_view=True)

    with search_tab:
        if not has_embeddings:
            st.warning(
                "Embeddings not found. Semantic search is disabled. "
                "Use metadata-only search or rebuild embeddings from the sidebar."
            )
        metadata_only_default = st.session_state.get("metadata_only", False)
        metadata_only_disabled = False
        if not has_embeddings:
            metadata_only_default = True
            metadata_only_disabled = True
        metadata_only = st.toggle(
            "Metadata-only search",
            value=metadata_only_default,
            help="Search by title/author metadata without semantic embeddings.",
            disabled=metadata_only_disabled,
        )
        if not has_embeddings:
            metadata_only = True
        st.session_state["metadata_only"] = metadata_only

        # Recent queries
        if "query_history" not in st.session_state:
            st.session_state["query_history"] = []

        query_history = st.session_state["query_history"]
        if query_history:
            hist_cols = st.columns([0.7, 0.3])
            with hist_cols[0]:
                selected_query = st.selectbox(
                    "Recent queries",
                    options=[""] + query_history,
                    index=0,
                    key="recent_query_select",
                    label_visibility="collapsed",
                    placeholder="Recent queries...",
                )
                if selected_query:
                    st.session_state["last_query"] = selected_query
                    st.rerun()
            with hist_cols[1]:
                if st.button("Clear history", use_container_width=True):
                    st.session_state["query_history"] = []
                    st.rerun()

        with st.form("search_form", clear_on_submit=False):
            title_query = ""
            author_query = ""
            query = ""
            if metadata_only:
                title_query = st.text_input(
                    "Title contains",
                    value=st.session_state.get("metadata_title", ""),
                    placeholder="e.g., neural network",
                )
                author_query = st.text_input(
                    "Author contains",
                    value=st.session_state.get("metadata_author", ""),
                    placeholder="e.g., Smith",
                )
            else:
                query = st.text_input(
                    "Search query",
                    value=st.session_state.get("last_query", ""),
                    placeholder="Enter a research question or concept",
                )
            submitted = st.form_submit_button("Run search", type="primary")

        if submitted:
            st.session_state["search_error"] = None
            if metadata_only:
                title_text = title_query.strip()
                author_text = author_query.strip()
                if not title_text and not author_text:
                    st.warning("Enter a title or author to search.")
                else:
                    match_parts = []
                    if title_text:
                        match_parts.append(f"title: {title_text}")
                    if author_text:
                        match_parts.append(f"author: {author_text}")
                    match_label = "Metadata match"
                    if match_parts:
                        match_label = f"Metadata match ({', '.join(match_parts)})"
                    search_label = "metadata " + " | ".join(match_parts)
                    with st.spinner("Running metadata search..."):
                        try:
                            results = execute_metadata_search(
                                engine=engine,
                                title_contains=title_text or None,
                                author_contains=author_text or None,
                                year_min=year_range[0] if use_year_filter else None,
                                year_max=year_range[1] if use_year_filter else None,
                                collections=collections or None,
                                item_types=item_types or None,
                                top_k=top_k,
                                match_label=match_label,
                            )
                            results = sort_results(results, sort_key)
                        except Exception as e:
                            st.session_state["search_error"] = f"Metadata search failed: {e}"
                            st.error(f"Metadata search failed: {e}")
                            results = []
                    st.session_state["search_results"] = results
                    st.session_state["last_query"] = search_label.strip()
                    st.session_state["metadata_title"] = title_text
                    st.session_state["metadata_author"] = author_text
                    st.session_state["selected_ids"] = set()  # Clear selection on new search
                    st.session_state["results_page"] = 0  # Reset to first page
                    st.session_state["selected_paper_id"] = (
                        results[0].paper_id if results else None
                    )
                    st.session_state.pop("similar_results", None)
                    st.session_state.pop("pdf_export_bytes", None)
                    st.session_state.pop("pdf_export_name", None)
            else:
                query_text = query.strip()
                if not query_text:
                    st.warning("Enter a query to search.")
                elif len(query_text) < 3:
                    st.warning("Query too short. Enter at least 3 characters.")
                else:
                    with st.spinner("Running semantic search..."):
                        try:
                            results = execute_search(
                                engine=engine,
                                query=query_text,
                                top_k=top_k,
                                chunk_types=normalize_chunk_filter(chunk_types),
                                year_min=year_range[0] if use_year_filter else None,
                                year_max=year_range[1] if use_year_filter else None,
                                collections=collections or None,
                                item_types=item_types or None,
                                include_extraction=include_extraction,
                                deduplicate_papers=deduplicate_papers,
                            )
                            results = sort_results(results, sort_key)
                        except Exception as e:
                            st.session_state["search_error"] = f"Search failed: {e}"
                            st.error(f"Search failed: {e}")
                            results = []
                    st.session_state["search_results"] = results
                    st.session_state["last_query"] = query_text
                    # Add to query history (dedupe, limit 10)
                    history = st.session_state.get("query_history", [])
                    if query_text in history:
                        history.remove(query_text)
                    history.insert(0, query_text)
                    st.session_state["query_history"] = history[:10]
                    st.session_state["selected_ids"] = set()  # Clear selection on new search
                    st.session_state["results_page"] = 0  # Reset to first page
                    st.session_state["selected_paper_id"] = (
                        results[0].paper_id if results else None
                    )
                    # Clear similar results when new search is run
                    st.session_state.pop("similar_results", None)
                    st.session_state.pop("pdf_export_bytes", None)
                    st.session_state.pop("pdf_export_name", None)

        # Handle quick filter trigger - re-run search with current query and updated filters
        if st.session_state.pop("trigger_search", False):
            st.session_state["search_error"] = None
            last_query = st.session_state.get("last_query", "")
            if metadata_only:
                title_text = st.session_state.get("metadata_title", "").strip()
                author_text = st.session_state.get("metadata_author", "").strip()
                if title_text or author_text:
                    match_parts = []
                    if title_text:
                        match_parts.append(f"title: {title_text}")
                    if author_text:
                        match_parts.append(f"author: {author_text}")
                    match_label = "Metadata match"
                    if match_parts:
                        match_label = f"Metadata match ({', '.join(match_parts)})"
                    with st.spinner("Applying filters..."):
                        try:
                            results = execute_metadata_search(
                                engine=engine,
                                title_contains=title_text or None,
                                author_contains=author_text or None,
                                year_min=year_range[0] if use_year_filter else None,
                                year_max=year_range[1] if use_year_filter else None,
                                collections=collections or None,
                                item_types=item_types or None,
                                top_k=top_k,
                                match_label=match_label,
                            )
                            results = sort_results(results, sort_key)
                            st.session_state["search_results"] = results
                            st.session_state["selected_ids"] = set()
                            st.session_state["results_page"] = 0
                            st.session_state["selected_paper_id"] = (
                                results[0].paper_id if results else None
                            )
                        except Exception as e:
                            st.session_state["search_error"] = f"Filter search failed: {e}"
                            st.error(f"Filter search failed: {e}")
                else:
                    st.warning("Enter a title or author to search.")
            elif last_query:
                with st.spinner("Applying filters..."):
                    try:
                        results = execute_search(
                            engine=engine,
                            query=last_query,
                            top_k=top_k,
                            chunk_types=normalize_chunk_filter(chunk_types),
                            year_min=year_range[0] if use_year_filter else None,
                            year_max=year_range[1] if use_year_filter else None,
                            collections=collections or None,
                            item_types=item_types or None,
                            include_extraction=include_extraction,
                            deduplicate_papers=deduplicate_papers,
                        )
                        st.session_state["search_results"] = results
                        st.session_state["selected_ids"] = set()
                        st.session_state["results_page"] = 0
                    except Exception as e:
                        st.session_state["search_error"] = f"Filter search failed: {e}"
                        st.error(f"Filter search failed: {e}")

        results = sort_results(
            st.session_state.get("search_results", []),
            sort_key,
        )
        last_query = st.session_state.get("last_query", "")
        search_error = st.session_state.get("search_error")

        if results:
            # Escape query to prevent markdown injection
            safe_query = escape_markdown(last_query)
            st.markdown(f"**{len(results)} results** for `{safe_query}`")

            # Show active filters and reset option
            if render_active_filters(
                chunk_types=chunk_types,
                collections=collections,
                item_types=item_types,
                use_year_filter=use_year_filter,
                year_range=year_range,
                year_bounds=year_bounds,
                deduplicate=deduplicate_papers,
                metadata_only=metadata_only,
            ):
                st.session_state["reset_filters"] = True
                st.rerun()

        elif last_query:
            # Search was run but no results found
            if search_error:
                return
            st.warning(f"No results found for: `{last_query}`")
            st.markdown("""
**Suggestions:**
- Try broader or different search terms
- Remove some filters (collections, year range, item types)
- Check if the chunk types filter is too restrictive
- Use the "Find Similar Papers" feature from a known paper
            """)
            return
        else:
            st.info("Run a search to see results. Enter a research question or concept above.")
            # Show some example queries
            with st.expander("Example queries", expanded=False):
                examples = [
                    "neural network architectures for image classification",
                    "qualitative research methods in social sciences",
                    "climate change impact on biodiversity",
                    "machine learning in healthcare diagnostics",
                ]
                for ex in examples:
                    st.code(ex)

        if not results:
            return

        export_col, spacer_col, detail_col = st.columns([1.2, 0.1, 1.1], gap="large")

        with export_col:
            st.subheader("Results")

            # Selection controls
            if "selected_ids" not in st.session_state:
                st.session_state["selected_ids"] = set()

            sel_cols = st.columns([0.3, 0.3, 0.4])
            with sel_cols[0]:
                if st.button("Select all", use_container_width=True):
                    # Clear checkbox widget keys so they reinitialize from selected_ids
                    for key in list(st.session_state.keys()):
                        if key.startswith("sel_"):
                            del st.session_state[key]
                    st.session_state["selected_ids"] = {r.paper_id for r in results}
                    st.rerun()
            with sel_cols[1]:
                if st.button("Clear selection", use_container_width=True):
                    # Clear checkbox widget keys so they reinitialize from selected_ids
                    for key in list(st.session_state.keys()):
                        if key.startswith("sel_"):
                            del st.session_state[key]
                    st.session_state["selected_ids"] = set()
                    st.rerun()
            with sel_cols[2]:
                num_selected = len(st.session_state["selected_ids"] & {r.paper_id for r in results})
                if num_selected > 0:
                    st.caption(f"{num_selected} of {len(results)} selected")

            # Determine which results to export
            selected_ids = st.session_state["selected_ids"]
            export_results = [r for r in results if r.paper_id in selected_ids] if selected_ids else results
            export_label = f" ({len(export_results)} selected)" if selected_ids else ""

            # Export options
            export_format = st.selectbox(
                "Export format",
                options=["markdown", "json", "brief", "pdf", "csv", "bibtex"],
                index=0,
                help="Choose format for exporting search results",
            )

            export_cols = st.columns(2)
            with export_cols[0]:
                if st.button(f"Save to disk{export_label}", use_container_width=True):
                    if export_format in ("csv", "bibtex"):
                        # Handle CSV/BibTeX separately
                        query_slug = sanitize_filename_slug(
                            st.session_state.get("last_query", "search")
                        )
                        date_str = datetime.now().strftime("%Y-%m-%d")
                        if export_format == "csv":
                            content = results_to_csv(export_results)
                            filename = f"{date_str}_{query_slug}.csv"
                        else:
                            content = results_to_bibtex(export_results)
                            filename = f"{date_str}_{query_slug}.bib"
                        export_path = results_dir / filename
                        results_dir.mkdir(parents=True, exist_ok=True)
                        export_path.write_text(content, encoding="utf-8")
                        st.success(f"Saved to {export_path}")
                    else:
                        export_path = save_export(
                            results=export_results,
                            query=st.session_state.get("last_query", "search"),
                            export_format=cast(OutputFormat, export_format),
                            results_dir=results_dir,
                            include_extraction=include_extraction,
                        )
                        st.success(f"Saved to {export_path}")

            with export_cols[1]:
                # Direct download button
                query_slug = sanitize_filename_slug(
                    st.session_state.get("last_query", "search")
                )
                date_str = datetime.now().strftime("%Y-%m-%d")

                if export_format == "csv":
                    content = results_to_csv(export_results)
                    st.download_button(
                        f"Download CSV{export_label}",
                        data=content,
                        file_name=f"{date_str}_{query_slug}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                elif export_format == "bibtex":
                    content = results_to_bibtex(export_results)
                    st.download_button(
                        f"Download BibTeX{export_label}",
                        data=content,
                        file_name=f"{date_str}_{query_slug}.bib",
                        mime="application/x-bibtex",
                        use_container_width=True,
                    )
                elif export_format == "pdf":
                    if st.button(f"Prepare PDF{export_label}", use_container_width=True, key="prepare_pdf"):
                        temp_path = save_export(
                            results=export_results,
                            query=st.session_state.get("last_query", "search"),
                            export_format="pdf",
                            results_dir=results_dir,
                            include_extraction=include_extraction,
                        )
                        if temp_path.exists():
                            st.session_state["pdf_export_bytes"] = temp_path.read_bytes()
                            st.session_state["pdf_export_name"] = temp_path.name

                    pdf_bytes = st.session_state.get("pdf_export_bytes")
                    pdf_name = st.session_state.get("pdf_export_name", "results.pdf")
                    if pdf_bytes:
                        st.download_button(
                            "Download PDF",
                            data=pdf_bytes,
                            file_name=pdf_name,
                            mime="application/pdf",
                            use_container_width=True,
                        )
                else:
                    content = format_results(
                        results=export_results,
                        query=st.session_state.get("last_query", "search"),
                        output_format=cast(OutputFormat, export_format),
                        include_extraction=include_extraction,
                    )
                    ext = {"markdown": "md", "json": "json", "brief": "txt"}.get(export_format, "txt")
                    st.download_button(
                        f"Download {export_format.upper()}",
                        data=content,
                        file_name=f"{date_str}_{query_slug}.{ext}",
                        use_container_width=True,
                    )

            # Pagination controls
            PAGE_SIZE = 10
            total_results = len(results)
            total_pages = max(1, (total_results + PAGE_SIZE - 1) // PAGE_SIZE)

            if "results_page" not in st.session_state:
                st.session_state["results_page"] = 0

            current_page = st.session_state["results_page"]
            # Clamp to valid range
            if current_page >= total_pages:
                current_page = total_pages - 1
                st.session_state["results_page"] = current_page

            start_idx = current_page * PAGE_SIZE
            end_idx = min(start_idx + PAGE_SIZE, total_results)
            page_results = results[start_idx:end_idx]

            if total_pages > 1:
                pg_cols = st.columns([0.3, 0.4, 0.3])
                with pg_cols[0]:
                    if st.button("Prev", disabled=current_page == 0, use_container_width=True):
                        st.session_state["results_page"] = current_page - 1
                        st.rerun()
                with pg_cols[1]:
                    st.caption(f"Page {current_page + 1} of {total_pages} ({start_idx + 1}-{end_idx} of {total_results})")
                with pg_cols[2]:
                    if st.button("Next", disabled=current_page >= total_pages - 1, use_container_width=True):
                        st.session_state["results_page"] = current_page + 1
                        st.rerun()

            for idx, result in enumerate(page_results, start_idx + 1):
                year_label = f" ({result.year})" if result.year else ""
                authors = escape(result.authors or "Unknown")
                title = escape(result.title)
                chunk_label = escape(result.chunk_type)
                item_label = escape(result.item_type or "unknown")
                matched_full = result.matched_text.strip()
                is_truncated = len(matched_full) > 280
                matched_display = matched_full[:280] + "..." if is_truncated else matched_full

                # Highlight query terms for semantic search (not metadata search)
                if last_query and not st.session_state.get("metadata_only", False):
                    matched = highlight_query_terms(matched_display, last_query)
                else:
                    matched = escape(matched_display)
                # Show source index for federated results
                source_tag = ""
                if hasattr(result, "source_index") and result.source_index != "primary":
                    source_tag = f'<span class="result-tag" style="background: #6366f1; color: white;">{escape(result.source_index)}</span>'

                card_html = f"""
                <div class="result-card" style="animation-delay: {idx * 70}ms;">
                  <div class="result-title">{idx}. {title}{year_label}</div>
                  <div class="result-meta">{authors} | Score {result.score:.3f}</div>
                  <div class="result-tags">
                    <span class="result-tag">{chunk_label}</span>
                    <span class="result-tag">{item_label}</span>
                    {source_tag}
                  </div>
                  <div class="matched-text">{matched}</div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)

                # Expand option for truncated text
                if is_truncated:
                    with st.expander("Show full matched text", expanded=False):
                        # Limit expansion to 1000 chars for very long chunks
                        full_display = matched_full[:1000] + "..." if len(matched_full) > 1000 else matched_full
                        if last_query and not st.session_state.get("metadata_only", False):
                            st.markdown(highlight_query_terms(full_display, last_query), unsafe_allow_html=True)
                        else:
                            st.text(full_display)
                button_cols = st.columns([0.15, 0.35, 0.5])
                with button_cols[0]:
                    is_selected = result.paper_id in st.session_state["selected_ids"]
                    if st.checkbox("", value=is_selected, key=f"sel_{result.paper_id}_{idx}", label_visibility="collapsed"):
                        st.session_state["selected_ids"].add(result.paper_id)
                    else:
                        st.session_state["selected_ids"].discard(result.paper_id)
                with button_cols[1]:
                    if st.button("Focus", key=f"focus_{result.paper_id}_{idx}"):
                        st.session_state["selected_paper_id"] = result.paper_id
                with button_cols[2]:
                    if result.collections:
                        st.caption(", ".join(result.collections))

                # Quick filter buttons
                filter_cols = st.columns([0.25, 0.25, 0.5])
                with filter_cols[0]:
                    if result.year and st.button(f"Year: {result.year}", key=f"qf_year_{result.paper_id}_{idx}", type="secondary"):
                        st.session_state["filter_use_year"] = True
                        st.session_state["filter_year_range"] = (result.year, result.year)
                        st.session_state["trigger_search"] = True
                        st.rerun()
                with filter_cols[1]:
                    if result.item_type and st.button(f"Type: {result.item_type[:12]}", key=f"qf_type_{result.paper_id}_{idx}", type="secondary"):
                        current_types = list(st.session_state.get("filter_item_types", []))
                        if result.item_type not in current_types:
                            current_types.append(result.item_type)
                            st.session_state["filter_item_types"] = current_types
                            st.session_state["trigger_search"] = True
                            st.rerun()
                with filter_cols[2]:
                    if result.collections:
                        first_coll = result.collections[0]
                        coll_label = first_coll[:15] + "..." if len(first_coll) > 15 else first_coll
                        if st.button(f"Coll: {coll_label}", key=f"qf_coll_{result.paper_id}_{idx}", type="secondary"):
                            current_colls = list(st.session_state.get("filter_collections", []))
                            if first_coll not in current_colls:
                                current_colls.append(first_coll)
                                st.session_state["filter_collections"] = current_colls
                                st.session_state["trigger_search"] = True
                                st.rerun()

        with detail_col:
            st.subheader("Detail")
            selected_id = st.session_state.get("selected_paper_id")
            selected = next((r for r in results if r.paper_id == selected_id), results[0])
            detail_extraction = st.toggle(
                "Load extraction for focused paper",
                value=st.session_state.get("detail_extraction", False),
                help="Fetch extraction data only for the focused paper.",
            )
            st.session_state["detail_extraction"] = detail_extraction
            detail_md = resolve_detail_markdown(
                engine=engine,
                result=selected,
                detail_extraction=detail_extraction,
            )
            matched_preview = escape(selected.matched_text[:320])
            st.markdown(
                f"""
                <div class="detail-panel">
                  <div class="detail-title">Focused paper</div>
                  <div class="matched-text">
                    {matched_preview}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(detail_md)

            # Clickable DOI and URL links
            paper = selected.paper_data or {}
            doi_text = str(paper.get("doi") or "").strip()
            url_text = str(paper.get("url") or "").strip()
            if doi_text or url_text:
                link_cols = st.columns(2)
                with link_cols[0]:
                    if doi_text:
                        doi_url = normalize_doi_url(doi_text)
                        if doi_url:
                            st.markdown(f"[Open DOI]({doi_url})")
                        else:
                            st.caption("DOI omitted (invalid format).")
                with link_cols[1]:
                    if url_text:
                        if is_safe_http_url(url_text):
                            st.markdown(f"[Open URL]({url_text})")
                        else:
                            st.caption("URL omitted (invalid scheme).")

            st.subheader("Citation")
            citation_format = st.selectbox(
                "Format",
                ["APA", "MLA", "Chicago", "BibTeX"],
                key="citation_format",
            )
            citation_text = format_citation(
                selected.paper_data or {},
                selected,
                citation_format,
            )
            code_lang = "bibtex" if citation_format == "BibTeX" else "text"
            st.code(citation_text, language=code_lang)
            if st.button("Copy citation", key="copy_citation"):
                st.session_state["citation_copy_pending"] = citation_text
                if hasattr(st, "toast"):
                    st.toast("Citation copied to clipboard.")
                else:
                    st.success("Citation copied to clipboard.")
            copy_pending = st.session_state.get("citation_copy_pending")
            if copy_pending:
                import streamlit.components.v1 as components

                components.html(
                    f"<script>navigator.clipboard.writeText({json.dumps(copy_pending)});</script>",
                    height=0,
                )
                st.session_state.pop("citation_copy_pending", None)

            # PDF actions
            pdf_path_str = selected.paper_data.get("pdf_path", "")
            if pdf_path_str:
                pdf_path = Path(pdf_path_str).expanduser()
                if pdf_path.is_file():
                    pdf_cols = st.columns(2)
                    with pdf_cols[0]:
                        st.download_button(
                            "Download PDF",
                            data=pdf_path.read_bytes(),
                            file_name=pdf_path.name,
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    with pdf_cols[1]:
                        st.caption(f"Local: {pdf_path.name}")
                else:
                    st.caption(f"PDF not found: {pdf_path_str}")

            # Similar papers
            with st.expander("Find Similar Papers", expanded=False):
                num_similar = st.slider("Number of similar papers", 3, 10, 5, key="similar_k")
                if st.button("Find Similar", use_container_width=True):
                    with st.spinner("Finding similar papers..."):
                        similar = find_similar_papers(engine, selected.paper_id, num_similar)
                    if similar:
                        st.session_state["similar_results"] = similar
                        st.success(f"Found {len(similar)} similar papers")
                    else:
                        st.info("No similar papers found")

                if "similar_results" in st.session_state:
                    for i, sim in enumerate(st.session_state["similar_results"], 1):
                        year_str = f" ({sim.year})" if sim.year else ""
                        # Escape markdown to prevent injection from title/authors
                        safe_title = escape_markdown(sim.title) if sim.title else ""
                        safe_authors = escape_markdown(sim.authors) if sim.authors else ""
                        st.markdown(f"**{i}. {safe_title}{year_str}**")
                        st.caption(f"{safe_authors} | Score: {sim.score:.3f}")
                        if st.button("Focus", key=f"sim_focus_{sim.paper_id}_{i}"):
                            st.session_state["selected_paper_id"] = sim.paper_id
                            st.session_state.pop("similar_results", None)
                            st.rerun()

            # Similarity network visualization
            with st.expander("Similarity Network", expanded=False):
                if PYVIS_AVAILABLE:
                    net_similar = st.slider(
                        "Papers in network",
                        3,
                        15,
                        7,
                        key="network_k",
                        help="Number of similar papers to show in the network.",
                    )
                    if st.button("Build Network", use_container_width=True):
                        render_similarity_network(
                            engine=engine,
                            paper=selected,
                            num_similar=net_similar,
                            dark_mode=dark_mode,
                        )
                else:
                    st.info(
                        "Network visualization requires pyvis. "
                        "Install with: `pip install pyvis networkx`"
                    )

            # Formatted output preview
            if st.checkbox("Show formatted output preview", value=False):
                preview = format_results(
                    results=[selected],
                    query=st.session_state.get("last_query", "search"),
                    output_format="markdown",
                    include_extraction=include_extraction,
                )
                st.markdown(preview)


if __name__ == "__main__":
    main()
