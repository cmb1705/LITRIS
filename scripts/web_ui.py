#!/usr/bin/env python
"""Launch the local web UI for LITRIS search using Streamlit."""

from __future__ import annotations

import csv
import io
import subprocess
import sys
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Iterable, cast

import streamlit as st

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.indexing.embeddings import CHUNK_TYPES, ChunkType
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


def inject_styles() -> None:
    """Inject global CSS styles into the Streamlit app."""
    st.markdown(STYLE, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_engine(config_path: str | None) -> tuple[SearchEngine, Config, Path, Path]:
    """Load config and initialize the search engine.

    Args:
        config_path: Optional path to config.yaml.

    Returns:
        Tuple of (SearchEngine, Config, index_dir, results_dir).
    """
    config = Config.load(config_path)
    index_dir = project_root / "data" / "index"
    results_dir = project_root / "data" / "query_results"

    if not index_dir.exists():
        raise FileNotFoundError(
            f"Index directory not found: {index_dir}\n"
            "Run scripts/build_index.py first to create the index."
        )

    engine = SearchEngine(
        index_dir=index_dir,
        chroma_dir=index_dir / "chroma",
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


def normalize_chunk_filter(selected: Iterable[str]) -> list[ChunkType] | None:
    """Normalize chunk type selections into a search filter."""
    selected_list = [c for c in selected if c in CHUNK_TYPES]
    if not selected_list or len(selected_list) == len(CHUNK_TYPES):
        return None
    return cast(list[ChunkType], selected_list)


def resolve_detail_markdown(
    engine: SearchEngine,
    result: EnrichedResult,
    include_extraction: bool,
) -> str:
    """Build the detail panel markdown for a selected result."""
    paper_data = result.paper_data or {}
    extraction = result.extraction_data or None

    if include_extraction and not extraction:
        combined = engine.get_paper(result.paper_id) or {}
        paper_data = combined.get("paper", paper_data)
        extraction = combined.get("extraction")

    if not paper_data:
        combined = engine.get_paper(result.paper_id) or {}
        paper_data = combined.get("paper", {})
        if include_extraction:
            extraction = combined.get("extraction")

    return format_paper_detail(paper_data, extraction)


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

        # Generate citation key
        author_part = result.authors.split(",")[0].split()[-1] if result.authors else "Unknown"
        year_part = result.year or "nd"
        cite_key = f"{author_part}{year_part}".replace(" ", "")
        cite_key = "".join(c for c in cite_key if c.isalnum())

        lines = [f"@{bibtex_type}{{{cite_key},"]
        lines.append(f'  title = {{{escape_bibtex(result.title)}}},')

        if result.authors:
            lines.append(f'  author = {{{escape_bibtex(result.authors)}}},')
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


def check_index_exists() -> tuple[bool, Path]:
    """Check if the index directory exists and has required files."""
    index_dir = project_root / "data" / "index"
    required_files = ["papers.json", "extractions.json"]
    chroma_dir = index_dir / "chroma"

    if not index_dir.exists():
        return False, index_dir

    for fname in required_files:
        if not (index_dir / fname).exists():
            return False, index_dir

    if not chroma_dir.exists():
        return False, index_dir

    return True, index_dir


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

See the [documentation](docs/guides/getting-started.md) for detailed setup instructions.
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


def render_index_summary(engine: SearchEngine) -> None:
    """Render index summary stats in the sidebar."""
    if "summary" not in st.session_state:
        st.session_state.summary = engine.get_summary()

    summary = st.session_state.summary
    st.metric("Papers", summary.get("total_papers", 0))
    st.metric("Extractions", summary.get("total_extractions", 0))
    generated_at = summary.get("generated_at")
    if generated_at:
        st.caption(f"Summary generated: {generated_at}")
    if summary.get("vector_store"):
        st.caption(
            f"Vector chunks: {summary['vector_store'].get('total_chunks', 0)}"
        )


def render_build_controls() -> None:
    """Render index build/rebuild controls in an expander."""
    with st.expander("Index Build Controls", expanded=False):
        st.caption("Build or rebuild the literature index from your reference source.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Build Index", use_container_width=True):
                st.session_state["confirm_build"] = True
        with col2:
            if st.button("Rebuild Index", use_container_width=True):
                st.session_state["confirm_rebuild"] = True

        if st.session_state.get("confirm_build"):
            st.warning("This will build the index. Proceed?")
            if st.button("Confirm Build"):
                with st.spinner("Building index... This may take a while."):
                    try:
                        result = subprocess.run(
                            [sys.executable, str(project_root / "scripts" / "build_index.py")],
                            capture_output=True,
                            text=True,
                            cwd=str(project_root),
                            timeout=3600,
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
                        st.error("Build timed out after 1 hour.")
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
                        result = subprocess.run(
                            [sys.executable, str(project_root / "scripts" / "build_index.py"), "--force"],
                            capture_output=True,
                            text=True,
                            cwd=str(project_root),
                            timeout=3600,
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
                        st.error("Rebuild timed out after 1 hour.")
                    except Exception as e:
                        st.error(f"Rebuild error: {e}")
                st.session_state.pop("confirm_rebuild", None)
            if st.button("Cancel", key="cancel_rebuild"):
                st.session_state.pop("confirm_rebuild", None)


def main() -> None:
    """Main UI entry point."""
    st.set_page_config(
        page_title="LITRIS Search Workbench",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    render_header()

    # Check if index exists before trying to load
    index_exists, index_dir = check_index_exists()

    config_path = st.sidebar.text_input(
        "Config path (optional)",
        value="",
        help="Leave blank to auto-detect config.yaml.",
    )

    # Show build controls in sidebar regardless of index state
    with st.sidebar:
        render_build_controls()

    if not index_exists:
        render_no_index_message()
        st.stop()

    try:
        engine, _, _, results_dir = load_engine(config_path or None)
    except FileNotFoundError as exc:
        render_no_index_message()
        st.caption(f"Details: {exc}")
        st.stop()
    except Exception as exc:  # noqa: BLE001 - user-facing error display
        st.error(f"Failed to load search engine: {exc}")
        st.info("Try rebuilding the index using the controls in the sidebar.")
        st.stop()

    with st.sidebar:
        st.subheader("Filters")
        filter_options = load_filter_options(engine)

        chunk_types = st.multiselect(
            "Chunk types",
            options=CHUNK_TYPES,
            default=CHUNK_TYPES,
        )
        collections = st.multiselect(
            "Collections",
            options=filter_options["collections"],
        )
        item_types = st.multiselect(
            "Item types",
            options=filter_options["item_types"],
        )

        year_min, year_max = filter_options["year_range"]
        year_filter_available = year_min is not None and year_max is not None
        use_year_filter = st.checkbox(
            "Filter by year",
            value=year_filter_available,
            disabled=not year_filter_available,
        )
        selected_year_min = year_min or 1900
        selected_year_max = year_max or 2100
        year_range = st.slider(
            "Year range",
            min_value=selected_year_min,
            max_value=selected_year_max,
            value=(selected_year_min, selected_year_max),
            disabled=not use_year_filter or not year_filter_available,
        )

        top_k = st.slider(
            "Results",
            min_value=3,
            max_value=MAX_TOP_K,
            value=DEFAULT_TOP_K,
        )
        include_extraction = st.checkbox(
            "Include extraction in results",
            value=False,
        )
        deduplicate_papers = st.checkbox(
            "Deduplicate by paper",
            value=True,
        )

        with st.expander("Index summary", expanded=False):
            render_index_summary(engine)

        if st.button("Refresh index metadata", use_container_width=True):
            st.session_state.pop("filter_options", None)
            st.session_state.pop("summary", None)

    with st.form("search_form", clear_on_submit=False):
        query = st.text_input(
            "Search query",
            value=st.session_state.get("last_query", ""),
            placeholder="Enter a research question or concept",
        )
        submitted = st.form_submit_button("Run search", type="primary")

    if submitted:
        if not query.strip():
            st.warning("Enter a query to search.")
        else:
            with st.spinner("Running semantic search..."):
                results = execute_search(
                    engine=engine,
                    query=query.strip(),
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
            st.session_state["last_query"] = query.strip()
            st.session_state["selected_paper_id"] = (
                results[0].paper_id if results else None
            )

    results = st.session_state.get("search_results", [])
    if results:
        st.markdown(f"**{len(results)} results** for `{st.session_state.get('last_query', '')}`")
    else:
        st.info("Run a search to see results.")

    if not results:
        return

    export_col, spacer_col, detail_col = st.columns([1.2, 0.1, 1.1], gap="large")

    with export_col:
        st.subheader("Results")

        # Export options
        export_format = st.selectbox(
            "Export format",
            options=["markdown", "json", "brief", "pdf", "csv", "bibtex"],
            index=0,
            help="Choose format for exporting search results",
        )

        export_cols = st.columns(2)
        with export_cols[0]:
            if st.button("Save to disk", use_container_width=True):
                if export_format in ("csv", "bibtex"):
                    # Handle CSV/BibTeX separately
                    query_slug = st.session_state.get("last_query", "search")[:30].replace(" ", "-")
                    date_str = datetime.now().strftime("%Y-%m-%d")
                    if export_format == "csv":
                        content = results_to_csv(results)
                        filename = f"{date_str}_{query_slug}.csv"
                    else:
                        content = results_to_bibtex(results)
                        filename = f"{date_str}_{query_slug}.bib"
                    export_path = results_dir / filename
                    results_dir.mkdir(parents=True, exist_ok=True)
                    export_path.write_text(content, encoding="utf-8")
                    st.success(f"Saved to {export_path}")
                else:
                    export_path = save_export(
                        results=results,
                        query=st.session_state.get("last_query", "search"),
                        export_format=cast(OutputFormat, export_format),
                        results_dir=results_dir,
                        include_extraction=include_extraction,
                    )
                    st.success(f"Saved to {export_path}")

        with export_cols[1]:
            # Direct download button
            query_slug = st.session_state.get("last_query", "search")[:30].replace(" ", "-")
            date_str = datetime.now().strftime("%Y-%m-%d")

            if export_format == "csv":
                content = results_to_csv(results)
                st.download_button(
                    "Download CSV",
                    data=content,
                    file_name=f"{date_str}_{query_slug}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            elif export_format == "bibtex":
                content = results_to_bibtex(results)
                st.download_button(
                    "Download BibTeX",
                    data=content,
                    file_name=f"{date_str}_{query_slug}.bib",
                    mime="application/x-bibtex",
                    use_container_width=True,
                )
            elif export_format == "pdf":
                # Generate PDF and offer download
                temp_path = save_export(
                    results=results,
                    query=st.session_state.get("last_query", "search"),
                    export_format="pdf",
                    results_dir=results_dir,
                    include_extraction=include_extraction,
                )
                if temp_path.exists():
                    st.download_button(
                        "Download PDF",
                        data=temp_path.read_bytes(),
                        file_name=temp_path.name,
                        mime="application/pdf",
                        use_container_width=True,
                    )
            else:
                content = format_results(
                    results=results,
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

        for idx, result in enumerate(results, 1):
            year_label = f" ({result.year})" if result.year else ""
            authors = escape(result.authors or "Unknown")
            title = escape(result.title)
            chunk_label = escape(result.chunk_type)
            item_label = escape(result.item_type or "unknown")
            matched = escape(result.matched_text.strip())
            if len(matched) > 280:
                matched = matched[:280] + "..."
            card_html = f"""
            <div class="result-card" style="animation-delay: {idx * 70}ms;">
              <div class="result-title">{idx}. {title}{year_label}</div>
              <div class="result-meta">{authors} | Score {result.score:.3f}</div>
              <div class="result-tags">
                <span class="result-tag">{chunk_label}</span>
                <span class="result-tag">{item_label}</span>
              </div>
              <div class="matched-text">{matched}</div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            button_cols = st.columns([0.4, 0.6])
            with button_cols[0]:
                if st.button("Focus", key=f"focus_{result.paper_id}_{idx}"):
                    st.session_state["selected_paper_id"] = result.paper_id
            with button_cols[1]:
                if result.collections:
                    st.caption(", ".join(result.collections))

    with detail_col:
        st.subheader("Detail")
        selected_id = st.session_state.get("selected_paper_id")
        selected = next((r for r in results if r.paper_id == selected_id), results[0])
        detail_md = resolve_detail_markdown(
            engine=engine,
            result=selected,
            include_extraction=include_extraction,
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

        pdf_path = Path(selected.paper_data.get("pdf_path", "")).expanduser()
        if pdf_path.is_file():
            st.download_button(
                "Download PDF",
                data=pdf_path.read_bytes(),
                file_name=pdf_path.name,
                mime="application/pdf",
            )
        elif selected.paper_data.get("pdf_path"):
            st.caption(f"PDF path: {selected.paper_data.get('pdf_path')}")

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
