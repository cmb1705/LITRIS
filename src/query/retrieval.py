"""Result formatting and retrieval utilities."""

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from src.query.search import EnrichedResult
from src.utils.file_utils import safe_write_json
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

OutputFormat = Literal["json", "markdown", "brief", "pdf"]


def format_results(
    results: list[EnrichedResult],
    query: str,
    output_format: OutputFormat = "json",
    include_extraction: bool = False,
) -> str:
    """Format search results for output.

    Args:
        results: List of search results.
        query: Original search query.
        output_format: Output format (json, markdown, brief, pdf).
        include_extraction: Whether to include full extraction data.

    Returns:
        Formatted string output (for pdf, returns markdown for display).
    """
    if output_format == "json":
        return format_json(results, query, include_extraction)
    elif output_format == "markdown" or output_format == "pdf":
        return format_markdown(results, query, include_extraction)
    else:
        return format_brief(results, query)


def format_json(
    results: list[EnrichedResult],
    query: str,
    include_extraction: bool = False,
) -> str:
    """Format results as JSON.

    Args:
        results: List of search results.
        query: Original search query.
        include_extraction: Whether to include full extraction data.

    Returns:
        JSON string.
    """
    output = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "result_count": len(results),
        "results": [],
    }

    for i, result in enumerate(results, 1):
        result_data = {
            "rank": i,
            "score": round(result.score, 4),
            "paper_id": result.paper_id,
            "title": result.title,
            "authors": result.authors,
            "year": result.year,
            "collections": result.collections,
            "item_type": result.item_type,
            "chunk_type": result.chunk_type,
            "matched_text": result.matched_text[:500] + "..."
            if len(result.matched_text) > 500
            else result.matched_text,
        }

        if include_extraction and result.extraction_data:
            result_data["extraction"] = result.extraction_data

        output["results"].append(result_data)

    return json.dumps(output, indent=2, ensure_ascii=False)


def format_markdown(
    results: list[EnrichedResult],
    query: str,
    include_extraction: bool = False,
) -> str:
    """Format results as Markdown.

    Args:
        results: List of search results.
        query: Original search query.
        include_extraction: Whether to include extraction snippets.

    Returns:
        Markdown string.
    """
    lines = [
        "# Literature Search Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Results:** {len(results)}",
        "",
        "---",
        "",
        "## Search Query",
        "",
        f"> {query}",
        "",
        "---",
        "",
    ]

    for i, result in enumerate(results, 1):
        # Title and metadata
        year_str = f" ({result.year})" if result.year else ""
        lines.extend(
            [
                f"## {i}. {result.title}{year_str}",
                "",
                f"**Authors:** {result.authors or 'Unknown'}",
                f"**Score:** {result.score:.4f}",
                f"**Type:** {result.item_type}",
                f"**Match:** {result.chunk_type}",
            ]
        )

        if result.collections:
            lines.append(f"**Collections:** {', '.join(result.collections)}")

        lines.append(f"**Paper ID:** `{result.paper_id}`")
        lines.append("")

        # Matched text
        lines.append("### Matched Text")
        lines.append("")
        matched = result.matched_text
        if len(matched) > 800:
            matched = matched[:800] + "..."
        lines.append(f"> {matched}")
        lines.append("")

        # Extraction snippets if requested
        if include_extraction and result.extraction_data:
            ext = result.extraction_data.get("extraction", result.extraction_data)

            if ext.get("thesis_statement"):
                lines.extend(
                    [
                        "### Thesis",
                        "",
                        ext["thesis_statement"],
                        "",
                    ]
                )

            if ext.get("contribution_summary"):
                lines.extend(
                    [
                        "### Contribution",
                        "",
                        ext["contribution_summary"],
                        "",
                    ]
                )

            if ext.get("key_findings"):
                lines.append("### Key Findings")
                lines.append("")
                for finding in ext["key_findings"][:3]:
                    if isinstance(finding, dict):
                        lines.append(f"- {finding.get('finding', finding)}")
                    else:
                        lines.append(f"- {finding}")
                lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def format_brief(results: list[EnrichedResult], query: str) -> str:
    """Format results as brief text output.

    Args:
        results: List of search results.
        query: Original search query.

    Returns:
        Brief text string.
    """
    lines = [
        f"Query: {query}",
        f"Found {len(results)} results:",
        "",
    ]

    for i, result in enumerate(results, 1):
        year_str = f" ({result.year})" if result.year else ""
        score_pct = result.score * 100
        lines.append(
            f"  {i}. [{score_pct:.1f}%] {result.title[:60]}...{year_str}"
        )
        lines.append(f"      {result.authors[:50]}..." if len(result.authors) > 50 else f"      {result.authors}")
        lines.append(f"      Match: {result.chunk_type}")
        lines.append("")

    return "\n".join(lines)


def slugify_query(query: str, max_length: int = 50) -> str:
    """Convert query to a safe filename slug.

    Args:
        query: Original search query.
        max_length: Maximum length of slug.

    Returns:
        Safe filename string.
    """
    import re

    # Lowercase and replace spaces with hyphens
    slug = query.lower().strip()
    # Remove special characters
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    # Replace whitespace with hyphens
    slug = re.sub(r"[\s_]+", "-", slug)
    # Remove multiple consecutive hyphens
    slug = re.sub(r"-+", "-", slug)
    # Trim hyphens from ends
    slug = slug.strip("-")
    # Truncate to max length at word boundary
    if len(slug) > max_length:
        slug = slug[:max_length].rsplit("-", 1)[0]
    return slug or "search"


def _get_pdf_css() -> str:
    """Return CSS styles for PDF HTML rendering."""
    return """
    body { font-family: Helvetica, Arial, sans-serif; font-size: 10pt; line-height: 1.4; color: #333; }
    h1 { font-size: 16pt; color: #1a1a1a; margin-bottom: 10px; border-bottom: 2px solid #3498db; padding-bottom: 5px; }
    h2 { font-size: 12pt; color: #2c3e50; margin-top: 12px; margin-bottom: 6px; }
    h3 { font-size: 11pt; color: #34495e; margin-top: 8px; margin-bottom: 4px; }
    .meta { font-size: 9pt; color: #666; margin-bottom: 12px; }
    .query { background-color: #f0f4f8; border-left: 3px solid #3498db; padding: 8px 12px; margin: 10px 0; font-style: italic; }
    .result { margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #ddd; }
    .title { font-size: 11pt; font-weight: bold; color: #2c3e50; margin-bottom: 4px; }
    .info { font-size: 9pt; color: #666; margin-left: 10px; }
    .paper-id { font-family: Courier, monospace; font-size: 8pt; color: #888; }
    .matched { margin-top: 6px; margin-left: 10px; padding: 6px; background-color: #fafafa; font-size: 9pt; }
    blockquote { border-left: 2px solid #ccc; margin-left: 10px; padding-left: 8px; color: #555; font-style: italic; }
    table { border-collapse: collapse; width: 100%; margin: 8px 0; }
    th, td { border: 1px solid #ddd; padding: 4px 6px; text-align: left; font-size: 9pt; }
    th { background-color: #f5f5f5; font-weight: bold; }
    """


def _results_to_html(
    results: list[EnrichedResult],
    query: str,
    include_extraction: bool = False,
) -> str:
    """Convert search results to HTML for PDF rendering."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = [
        "<html><body>",
        "<h1>Literature Search Report</h1>",
        f'<p class="meta">Generated: {timestamp} | Results: {len(results)}</p>',
        "<h2>Search Query</h2>",
        f'<div class="query">{query}</div>',
    ]

    for i, result in enumerate(results, 1):
        year_str = f" ({result.year})" if result.year else ""
        html.append(f'<div class="result">')
        html.append(f'<p class="title">{i}. {result.title}{year_str}</p>')
        html.append(f'<p class="info"><b>Authors:</b> {result.authors or "Unknown"}</p>')
        html.append(f'<p class="info"><b>Score:</b> {result.score:.4f} | <b>Type:</b> {result.item_type} | <b>Match:</b> {result.chunk_type}</p>')

        if result.collections:
            colls = ", ".join(result.collections[:3])
            if len(result.collections) > 3:
                colls += f" (+{len(result.collections) - 3} more)"
            html.append(f'<p class="info"><b>Collections:</b> {colls}</p>')

        html.append(f'<p class="info paper-id">Paper ID: {result.paper_id}</p>')

        matched = result.matched_text[:600] + "..." if len(result.matched_text) > 600 else result.matched_text
        html.append(f'<div class="matched"><b>Matched:</b> {matched}</div>')

        if include_extraction and result.extraction_data:
            ext = result.extraction_data.get("extraction", result.extraction_data)
            if ext.get("thesis_statement"):
                html.append(f'<div class="matched"><b>Thesis:</b> {ext["thesis_statement"][:400]}</div>')

        html.append("</div>")

    html.append("</body></html>")
    return "\n".join(html)


def generate_pdf(
    results: list[EnrichedResult],
    query: str,
    output_path: Path,
    include_extraction: bool = False,
) -> None:
    """Generate PDF report from search results using PyMuPDF Story API.

    Args:
        results: List of search results.
        query: Original search query.
        output_path: Path to save PDF file.
        include_extraction: Whether to include extraction snippets.
    """
    import fitz  # PyMuPDF

    html_content = _results_to_html(results, query, include_extraction)
    css = _get_pdf_css()

    # Page settings
    page_width, page_height = 612, 792  # Letter size
    margin = 54  # 0.75 inch

    try:
        story = fitz.Story(html=html_content, user_css=css)
        writer = fitz.DocumentWriter(str(output_path))

        # Define rectangle function for page layout
        mediabox = fitz.Rect(0, 0, page_width, page_height)
        where = fitz.Rect(margin, margin, page_width - margin, page_height - margin)

        more = True
        while more:
            device = writer.begin_page(mediabox)
            more, _ = story.place(where)
            story.draw(device)
            writer.end_page()

        writer.close()

    except Exception as e:
        logger.warning(f"HTML rendering failed, using simple format: {e}")
        _generate_simple_pdf(results, query, output_path, include_extraction)

    logger.info(f"PDF saved to {output_path}")


def _generate_simple_pdf(
    results: list[EnrichedResult],
    query: str,
    output_path: Path,
    include_extraction: bool = False,
) -> None:
    """Fallback simple PDF generation without HTML rendering."""
    import fitz

    doc = fitz.open()
    page_width, page_height = 612, 792
    margin = 72

    page = doc.new_page(width=page_width, height=page_height)
    y = margin

    # Simple text rendering
    page.insert_text(fitz.Point(margin, y), "Literature Search Report", fontsize=18)
    y += 30
    page.insert_text(fitz.Point(margin, y), f"Query: {query}", fontsize=11)
    y += 20
    page.insert_text(fitz.Point(margin, y), f"Results: {len(results)}", fontsize=10)
    y += 30

    for i, result in enumerate(results, 1):
        if y > page_height - 100:
            page = doc.new_page(width=page_width, height=page_height)
            y = margin

        title = f"{i}. {result.title[:60]}..."
        page.insert_text(fitz.Point(margin, y), title, fontsize=11)
        y += 15
        page.insert_text(fitz.Point(margin + 15, y), f"Authors: {result.authors[:50]}", fontsize=9)
        y += 12
        page.insert_text(fitz.Point(margin + 15, y), f"Score: {result.score:.4f}", fontsize=9)
        y += 20

    doc.save(output_path)
    doc.close()


def save_results(
    results: list[EnrichedResult],
    query: str,
    output_dir: Path | str,
    output_format: OutputFormat = "markdown",
    filename_prefix: str = "search",
    include_extraction: bool = False,
) -> Path:
    """Save results to file.

    Args:
        results: List of search results.
        query: Original search query.
        output_dir: Directory to save results.
        output_format: Output format (default: markdown, also supports pdf).
        filename_prefix: Prefix for output filename.
        include_extraction: Whether to include extraction data.

    Returns:
        Path to saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with date and query slug
    date_str = datetime.now().strftime("%Y-%m-%d")
    query_slug = slugify_query(query)

    # Determine file extension
    if output_format == "json":
        ext = "json"
    elif output_format == "pdf":
        ext = "pdf"
    elif output_format == "markdown":
        ext = "md"
    else:
        ext = "txt"

    filename = f"{date_str}_{query_slug}.{ext}"
    filepath = output_dir / filename

    # Handle PDF separately
    if output_format == "pdf":
        generate_pdf(results, query, filepath, include_extraction)
        # Also save latest
        latest_path = output_dir / f"latest.{ext}"
        generate_pdf(results, query, latest_path, include_extraction)
    else:
        # Format and save text-based formats
        content = format_results(results, query, output_format, include_extraction)
        filepath.write_text(content, encoding="utf-8")
        # Also save as "latest" for easy access
        latest_path = output_dir / f"latest.{ext}"
        latest_path.write_text(content, encoding="utf-8")

    logger.info(f"Saved results to {filepath}")
    return filepath


def convert_markdown_to_pdf(md_path: Path, pdf_path: Path | None = None) -> Path:
    """Convert a markdown file to PDF using proper markdown rendering.

    Args:
        md_path: Path to the markdown file.
        pdf_path: Optional output path. Defaults to same name with .pdf extension.

    Returns:
        Path to the created PDF file.
    """
    import fitz  # PyMuPDF
    import markdown

    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    md_content = md_path.read_text(encoding="utf-8")

    if pdf_path is None:
        pdf_path = md_path.with_suffix(".pdf")

    # Convert markdown to HTML with extensions for tables
    html_content = markdown.markdown(
        md_content,
        extensions=["tables", "fenced_code", "nl2br"],
    )
    # Wrap in html/body tags
    html_content = f"<html><body>{html_content}</body></html>"

    css = _get_pdf_css()

    # Page settings
    page_width, page_height = 612, 792  # Letter size
    margin = 54

    try:
        story = fitz.Story(html=html_content, user_css=css)
        writer = fitz.DocumentWriter(str(pdf_path))

        mediabox = fitz.Rect(0, 0, page_width, page_height)
        where = fitz.Rect(margin, margin, page_width - margin, page_height - margin)

        more = True
        while more:
            device = writer.begin_page(mediabox)
            more, _ = story.place(where)
            story.draw(device)
            writer.end_page()

        writer.close()

    except Exception as e:
        logger.warning(f"HTML rendering failed, using simple format: {e}")
        _convert_markdown_simple(md_content, pdf_path)

    logger.info(f"Converted {md_path} to {pdf_path}")
    return pdf_path


def _convert_markdown_simple(md_content: str, pdf_path: Path) -> None:
    """Fallback simple markdown to PDF conversion."""
    import fitz

    doc = fitz.open()
    page_width, page_height = 612, 792
    margin = 72
    content_width = page_width - 2 * margin

    def add_page():
        return doc.new_page(width=page_width, height=page_height), margin

    def wrap_text(text: str, font_size: int) -> list[str]:
        char_width = font_size * 0.5
        max_chars = int(content_width / char_width)
        words = text.split()
        lines, current = [], []
        for word in words:
            test = " ".join(current + [word])
            if len(test) <= max_chars:
                current.append(word)
            else:
                if current:
                    lines.append(" ".join(current))
                current = [word]
        if current:
            lines.append(" ".join(current))
        return lines or [""]

    page, y = add_page()

    for line in md_content.split("\n"):
        stripped = line.strip()

        if y > page_height - margin - 20:
            page, y = add_page()

        if stripped.startswith("# "):
            text, font_size = stripped[2:], 18
            y += 10
        elif stripped.startswith("## "):
            text, font_size = stripped[3:], 14
            y += 8
        elif stripped.startswith("### "):
            text, font_size = stripped[4:], 12
            y += 5
        elif stripped.startswith("> "):
            text, font_size = "    " + stripped[2:], 10
        elif stripped == "---":
            page.draw_line(
                fitz.Point(margin, y),
                fitz.Point(page_width - margin, y),
                color=(0.7, 0.7, 0.7), width=0.5
            )
            y += 10
            continue
        elif stripped == "":
            y += 8
            continue
        else:
            text, font_size = stripped, 10

        for ln in wrap_text(text, font_size):
            if y > page_height - margin - font_size:
                page, y = add_page()
            page.insert_text(fitz.Point(margin, y), ln, fontsize=font_size)
            y += font_size + 3

    doc.save(pdf_path)
    doc.close()


def format_paper_detail(
    paper_data: dict,
    extraction_data: dict | None = None,
) -> str:
    """Format detailed paper information as Markdown.

    Args:
        paper_data: Paper metadata dictionary.
        extraction_data: Extraction data dictionary.

    Returns:
        Markdown formatted string.
    """
    lines = [
        f"# {paper_data.get('title', 'Unknown Title')}",
        "",
    ]

    # Metadata section
    lines.append("## Metadata")
    lines.append("")

    if paper_data.get("authors"):
        if isinstance(paper_data["authors"], list):
            author_names = [
                a.get("full_name", f"{a.get('first_name', '')} {a.get('last_name', '')}")
                for a in paper_data["authors"]
            ]
            lines.append(f"**Authors:** {', '.join(author_names)}")
        else:
            lines.append(f"**Authors:** {paper_data.get('author_string', 'Unknown')}")

    if paper_data.get("publication_year"):
        lines.append(f"**Year:** {paper_data['publication_year']}")

    if paper_data.get("journal"):
        lines.append(f"**Journal:** {paper_data['journal']}")

    if paper_data.get("item_type"):
        lines.append(f"**Type:** {paper_data['item_type']}")

    if paper_data.get("doi"):
        lines.append(f"**DOI:** {paper_data['doi']}")

    if paper_data.get("collections"):
        lines.append(f"**Collections:** {', '.join(paper_data['collections'])}")

    lines.append(f"**Paper ID:** `{paper_data.get('paper_id', 'Unknown')}`")
    lines.append("")

    # Abstract
    if paper_data.get("abstract"):
        lines.extend(
            [
                "## Abstract",
                "",
                paper_data["abstract"],
                "",
            ]
        )

    # Extraction data
    if extraction_data:
        ext = extraction_data.get("extraction", extraction_data)

        if ext.get("thesis_statement"):
            lines.extend(
                [
                    "## Thesis Statement",
                    "",
                    ext["thesis_statement"],
                    "",
                ]
            )

        if ext.get("research_questions"):
            lines.extend(
                [
                    "## Research Questions",
                    "",
                ]
            )
            for rq in ext["research_questions"]:
                lines.append(f"- {rq}")
            lines.append("")

        if ext.get("methodology"):
            method = ext["methodology"]
            lines.extend(
                [
                    "## Methodology",
                    "",
                ]
            )
            if isinstance(method, dict):
                if method.get("approach"):
                    lines.append(f"**Approach:** {method['approach']}")
                if method.get("design"):
                    lines.append(f"**Design:** {method['design']}")
                if method.get("data_sources"):
                    lines.append(f"**Data Sources:** {', '.join(method['data_sources'])}")
                if method.get("analysis_methods"):
                    lines.append(f"**Analysis:** {', '.join(method['analysis_methods'])}")
                if method.get("sample_size"):
                    lines.append(f"**Sample:** {method['sample_size']}")
            else:
                lines.append(str(method))
            lines.append("")

        if ext.get("key_findings"):
            lines.extend(
                [
                    "## Key Findings",
                    "",
                ]
            )
            for finding in ext["key_findings"]:
                if isinstance(finding, dict):
                    lines.append(f"- **{finding.get('finding', '')}**")
                    if finding.get("evidence_type"):
                        lines.append(f"  - Evidence: {finding['evidence_type']}")
                    if finding.get("significance"):
                        lines.append(f"  - Significance: {finding['significance']}")
                else:
                    lines.append(f"- {finding}")
            lines.append("")

        if ext.get("conclusions"):
            lines.extend(
                [
                    "## Conclusions",
                    "",
                    ext["conclusions"],
                    "",
                ]
            )

        if ext.get("limitations"):
            lines.extend(
                [
                    "## Limitations",
                    "",
                ]
            )
            for lim in ext["limitations"]:
                lines.append(f"- {lim}")
            lines.append("")

        if ext.get("future_directions"):
            lines.extend(
                [
                    "## Future Directions",
                    "",
                ]
            )
            for fd in ext["future_directions"]:
                lines.append(f"- {fd}")
            lines.append("")

        if ext.get("contribution_summary"):
            lines.extend(
                [
                    "## Contribution Summary",
                    "",
                    ext["contribution_summary"],
                    "",
                ]
            )

        if ext.get("discipline_tags"):
            lines.append(f"**Discipline Tags:** {', '.join(ext['discipline_tags'])}")
            lines.append("")

        if ext.get("extraction_confidence"):
            lines.append(f"**Extraction Confidence:** {ext['extraction_confidence']:.2f}")
            lines.append("")

    return "\n".join(lines)


def format_summary(summary: dict) -> str:
    """Format index summary as Markdown.

    Args:
        summary: Summary dictionary from structured store.

    Returns:
        Markdown formatted string.
    """
    lines = [
        "# Index Summary",
        "",
        f"**Generated:** {summary.get('generated_at', 'Unknown')}",
        f"**Total Papers:** {summary.get('total_papers', 0)}",
        f"**Total Extractions:** {summary.get('total_extractions', 0)}",
        "",
    ]

    # Papers by type
    if summary.get("papers_by_type"):
        lines.extend(
            [
                "## Papers by Type",
                "",
            ]
        )
        for item_type, count in sorted(
            summary["papers_by_type"].items(), key=lambda x: -x[1]
        ):
            lines.append(f"- {item_type}: {count}")
        lines.append("")

    # Papers by year
    if summary.get("papers_by_year"):
        lines.extend(
            [
                "## Papers by Year",
                "",
            ]
        )
        for year, count in sorted(summary["papers_by_year"].items(), reverse=True)[:10]:
            lines.append(f"- {year}: {count}")
        lines.append("")

    # Papers by collection
    if summary.get("papers_by_collection"):
        lines.extend(
            [
                "## Top Collections",
                "",
            ]
        )
        for coll, count in list(summary["papers_by_collection"].items())[:10]:
            lines.append(f"- {coll}: {count}")
        lines.append("")

    # Top disciplines
    if summary.get("top_disciplines"):
        lines.extend(
            [
                "## Top Disciplines",
                "",
            ]
        )
        for disc, count in list(summary["top_disciplines"].items())[:10]:
            lines.append(f"- {disc}: {count}")
        lines.append("")

    # Recent papers
    if summary.get("recent_papers"):
        lines.extend(
            [
                "## Recently Added",
                "",
            ]
        )
        for paper in summary["recent_papers"][:5]:
            year_str = f" ({paper.get('year')})" if paper.get("year") else ""
            lines.append(f"- {paper.get('title', 'Unknown')}{year_str}")
        lines.append("")

    # Vector store stats
    if summary.get("vector_store"):
        vs = summary["vector_store"]
        lines.extend(
            [
                "## Vector Store",
                "",
                f"**Total Chunks:** {vs.get('total_chunks', 0)}",
                f"**Unique Papers:** {vs.get('unique_papers', 0)}",
                "",
            ]
        )

    return "\n".join(lines)
