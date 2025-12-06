"""Result formatting and retrieval utilities."""

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from src.query.search import EnrichedResult
from src.utils.file_utils import safe_write_json
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

OutputFormat = Literal["json", "markdown", "brief"]


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
        output_format: Output format (json, markdown, brief).
        include_extraction: Whether to include full extraction data.

    Returns:
        Formatted string output.
    """
    if output_format == "json":
        return format_json(results, query, include_extraction)
    elif output_format == "markdown":
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
        f"# Search Results",
        "",
        f"**Query:** {query}",
        f"**Results:** {len(results)}",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
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


def save_results(
    results: list[EnrichedResult],
    query: str,
    output_dir: Path | str,
    output_format: OutputFormat = "json",
    filename_prefix: str = "results",
    include_extraction: bool = False,
) -> Path:
    """Save results to file.

    Args:
        results: List of search results.
        query: Original search query.
        output_dir: Directory to save results.
        output_format: Output format.
        filename_prefix: Prefix for output filename.
        include_extraction: Whether to include extraction data.

    Returns:
        Path to saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = "json" if output_format == "json" else "md" if output_format == "markdown" else "txt"
    filename = f"{filename_prefix}_{timestamp}.{ext}"
    filepath = output_dir / filename

    # Format and save
    content = format_results(results, query, output_format, include_extraction)
    filepath.write_text(content, encoding="utf-8")

    # Also save as "latest" for easy access
    latest_path = output_dir / f"latest.{ext}"
    latest_path.write_text(content, encoding="utf-8")

    logger.info(f"Saved results to {filepath}")
    return filepath


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
