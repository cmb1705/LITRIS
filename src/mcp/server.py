"""LITRIS MCP Server.

This module implements the MCP server that exposes LITRIS search
functionality as tools callable by Claude Code.
"""

import asyncio
import os
import signal
import sys
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.mcp.adapters import LitrisAdapter
from src.mcp.validators import (
    ValidationError,
    validate_chunk_types,
    validate_max_rounds,
    validate_n_variants,
    validate_paper_id,
    validate_quality_min,
    validate_query,
    validate_recency_boost,
    validate_top_k,
    validate_year,
)
from src.utils.logging_config import get_logger, setup_logging

# Initialize MCP-specific logging
_log_level = os.getenv("MCP_LOG_LEVEL", "INFO")
setup_logging(level=_log_level, log_file="mcp_server.log", console=False)

logger = get_logger(__name__)

# Create the MCP server instance
mcp = FastMCP(
    name="litris",
    instructions="LITRIS provides semantic search over an academic literature index. "
    "Use litris_search for finding papers, litris_search_rrf for comprehensive "
    "multi-query search with improved recall, litris_search_agentic for multi-round "
    "search with gap analysis, litris_deep_review for generating integrated "
    "literature reviews, litris_get_paper for full details, "
    "litris_similar for related papers, litris_summary for index stats, "
    "litris_collections for available collections, litris_save_query to save "
    "query results to the query_results folder (content must include the "
    "original query verbatim in a blockquote before analysis), "
    "litris_search_dimension to search within a specific SemanticAnalysis "
    "dimension (q01-q40, e.g. q07_methods for methodology), and "
    "litris_search_group to search across a group of dimensions by analysis "
    "pass (research_core, methodology, contribution, context, synthesis, deep).",
)

@lru_cache(maxsize=1)
def get_adapter() -> LitrisAdapter:
    """Get or create the LITRIS adapter instance.

    Uses lru_cache for thread-safe lazy initialization.
    """
    logger.info("Initializing LITRIS adapter...")
    adapter = LitrisAdapter()
    logger.info("LITRIS adapter initialized successfully")
    return adapter


@mcp.tool()
async def litris_search(
    query: str,
    top_k: int = 10,
    chunk_types: list[str] | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    collections: list[str] | None = None,
    item_types: list[str] | None = None,
    include_extraction: bool = True,
    recency_boost: float = 0.0,
    quality_min: int | None = None,
) -> dict[str, Any]:
    """Search the LITRIS literature index semantically.

    Args:
        query: Natural language search query
        top_k: Number of results to return (default: 10, max: 50)
        chunk_types: Filter by chunk type (dim_q01-q40, raptor_overview, raptor_core, abstract)
        year_min: Minimum publication year
        year_max: Maximum publication year
        collections: Filter by Zotero collection names
        item_types: Filter by item type (journalArticle, book, etc.)
        include_extraction: Include full extraction data (default: True)
        recency_boost: Boost recent papers 0.0-1.0 (default: 0.0)
        quality_min: Minimum paper quality rating 1-5 (only rated papers returned)

    Returns:
        Search results with paper metadata and extractions
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    logger.info(f"[{request_id}] litris_search called: query='{query[:50]}...' top_k={top_k}")

    try:
        # Validate inputs
        validate_query(query)
        top_k = validate_top_k(top_k)
        if year_min is not None:
            validate_year(year_min, "year_min")
        if year_max is not None:
            validate_year(year_max, "year_max")
        if chunk_types is not None:
            validate_chunk_types(chunk_types)
        recency_boost = validate_recency_boost(recency_boost)
        if quality_min is not None:
            validate_quality_min(quality_min)

        adapter = get_adapter()
        results = adapter.search(
            query=query,
            top_k=top_k,
            chunk_types=chunk_types,
            year_min=year_min,
            year_max=year_max,
            collections=collections,
            item_types=item_types,
            include_extraction=include_extraction,
            recency_boost=recency_boost,
            quality_min=quality_min,
        )

        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] litris_search returning {results.get('result_count', 0)} results in {elapsed:.3f}s")
        return results

    except ValidationError as e:
        elapsed = time.time() - start_time
        logger.warning(f"[{request_id}] Validation error in {elapsed:.3f}s: {e}")
        return {"error": "VALIDATION_ERROR", "message": str(e), "result_count": 0, "results": []}

    except FileNotFoundError as e:
        elapsed = time.time() - start_time
        logger.error(f"[{request_id}] Index not found in {elapsed:.3f}s: {e}")
        return {"error": "INDEX_NOT_FOUND", "message": "Literature index not found. Run /build to create.", "result_count": 0, "results": []}

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[{request_id}] Search failed in {elapsed:.3f}s: {e}")
        return {"error": "SEARCH_FAILED", "message": str(e), "result_count": 0, "results": []}


@mcp.tool()
async def litris_search_rrf(
    query: str,
    top_k: int = 10,
    n_variants: int = 4,
    chunk_types: list[str] | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    collections: list[str] | None = None,
    item_types: list[str] | None = None,
    include_extraction: bool = True,
    recency_boost: float = 0.0,
    quality_min: int | None = None,
) -> dict[str, Any]:
    """Multi-query search using Reciprocal Rank Fusion for improved recall.

    Generates query reformulations and fuses results across multiple searches.
    Slower than litris_search but finds more relevant papers by approaching the
    topic from multiple angles. Use for comprehensive literature searches.

    Args:
        query: Natural language search query
        top_k: Number of results to return (default: 10, max: 50)
        n_variants: Number of query reformulations (default: 4, max: 10)
        chunk_types: Filter by chunk type (dim_q01-q40, raptor_overview, raptor_core, abstract)
        year_min: Minimum publication year
        year_max: Maximum publication year
        collections: Filter by Zotero collection names
        item_types: Filter by item type (journalArticle, book, etc.)
        include_extraction: Include full extraction data (default: True)
        recency_boost: Boost recent papers 0.0-1.0 (default: 0.0)
        quality_min: Minimum paper quality rating 1-5 (only rated papers returned)

    Returns:
        Search results with query variants used and paper metadata
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    logger.info(f"[{request_id}] litris_search_rrf called: query='{query[:50]}...' top_k={top_k} variants={n_variants}")

    try:
        validate_query(query)
        top_k = validate_top_k(top_k)
        n_variants = validate_n_variants(n_variants)
        if year_min is not None:
            validate_year(year_min, "year_min")
        if year_max is not None:
            validate_year(year_max, "year_max")
        if chunk_types is not None:
            validate_chunk_types(chunk_types)
        recency_boost = validate_recency_boost(recency_boost)
        if quality_min is not None:
            validate_quality_min(quality_min)

        adapter = get_adapter()
        results = adapter.search_rrf(
            query=query,
            top_k=top_k,
            n_variants=n_variants,
            chunk_types=chunk_types,
            year_min=year_min,
            year_max=year_max,
            collections=collections,
            item_types=item_types,
            include_extraction=include_extraction,
            recency_boost=recency_boost,
            quality_min=quality_min,
        )

        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] litris_search_rrf returning {results.get('result_count', 0)} results in {elapsed:.3f}s")
        return results

    except ValidationError as e:
        elapsed = time.time() - start_time
        logger.warning(f"[{request_id}] Validation error in {elapsed:.3f}s: {e}")
        return {"error": "VALIDATION_ERROR", "message": str(e), "result_count": 0, "results": []}

    except FileNotFoundError as e:
        elapsed = time.time() - start_time
        logger.error(f"[{request_id}] Index not found in {elapsed:.3f}s: {e}")
        return {"error": "INDEX_NOT_FOUND", "message": "Literature index not found. Run /build to create.", "result_count": 0, "results": []}

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[{request_id}] RRF search failed in {elapsed:.3f}s: {e}")
        return {"error": "SEARCH_FAILED", "message": str(e), "result_count": 0, "results": []}


@mcp.tool()
async def litris_search_agentic(
    query: str,
    top_k: int = 10,
    max_rounds: int = 2,
    chunk_types: list[str] | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    collections: list[str] | None = None,
    item_types: list[str] | None = None,
    include_extraction: bool = True,
    recency_boost: float = 0.0,
    quality_min: int | None = None,
) -> dict[str, Any]:
    """Multi-round agentic search with gap analysis for thorough literature coverage.

    Performs an initial search, then uses an LLM to analyze results for topical
    gaps and generates follow-up queries to fill them. Repeats for up to
    max_rounds. Best for comprehensive literature reviews where recall matters
    more than speed.

    Args:
        query: Natural language search query
        top_k: Number of results to return (default: 10, max: 50)
        max_rounds: Maximum gap-analysis rounds (default: 2, max: 5)
        chunk_types: Filter by chunk type (dim_q01-q40, raptor_overview, raptor_core, abstract)
        year_min: Minimum publication year
        year_max: Maximum publication year
        collections: Filter by Zotero collection names
        item_types: Filter by item type (journalArticle, book, etc.)
        include_extraction: Include full extraction data (default: True)
        recency_boost: Boost recent papers 0.0-1.0 (default: 0.0)
        quality_min: Minimum paper quality rating 1-5 (only rated papers returned)

    Returns:
        Search results with round-by-round metadata including gaps identified
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    logger.info(
        f"[{request_id}] litris_search_agentic called: "
        f"query='{query[:50]}...' top_k={top_k} max_rounds={max_rounds}"
    )

    try:
        validate_query(query)
        top_k = validate_top_k(top_k)
        max_rounds = validate_max_rounds(max_rounds)
        if year_min is not None:
            validate_year(year_min, "year_min")
        if year_max is not None:
            validate_year(year_max, "year_max")
        if chunk_types is not None:
            validate_chunk_types(chunk_types)
        recency_boost = validate_recency_boost(recency_boost)
        if quality_min is not None:
            validate_quality_min(quality_min)

        adapter = get_adapter()
        results = adapter.search_agentic(
            query=query,
            top_k=top_k,
            max_rounds=max_rounds,
            chunk_types=chunk_types,
            year_min=year_min,
            year_max=year_max,
            collections=collections,
            item_types=item_types,
            include_extraction=include_extraction,
            recency_boost=recency_boost,
            quality_min=quality_min,
        )

        elapsed = time.time() - start_time
        logger.info(
            f"[{request_id}] litris_search_agentic returning "
            f"{results.get('result_count', 0)} results in {elapsed:.3f}s"
        )
        return results

    except ValidationError as e:
        elapsed = time.time() - start_time
        logger.warning(f"[{request_id}] Validation error in {elapsed:.3f}s: {e}")
        return {"error": "VALIDATION_ERROR", "message": str(e), "result_count": 0, "results": []}

    except FileNotFoundError as e:
        elapsed = time.time() - start_time
        logger.error(f"[{request_id}] Index not found in {elapsed:.3f}s: {e}")
        return {"error": "INDEX_NOT_FOUND", "message": "Literature index not found. Run /build to create.", "result_count": 0, "results": []}

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[{request_id}] Agentic search failed in {elapsed:.3f}s: {e}")
        return {"error": "SEARCH_FAILED", "message": str(e), "result_count": 0, "results": []}


@mcp.tool()
async def litris_deep_review(
    topic: str,
    top_k: int = 20,
    max_rounds: int = 2,
    verify: bool = True,
) -> dict[str, Any]:
    """Generate an integrated literature review on a topic.

    Executes a 4-phase pipeline: discovery (agentic search), reading
    (load extractions), synthesis (LLM-generated review), and QA
    (citation verification). Produces a 3000-5000 word review.

    This is a long-running operation (30-120 seconds depending on corpus size).

    Args:
        topic: Research topic or question for the review
        top_k: Number of papers to synthesize (default: 20, max: 50)
        max_rounds: Gap-analysis rounds for discovery (default: 2, max: 5)
        verify: Run citation QA verification (default: True)

    Returns:
        Literature review with source papers and QA results
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    logger.info(
        f"[{request_id}] litris_deep_review called: "
        f"topic='{topic[:50]}...' top_k={top_k}"
    )

    try:
        validate_query(topic)
        top_k = validate_top_k(top_k)
        max_rounds = validate_max_rounds(max_rounds)

        adapter = get_adapter()
        result = adapter.deep_review(
            topic=topic,
            top_k=top_k,
            max_rounds=max_rounds,
            verify=verify,
        )

        elapsed = time.time() - start_time
        logger.info(
            f"[{request_id}] litris_deep_review complete: "
            f"{result.get('papers_used', 0)} papers in {elapsed:.3f}s"
        )
        return result

    except ValidationError as e:
        elapsed = time.time() - start_time
        logger.warning(f"[{request_id}] Validation error in {elapsed:.3f}s: {e}")
        return {"error": "VALIDATION_ERROR", "message": str(e)}

    except FileNotFoundError as e:
        elapsed = time.time() - start_time
        logger.error(f"[{request_id}] Index not found in {elapsed:.3f}s: {e}")
        return {"error": "INDEX_NOT_FOUND", "message": "Literature index not found. Run /build to create."}

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[{request_id}] Deep review failed in {elapsed:.3f}s: {e}")
        return {"error": "REVIEW_FAILED", "message": str(e)}


@mcp.tool()
async def litris_get_paper(paper_id: str) -> dict[str, Any]:
    """Get full details for a specific paper.

    Args:
        paper_id: LITRIS paper identifier

    Returns:
        Complete paper metadata and extraction data
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    logger.info(f"[{request_id}] litris_get_paper called: paper_id='{paper_id}'")

    try:
        validate_paper_id(paper_id)

        adapter = get_adapter()
        result = adapter.get_paper(paper_id)

        elapsed = time.time() - start_time
        if result.get("found"):
            logger.info(f"[{request_id}] litris_get_paper: found paper '{result.get('paper', {}).get('title', 'Unknown')[:50]}' in {elapsed:.3f}s")
        else:
            logger.warning(f"[{request_id}] litris_get_paper: paper not found: {paper_id} in {elapsed:.3f}s")

        return result

    except ValidationError as e:
        elapsed = time.time() - start_time
        logger.warning(f"[{request_id}] Validation error in {elapsed:.3f}s: {e}")
        return {"error": "VALIDATION_ERROR", "message": str(e), "paper_id": paper_id, "found": False}

    except FileNotFoundError as e:
        elapsed = time.time() - start_time
        logger.error(f"[{request_id}] Index not found in {elapsed:.3f}s: {e}")
        return {"error": "INDEX_NOT_FOUND", "message": "Literature index not found. Run /build to create.", "paper_id": paper_id, "found": False}

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[{request_id}] Get paper failed in {elapsed:.3f}s: {e}")
        return {"error": "SEARCH_FAILED", "message": str(e), "paper_id": paper_id, "found": False}


@mcp.tool()
async def litris_similar(paper_id: str, top_k: int = 10) -> dict[str, Any]:
    """Find papers similar to a given paper.

    Args:
        paper_id: Source paper identifier
        top_k: Number of similar papers to return (default: 10)

    Returns:
        List of similar papers with similarity scores
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    logger.info(f"[{request_id}] litris_similar called: paper_id='{paper_id}' top_k={top_k}")

    try:
        validate_paper_id(paper_id)
        top_k = validate_top_k(top_k)

        adapter = get_adapter()
        results = adapter.find_similar(paper_id, top_k)

        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] litris_similar returning {results.get('result_count', 0)} similar papers in {elapsed:.3f}s")
        return results

    except ValidationError as e:
        elapsed = time.time() - start_time
        logger.warning(f"[{request_id}] Validation error in {elapsed:.3f}s: {e}")
        return {"error": "VALIDATION_ERROR", "message": str(e), "source_paper_id": paper_id, "result_count": 0, "similar_papers": []}

    except FileNotFoundError as e:
        elapsed = time.time() - start_time
        logger.error(f"[{request_id}] Index not found in {elapsed:.3f}s: {e}")
        return {"error": "INDEX_NOT_FOUND", "message": "Literature index not found. Run /build to create.", "source_paper_id": paper_id, "result_count": 0, "similar_papers": []}

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[{request_id}] Similar search failed in {elapsed:.3f}s: {e}")
        return {"error": "SEARCH_FAILED", "message": str(e), "source_paper_id": paper_id, "result_count": 0, "similar_papers": []}


@mcp.tool()
async def litris_clusters(
    min_cluster_size: int = 5,
) -> dict[str, Any]:
    """Run topic clustering on paper embeddings using UMAP + HDBSCAN.

    Identifies topic groups in the corpus by clustering raptor_overview embeddings.
    Returns cluster assignments with representative paper titles.

    Args:
        min_cluster_size: Minimum papers per cluster (default: 5)

    Returns:
        Cluster assignments, sizes, and representative papers
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    logger.info(f"[{request_id}] litris_clusters called: min_cluster_size={min_cluster_size}")

    try:
        adapter = get_adapter()
        result = adapter.get_clusters(min_cluster_size=min_cluster_size)

        elapsed = time.time() - start_time
        logger.info(
            f"[{request_id}] litris_clusters: {result.get('n_clusters', 0)} clusters "
            f"in {elapsed:.3f}s"
        )
        return result

    except FileNotFoundError as e:
        elapsed = time.time() - start_time
        logger.error(f"[{request_id}] Index not found in {elapsed:.3f}s: {e}")
        return {"error": "INDEX_NOT_FOUND", "message": "Literature index not found. Run /build to create."}

    except ImportError as e:
        elapsed = time.time() - start_time
        logger.error(f"[{request_id}] Missing dependency in {elapsed:.3f}s: {e}")
        return {"error": "MISSING_DEPENDENCY", "message": f"Install required packages: {e}"}

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[{request_id}] Clustering failed in {elapsed:.3f}s: {e}")
        return {"error": "CLUSTERING_FAILED", "message": str(e)}


@mcp.tool()
async def litris_summary() -> dict[str, Any]:
    """Get summary statistics for the LITRIS index.

    Returns:
        Index statistics including paper counts, collections, and disciplines
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    logger.info(f"[{request_id}] litris_summary called")

    try:
        adapter = get_adapter()
        summary = adapter.get_summary()

        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] litris_summary: {summary.get('total_papers', 0)} papers indexed in {elapsed:.3f}s")
        return summary

    except FileNotFoundError as e:
        elapsed = time.time() - start_time
        logger.error(f"[{request_id}] Index not found in {elapsed:.3f}s: {e}")
        return {"error": "INDEX_NOT_FOUND", "message": "Literature index not found. Run /build to create.", "total_papers": 0}

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[{request_id}] Summary failed in {elapsed:.3f}s: {e}")
        return {"error": "SEARCH_FAILED", "message": str(e), "total_papers": 0}


@mcp.tool()
async def litris_collections() -> dict[str, Any]:
    """List all collections in the LITRIS index.

    Returns:
        List of collection names with paper counts
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    logger.info(f"[{request_id}] litris_collections called")

    try:
        adapter = get_adapter()
        collections = adapter.get_collections()

        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] litris_collections: {len(collections.get('collections', []))} collections in {elapsed:.3f}s")
        return collections

    except FileNotFoundError as e:
        elapsed = time.time() - start_time
        logger.error(f"[{request_id}] Index not found in {elapsed:.3f}s: {e}")
        return {"error": "INDEX_NOT_FOUND", "message": "Literature index not found. Run /build to create.", "collections": []}

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[{request_id}] Collections failed in {elapsed:.3f}s: {e}")
        return {"error": "SEARCH_FAILED", "message": str(e), "collections": []}


@mcp.tool()
async def litris_save_query(
    content: str,
    query: str,
    title: str | None = None,
    generate_pdf: bool = True,
) -> dict[str, Any]:
    """Save query results to the query_results folder.

    IMPORTANT: The content must include the original query/prompt verbatim
    in a "Search Query" section (blockquote format) immediately after the
    metadata header and before any analysis. Reports must be self-contained
    so any reader can understand what questions are being answered.
    See CLAUDE.md "Report Standards" for the full format specification.

    Args:
        content: Markdown content to save. Must begin with a title, metadata
            header, and the original query in a blockquote before analysis.
        query: The original search query (used for filename generation)
        title: Optional custom title for the file (overrides query-based slug)
        generate_pdf: Whether to also generate a PDF version (default: True)

    Returns:
        Paths to saved files and status
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    logger.info(f"[{request_id}] litris_save_query called: query='{query[:50]}...'")

    try:
        from datetime import datetime

        from src.query.retrieval import _get_pdf_css, slugify_query

        # Determine output directory
        output_dir = Path("data/query_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        date_str = datetime.now().strftime("%Y-%m-%d")
        if title:
            slug = slugify_query(title)
        else:
            slug = slugify_query(query)

        md_filename = f"{date_str}_{slug}.md"
        md_path = output_dir / md_filename

        # Save markdown
        md_path.write_text(content, encoding="utf-8")

        # Update latest.md
        latest_md = output_dir / "latest.md"
        latest_md.write_text(content, encoding="utf-8")

        saved_files = [str(md_path), str(latest_md)]

        # Generate PDF if requested
        if generate_pdf:
            try:
                import fitz
                import markdown

                html_content = markdown.markdown(
                    content,
                    extensions=["tables", "fenced_code", "nl2br"],
                )
                html_content = f"<html><body>{html_content}</body></html>"

                css = _get_pdf_css()
                page_width, page_height = 612, 792
                margin = 54

                # Save dated PDF
                pdf_path = md_path.with_suffix(".pdf")
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

                # Save latest PDF
                latest_pdf = output_dir / "latest.pdf"
                story = fitz.Story(html=html_content, user_css=css)
                writer = fitz.DocumentWriter(str(latest_pdf))

                more = True
                while more:
                    device = writer.begin_page(mediabox)
                    more, _ = story.place(where)
                    story.draw(device)
                    writer.end_page()
                writer.close()

                saved_files.extend([str(pdf_path), str(latest_pdf)])
                logger.info(f"[{request_id}] Generated PDF: {pdf_path}")

            except ImportError as e:
                logger.warning(f"[{request_id}] PDF generation skipped (missing deps): {e}")
            except Exception as e:
                logger.warning(f"[{request_id}] PDF generation failed: {e}")

        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] litris_save_query saved {len(saved_files)} files in {elapsed:.3f}s")

        return {
            "success": True,
            "saved_files": saved_files,
            "primary_file": str(md_path),
            "message": f"Query results saved to {md_path}",
        }

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[{request_id}] Save query failed in {elapsed:.3f}s: {e}")
        return {
            "success": False,
            "error": "SAVE_FAILED",
            "message": str(e),
            "saved_files": [],
        }


@mcp.tool()
async def litris_search_dimension(
    query: str,
    dimension: str,
    top_k: int = 10,
    year_min: int | None = None,
    year_max: int | None = None,
    collections: list[str] | None = None,
    quality_min: int | None = None,
) -> dict[str, Any]:
    """Search within a specific SemanticAnalysis dimension.

    Searches only embeddings from a single question dimension (q01-q40),
    enabling targeted retrieval like "find papers with similar methodology"
    by searching dim_q07 only.

    Args:
        query: Natural language search query
        dimension: Dimension identifier, e.g. "q07" or "q07_methods".
            Short form (q01-q40) or full field name accepted.
        top_k: Number of results to return (default: 10, max: 50)
        year_min: Minimum publication year
        year_max: Maximum publication year
        collections: Filter by Zotero collection names
        quality_min: Minimum paper quality rating 1-5

    Returns:
        Search results filtered to the specified dimension
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    logger.info(f"[{request_id}] litris_search_dimension called: query='{query[:50]}...' dimension={dimension}")

    try:
        from src.query.dimension_search import search_dimension

        validate_query(query)
        top_k = validate_top_k(top_k)
        if year_min is not None:
            validate_year(year_min, "year_min")
        if year_max is not None:
            validate_year(year_max, "year_max")
        if quality_min is not None:
            validate_quality_min(quality_min)

        adapter = get_adapter()
        results = search_dimension(
            engine=adapter.engine,
            query=query,
            dimension=dimension,
            top_k=top_k,
            year_min=year_min,
            year_max=year_max,
            collections=collections,
            quality_min=quality_min,
        )

        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append({
                "rank": i,
                "score": round(result.score, 4),
                "paper_id": result.paper_id,
                "title": result.title,
                "authors": result.authors,
                "year": result.year,
                "chunk_type": result.chunk_type,
                "matched_text": result.matched_text[:500] if result.matched_text else "",
            })

        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] litris_search_dimension returning {len(formatted)} results in {elapsed:.3f}s")
        return {"query": query, "dimension": dimension, "result_count": len(formatted), "results": formatted}

    except ValueError as e:
        return {"error": "VALIDATION_ERROR", "message": str(e), "result_count": 0, "results": []}
    except Exception as e:
        logger.error(f"[{request_id}] Dimension search failed: {e}")
        return {"error": "SEARCH_FAILED", "message": str(e), "result_count": 0, "results": []}


@mcp.tool()
async def litris_search_group(
    query: str,
    group: str,
    top_k: int = 10,
    year_min: int | None = None,
    year_max: int | None = None,
    collections: list[str] | None = None,
    quality_min: int | None = None,
) -> dict[str, Any]:
    """Search across all dimensions in a thematic group.

    Searches embeddings from all questions in a dimension group, enabling
    broad thematic queries like "search only methodology dimensions."

    Args:
        query: Natural language search query
        group: Group name: research_core, methodology, context, meta,
            scholarly, or impact
        top_k: Number of results to return (default: 10, max: 50)
        year_min: Minimum publication year
        year_max: Maximum publication year
        collections: Filter by Zotero collection names
        quality_min: Minimum paper quality rating 1-5

    Returns:
        Search results filtered to the specified dimension group
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    logger.info(f"[{request_id}] litris_search_group called: query='{query[:50]}...' group={group}")

    try:
        from src.query.dimension_search import search_group

        validate_query(query)
        top_k = validate_top_k(top_k)
        if year_min is not None:
            validate_year(year_min, "year_min")
        if year_max is not None:
            validate_year(year_max, "year_max")
        if quality_min is not None:
            validate_quality_min(quality_min)

        adapter = get_adapter()
        results = search_group(
            engine=adapter.engine,
            query=query,
            group=group,
            top_k=top_k,
            year_min=year_min,
            year_max=year_max,
            collections=collections,
            quality_min=quality_min,
        )

        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append({
                "rank": i,
                "score": round(result.score, 4),
                "paper_id": result.paper_id,
                "title": result.title,
                "authors": result.authors,
                "year": result.year,
                "chunk_type": result.chunk_type,
                "matched_text": result.matched_text[:500] if result.matched_text else "",
            })

        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] litris_search_group returning {len(formatted)} results in {elapsed:.3f}s")
        return {"query": query, "group": group, "result_count": len(formatted), "results": formatted}

    except ValueError as e:
        return {"error": "VALIDATION_ERROR", "message": str(e), "result_count": 0, "results": []}
    except Exception as e:
        logger.error(f"[{request_id}] Group search failed: {e}")
        return {"error": "SEARCH_FAILED", "message": str(e), "result_count": 0, "results": []}


def create_server() -> FastMCP:
    """Create and return the MCP server instance.

    Returns:
        Configured FastMCP server
    """
    return mcp


async def run_server_async():
    """Run the MCP server asynchronously."""
    logger.info("Starting LITRIS MCP server...")

    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()
    shutdown_event = asyncio.Event()

    def handle_shutdown(sig=None):
        sig_name = sig.name if sig else "CTRL"
        logger.info(f"Received {sig_name} signal, initiating graceful shutdown...")
        shutdown_event.set()

    # Register signal handlers based on platform
    if sys.platform == "win32":
        # Windows: Use SetConsoleCtrlHandler for graceful shutdown
        try:
            import ctypes
            from ctypes import wintypes

            # Windows console control handler
            CTRL_C_EVENT = 0
            CTRL_BREAK_EVENT = 1
            CTRL_CLOSE_EVENT = 2

            @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
            def console_ctrl_handler(ctrl_type):
                if ctrl_type in (CTRL_C_EVENT, CTRL_BREAK_EVENT, CTRL_CLOSE_EVENT):
                    logger.info(f"Received Windows control event {ctrl_type}")
                    # Schedule shutdown on the event loop
                    loop.call_soon_threadsafe(shutdown_event.set)
                    return True  # Signal handled
                return False

            kernel32 = ctypes.windll.kernel32
            if not kernel32.SetConsoleCtrlHandler(console_ctrl_handler, True):
                logger.warning("Failed to set Windows console control handler")
            else:
                logger.debug("Windows console control handler registered")

        except (ImportError, AttributeError, OSError) as e:
            logger.warning(f"Could not set up Windows signal handler: {e}")
    else:
        # Unix: Use standard signal handlers
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda s=sig: handle_shutdown(s))

    try:
        await mcp.run_stdio_async()
    except KeyboardInterrupt:
        logger.info("Server stopped by keyboard interrupt")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        logger.info("LITRIS MCP server shutdown complete")


def run_server():
    """Run the MCP server (blocking)."""
    asyncio.run(run_server_async())


if __name__ == "__main__":
    run_server()
