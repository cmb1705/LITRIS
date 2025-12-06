"""LITRIS MCP Server.

This module implements the MCP server that exposes LITRIS search
functionality as tools callable by Claude Code.
"""

import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.mcp.adapters import LitrisAdapter
from src.mcp.validators import (
    validate_query,
    validate_paper_id,
    validate_top_k,
    validate_year,
    validate_chunk_types,
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
    "Use litris_search for finding papers, litris_get_paper for full details, "
    "litris_similar for related papers, litris_summary for index stats, "
    "and litris_collections for available collections.",
)

# Global adapter instance (lazy initialized)
_adapter: LitrisAdapter | None = None


def get_adapter() -> LitrisAdapter:
    """Get or create the LITRIS adapter instance."""
    global _adapter
    if _adapter is None:
        logger.info("Initializing LITRIS adapter...")
        _adapter = LitrisAdapter()
        logger.info("LITRIS adapter initialized successfully")
    return _adapter


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
) -> dict[str, Any]:
    """Search the LITRIS literature index semantically.

    Args:
        query: Natural language search query
        top_k: Number of results to return (default: 10, max: 50)
        chunk_types: Filter by extraction section (thesis, methodology, etc.)
        year_min: Minimum publication year
        year_max: Maximum publication year
        collections: Filter by Zotero collection names
        item_types: Filter by item type (journalArticle, book, etc.)
        include_extraction: Include full extraction data (default: True)
        recency_boost: Boost recent papers 0.0-1.0 (default: 0.0)

    Returns:
        Search results with paper metadata and extractions
    """
    logger.info(f"litris_search called: query='{query[:50]}...' top_k={top_k}")

    # Validate inputs
    validate_query(query)
    top_k = validate_top_k(top_k)
    if year_min is not None:
        validate_year(year_min, "year_min")
    if year_max is not None:
        validate_year(year_max, "year_max")
    if chunk_types is not None:
        validate_chunk_types(chunk_types)

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
    )

    logger.info(f"litris_search returning {results.get('result_count', 0)} results")
    return results


@mcp.tool()
async def litris_get_paper(paper_id: str) -> dict[str, Any]:
    """Get full details for a specific paper.

    Args:
        paper_id: LITRIS paper identifier

    Returns:
        Complete paper metadata and extraction data
    """
    logger.info(f"litris_get_paper called: paper_id='{paper_id}'")

    validate_paper_id(paper_id)

    adapter = get_adapter()
    result = adapter.get_paper(paper_id)

    if result.get("found"):
        logger.info(f"litris_get_paper: found paper '{result.get('paper', {}).get('title', 'Unknown')[:50]}'")
    else:
        logger.warning(f"litris_get_paper: paper not found: {paper_id}")

    return result


@mcp.tool()
async def litris_similar(paper_id: str, top_k: int = 10) -> dict[str, Any]:
    """Find papers similar to a given paper.

    Args:
        paper_id: Source paper identifier
        top_k: Number of similar papers to return (default: 10)

    Returns:
        List of similar papers with similarity scores
    """
    logger.info(f"litris_similar called: paper_id='{paper_id}' top_k={top_k}")

    validate_paper_id(paper_id)
    top_k = validate_top_k(top_k)

    adapter = get_adapter()
    results = adapter.find_similar(paper_id, top_k)

    logger.info(f"litris_similar returning {results.get('result_count', 0)} similar papers")
    return results


@mcp.tool()
async def litris_summary() -> dict[str, Any]:
    """Get summary statistics for the LITRIS index.

    Returns:
        Index statistics including paper counts, collections, and disciplines
    """
    logger.info("litris_summary called")

    adapter = get_adapter()
    summary = adapter.get_summary()

    logger.info(f"litris_summary: {summary.get('total_papers', 0)} papers indexed")
    return summary


@mcp.tool()
async def litris_collections() -> dict[str, Any]:
    """List all collections in the LITRIS index.

    Returns:
        List of collection names with paper counts
    """
    logger.info("litris_collections called")

    adapter = get_adapter()
    collections = adapter.get_collections()

    logger.info(f"litris_collections: {len(collections.get('collections', []))} collections")
    return collections


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

    def handle_shutdown(sig):
        logger.info(f"Received signal {sig}, shutting down...")
        loop.stop()

    # Register signal handlers (Unix only)
    if sys.platform != "win32":
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
