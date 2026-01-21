"""LITRIS MCP Server.

This module implements the MCP server that exposes LITRIS search
functionality as tools callable by Claude Code.
"""

import asyncio
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any
import uuid

from mcp.server.fastmcp import FastMCP

from src.mcp.adapters import LitrisAdapter
from src.mcp.validators import (
    ValidationError,
    validate_query,
    validate_paper_id,
    validate_top_k,
    validate_year,
    validate_chunk_types,
    validate_recency_boost,
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
