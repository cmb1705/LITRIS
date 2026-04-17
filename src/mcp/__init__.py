"""LITRIS MCP Server Module.

This module provides Model Context Protocol (MCP) integration for LITRIS,
enabling Claude Code to directly query the literature index through
semantic search, paper retrieval, and index analysis tools.

Tools:
    litris_search: Semantic search with filters
    litris_get_paper: Retrieve full paper extraction
    litris_get_fulltext_context: Retrieve verbatim full-text contexts
    litris_similar: Find similar papers
    litris_summary: Index statistics
    litris_collections: List collections
"""


def create_server():
    """Create and return the MCP server instance (lazy import)."""
    from src.mcp.server import create_server as _create

    return _create()


def run_server():
    """Run the MCP server (lazy import)."""
    from src.mcp.server import run_server as _run

    _run()


__all__ = ["create_server", "run_server"]
