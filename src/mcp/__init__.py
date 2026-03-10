"""LITRIS MCP Server Module.

This module provides Model Context Protocol (MCP) integration for LITRIS,
enabling Claude Code to directly query the literature index through
semantic search, paper retrieval, and index analysis tools.

Tools:
    litris_search: Semantic search with filters
    litris_get_paper: Retrieve full paper extraction
    litris_similar: Find similar papers
    litris_summary: Index statistics
    litris_collections: List collections
"""

from src.mcp.server import create_server, run_server

__all__ = ["create_server", "run_server"]
