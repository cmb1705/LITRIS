"""Entry point for running the LITRIS MCP server as a package.

Usage:
    python -m src.mcp
"""

from src.mcp.server import run_server

if __name__ == "__main__":
    run_server()
