@echo off
rem LITRIS MCP server launcher.
rem Self-locates relative to this script so callers (Claude Code, other MCP
rem clients) do not need to set cwd or PYTHONPATH explicitly. Needed because
rem Claude Code on Windows does not reliably honor .mcp.json cwd/env fields.
cd /d "%~dp0.."
set "PYTHONPATH=%~dp0.."
"%~dp0..\.venv\Scripts\python.exe" -m src.mcp %*
