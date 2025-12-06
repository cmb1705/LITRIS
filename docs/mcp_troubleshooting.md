# LITRIS MCP Troubleshooting Guide

## Common Issues

### Server Not Starting

**Symptoms**: MCP server fails to initialize or Claude Code doesn't discover tools.

**Causes and Solutions**:

| Cause | Solution |
|-------|----------|
| Missing dependencies | Run `pip install -r requirements.txt` |
| Wrong Python path | Ensure `.mcp.json` uses correct Python executable |
| Config not found | Verify `config.yaml` exists in project root |

**Diagnostic Command**:
```bash
python -m src.mcp.server
```

### Tools Not Discovered

**Symptoms**: Claude Code doesn't show LITRIS tools.

**Solutions**:

1. Verify `.mcp.json` exists in project root
2. Check `.claude/settings.json` has `"enableAllProjectMcpServers": true`
3. Restart Claude Code session
4. Check server logs in `data/logs/mcp_server.log`

### Search Returns No Results

**Symptoms**: `litris_search` returns empty results.

**Causes and Solutions**:

| Cause | Solution |
|-------|----------|
| Empty vector store | Run `/build` to rebuild index |
| Wrong index path | Verify `data/index/chroma` exists |
| Query too specific | Try broader search terms |

**Diagnostic Command**:
```python
from src.mcp.adapters import LitrisAdapter
adapter = LitrisAdapter()
print(adapter.get_summary())
```

### Paper Not Found

**Symptoms**: `litris_get_paper` returns "paper not found".

**Solutions**:

1. Verify paper ID format (alphanumeric with underscores/hyphens)
2. Use `litris_search` first to find valid paper IDs
3. Check if paper exists in `data/index/papers_index.json`

### Slow Response Times

**Symptoms**: Tools take more than 5 seconds to respond.

**Causes and Solutions**:

| Cause | Solution |
|-------|----------|
| First query (cold start) | Model loading is slow on first use |
| Large result sets | Reduce `top_k` parameter |
| CPU-only mode | GPU acceleration not available |

### Configuration Issues

**Check Config Loading**:
```python
from src.config import Config
config = Config.load()
print(f"Project root: {config._project_root}")
print(f"Index path: {config._project_root / 'data/index'}")
```

## Log Locations

| Log | Location | Contents |
|-----|----------|----------|
| MCP Server | `data/logs/mcp_server.log` | Tool calls, errors |
| Operations | `data/logs/operations.log` | File operations |
| Application | `data/logs/lit_review_*.log` | General logging |

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `MCP_LOG_LEVEL` | Server log level | INFO |
| `ANTHROPIC_API_KEY` | API key (not needed for MCP) | - |

## Verification Checklist

- [ ] Python 3.10+ installed
- [ ] Virtual environment activated
- [ ] Dependencies installed
- [ ] `config.yaml` present and valid
- [ ] Index built (`data/index/` populated)
- [ ] ChromaDB populated (`data/index/chroma/`)
- [ ] `.mcp.json` configured
- [ ] Claude Code settings enable MCP servers

## Getting Help

- Check logs in `data/logs/`
- Run diagnostic commands above
- Review [STATE.md](../STATE.md) for current status
