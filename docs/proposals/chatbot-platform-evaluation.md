# Chatbot Platform Evaluation for LITRIS

**Issue**: LITRIS-uys
**Date**: 2026-03-06
**Status**: Research complete

## Context

LITRIS needs a conversational interface for semantic search over academic papers. The system has an MCP server (FastMCP, 6 tools) and a Streamlit web UI. Target deployment: home NAS, always-on.

## Platforms Evaluated

### Comparison Matrix

| Criterion | Discord | Teams | Matrix | Telegram | Slack | Streamlit | Chainlit | Open WebUI |
|-----------|---------|-------|--------|----------|-------|-----------|----------|------------|
| Cost | Free | Free* | Free | Free | $7.25+/mo | Free | Free | Free |
| Self-hosted | Partial | No | Full | Partial | No | Full | Full | Full |
| NAS-friendly | Yes | No | Heavy | Yes | No | Yes | Yes | Yes |
| Python SDK | Excellent | Preview | Good | Excellent | Good | IS Python | IS Python | REST only |
| Message limits | 2K/6K | 28K | 65K | 4K | 40K | None | None | None |
| Rich formatting | Good | Good | Moderate | Good | Excellent | Excellent | Excellent | Excellent |
| Setup complexity | Low | High | High | Very Low | Moderate | None | Low | Low |
| MCP integration | Manual | No | Manual | Manual | Manual | Manual | Native | Native |
| Mobile access | Yes | Yes | Yes | Yes | Yes | Browser | Browser | Browser |
| Long-term risk | Low | Medium | Low | Low | Medium | Low | High** | Low |

\* Teams requires Azure account. \*\* Chainlit original team left May 2025; now community-maintained.

### Discord (Recommended for messaging)

- Free, zero ongoing cost, `discord.py` is mature and async
- Runs as lightweight Python process on NAS
- Slash commands + embeds format paper results well
- Mobile app for research access anywhere
- 2K/6K char limit manageable with pagination
- Private server = personal research assistant

### Open WebUI (Recommended for web chat)

- Polished ChatGPT-like UI, Docker-first design
- Native MCP support (streamable HTTP) since v0.6.31
- Built-in RAG complements LITRIS semantic search
- Multi-model support (Ollama local + cloud APIs)
- Active development, frequent releases
- LITRIS MCP server needs streamable HTTP adapter (or use `mcpo` proxy)

### Not Recommended

- **Teams**: Azure dependency, enterprise complexity disproportionate for personal NAS
- **Slack**: $7.25/user/mo for features needed, free plan has 90-day history and 10-integration limits
- **Matrix**: Synapse requires 2-4 GB RAM -- too heavy for NAS running LITRIS + ChromaDB + LLM

### Chainlit (Not Recommended Despite MCP Fit)

- Native MCP support and tool-calling visualization are ideal for LITRIS
- Purpose-built for LLM + tool calling + chat
- **Risk**: Original team abandoned active development May 2025; community-maintained under Maintainer Agreement
- For always-on NAS deployment, long-term maintenance uncertainty is disqualifying

### Simplest Single Option: Telegram

- `python-telegram-bot` (27k+ GitHub stars) is the best-documented Python bot SDK
- Setup takes minutes via BotFather, good mobile access
- 4,096 char limit is main drawback
- Can self-host the Bot API server for local webhooks and no file size limits

## Recommendation

**Dual-interface approach**: Discord (mobile/messaging) + Open WebUI (web chat/deep research)

Both integrate with LITRIS tools, run on NAS via Docker, and have zero ongoing cost. Discord handles quick queries from mobile; Open WebUI handles longer research sessions with full MCP tool orchestration.

## Integration Architecture

```
NAS Docker Compose:
  - litris-mcp-server (FastMCP, streamable HTTP via mcpo)
  - open-webui (connects to MCP server + Ollama)
  - ollama (local LLM for conversation)
  - discord-bot (calls LITRIS search directly)
  - chromadb (vector store)
```

## Sources

- Discord Developer Docs, discord.py API Reference
- Discord Character Limit Guide 2026 (lettercounter.org)
- Open WebUI MCP Documentation (docs.openwebui.com)
- Chainlit MCP Docs, GitHub (Maintainer Agreement post-May 2025)
- Microsoft Teams Bot Development, Azure Bot Service Pricing
- Microsoft Teams SDK Evolution 2025 (voitanos.io)
- Slack Pricing 2026, Rate Limits (viewexport.com)
- Matrix Self-Hosting Guide 2026 (selfhosthero.com)
- Telegram Bot API, python-telegram-bot, Local Bot API Server (tdlib)
- LLM Chat UIs that Support MCP (clickhouse.com)
