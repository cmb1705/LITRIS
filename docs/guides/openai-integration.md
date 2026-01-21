# OpenAI Integration Guide

LITRIS supports OpenAI GPT models for paper extraction, offering two modes of operation:

- **API Mode**: Direct OpenAI API calls (requires API key, pay-per-use)
- **CLI Mode**: Codex CLI with ChatGPT authentication (included with ChatGPT Plus/Pro/Max subscriptions)

## Quick Start

### Option A: Codex CLI (Recommended for ChatGPT Subscribers)

```powershell
# 1. Install Codex CLI
npm i -g @openai/codex

# 2. Authenticate with ChatGPT
codex login

# 3. Verify authentication
codex login status
# Output: Logged in using ChatGPT

# 4. Build index with OpenAI
python scripts/build_index.py --provider openai --mode cli
```

### Option B: OpenAI API Key

```powershell
# 1. Set API key
$env:OPENAI_API_KEY = "sk-..."

# 2. Build index with OpenAI
python scripts/build_index.py --provider openai --mode api --model gpt-4o-mini
```

## Codex CLI Setup (Detailed)

### Installation

**Windows (npm)**:
```powershell
npm i -g @openai/codex
```

**macOS (Homebrew)**:
```bash
brew install --cask codex
```

**Verify installation**:
```powershell
codex --version
# Output: OpenAI Codex v0.88.0 (or newer)
```

### Authentication

Codex CLI authenticates via your ChatGPT account:

```powershell
# Standard OAuth login (opens browser)
codex login

# Headless/server environments
codex login --device-auth

# Check status
codex login status
```

### Supported Models

With ChatGPT authentication, only `gpt-5.2` is currently supported:

| Model | Description | Availability |
|-------|-------------|--------------|
| gpt-5.2 | Latest flagship model | CLI + API |
| gpt-4o | Multimodal, fast | API only |
| gpt-4o-mini | Cost-effective | API only |

LITRIS automatically defaults to `gpt-5.2` when using CLI mode.

## MCP Integration (Codex + LITRIS)

Codex CLI can use LITRIS as an MCP (Model Context Protocol) server, enabling GPT-5.2 to search your literature index directly.

### Setup

```powershell
# Add LITRIS MCP server to Codex (run once)
codex mcp add litris --env "PYTHONPATH=D:\Git_Repos\LITRIS" -- "D:\Git_Repos\LITRIS\.venv\Scripts\python.exe" -m src.mcp.server

# Verify
codex mcp list
```

### Usage

```powershell
# Search your library via Codex
codex "Find papers about network analysis methods in my library"

# Get index statistics
codex "Use litris_summary to show my index statistics"

# Find similar papers
codex "Find papers similar to the one about graph neural networks"
```

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `litris_search` | Semantic search with filters (year, collection, type) |
| `litris_get_paper` | Get full paper details by ID |
| `litris_similar` | Find papers similar to a given paper |
| `litris_summary` | Index statistics (paper counts, disciplines, etc.) |
| `litris_collections` | List available Zotero collections |

## API Mode Setup

### Get an API Key

1. Go to [platform.openai.com](https://platform.openai.com)
2. Navigate to API Keys
3. Create a new secret key

### Configuration

Set the environment variable:

**PowerShell**:
```powershell
$env:OPENAI_API_KEY = "sk-..."
```

**Bash**:
```bash
export OPENAI_API_KEY="sk-..."
```

**Persistent (Windows)**:
```powershell
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "sk-...", "User")
```

### Model Selection

```powershell
# Cost-effective (recommended for large libraries)
python scripts/build_index.py --provider openai --mode api --model gpt-4o-mini

# Higher quality
python scripts/build_index.py --provider openai --mode api --model gpt-4o

# Latest flagship
python scripts/build_index.py --provider openai --mode api --model gpt-5.2
```

## Cost Estimation

### API Mode Pricing (per paper, ~10k chars)

| Model | Estimated Cost |
|-------|---------------|
| gpt-4o-mini | $0.002 |
| gpt-4o | $0.03 |
| gpt-5.2 | $0.09 |

### CLI Mode

Included with ChatGPT subscription:
- **Plus**: $20/month
- **Pro**: $200/month (higher limits)
- **Max**: Higher rate limits

## Troubleshooting

### Codex CLI Not Found

If Python can't find Codex CLI:

```powershell
# Check where it's installed
where codex

# Common location on Windows
C:\Users\<username>\AppData\Roaming\npm\codex.cmd
```

LITRIS automatically searches npm global directories. If issues persist, ensure the npm bin directory is in your PATH.

### Model Not Supported Error

```
The 'gpt-4o-mini' model is not supported when using Codex with a ChatGPT account.
```

This means you're using CLI mode with a model that requires API access. Solutions:
1. Use `gpt-5.2` (default for CLI mode)
2. Switch to API mode with `--mode api`

### Authentication Failed

```powershell
# Re-authenticate
codex logout
codex login

# Check status
codex login status
```

### API Key Invalid

```
Error code: 401 - Incorrect API key provided
```

Verify your API key:
```powershell
# Check if set
echo $env:OPENAI_API_KEY

# Test with curl
curl https://api.openai.com/v1/models -H "Authorization: Bearer $env:OPENAI_API_KEY"
```

## Comparison: Claude vs GPT

| Feature | Claude (Anthropic) | GPT (OpenAI) |
|---------|-------------------|--------------|
| Default Model | claude-sonnet-4-20250514 | gpt-5.2 |
| CLI Tool | claude | codex |
| Subscription Auth | Claude Max | ChatGPT Plus/Pro |
| API Pricing | Similar | Similar |
| Extraction Quality | Excellent | Excellent |

Both providers produce high-quality extractions. Choose based on:
- Existing subscriptions
- API pricing preferences
- Model availability

## Programmatic Usage

```python
from src.analysis.llm_factory import create_llm_client

# CLI mode (ChatGPT subscription)
client = create_llm_client(
    provider="openai",
    mode="cli",
    model="gpt-5.2"
)

# API mode
client = create_llm_client(
    provider="openai",
    mode="api",
    model="gpt-4o-mini"
)

# Extract from a paper
result = client.extract(
    paper_id="paper_001",
    title="Example Paper",
    authors="Smith, J.",
    year=2024,
    item_type="journalArticle",
    text="Full paper text..."
)

if result.success:
    print(f"Thesis: {result.extraction.thesis_statement}")
    print(f"Confidence: {result.extraction.extraction_confidence}")
```

## Running the Smoketest

Verify your OpenAI integration:

```powershell
# Run smoketest
python scripts/smoketest_openai.py

# Expected output:
# models: PASS
# costs: PASS
# factory: PASS
# cli: PASS (if Codex installed)
# api: PASS (if API key set)
```
