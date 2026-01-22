# Google Gemini Integration Guide

LITRIS supports Google Gemini models for paper extraction via the Google Gen AI SDK.

Unlike Claude and OpenAI, Gemini currently only supports API mode (no CLI option with subscription authentication).

## Quick Start

```powershell
# 1. Install Google Gen AI package
pip install google-genai

# 2. Set API key
$env:GOOGLE_API_KEY = "AIza..."

# 3. Build index with Gemini
python scripts/build_index.py --provider google --mode api
```

## API Setup

### Get an API Key

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Click "Get API key" in the left sidebar
3. Create a new API key or use an existing one

### Configuration

Set the environment variable (either works):

**PowerShell**:

```powershell
$env:GOOGLE_API_KEY = "AIza..."
# or
$env:GEMINI_API_KEY = "AIza..."
```

**Bash**:

```bash
export GOOGLE_API_KEY="AIza..."
# or
export GEMINI_API_KEY="AIza..."
```

**Persistent (Windows)**:

```powershell
[Environment]::SetEnvironmentVariable("GOOGLE_API_KEY", "AIza...", "User")
```

## Available Models

| Model | Description | Best For |
|-------|-------------|----------|
| gemini-3-pro | Gemini 3 Pro Preview (highest capability) | Complex papers, highest quality |
| gemini-3-flash | Gemini 3 Flash (Pro intelligence at Flash speed) | Balance of quality and speed |
| gemini-2.5-pro | Gemini 2.5 Pro (state-of-the-art reasoning) | Deep analysis, complex methodology |
| gemini-2.5-flash | Gemini 2.5 Flash (best price-performance) | Large libraries, good quality |
| gemini-2.5-flash-lite | Gemini 2.5 Flash-Lite (fastest, cost-effective) | Testing, rapid iteration |

LITRIS defaults to `gemini-3-pro` for highest extraction quality.

### Model Selection

```powershell
# Cost-effective (recommended for large libraries)
python scripts/build_index.py --provider google --mode api --model gemini-2.5-flash

# Fastest and cheapest
python scripts/build_index.py --provider google --mode api --model gemini-2.5-flash-lite

# Highest quality
python scripts/build_index.py --provider google --mode api --model gemini-3-pro
```

## Cost Estimation

### API Pricing (per 1M tokens, January 2026)

| Model | Input | Output | Est. per paper (~10k chars) |
|-------|-------|--------|----------------------------|
| gemini-2.5-flash-lite | $0.10 | $0.40 | ~$0.001 |
| gemini-2.5-flash | $0.30 | $2.50 | ~$0.006 |
| gemini-2.0-flash | $0.10 | $0.40 | ~$0.001 |
| gemini-2.5-pro (<=200k) | $1.25 | $10.00 | ~$0.023 |
| gemini-3-flash-preview | $0.50 | $3.00 | ~$0.007 |
| gemini-3-pro-preview (<=200k) | $2.00 | $12.00 | ~$0.029 |

**Context length pricing**: For contexts exceeding 200K tokens, prices typically double for Pro models.

**Batch API**: 50% discount available for asynchronous batch processing.

Gemini models are generally the most cost-effective option for large libraries.
See [official pricing](https://ai.google.dev/gemini-api/docs/pricing) for current rates.

### Cost Estimation Command

```powershell
# Estimate costs before processing
python scripts/build_index.py --provider google --model gemini-2.5-flash --estimate-cost
```

## Configuration File

You can set Gemini as the default provider in `config.yaml`:

```yaml
extraction:
  provider: "google"
  mode: "api"
  model: "gemini-2.5-flash"  # or leave empty for default (gemini-3-pro)
  max_tokens: 100000
  timeout: 120
```

## Programmatic Usage

```python
from src.analysis.llm_factory import create_llm_client

# Create Gemini client
client = create_llm_client(
    provider="google",
    mode="api",
    model="gemini-2.5-flash"
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

## Troubleshooting

### Missing Package Error

```text
ImportError: Google Gen AI package not installed
```

Install the required package:

```powershell
pip install google-genai
```

### API Key Invalid

```text
Error code: 401 - API key not valid
```

Verify your API key:

1. Check if set: `echo $env:GOOGLE_API_KEY`
2. Ensure key is from [Google AI Studio](https://aistudio.google.com/)
3. Verify key is not expired or revoked

### Rate Limits

Gemini has rate limits based on your API tier:

- Free tier: 15 RPM (requests per minute)
- Pay-as-you-go: Higher limits

If hitting rate limits, LITRIS will automatically retry with backoff.

### Model Not Found

```text
Model 'gemini-xyz' not found
```

Use one of the supported models listed above. Model names are case-sensitive.

## Running the Smoketest

Verify your Gemini integration:

```powershell
python scripts/smoketest_gemini.py
```

Expected output:

```text
models: PASS
costs: PASS
factory: PASS
api: PASS (if API key set)
```

## Comparison: Claude vs GPT vs Gemini

| Feature | Claude (Anthropic) | GPT (OpenAI) | Gemini (Google) |
|---------|-------------------|--------------|-----------------|
| Default Model | claude-opus-4-5-20251101 | gpt-5.2 | gemini-3-pro |
| CLI Mode | Yes (claude) | Yes (codex) | No |
| Subscription Auth | Claude Max | ChatGPT Plus/Pro | No |
| API Pricing | $$$ | $$$ | $ (cheapest) |
| Extraction Quality | Excellent | Excellent | Excellent |
| Context Window | 200K | 128K | 1M+ |

### When to Choose Gemini

- **Large libraries**: Lowest cost per paper with gemini-2.5-flash-lite
- **No subscription**: Only requires API key, no CLI subscription needed
- **Long papers**: 1M+ token context window handles very long documents
- **Budget-conscious**: Flash-Lite model is the cheapest option across all providers

### When to Choose Claude or GPT

- **Subscription-based**: If you have Claude Max or ChatGPT Plus/Pro
- **CLI preference**: Gemini has no CLI mode with subscription auth
- **Specific quality needs**: All providers produce excellent extractions
