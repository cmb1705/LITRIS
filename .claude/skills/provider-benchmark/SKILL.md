---
name: provider-benchmark
description: Run side-by-side extraction comparison between LLM providers
disable-model-invocation: true
---

# Provider Benchmark

Compare extraction quality between Anthropic and OpenAI on a paper.

## Usage

```
/provider-benchmark <zotero_key>
```

## Arguments

- `zotero_key`: The Zotero item key to compare (e.g., `CLCNNCAH`)

## Workflow

1. Run the comparison script with the provided key
2. Use `--anthropic-from-index` to load existing Anthropic extraction (faster)
3. Run fresh OpenAI extraction for comparison
4. Save both extractions to `data/comparison/`
5. Report quality differences

## Command

```bash
python scripts/compare_providers.py --key $ARGUMENTS --anthropic-from-index --save
```

## Output

The script produces:
- Side-by-side comparison of metrics (confidence, findings, claims)
- Thesis statement comparison
- Keywords comparison
- Saved JSON files in `data/comparison/<key>_anthropic.json` and `<key>_openai.json`

## Examples

Compare extraction for a specific paper:
```
/provider-benchmark CLCNNCAH
```

## Notes

- Uses existing Anthropic extraction from index to avoid redundant API calls
- Runs fresh OpenAI extraction via Codex CLI
- Requires valid Zotero key with existing extraction in index
