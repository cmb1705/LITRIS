<!-- markdownlint-disable MD031 MD032 -->
# Troubleshooting Guide

This guide covers common issues and their solutions.

## Installation Issues

### Python Version

**Error**: `SyntaxError` or `ImportError` on startup

**Solution**: Ensure Python 3.10+ is installed:
```bash
python --version
```

### Missing Dependencies

**Error**: `ModuleNotFoundError: No module named 'xxx'`

**Solution**: Install requirements:
```bash
pip install -r requirements.txt
```

### Virtual Environment

**Error**: Dependencies not found despite installing

**Solution**: Ensure virtual environment is activated:
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

## Zotero Connection Issues

### Database Not Found

**Error**: `FileNotFoundError: Zotero database not found`

**Solution**: Check `config.yaml` paths:
```yaml
zotero:
  database_path: "D:/Zotero/zotero.sqlite"  # Adjust to your path
  storage_path: "D:/Zotero/storage"
```

Find your Zotero data directory:
- Windows: `%APPDATA%\Zotero\Zotero\Profiles\xxx.default\zotero`
- Mac: `~/Zotero`
- Linux: `~/.zotero` or `~/Zotero`

### Database Locked

**Error**: `sqlite3.OperationalError: database is locked`

**Solution**: Close Zotero desktop application before running:
```bash
# Verify Zotero is closed, then retry
python scripts/build_index.py
```

### No Papers Found

**Error**: `Found 0 items with PDF attachments`

**Causes**:
1. Papers don't have PDF attachments
2. PDFs are linked files that don't exist
3. Wrong Zotero library path

**Solution**: Verify PDFs exist in Zotero storage:
```bash
dir "D:\Zotero\storage"  # Windows
ls ~/Zotero/storage      # Linux/Mac
```

## CLI Extraction Issues

### CLI Not Found

**Error**: `CliExecutionError: Claude CLI not found in PATH`

**Solution**: Install and log in to Claude Code:
```bash
# Install
npm install -g @anthropic-ai/claude-code

# Verify
claude --version
```

For OpenAI CLI mode, verify Codex separately:

```bash
# Install
npm i -g @openai/codex

# Verify
codex --version
```

### Rate Limit Hit

**Error**: `RateLimitError: Rate limit exceeded`

**Solution**: Wait for the reset window, then resume the same command:
```bash
python scripts/build_index.py --resume
```

For portable-dimension runs:

```bash
python scripts/dimensions.py backfill --index-dir data/index \
  --dimension-profile ./profiles/your_profile.yaml --resume
```

If you only want to finish canonical full-text capture without spending more
provider quota, use:

```bash
python scripts/dimensions.py backfill --index-dir data/index \
  --dimension-profile ./profiles/your_profile.yaml \
  --fulltext-only --resume
```

### CLI Session/Auth Failures

**Error**: `Authentication failed during extraction` or repeated
`Please re-authenticate` messages

**Solution**:
1. Re-run `claude` in the same shell and confirm the session is still valid.
2. Resume the interrupted LITRIS run with `--resume`.
3. If the failure actually reflects subscription quota exhaustion, wait for the
   quota window to reset before resuming.

LITRIS now preserves the raw CLI stderr/stdout summary in failure messages so
session expiry, rate limits, and prompt-size failures are easier to distinguish.

### Prompt Too Long

**Error**: `Prompt too long` or `prompt is too long: ... tokens > ... maximum`

**Solution**:
1. Resume the run after the failing paper is handled; LITRIS checkpoints
   completed work.
2. If you are running a semantic backfill, populate canonical full-text first
   with `--fulltext-only` so later retries reuse stored snapshots.
3. Reprocess the specific paper later with a smaller provider input budget or
   a different extraction stack if needed.

### Timeout Errors

**Error**: `ExtractionTimeoutError`

**Solution**: Increase timeout in `config.yaml`:
```yaml
extraction:
  timeout: 180  # Increase from 120
```

## Batch API Issues

### Missing API Key

**Error**: `ANTHROPIC_API_KEY environment variable is required`

**Solution**: Set your API key:
```bash
# Windows
set ANTHROPIC_API_KEY=sk-ant-xxx

# Linux/Mac
export ANTHROPIC_API_KEY=sk-ant-xxx
```

### API Errors

**Error**: `APIError` or `AuthenticationError`

**Solution**: Verify your API key is valid:
1. Check for typos in the key
2. Verify key hasn't expired
3. Check API account status

## PDF Processing Issues

### PDF Text Extraction Failed

**Error**: `Failed to extract text from PDF` or empty text

**Causes**:
1. Scanned PDF (image-based)
2. Encrypted/protected PDF
3. Corrupted PDF file

**Solutions**:

For scanned PDFs, enable OCR:
```yaml
processing:
  ocr_enabled: true
```

Install Tesseract OCR:
```bash
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
# Mac
brew install tesseract
# Linux
sudo apt install tesseract-ocr
```

### Minimum Text Length

**Error**: Paper skipped due to short text

**Solution**: Adjust minimum length:
```yaml
processing:
  min_text_length: 50  # Lower from 100
```

## Search Issues

### No Results Found

**Causes**:
1. Index is empty
2. Query doesn't match any content
3. Filters too restrictive

**Solutions**:

Check index status:
```bash
python scripts/query_index.py --stats
```

Try broader query:
```bash
python scripts/query_index.py -q "research" -n 100
```

Remove filters:
```bash
python scripts/query_index.py -q "your query"  # No year filters
```

### Poor Quality Results

**Causes**:
1. Query too vague
2. Extraction quality issues
3. Not enough papers indexed

**Solutions**:

Be more specific:
```bash
# Instead of: "networks"
python scripts/query_index.py -q "social network analysis methods"
```

Check extraction quality:
```bash
python scripts/export_results.py summary
```

### Vector Store Errors

**Error**: `ChromaDB` errors

**Solution**: Rebuild embeddings:
```bash
python scripts/build_index.py --sync-mode full --skip-extraction
```

## Incremental Update Issues

### Index Directory Not Found

**Error**: `Index directory not found`

**Solution**: Run initial build first:
```bash
python scripts/build_index.py --sync-mode full
```

### Stale Changes

**Error**: Changes detected but not real

**Solution**: Inspect the resolved sync plan, then force a full rebuild if needed:
```bash
# View the update-only plan
python scripts/build_index.py --sync-mode update --explain-plan --dry-run

# If issues persist, rebuild:
python scripts/build_index.py --sync-mode full
```

## Performance Issues

### Slow Extraction

**Causes**:
1. Large PDFs
2. Network latency (CLI mode)
3. Rate limiting

**Solutions**:

Process in batches:
```bash
python scripts/build_index.py --limit 50
# Wait, then continue
python scripts/build_index.py --resume
```

### High Memory Usage

**Causes**:
1. Large embedding model
2. Many papers loaded at once

**Solutions**:

Use smaller batch size:
```yaml
embeddings:
  batch_size: 16  # Or use "auto" to probe locally
```

For one-off runs, you can also override this without editing config:

```bash
python scripts/build_index.py --embedding-batch-size 16
```

`auto` intentionally will not target the largest batch that happens to succeed;
it chooses the largest successful batch that stays within the latency budget for
the current run.

Use lighter embedding model:
```yaml
embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"  # Smaller model
```

### Slow Searches

**Causes**:
1. Large vector store
2. Many results requested

**Solutions**:

Reduce result count:
```bash
python scripts/query_index.py -q "query" -n 10  # Instead of 50
```

## Data Recovery

### Corrupted Progress File

**Error**: `Failed to load progress file`

**Solution**: Reset checkpoint:
```bash
python scripts/build_index.py --reset-checkpoint
```

### Lost Extractions

**Solution**: Rerun extraction for specific papers:
```bash
python scripts/build_index.py --retry-failed
```

For portable-dimension backfills, rerun the same command with `--resume` to
continue from the saved checkpoint instead of restarting from scratch.

### Full Rebuild

If all else fails, rebuild from scratch:
```bash
# Backup existing data
mv data/index data/index.bak

# Full rebuild
python scripts/build_index.py --sync-mode full
```

## Getting Help

If issues persist:

1. Check logs in console output with `-v` flag
2. Review [GitHub Issues](https://github.com/YOUR_USERNAME/LITRIS/issues)
3. Open a new issue with:
   - Error message
   - Steps to reproduce
   - Config file (redact sensitive paths)
   - Python and package versions
