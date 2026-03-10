---
name: smoketest
description: Run a targeted smoketest for a specific integration
disable-model-invocation: true
---

# Smoketest Runner

Run targeted integration smoketests by keyword.

## Usage

```
/smoketest <integration>
/smoketest list
```

## Arguments

- `integration`: One of the keywords below, or `list` to show available tests
- `all`: Run all smoketests sequentially

## Available Smoketests

| Keyword | Script | Tests |
|---------|--------|-------|
| bibtex | smoketest_bibtex.py | BibTeX file import |
| config | smoketest_config_migration.py | Config migration logic |
| endnote | smoketest_endnote.py | EndNote XML import |
| gaps | smoketest_gap_analysis.py | Gap detection pipeline |
| gemini | smoketest_gemini.py | Google Gemini provider |
| local | smoketest_local_llm.py | Ollama/llama.cpp provider |
| mendeley | smoketest_mendeley.py | Mendeley SQLite import |
| ocr | smoketest_ocr.py | OCR fallback extraction |
| openai | smoketest_openai.py | OpenAI provider |
| paperpile | smoketest_paperpile.py | Paperpile BibTeX import |
| pdffolder | smoketest_pdffolder.py | PDF folder source |
| webui | smoketest_web_ui.py | Streamlit web interface |

## Workflow

1. If argument is `list`, display the table above and stop
2. Map the keyword to the corresponding script filename
3. Run the script:
   ```bash
   python scripts/smoketest_<mapped_name>.py
   ```
4. Report the result (pass/fail, any errors)
5. If argument is `all`, run each script sequentially and report a summary

## Keyword Mapping

- `bibtex` -> `smoketest_bibtex.py`
- `config` -> `smoketest_config_migration.py`
- `endnote` -> `smoketest_endnote.py`
- `gaps` -> `smoketest_gap_analysis.py`
- `gemini` -> `smoketest_gemini.py`
- `local` -> `smoketest_local_llm.py`
- `mendeley` -> `smoketest_mendeley.py`
- `ocr` -> `smoketest_ocr.py`
- `openai` -> `smoketest_openai.py`
- `paperpile` -> `smoketest_paperpile.py`
- `pdffolder` -> `smoketest_pdffolder.py`
- `webui` -> `smoketest_web_ui.py`

## Notes

- All smoketests are safe to run (read-only or use test data)
- Some require API keys (openai, gemini) or external tools (ocr requires Tesseract)
- If a required dependency is missing, the script will report it clearly
