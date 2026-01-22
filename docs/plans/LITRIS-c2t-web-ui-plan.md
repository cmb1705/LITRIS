# LITRIS-c2t Web UI Plan

## Task Summary

Build a local web UI for semantic search with filters and export. The UI should reuse the existing search and formatting stack where possible and avoid duplicating CLI logic.

## Review Notes (Repo Evidence)

- Search API accepts query, filters, and extraction toggles through `SearchEngine.search`, which is the core entrypoint to reuse in a UI. `src/query/search.py:74` `src/query/search.py:83` `src/query/search.py:138`
- The CLI query tool exposes the current filter set, output formats, and auto-save behavior that the UI should mirror or supersede. `scripts/query_index.py:31` `scripts/query_index.py:44` `scripts/query_index.py:73` `scripts/query_index.py:176` `scripts/query_index.py:288`
- Documented search filters and output formats are in the Query Guide and should map to UI controls. `docs/query_guide.md:68` `docs/query_guide.md:100`
- Supported chunk types for filtering are defined in the embedding layer and should drive UI filter options. `src/indexing/embeddings.py:15`
- Collections, item types, and year range can be derived from SearchEngine helpers for filter population. `src/query/search.py:284` `src/query/search.py:293` `src/query/search.py:302`
- Structured index files live under `data/index` and include `papers.json`, `extractions.json`, and `summary.json`, which the UI will read via existing stores. `src/indexing/structured_store.py:30` `src/indexing/structured_store.py:151`
- The vector store search supports year and item type filters in Chroma queries, plus post-filtering by collections. `src/indexing/vector_store.py:143` `src/indexing/vector_store.py:217`
- Result formatting and export are already implemented for JSON, Markdown, brief text, and PDF. `src/query/retrieval.py:17` `src/query/retrieval.py:399`
- Export tooling includes CSV/BibTeX and CSV formula injection mitigation that the UI should reuse. `scripts/export_results.py:51` `scripts/export_results.py:92`
- Paper metadata includes `pdf_path`, which the UI can surface for open and reveal actions. `src/zotero/models.py:66` `src/zotero/models.py:165`
- `data/` is an indexed and cached workspace folder; it is already the default location for query outputs. `README.md:226` `scripts/query_index.py:178`
- Storage paths (chroma, cache) are configurable and should be respected by the UI. `config.example.yaml:90`
- Secrets are loaded from env or OS keyring and should not be re-stored by the UI. `src/utils/secrets.py:16`

## UI Concept and Layout

The UI is a local, single-page search workspace with a left filter rail, a main results list, and a detail panel. It should feel like a research workbench: a focused query bar at the top, filters always visible, and results that emphasize title, authors, year, and matched text.

Visual direction requirements for the implementation phase:

- Use a non-default, expressive font pairing that still reads well for dense text.
- Use a neutral base palette with a deliberate accent color and subtle background texture or gradient.
- Add a single load animation for results and a staggered reveal for result cards.
- Ensure the layout remains readable on both desktop and mobile.

## Primary User Flows

- Search and refine: enter a query, adjust filters, see ranked results, and tweak without full-page reload.
- Inspect and save: open a result detail view, optionally show extraction sections, and export in a chosen format.
- Index status check: view summary stats and last index info to verify coverage and freshness.

## Data and Integration Strategy

- Instantiate `SearchEngine` once and cache it for the session to avoid repeated embedding model loads. `src/query/search.py:72`
- Use `SearchEngine.search` for the main search, `get_collections`, `get_item_types`, and `get_year_range` to build filter UI options. `src/query/search.py:74` `src/query/search.py:284`
- Reuse `format_results` and `save_results` for exports and consistent formatting. `src/query/retrieval.py:17` `src/query/retrieval.py:399`
- For paper details, reuse the existing formatter to show metadata and extraction sections. `src/query/retrieval.py:589`
- Use the same index location and results output directory as the CLI unless configured otherwise. `scripts/query_index.py:176` `scripts/query_index.py:178`
- Provide a UI action to trigger index build or rebuild via `scripts/build_index.py`, surfaced as a controlled, confirm-required operation.

## Scripts and Modules Likely Affected

- New UI entrypoint script (for example `scripts/web_ui.py` or `apps/web_ui/app.py`) to launch Streamlit or Gradio.
- `src/query/search.py` and `src/query/retrieval.py` will be imported directly, no changes expected beyond optional shared helpers. `src/query/search.py:74` `src/query/retrieval.py:17`
- `scripts/query_index.py` for parity checks and potential refactoring of shared query setup. `scripts/query_index.py:169`
- `scripts/export_results.py` for exporting CSV or BibTeX if added to UI export options. `scripts/export_results.py:92`
- `src/indexing/structured_store.py` and `src/indexing/vector_store.py` for underlying data access; no UI-specific changes expected. `src/indexing/structured_store.py:18` `src/indexing/vector_store.py:41`
- `src/utils/secrets.py` and `src/config.py` for safe config and credential handling in the UI. `src/utils/secrets.py:16` `src/config.py:204`

## Security, Privacy, and Data Handling Checklist

- Bind the UI server to localhost by default to avoid accidental exposure.
- Do not store API keys in the UI layer; rely on env variables or OS keyring. `src/utils/secrets.py:16`
- Avoid writing query history unless the user opts in, and default to `data/query_results` for exports only. `scripts/query_index.py:178`
- Sanitize CSV outputs to prevent formula injection if CSV export is exposed in the UI. `scripts/export_results.py:51`
- Redact sensitive fields from logs and avoid logging raw full-text extraction.
- Validate file paths before opening PDFs to avoid path traversal or invalid path errors.

## Implementation Milestones

- Milestone 1: Streamlit app scaffold with layout shell (filters, results list, detail panel).
- Milestone 2: Wire search execution and results rendering using `SearchEngine` and `format_results`.
- Milestone 3: Add detail view, extraction toggle, and export actions (Markdown, JSON, CSV, BibTeX).
- Milestone 4: Add index summary panel, caching, and basic error handling for missing index or empty results.
- Milestone 5: Add index build/rebuild controls with confirmations and clear status output.

## Edge Cases and Additional Considerations

- **No index exists**: Show clear onboarding message with build instructions
- **Empty search results**: Display helpful message with query refinement suggestions
- **Query validation**: Handle empty queries, minimum length, special characters
- **PDF path handling**: Validate paths exist before offering open/reveal actions
- **Similar papers**: Expose `search_similar_papers` from result detail view
- **Metadata-only search**: Option to search by title/author without semantic matching
- **Session state**: Use Streamlit session state for SearchEngine caching and filter persistence
- **Loading indicators**: Show spinner during search, staggered card reveal for results
- **Error states**: Handle ChromaDB connection issues, corrupted index, missing files
- **Export downloads**: Provide in-browser download for all export formats
- **Mobile layout**: Stack filter rail below results on narrow screens

## Open Questions

- Do we need a search history panel, or keep the UI stateless by default?
  - **Decision**: Start stateless, add opt-in history if requested
