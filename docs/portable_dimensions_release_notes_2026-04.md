# Portable Dimensions Release Notes

Date: 2026-04-01
Release: v0.3.0

## Summary

This release completes the portable semantic dimensions migration for LITRIS.

Semantic analysis is no longer bound to a single hard-coded 40-field schema at
 runtime. Each index can now carry a profile snapshot, store canonical
 dimension maps keyed by stable IDs, accept profile-driven search and MCP
 filtering, and backfill new or changed dimensions without rebuilding the
 entire corpus from scratch.

## Shipped Changes

- Added versioned dimension profiles with registry-driven resolution for
  prompts, storage, embeddings, search, MCP, and downstream analysis.
- Preserved legacy compatibility for `qNN`, `qNN_field_name`, and `dim_qNN`
  aliases.
- Added canonical `DimensionedExtraction` storage and index-level
  `dimension_profile.json` snapshots.
- Added `scripts/dimensions.py` commands for `migrate-store`, `diff`,
  `suggest`, `approve`, and `backfill`.
- Added section-scoped backfill so `added`, `reworded`, and `replaced`
  dimensions only re-run the affected passes.
- Added failure-safe partial backfill behavior so pilot runs do not advance the
  live profile snapshot.
- Added a semantic suggestion workflow that uses sampled extraction content
  plus vectorstore-derived similarity neighbors to generate new candidate
  dimensions, with heuristic fallback when the semantic LLM pass is disabled
  or fails.

## Validation

Code-level validation completed:

- `369 passed`
- `153 passed, 1 skipped`

Operational validation completed on clean cloned indexes:

- storage migration dry-run and apply
- five-paper backfill pilot
- four-paper prompt-tuned pilot
- semantic suggestion generation using a real cloned index and
  `similarity_pairs.json`

Live-index safety confirmed:

- `data/index` remains on the legacy snapshot
- `data/index/dimension_profile.json` is still absent
- the live Chroma collection count remains `64364`

## Known Constraints

- Provider safety policies can still block individual passes. During
  validation, OpenAI rejected the CRISPR paper's pass 6 prompt as
  biology-sensitive content. LITRIS now handles this correctly by aborting the
  targeted backfill before mutating the index snapshot.
- Existing Chroma rows can contain broken embedding entries for older chunks.
  The vector-store refresh path now tolerates backup export failures and still
  performs a clean replacement when possible.
- `README.md` and `docs/usage.md` still have many pre-existing markdownlint
  violations outside the portable-dimensions scope. The new
  `docs/dimension_profiles.md` and the new live-test guide pass markdownlint.

## Recommended Release Checklist

1. Run the live-index migration and Anthropic CLI pilot using
   [portable-dimensions-live-test.md](d:/Git_Repos/LITRIS/docs/guides/portable-dimensions-live-test.md).
2. Review the pilot outputs for `stp_cas_linkage` and
   `intervention_leverage_point`.
3. Run the full-corpus backfill only after pilot approval.
4. Capture a short release summary and the exact commit hash in the GitHub
   release body.
5. Follow with a narrower docs cleanup pass for the remaining markdownlint debt
   in `README.md` and `docs/usage.md`.
