# Dimension Profiles

LITRIS now treats semantic dimensions as a versioned profile rather than a
fixed hard-coded list. Each index carries one active profile snapshot in
`dimension_profile.json`, and extraction records store canonical answers in a
`dimensions` map keyed by stable dimension IDs.

The built-in fallback profile is `legacy_semantic_v1`. It preserves the
historical `q01` through `q40` aliases, `qNN_field_name` storage, and
`dim_qNN` chunk types so existing indexes continue to load without a forced
rebuild.

## Files

- `data/index/dimension_profile.json`: Exact profile snapshot used by the index.
- `data/index/semantic_analyses.json`: Extraction store. Schema `2.0.0`
  records canonical `dimensions` plus profile metadata.
- `data/index/index_manifest.json`: Operational manifest. Extraction metadata
  now includes `profile_id`, `profile_version`, and `profile_fingerprint`.

## Config

Add the `dimensions` section to `config.yaml`:

```yaml
dimensions:
  active_profile: "legacy_semantic_v1"
  profile_paths:
    - "./profiles/sts_policy.yaml"
  approval_required: true
  suggestion_sample_size: 25
```

- `active_profile`: Profile id to activate for a build or update run.
- `profile_paths`: Additional YAML or JSON profiles to register.
- `approval_required`: Whether generated proposals need explicit approval.
- `suggestion_sample_size`: Corpus sample size for
  `scripts/dimensions.py suggest`.

You can also override the active profile for a single build or update:

```bash
python scripts/build_index.py --dimension-profile ./profiles/sts_policy.yaml
python scripts/update_index.py --dimension-profile ./profiles/sts_policy.yaml
```

## Migration Workflow

For an existing index, use this order:

1. Dry-run the storage migration.
2. Back up the full index directory.
3. Apply the storage migration.
4. Diff the current index profile against a target profile.
5. Dry-run backfill.
6. Pilot on a small set of papers before a full run.

```bash
python scripts/dimensions.py migrate-store --index-dir data/index --dry-run
python scripts/dimensions.py migrate-store --index-dir data/index
python scripts/dimensions.py diff --index-dir data/index --new-profile \
  ./profiles/sts_policy.yaml
python scripts/dimensions.py backfill --index-dir data/index \
  --dimension-profile ./profiles/sts_policy.yaml --dry-run
python scripts/dimensions.py backfill --index-dir data/index \
  --dimension-profile ./profiles/sts_policy.yaml \
  --paper PAPER_001 --paper PAPER_002
```

## Backfill Semantics

- `added`, `reworded`, and `replaced` dimensions trigger section-scoped
  re-extraction only for the affected sections.
- `disabled` dimensions do not trigger LLM extraction. LITRIS retargets the
  stored records to the new profile and regenerates embeddings so retired
  chunks disappear from active search.
- Retired answers stay in storage by default. The active profile controls
  what is visible and searchable.
- If a targeted re-extraction fails, the backfill command aborts before
  advancing the profile snapshot.

## Suggestions And Approval

The current suggestion flow is heuristic. It samples stored papers and
extractions, looks for repeated keyword patterns, and writes proposals to
`dimension_proposals.json`.

```bash
python scripts/dimensions.py suggest --index-dir data/index
python scripts/dimensions.py approve --profile ./profiles/sts_policy.yaml \
  --proposals data/index/dimension_proposals.json \
  --dimension-id stakeholders
```

- `suggest` is useful for initial discovery, but it is not yet a corpus-wide
  LLM reasoning workflow.
- `approve` appends approved dimensions to a profile file and bumps the patch version.

## Compatibility

- Existing indexes without `dimension_profile.json` automatically load `legacy_semantic_v1`.
- Search and MCP dimension filters still accept canonical IDs, legacy `qNN`
  aliases, legacy `qNN_field_name` aliases, and `dim_qNN` chunk types.
- Legacy `SemanticAnalysis` callers continue to work, but new storage and
  backfill logic use canonical `DimensionedExtraction` internally.
