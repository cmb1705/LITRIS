# Portable Dimensions Live Test

This guide captures the recommended operator workflow for validating a
portable-dimensions migration and backfill on a live LITRIS index.

The commands below assume:

- Repository root: `d:\Git_Repos\LITRIS`
- Live index: `d:\Git_Repos\LITRIS\data\index`
- Target profile: `d:\Git_Repos\LITRIS\profiles\stp_cas.yaml`
- Extraction stack for backfill: `anthropic` provider in `cli` mode

## Preconditions

Before running the backfill, verify:

- `claude --version` succeeds in the shell you will use for the test.
- Your full index backup exists. The earlier backup path is:
  `d:\Git_Repos\LITRIS\data\archive\index_backup_20260331_210229`
- You are comfortable with the profile diff in
  [stp_cas.yaml](d:/Git_Repos/LITRIS/profiles/stp_cas.yaml).

## Commands

Run these from `d:\Git_Repos\LITRIS`.

### 1. Dry-run the storage migration

```powershell
.\.venv\Scripts\python.exe scripts\dimensions.py migrate-store `
  --index-dir "d:\Git_Repos\LITRIS\data\index" `
  --dry-run
```

### 2. Apply the storage migration

```powershell
.\.venv\Scripts\python.exe scripts\dimensions.py migrate-store `
  --index-dir "d:\Git_Repos\LITRIS\data\index"
```

### 3. Diff the live index against the STP/CAS profile

```powershell
.\.venv\Scripts\python.exe scripts\dimensions.py diff `
  --index-dir "d:\Git_Repos\LITRIS\data\index" `
  --new-profile "d:\Git_Repos\LITRIS\profiles\stp_cas.yaml"
```

Expected result:

- `stp_cas_linkage` is reported as `added` in `scholarly`
- `intervention_leverage_point` is reported as `added` in `impact`

### 4. Dry-run the backfill plan

```powershell
.\.venv\Scripts\python.exe scripts\dimensions.py backfill `
  --index-dir "d:\Git_Repos\LITRIS\data\index" `
  --dimension-profile "d:\Git_Repos\LITRIS\profiles\stp_cas.yaml" `
  --provider anthropic `
  --mode cli `
  --model claude-opus-4-6 `
  --dry-run
```

Expected result:

- `changed_sections` contains `scholarly` and `impact`
- `reextract_sections` contains `scholarly` and `impact`
- `disable_only` is `false`

### 5. Pilot on the five validation papers

Use `--no-cache` on the pilot so the new profile wording is re-run directly.

```powershell
.\.venv\Scripts\python.exe scripts\dimensions.py backfill `
  --index-dir "d:\Git_Repos\LITRIS\data\index" `
  --dimension-profile "d:\Git_Repos\LITRIS\profiles\stp_cas.yaml" `
  --paper "ZZIBFY4G_IIHDQH2R" `
  --paper "ZVTXLNKK_BE6UNPS5" `
  --paper "ZU8RKD2E_BWFKFHZM" `
  --paper "ZX7ECSSD_ESVU5SJB" `
  --paper "ZYHCT2VV_988LV28A" `
  --provider anthropic `
  --mode cli `
  --model claude-opus-4-6 `
  --parallel 4 `
  --skip-similarity `
  --no-cache
```

What to check after the pilot:

- The run completes without advancing `dimension_profile.json`.
- Only the five targeted papers change in `semantic_analyses.json`.
- `stp_cas_linkage` and `intervention_leverage_point` are populated where the
  paper supports them.
- The vector store refresh completes successfully.

### 6. Inspect pilot results

The fastest local check is to inspect the two new dimensions directly:

```powershell
@'
import json
from pathlib import Path

path = Path(r"d:\Git_Repos\LITRIS\data\index\semantic_analyses.json")
data = json.loads(path.read_text(encoding="utf-8"))["extractions"]
paper_ids = [
    "ZZIBFY4G_IIHDQH2R",
    "ZVTXLNKK_BE6UNPS5",
    "ZU8RKD2E_BWFKFHZM",
    "ZX7ECSSD_ESVU5SJB",
    "ZYHCT2VV_988LV28A",
]

for paper_id in paper_ids:
    extraction = data[paper_id]["extraction"]
    dims = extraction.get("dimensions", {})
    print(f"[{paper_id}]")
    print("stp_cas_linkage =", repr(dims.get("stp_cas_linkage")))
    print(
        "intervention_leverage_point =",
        repr(dims.get("intervention_leverage_point")),
    )
    print()
'@ | .\.venv\Scripts\python.exe -
```

### 7. Full-corpus backfill after pilot approval

Run this only after you are satisfied with the pilot outputs.

```powershell
.\.venv\Scripts\python.exe scripts\dimensions.py backfill `
  --index-dir "d:\Git_Repos\LITRIS\data\index" `
  --dimension-profile "d:\Git_Repos\LITRIS\profiles\stp_cas.yaml" `
  --provider anthropic `
  --mode cli `
  --model claude-opus-4-6 `
  --parallel 4 `
  --no-cache
```

On a successful full-corpus run, LITRIS will:

- update `semantic_analyses.json`
- refresh embeddings
- write `dimension_profile.json`
- update `index_manifest.json` with the new profile metadata

## Notes

- Partial pilot runs do not advance the index-level profile snapshot by design.
- If any targeted section fails during backfill, the run aborts before
  advancing the profile snapshot.
- If `claude` is not available in your shell, fix the CLI installation or PATH
  before running the backfill.
