# Alternative Reference Sources Guide

LITRIS supports multiple reference sources beyond Zotero through the
`--source` and `--source-path` flags on `scripts/build_index.py`.

Currently supported source types are:

- `zotero`
- `bibtex`
- `pdffolder`
- `mendeley`
- `endnote`
- `paperpile`

## Current CLI Surface

The top-level build CLI currently exposes one required path per non-Zotero
source:

```powershell
# BibTeX
python scripts/build_index.py --source bibtex --source-path .\references.bib

# Folder of PDFs
python scripts/build_index.py --source pdffolder --source-path .\papers

# Mendeley Desktop database
python scripts/build_index.py --source mendeley --source-path .\mendeley.sqlite

# EndNote XML export
python scripts/build_index.py --source endnote --source-path .\library.xml

# Paperpile BibTeX export
python scripts/build_index.py --source paperpile --source-path .\paperpile.bib
```

The build CLI does not currently expose every adapter-specific option from the
lower-level reference adapters. For example, PDF side directories, recursive
PDF-folder toggles, and Paperpile sync-folder wiring are available in the
adapter layer, but not as first-class `build_index.py` flags yet.

## Source Mapping

`build_index.py` maps `--source-path` to the adapter entrypoint as follows:

| `--source` | Adapter argument |
| ---------- | ---------------- |
| `bibtex` | `bibtex_path` |
| `pdffolder` | `folder_path` |
| `mendeley` | `db_path` |
| `endnote` | `xml_path` |
| `paperpile` | `bibtex_path` |

## Notes By Source

### BibTeX

Use this when you have a `.bib` export from another reference manager.

- Metadata comes from the BibTeX file.
- PDF attachment matching through extra directories is supported in the adapter
  layer, but is not currently exposed through `build_index.py`.

### PDF Folder

Use this when you have a directory of PDFs and no reference manager.

- LITRIS derives metadata from filenames and PDF document metadata.
- Subfolders become collections.
- Recursive scanning and PDF-metadata extraction behavior are adapter features;
  the current build CLI uses adapter defaults.

### Mendeley

Use this with a local Mendeley Desktop SQLite database.

- The adapter reads metadata, collections, tags, and file attachments directly
  from the database.
- If you need custom storage-path resolution, use the adapter layer directly or
  extend the top-level CLI.

### EndNote

Use this with an EndNote XML export.

- Metadata is parsed from XML.
- Extra PDF-directory matching exists in the adapter layer, not the current
  top-level build flags.

### Paperpile

Use this with a BibTeX export from Paperpile.

- Metadata is parsed from the export file.
- Paperpile-specific PDF sync-folder behavior exists in the adapter layer, but
  the current build CLI only accepts the exported `.bib` path directly.

## Programmatic Use

If you need adapter-specific options that are not yet exposed by
`scripts/build_index.py`, use the adapter factory directly:

```python
from pathlib import Path

from src.references.factory import create_reference_db

db = create_reference_db(
    provider="bibtex",
    bibtex_path=Path("references.bib"),
    pdf_dir=Path("papers"),
)

for paper in db.get_all_papers():
    print(paper.paper_id, paper.title)
```

See [src/references/factory.py](d:/Git_Repos/LITRIS/src/references/factory.py)
for the adapter-level arguments currently available.
