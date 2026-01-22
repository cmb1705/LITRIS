# Alternative Reference Sources Guide

LITRIS supports multiple reference sources beyond Zotero:

- **BibTeX**: Import from `.bib` files exported from any reference manager
- **PDF Folder**: Scan folders of PDF files without any reference manager
- **Mendeley**: Read directly from Mendeley Desktop database

## Quick Reference

| Source | Input | Best For |
| ------ | ----- | -------- |
| Zotero | SQLite database | Primary use case, full metadata |
| BibTeX | `.bib` file + PDF folder | Cross-platform, any ref manager |
| PDF Folder | Folder of PDFs | No reference manager needed |
| Mendeley | SQLite database | Mendeley Desktop users |

## BibTeX Import

### Overview

The BibTeX adapter parses `.bib` files and optionally matches PDFs from a directory. This works with exports from any reference manager (Zotero, Mendeley, EndNote, etc.).

### Setup

1. Export your library to BibTeX format from your reference manager
2. Optionally organize PDFs in a folder (named by citation key)

### Usage

```powershell
# Build index from BibTeX file
python scripts/build_index.py --provider bibtex --bibtex-path references.bib

# With PDF directory
python scripts/build_index.py --provider bibtex --bibtex-path references.bib --pdf-dir ./papers/
```

### PDF Matching

When a `pdf_dir` is specified, the adapter looks for PDFs matching the citation key:

```text
references.bib contains:
  @article{smith2020example, ...}

papers/ contains:
  smith2020example.pdf  -> Matched!
  Smith2020Example.pdf  -> Matched (case-insensitive)
```

### Supported Entry Types

| BibTeX Type | LITRIS Type |
| ----------- | ----------- |
| `@article` | journalArticle |
| `@inproceedings` | conferencePaper |
| `@book` | book |
| `@incollection` | bookSection |
| `@phdthesis`, `@mastersthesis` | thesis |
| `@techreport` | report |
| `@misc` | document |

### Metadata Extraction

The adapter extracts:

- Title, authors, year
- Journal/booktitle
- Volume, issue, pages
- DOI, ISBN
- Abstract
- Keywords (from `keywords` field)

### Example BibTeX Entry

```bibtex
@article{smith2020example,
    author = {Smith, John and Doe, Jane},
    title = {An Example Article Title},
    journal = {Journal of Examples},
    year = {2020},
    volume = {10},
    number = {2},
    pages = {100-120},
    doi = {10.1234/example.2020},
    abstract = {This is the abstract.},
    keywords = {machine learning, NLP}
}
```

### Programmatic Usage

```python
from src.references.factory import create_reference_db

db = create_reference_db(
    provider="bibtex",
    bibtex_path="references.bib",
    pdf_dir="papers/"  # Optional
)

# Iterate papers
for paper in db.get_all_papers():
    print(f"{paper.title} ({paper.publication_year})")
    if paper.pdf_path:
        print(f"  PDF: {paper.pdf_path}")
```

## PDF Folder Import

### Overview

The PDF folder adapter scans a directory for PDF files and extracts metadata from:

1. PDF document properties (embedded title, author, creation date)
2. Filename parsing (author, year, title patterns)
3. Subfolder names (mapped to collections)

This is ideal when you have PDFs but no reference management software.

### Setup

Organize your PDFs in a folder. Subfolder structure is preserved as collections.

```text
papers/
  Methods/
    Smith - 2020 - Network Analysis.pdf
    Jones_2019_Graph Theory.pdf
  Theory/
    Brown2018ReviewPaper.pdf
```

### Usage

```powershell
# Build index from PDF folder
python scripts/build_index.py --provider pdffolder --folder-path ./papers/

# Non-recursive (top-level only)
python scripts/build_index.py --provider pdffolder --folder-path ./papers/ --no-recursive

# Skip PDF metadata extraction (faster, filename only)
python scripts/build_index.py --provider pdffolder --folder-path ./papers/ --no-pdf-metadata
```

### Filename Patterns

The adapter recognizes these naming conventions:

| Pattern | Example |
| ------- | ------- |
| `Author - Year - Title.pdf` | `Smith - 2020 - Network Analysis.pdf` |
| `Author_Year_Title.pdf` | `Smith_2020_Network_Analysis.pdf` |
| `Year_Author_Title.pdf` | `2020_Smith_Network_Analysis.pdf` |
| `Year - Author - Title.pdf` | `2020 - Smith - Network Analysis.pdf` |
| `Title.pdf` (fallback) | `Network Analysis Methods.pdf` |

### Author Parsing

Multiple authors can be specified:

- `Smith and Jones - 2020 - Title.pdf`
- `Smith, Jones - 2020 - Title.pdf`
- `Smith; Jones; Brown - 2020 - Title.pdf`
- `Smith et al - 2020 - Title.pdf`

### PDF Metadata

When `extract_pdf_metadata` is enabled (default), the adapter reads:

- Document title
- Author field
- Creation date
- Subject/keywords

PDF metadata takes precedence over filename parsing when available.

### Collections from Folders

Subfolder names become collections:

```text
papers/
  Machine Learning/    -> Collection: "Machine Learning"
    paper1.pdf
  Statistics/          -> Collection: "Statistics"
    paper2.pdf
```

### Programmatic Usage

```python
from src.references.factory import create_reference_db

db = create_reference_db(
    provider="pdffolder",
    folder_path="papers/",
    recursive=True,           # Scan subfolders
    extract_pdf_metadata=True # Read PDF properties
)

# Get paper count
print(f"Found {db.get_paper_count()} papers")

# Filter by collection (subfolder)
for paper in db.filter_papers(collections=["Methods"]):
    print(paper.title)
```

### Running the Smoketest

```powershell
python scripts/smoketest_pdffolder.py
```

## Mendeley Import

### Overview

The Mendeley adapter reads directly from the Mendeley Desktop SQLite database, providing full access to your library metadata including folders, tags, and file attachments.

### Finding Your Database

**Windows**:

```text
%LOCALAPPDATA%\Mendeley Ltd\Mendeley Desktop\<your-email>@www.mendeley.com.sqlite
```

**macOS**:

```text
~/Library/Application Support/Mendeley Desktop/<your-email>@www.mendeley.com.sqlite
```

**Linux**:

```text
~/.local/share/data/Mendeley Ltd./Mendeley Desktop/<your-email>@www.mendeley.com.sqlite
```

### Usage

```powershell
# Build index from Mendeley database
python scripts/build_index.py --provider mendeley --db-path "path/to/mendeley.sqlite"

# With storage path for resolving relative file paths
python scripts/build_index.py --provider mendeley --db-path "path/to/mendeley.sqlite" --storage-path "path/to/mendeley/files"
```

### Supported Metadata

The adapter extracts:

- Title, authors, year
- Document type (journal article, book, thesis, etc.)
- Journal/publication details
- DOI, ISBN, PMID
- Abstract
- Keywords/tags
- Folder membership (as collections)
- File attachments

### Programmatic Usage

```python
from src.references.factory import create_reference_db

db = create_reference_db(
    provider="mendeley",
    db_path="path/to/mendeley.sqlite",
    storage_path="path/to/mendeley/files"  # Optional
)

for paper in db.get_all_papers():
    print(f"{paper.title}")
    print(f"  Collections: {paper.collections}")
```

## Comparison

| Feature | Zotero | BibTeX | PDF Folder | Mendeley |
| ------- | ------ | ------ | ---------- | -------- |
| Full metadata | Yes | Yes | Partial | Yes |
| PDF attachments | Yes | Manual match | Automatic | Yes |
| Collections/folders | Yes | No | From subfolders | Yes |
| Tags/keywords | Yes | From field | From PDF | Yes |
| Requires software | Yes | No | No | Yes |
| Cross-platform | Yes | Yes | Yes | Yes |

## Troubleshooting

### BibTeX Parse Errors

If your `.bib` file fails to parse:

1. Check for unescaped special characters (`&`, `%`, `_`)
2. Ensure all entries have closing braces
3. Try exporting fresh from your reference manager

### PDF Folder: Missing Metadata

If papers show minimal metadata:

1. Ensure PDF document properties are set
2. Use descriptive filenames with author/year/title
3. Enable `extract_pdf_metadata=True`

### Mendeley: Database Locked

If you get a "database is locked" error:

1. Close Mendeley Desktop
2. Wait a few seconds for locks to release
3. Retry the import

The adapter opens the database read-only, but Mendeley may hold locks.

### Mixed Sources

Currently, LITRIS supports one source per index build. To combine sources:

1. Export all to BibTeX format
2. Merge `.bib` files
3. Build from the combined BibTeX
