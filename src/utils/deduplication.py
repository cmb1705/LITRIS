"""DOI-based deduplication utilities for cross-database paper matching."""

from pathlib import Path

from src.utils.file_utils import safe_read_json
from src.zotero.models import PaperMetadata


def normalize_doi(doi: str | None) -> str | None:
    """Normalize DOI for consistent comparison.

    Args:
        doi: Raw DOI string.

    Returns:
        Normalized lowercase DOI without URL prefix, or None if invalid.
    """
    if not doi:
        return None

    doi = str(doi).strip().lower()

    # Remove common URL prefixes
    prefixes = [
        "https://doi.org/",
        "http://doi.org/",
        "doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "dx.doi.org/",
        "doi:",
    ]
    for prefix in prefixes:
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
            break

    # Remove leading/trailing whitespace after prefix removal
    doi = doi.strip()

    # Basic validation: DOI should start with "10."
    if not doi.startswith("10."):
        return None

    return doi


def extract_existing_dois(index_dir: Path) -> set[str]:
    """Extract normalized DOIs from existing index.

    Args:
        index_dir: Path to index directory containing papers.json.

    Returns:
        Set of normalized DOIs.
    """
    dois = set()
    papers_file = index_dir / "papers.json"

    if not papers_file.exists():
        return dois

    data = safe_read_json(papers_file, default={})

    # Handle structured format
    if isinstance(data, dict) and "papers" in data:
        papers = data["papers"]
    elif isinstance(data, list):
        papers = data
    else:
        return dois

    for paper in papers:
        doi = paper.get("doi")
        normalized = normalize_doi(doi)
        if normalized:
            dois.add(normalized)

    return dois


def filter_by_doi(
    papers: list[PaperMetadata],
    existing_dois: set[str],
) -> tuple[list[PaperMetadata], list[PaperMetadata]]:
    """Filter papers by DOI, removing those already in index.

    Args:
        papers: List of papers to filter.
        existing_dois: Set of normalized DOIs already in index.

    Returns:
        Tuple of (new_papers, duplicate_papers).
    """
    new_papers = []
    duplicate_papers = []

    for paper in papers:
        normalized = normalize_doi(paper.doi)

        if normalized and normalized in existing_dois:
            duplicate_papers.append(paper)
        else:
            new_papers.append(paper)

    return new_papers, duplicate_papers


def analyze_doi_overlap(
    new_papers: list[PaperMetadata],
    index_dir: Path,
) -> dict:
    """Analyze DOI overlap between new papers and existing index.

    Args:
        new_papers: Papers from new/current Zotero database.
        index_dir: Path to existing index.

    Returns:
        Analysis dictionary with counts and details.
    """
    existing_dois = extract_existing_dois(index_dir)

    new_with_doi = []
    new_without_doi = []
    duplicates = []
    genuinely_new = []

    for paper in new_papers:
        normalized = normalize_doi(paper.doi)

        if normalized:
            new_with_doi.append(paper)
            if normalized in existing_dois:
                duplicates.append(paper)
            else:
                genuinely_new.append(paper)
        else:
            new_without_doi.append(paper)

    return {
        "existing_index_dois": len(existing_dois),
        "new_papers_total": len(new_papers),
        "new_with_doi": len(new_with_doi),
        "new_without_doi": len(new_without_doi),
        "duplicates_by_doi": len(duplicates),
        "genuinely_new_with_doi": len(genuinely_new),
        "duplicate_papers": duplicates,
        "new_papers_filtered": genuinely_new + new_without_doi,
    }
