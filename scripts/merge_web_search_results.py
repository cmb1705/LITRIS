#!/usr/bin/env python
"""Merge web search DOI results into orphan PDFs report.

Reads web_search_dois.json and updates orphan_pdfs_report with newly found DOIs
and metadata from web searches.
"""

import csv
import json
from pathlib import Path


def load_web_search_results(json_path: Path) -> dict:
    """Load web search results and build lookup dictionaries."""
    with open(json_path) as f:
        data = json.load(f)

    # Build lookup by pdf_key
    pdf_matches = {}
    for match in data.get("matched_to_pdfs", []):
        if match.get("pdf_key") and match["pdf_key"] != "unknown":
            pdf_matches[match["pdf_key"]] = match

    # Build lookup by title (normalized) for metadata_only entries
    metadata_by_title = {}
    for entry in data.get("metadata_only", []):
        if entry.get("title"):
            # Normalize title for matching
            normalized = entry["title"].lower().strip()
            metadata_by_title[normalized] = entry

    # Build lookup for DOI results by title
    doi_by_title = {}
    results_by_doi = {}
    for entry in data.get("results", []):
        if entry.get("doi"):
            results_by_doi[entry["doi"]] = entry
        if entry.get("title") and entry.get("doi"):
            normalized = entry["title"].lower().strip()
            doi_by_title[normalized] = entry

    return {
        "pdf_matches": pdf_matches,
        "metadata_by_title": metadata_by_title,
        "doi_by_title": doi_by_title,
        "results_by_doi": results_by_doi,
        "total_dois": data.get("total_dois_found", 0),
    }


def normalize_title(title: str) -> str:
    """Normalize title for matching."""
    if not title:
        return ""
    # Remove common prefixes/suffixes
    normalized = title.lower().strip()
    # Remove year patterns
    import re
    normalized = re.sub(r"\s*\(\d{4}\)\s*$", "", normalized)
    normalized = re.sub(r"^\d{4}\s*[-_]\s*", "", normalized)
    return normalized


def update_report(
    input_csv: Path,
    output_csv: Path,
    web_results: dict,
) -> dict:
    """Update orphan PDFs report with web search results.

    Returns:
        Statistics about updates made.
    """
    stats = {
        "total_rows": 0,
        "doi_added": 0,
        "metadata_added": 0,
        "already_had_doi": 0,
    }

    rows = []
    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        for row in reader:
            stats["total_rows"] += 1
            updated = False

            attachment_key = row.get("attachment_key", "")
            current_doi = row.get("doi", "").strip()
            current_title = row.get("title", "").strip()

            # Check if we have a direct PDF match
            if attachment_key in web_results["pdf_matches"]:
                match = web_results["pdf_matches"][attachment_key]
                if match.get("doi") and not current_doi:
                    row["doi"] = match["doi"]
                    row["enrichment_source"] = "web_search"
                    row["enrichment_confidence"] = "0.90"
                    stats["doi_added"] += 1
                    updated = True
                    # Also get metadata from the matched results entry
                    result_entry = web_results.get("results_by_doi", {}).get(match["doi"])
                    if result_entry:
                        if not row.get("authors") and result_entry.get("authors"):
                            row["authors"] = result_entry["authors"]
                        if not row.get("year") and result_entry.get("year"):
                            row["year"] = str(result_entry["year"])
                elif current_doi:
                    stats["already_had_doi"] += 1

            # If still no DOI, try title matching
            if not row.get("doi") and current_title:
                normalized = normalize_title(current_title)

                # Check DOI results by title
                for title_key, entry in web_results["doi_by_title"].items():
                    if normalized in title_key or title_key in normalized:
                        row["doi"] = entry["doi"]
                        row["enrichment_source"] = "web_search_title_match"
                        row["enrichment_confidence"] = "0.85"
                        # Also add authors/year from the matched entry
                        if not row.get("authors") and entry.get("authors"):
                            row["authors"] = entry["authors"]
                        if not row.get("year") and entry.get("year"):
                            row["year"] = str(entry["year"])
                        stats["doi_added"] += 1
                        break

            # If still no DOI but we have metadata, update other fields
            if not row.get("doi") and current_title:
                normalized = normalize_title(current_title)

                for title_key, entry in web_results["metadata_by_title"].items():
                    if normalized in title_key or title_key in normalized:
                        # Update year if missing
                        if not row.get("year") and entry.get("year"):
                            row["year"] = str(entry["year"])
                        # Update authors if missing
                        if not row.get("authors") and entry.get("authors"):
                            row["authors"] = entry["authors"]
                        stats["metadata_added"] += 1
                        break

            rows.append(row)

    # Write updated report
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return stats


def main():
    base_path = Path(__file__).parent.parent / "data" / "query_results"
    json_path = base_path / "web_search_dois.json"
    input_csv = base_path / "orphan_pdfs_report_v3.csv"
    output_csv = base_path / "orphan_pdfs_report_v4.csv"

    print(f"Loading web search results from {json_path}...")
    web_results = load_web_search_results(json_path)
    print(f"  Total DOIs in web search: {web_results['total_dois']}")
    print(f"  PDF matches: {len(web_results['pdf_matches'])}")
    print(f"  Metadata entries: {len(web_results['metadata_by_title'])}")

    print(f"\nUpdating report from {input_csv}...")
    stats = update_report(input_csv, output_csv, web_results)

    print(f"\nResults:")
    print(f"  Total rows: {stats['total_rows']}")
    print(f"  DOIs added from web search: {stats['doi_added']}")
    print(f"  Metadata updated: {stats['metadata_added']}")
    print(f"  Already had DOI: {stats['already_had_doi']}")

    print(f"\nUpdated report saved to: {output_csv}")


if __name__ == "__main__":
    main()
