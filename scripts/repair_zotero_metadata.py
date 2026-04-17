#!/usr/bin/env python
"""Repair Zotero metadata for items with missing authors/dates.

Fills in missing metadata using:
1. CrossRef API (DOI lookup -- best quality)
2. CrossRef title search (for known papers without DOIs)
3. PDF file metadata (author/title/date from PDF properties)
4. Filename parsing (Author - Year - Title patterns)

Usage:
    python scripts/repair_zotero_metadata.py --dry-run          # Preview changes
    python scripts/repair_zotero_metadata.py --category A       # DOI lookups only
    python scripts/repair_zotero_metadata.py --category AB      # DOI + title search
    python scripts/repair_zotero_metadata.py                    # All categories
    python scripts/repair_zotero_metadata.py --apply            # Write to Zotero DB
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

CROSSREF_API = "https://api.crossref.org/works"
CROSSREF_EMAIL = os.environ.get("CROSSREF_EMAIL", "")  # For CrossRef polite pool

# Resolve Zotero paths from config (not hardcoded)
from src.config import Config as _Config

_cfg = _Config.load()
ZOTERO_DB = Path(_cfg.zotero.database_path)
ZOTERO_STORAGE = Path(_cfg.zotero.storage_path)


@dataclass
class MetadataFix:
    """A proposed metadata fix for a Zotero item."""

    item_id: int
    zotero_key: str
    current_title: str
    category: str  # A, B, C, D, E
    source: str  # crossref_doi, crossref_search, pdf_metadata, filename, manual

    # Proposed values (None = no change)
    new_title: str | None = None
    authors: list[dict] | None = None  # [{"first": "John", "last": "Doe"}, ...]
    date: str | None = None
    doi: str | None = None
    abstract: str | None = None
    publication: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    item_type: str | None = None  # Reclassify if needed

    confidence: float = 0.0  # 0-1 confidence in the fix
    notes: str = ""


def get_authorless_items(conn: sqlite3.Connection) -> list[dict]:
    """Get research items with no authors from Zotero database."""
    cursor = conn.cursor()

    target_types = (
        "journalArticle",
        "book",
        "bookSection",
        "report",
        "document",
        "conferencePaper",
        "thesis",
        "preprint",
    )
    placeholders = ",".join("?" * len(target_types))

    cursor.execute(
        f"""
        SELECT i.itemID, i.key, it.typeName, i.dateAdded
        FROM items i
        JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
        WHERE it.typeName IN ({placeholders})
        AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
        AND i.itemID NOT IN (SELECT itemID FROM itemCreators)
    """,
        target_types,
    )

    items = []
    for item_id, key, item_type, date_added in cursor.fetchall():
        # Get current field values
        fields = {}
        for field_name in (
            "title",
            "date",
            "abstractNote",
            "DOI",
            "publicationTitle",
            "volume",
            "issue",
            "pages",
            "url",
        ):
            fid = cursor.execute(
                "SELECT fieldID FROM fields WHERE fieldName = ?", (field_name,)
            ).fetchone()
            if fid:
                row = cursor.execute(
                    """
                    SELECT idv.value FROM itemData id
                    JOIN itemDataValues idv ON id.valueID = idv.valueID
                    WHERE id.itemID = ? AND id.fieldID = ?
                """,
                    (item_id, fid[0]),
                ).fetchone()
                fields[field_name] = row[0] if row else None

        # Get PDF attachment path
        pdf_row = cursor.execute(
            """
            SELECT i2.key, ia.path FROM itemAttachments ia
            JOIN items i2 ON ia.itemID = i2.itemID
            WHERE ia.parentItemID = ? AND ia.contentType = 'application/pdf'
        """,
            (item_id,),
        ).fetchone()

        pdf_path = None
        if pdf_row:
            att_key, att_path = pdf_row
            if att_path and att_path.startswith("storage:"):
                pdf_path = ZOTERO_STORAGE / att_key / att_path[8:]

        items.append(
            {
                "item_id": item_id,
                "key": key,
                "item_type": item_type,
                "date_added": date_added,
                "pdf_path": pdf_path,
                **fields,
            }
        )

    return items


def classify_item(item: dict) -> str:
    """Classify an item into category A-E."""
    if item.get("DOI"):
        return "A"

    title = item.get("title", "") or ""

    # Category E: garbled/scanner titles
    garbled_patterns = [
        r"^KMBT_",
        r"^214ps",
        r"^Scannable Document",
        r"^Scanned using",
        r"^pdf$",
        r"^\d+$",
        r"^__$",
    ]
    for pat in garbled_patterns:
        if re.match(pat, title, re.I):
            return "E"

    # Category D: non-publications (coursework, drafts, personal docs)
    non_pub_patterns = [
        r"^cc\d",
        r"^Activity \d",
        r"concept map",
        r"^DRAFT ",
        r"PROGRESS REPORT",
        r"Template-",
        r"cheatsheet",
        r"\.docx?$",
        r"\.pptx?$",
        r"\.jpg$",
        r"ACFT-Score",
        r"ARTIFACT ARCHIVE",
        r"User Agreement",
        r"biosketch",
        r"^annotated-",
        r"^bisht ",
        r"^moore ",
        r"Learning Outcomes",
        r"CHECKLIST",
        r"Recruitment-Script",
        r"STAC ActionPlan",
        r"Score-Cart",
        r"File Styles",
        r"Knitting Tips",
    ]
    for pat in non_pub_patterns:
        if re.search(pat, title, re.I):
            return "D"

    # Category C: government/policy reports (identifiable by org names)
    gov_patterns = [
        r"GAO",
        r"ODNI",
        r"NCSC",
        r"FAA",
        r"Biden",
        r"White House",
        r"National Strategy",
        r"CALL \d",
        r"Army",
        r"Defense",
        r"Strategic Guidance",
        r"Threat Assessment",
        r"NAS Roadmap",
        r"Rule Part 107",
        r"DOD",
        r"Emerging Tech.*Factsheet",
    ]
    for pat in gov_patterns:
        if re.search(pat, title, re.I):
            return "C"

    # Category B: likely real papers (has structured title)
    return "B"


def lookup_crossref_doi(doi: str) -> dict | None:
    """Look up metadata from CrossRef using DOI."""
    try:
        resp = requests.get(
            f"{CROSSREF_API}/{doi}",
            headers={"User-Agent": f"LITRIS/0.2.1 (mailto:{CROSSREF_EMAIL})"},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        data = resp.json().get("message", {})
        return _parse_crossref(data)
    except Exception as exc:
        logger.debug(f"CrossRef DOI lookup failed for {doi}: {exc}")
        return None


def search_crossref_title(title: str) -> dict | None:
    """Search CrossRef for a paper by title."""
    # Clean title: remove year prefixes, author prefixes, file extensions
    clean = re.sub(r"^\d{4}\s+", "", title)
    clean = re.sub(r"^[A-Z][a-z]+ (?:et al\.?\s*)?[-\u2013]\s*", "", clean)
    clean = re.sub(r"\.(pdf|txt|docx?)$", "", clean, flags=re.I)
    clean = clean.strip()

    if len(clean) < 10:
        return None

    try:
        resp = requests.get(
            CROSSREF_API,
            params={"query.title": clean[:200], "rows": 3},
            headers={"User-Agent": f"LITRIS/0.2.1 (mailto:{CROSSREF_EMAIL})"},
            timeout=15,
        )
        if resp.status_code != 200:
            return None

        items = resp.json().get("message", {}).get("items", [])
        if not items:
            return None

        # Check if top result title is a close match
        for item in items:
            result_title = item.get("title", [""])[0].lower()
            sim = _title_similarity(clean.lower(), result_title)
            if sim > 0.8:
                return _parse_crossref(item)

        return None
    except Exception as exc:
        logger.debug(f"CrossRef search failed for '{clean[:50]}': {exc}")
        return None


def extract_pdf_metadata(pdf_path: Path) -> dict | None:
    """Extract metadata from PDF file properties."""
    if not pdf_path or not pdf_path.exists():
        return None

    try:
        import fitz

        doc = fitz.open(str(pdf_path))
        meta = doc.metadata
        doc.close()

        result = {}
        if meta.get("author") and len(meta["author"]) > 2:
            result["authors_raw"] = meta["author"]
        if meta.get("title") and len(meta["title"]) > 5:
            result["title"] = meta["title"]
        if meta.get("creationDate"):
            # PDF dates: D:20200115120000
            date_match = re.search(r"(\d{4})", meta["creationDate"])
            if date_match:
                result["year"] = date_match.group(1)
        if meta.get("subject"):
            result["abstract"] = meta["subject"]

        return result if result else None
    except Exception as exc:
        logger.debug(f"PDF metadata extraction failed for {pdf_path}: {exc}")
        return None


def _parse_crossref(data: dict) -> dict:
    """Parse CrossRef API response into metadata dict."""
    result = {}

    # Title
    titles = data.get("title", [])
    if titles:
        result["title"] = titles[0]

    # Authors
    authors = []
    for a in data.get("author", []):
        authors.append(
            {
                "first": a.get("given", ""),
                "last": a.get("family", ""),
            }
        )
    if authors:
        result["authors"] = authors

    # Date
    date_parts = data.get("published-print", data.get("published-online", {}))
    parts = date_parts.get("date-parts", [[]])[0]
    if parts:
        if len(parts) >= 3:
            result["date"] = f"{parts[0]}-{parts[1]:02d}-{parts[2]:02d}"
        elif len(parts) >= 1:
            result["date"] = str(parts[0])

    # DOI
    if data.get("DOI"):
        result["doi"] = data["DOI"]

    # Abstract
    abstract = data.get("abstract", "")
    if abstract:
        # CrossRef abstracts often have JATS XML tags
        abstract = re.sub(r"<[^>]+>", "", abstract).strip()
        result["abstract"] = abstract

    # Journal
    containers = data.get("container-title", [])
    if containers:
        result["publication"] = containers[0]

    # Volume/Issue/Pages
    if data.get("volume"):
        result["volume"] = data["volume"]
    if data.get("issue"):
        result["issue"] = data["issue"]
    if data.get("page"):
        result["pages"] = data["page"]

    return result


def _title_similarity(a: str, b: str) -> float:
    """Simple word-overlap similarity between two titles."""
    words_a = set(re.findall(r"\w+", a.lower()))
    words_b = set(re.findall(r"\w+", b.lower()))
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    return len(intersection) / min(len(words_a), len(words_b))


def build_fixes(items: list[dict], categories: str = "ABCDE") -> list[MetadataFix]:
    """Build proposed fixes for all items."""
    fixes = []
    cat_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}

    for item in items:
        cat = classify_item(item)
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

        if cat not in categories:
            continue

        fix = MetadataFix(
            item_id=item["item_id"],
            zotero_key=item["key"],
            current_title=item.get("title", "(no title)"),
            category=cat,
            source="pending",
        )

        if cat == "A":
            # DOI lookup
            metadata = lookup_crossref_doi(item["DOI"])
            if metadata:
                fix.source = "crossref_doi"
                fix.confidence = 0.95
                _apply_metadata(fix, metadata, item)
                fixes.append(fix)
            else:
                fix.source = "crossref_doi_failed"
                fix.notes = f"DOI lookup failed for {item['DOI']}"
                fixes.append(fix)

        elif cat == "B":
            # Title search
            metadata = search_crossref_title(item.get("title", ""))
            if metadata:
                fix.source = "crossref_search"
                fix.confidence = 0.8
                _apply_metadata(fix, metadata, item)
                fixes.append(fix)
            else:
                # Fall back to PDF metadata
                pdf_meta = extract_pdf_metadata(item.get("pdf_path"))
                if pdf_meta:
                    fix.source = "pdf_metadata"
                    fix.confidence = 0.6
                    if pdf_meta.get("authors_raw"):
                        # Try to parse "First Last; First Last" or "First Last, First Last"
                        fix.authors = _parse_author_string(pdf_meta["authors_raw"])
                    if pdf_meta.get("title") and pdf_meta["title"] != item.get("title"):
                        fix.new_title = pdf_meta["title"]
                    if pdf_meta.get("year") and not item.get("date"):
                        fix.date = pdf_meta["year"]
                    fixes.append(fix)
                else:
                    fix.source = "no_match"
                    fix.notes = "No CrossRef match, no PDF metadata"
                    fixes.append(fix)

        elif cat == "C":
            # Government reports -- parse from title field
            pdf_meta = extract_pdf_metadata(item.get("pdf_path"))
            fix.source = "title_parse"
            fix.confidence = 0.5

            title = item.get("title", "")
            # Try to extract year from title
            year_match = re.search(r"\b(19|20)\d{2}\b", title)
            if year_match and not item.get("date"):
                fix.date = year_match.group()

            # Try to extract org author from title
            org_patterns = [
                (r"GAO", "U.S. Government Accountability Office"),
                (r"ODNI", "Office of the Director of National Intelligence"),
                (r"NCSC", "National Counterintelligence and Security Center"),
                (r"White House|Biden", "The White House"),
                (r"FAA", "Federal Aviation Administration"),
                (r"CALL \d", "Center for Army Lessons Learned"),
            ]
            for pat, org in org_patterns:
                if re.search(pat, title):
                    fix.authors = [{"first": "", "last": org}]
                    break

            if pdf_meta and pdf_meta.get("authors_raw"):
                fix.authors = _parse_author_string(pdf_meta["authors_raw"])

            fixes.append(fix)

        elif cat == "D":
            fix.source = "reclassify"
            fix.item_type = "document"
            fix.confidence = 0.7
            fix.notes = "Non-publication, reclassify as document"
            fixes.append(fix)

        elif cat == "E":
            pdf_meta = extract_pdf_metadata(item.get("pdf_path"))
            fix.source = "pdf_metadata" if pdf_meta else "manual_needed"
            fix.confidence = 0.3
            if pdf_meta:
                if pdf_meta.get("title"):
                    fix.new_title = pdf_meta["title"]
                if pdf_meta.get("authors_raw"):
                    fix.authors = _parse_author_string(pdf_meta["authors_raw"])
                if pdf_meta.get("year"):
                    fix.date = pdf_meta["year"]
            fixes.append(fix)

        # Rate limit CrossRef
        if cat in ("A", "B"):
            time.sleep(0.5)

    logger.info(f"Categories: {cat_counts}")
    return fixes


def _apply_metadata(fix: MetadataFix, metadata: dict, item: dict) -> None:
    """Apply CrossRef metadata to a fix."""
    if metadata.get("authors"):
        fix.authors = metadata["authors"]
    if metadata.get("date") and not item.get("date"):
        fix.date = metadata["date"]
    if metadata.get("doi") and not item.get("DOI"):
        fix.doi = metadata["doi"]
    if metadata.get("abstract") and not item.get("abstractNote"):
        fix.abstract = metadata["abstract"]
    if metadata.get("publication") and not item.get("publicationTitle"):
        fix.publication = metadata["publication"]
    if metadata.get("volume") and not item.get("volume"):
        fix.volume = metadata["volume"]
    if metadata.get("issue") and not item.get("issue"):
        fix.issue = metadata["issue"]
    if metadata.get("pages") and not item.get("pages"):
        fix.pages = metadata["pages"]
    if metadata.get("title"):
        # Only update title if current one looks garbled
        current = item.get("title", "")
        if current.endswith(".pdf") or re.match(r"^\d{4}\s", current) or len(current) > 100:
            fix.new_title = metadata["title"]


def _parse_author_string(raw: str) -> list[dict]:
    """Parse raw author string into structured author list."""
    authors = []
    # Try semicolon-separated
    parts = re.split(r"[;,]\s*", raw)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        words = part.split()
        if len(words) >= 2:
            authors.append({"first": " ".join(words[:-1]), "last": words[-1]})
        elif len(words) == 1:
            authors.append({"first": "", "last": words[0]})
    return authors


def apply_fixes_to_db(
    conn: sqlite3.Connection,
    fixes: list[MetadataFix],
) -> int:
    """Apply fixes to the Zotero database."""
    cursor = conn.cursor()
    applied = 0

    for fix in fixes:
        if fix.confidence < 0.5:
            continue

        try:
            # Add authors
            if fix.authors:
                # Get creator type ID for "author"
                ct_row = cursor.execute(
                    "SELECT creatorTypeID FROM creatorTypes WHERE creatorType = 'author'"
                ).fetchone()
                if ct_row:
                    ct_id = ct_row[0]
                    for idx, author in enumerate(fix.authors):
                        # Insert or find creator
                        cursor.execute(
                            "SELECT creatorID FROM creators WHERE firstName = ? AND lastName = ?",
                            (author["first"], author["last"]),
                        )
                        creator_row = cursor.fetchone()
                        if creator_row:
                            creator_id = creator_row[0]
                        else:
                            cursor.execute(
                                "INSERT INTO creators (firstName, lastName) VALUES (?, ?)",
                                (author["first"], author["last"]),
                            )
                            creator_id = cursor.lastrowid

                        # Link creator to item
                        cursor.execute(
                            "INSERT OR IGNORE INTO itemCreators (itemID, creatorID, creatorTypeID, orderIndex) VALUES (?, ?, ?, ?)",
                            (fix.item_id, creator_id, ct_id, idx),
                        )

            # Update fields
            field_updates = [
                ("title", fix.new_title),
                ("date", fix.date),
                ("DOI", fix.doi),
                ("abstractNote", fix.abstract),
                ("publicationTitle", fix.publication),
                ("volume", fix.volume),
                ("issue", fix.issue),
                ("pages", fix.pages),
            ]

            for field_name, value in field_updates:
                if value is None:
                    continue

                fid_row = cursor.execute(
                    "SELECT fieldID FROM fields WHERE fieldName = ?", (field_name,)
                ).fetchone()
                if not fid_row:
                    continue
                fid = fid_row[0]

                # Insert or update value
                val_row = cursor.execute(
                    "SELECT valueID FROM itemDataValues WHERE value = ?", (value,)
                ).fetchone()
                if val_row:
                    val_id = val_row[0]
                else:
                    cursor.execute("INSERT INTO itemDataValues (value) VALUES (?)", (value,))
                    val_id = cursor.lastrowid

                # Upsert itemData
                cursor.execute(
                    "INSERT OR REPLACE INTO itemData (itemID, fieldID, valueID) VALUES (?, ?, ?)",
                    (fix.item_id, fid, val_id),
                )

            applied += 1

        except Exception as exc:
            logger.error(f"Failed to apply fix for {fix.zotero_key}: {exc}")

    conn.commit()
    return applied


def print_report(fixes: list[MetadataFix]) -> None:
    """Print a summary report of proposed fixes."""
    by_cat = {}
    by_source = {}
    for fix in fixes:
        by_cat[fix.category] = by_cat.get(fix.category, 0) + 1
        by_source[fix.source] = by_source.get(fix.source, 0) + 1

    print(f"\n{'=' * 60}")
    print(f"Metadata Repair Report ({len(fixes)} items)")
    print(f"{'=' * 60}")
    print(f"\nBy category: {by_cat}")
    print(f"By source: {by_source}")

    high_conf = [f for f in fixes if f.confidence >= 0.8]
    med_conf = [f for f in fixes if 0.5 <= f.confidence < 0.8]
    low_conf = [f for f in fixes if f.confidence < 0.5]

    print(f"\nHigh confidence (>=0.8): {len(high_conf)} items")
    print(f"Medium confidence (0.5-0.8): {len(med_conf)} items")
    print(f"Low confidence (<0.5, skipped): {len(low_conf)} items")

    for fix in fixes:
        changes = []
        if fix.authors:
            author_str = "; ".join(f"{a['last']}, {a['first']}" for a in fix.authors[:3])
            if len(fix.authors) > 3:
                author_str += f" +{len(fix.authors) - 3} more"
            changes.append(f"authors={author_str}")
        if fix.date:
            changes.append(f"date={fix.date}")
        if fix.doi:
            changes.append(f"DOI={fix.doi}")
        if fix.new_title:
            changes.append(f"title={fix.new_title[:50]}...")
        if fix.abstract:
            changes.append(f"abstract={fix.abstract[:30]}...")
        if fix.publication:
            changes.append(f"journal={fix.publication[:30]}")
        if fix.item_type:
            changes.append(f"reclassify->{fix.item_type}")

        conf_label = f"[{fix.confidence:.0%}]"
        try:
            print(f"\n  {fix.zotero_key} (Cat {fix.category}) {conf_label} [{fix.source}]")
            print(f"    Current: {fix.current_title[:60]}")
            if changes:
                # Sanitize for Windows console encoding
                safe_changes = " | ".join(c.encode("ascii", "replace").decode() for c in changes)
                print(f"    Fix: {safe_changes}")
            elif fix.notes:
                print(f"    Note: {fix.notes}")
        except UnicodeEncodeError:
            print("    (output encoding error)")


def main():
    parser = argparse.ArgumentParser(description="Repair Zotero metadata")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes only")
    parser.add_argument("--apply", action="store_true", help="Write changes to Zotero DB")
    parser.add_argument("--category", default="ABCDE", help="Categories to process (e.g., AB)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(level="DEBUG" if args.verbose else "INFO")

    if not args.dry_run and not args.apply:
        print("Use --dry-run to preview or --apply to write changes.")
        return 1

    # Open Zotero DB
    mode = "ro" if args.dry_run else "rw"
    conn = sqlite3.connect(f"file:{ZOTERO_DB}?mode={mode}", uri=True)

    logger.info("Loading items with missing authors...")
    items = get_authorless_items(conn)
    logger.info(f"Found {len(items)} research items with no authors")

    logger.info(f"Building fixes for categories: {args.category}")
    fixes = build_fixes(items, categories=args.category.upper())

    print_report(fixes)

    if args.apply:
        print(f"\nApplying {len([f for f in fixes if f.confidence >= 0.5])} fixes...")
        applied = apply_fixes_to_db(conn, fixes)
        print(f"Applied {applied} fixes to Zotero database")

        # Save report
        report_path = Path("data/logs/zotero_metadata_repair.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_items": len(items),
            "fixes_proposed": len(fixes),
            "fixes_applied": applied,
            "fixes": [
                {
                    "key": f.zotero_key,
                    "category": f.category,
                    "source": f.source,
                    "confidence": f.confidence,
                    "authors": f.authors,
                    "date": f.date,
                    "doi": f.doi,
                    "title_changed": f.new_title is not None,
                }
                for f in fixes
                if f.confidence >= 0.5
            ],
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report saved: {report_path}")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
