#!/usr/bin/env python
"""Reversible cleanup utility for the CAS_mRNA Zotero collection.

This script is intentionally conservative:

- It never deletes items or attachments from Zotero.
- It creates a timestamped SQLite backup before any write operation.
- It moves duplicate or off-object parent items out of ``CAS_mRNA`` into
  subcollections instead of removing them from the library.
- It can tag known title-collision items for easier later review.

Use ``--dry-run`` first to inspect the proposed actions.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

KEY_CHARS = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
KEY_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"

OFF_OBJECT_DOIS = {
    "10.18564/jasss.5583",
    "10.1038/s41598-024-71248-y",
}

OFF_OBJECT_TITLE_PATTERNS = (
    "behavioral dynamics of epidemic trajectories and vaccination strategies",
    "conditional protein splicing of the mycobacterium tuberculosis reca intein",
)

BRANSTETTER_COLLISION_KEY = "FS7BX3GV"
BRANSTETTER_COLLISION_TAG = "checked-okuyama-not-branstetter"
NEEDS_PARENT_SPLIT_TAG = "needs-parent-split"
ARCHIVED_DUPLICATE_TAG = "archived-duplicate-cas-mrna"
OFF_OBJECT_TAG = "off-object-cas-mrna"


@dataclass
class CollectionRecord:
    """Minimal item state for collection cleanup decisions."""

    item_id: int
    zotero_key: str
    item_type: str
    title: str
    doi: str | None
    url: str | None
    date_added: str
    date_modified: str
    pdf_attachment_count: int
    html_attachment_count: int
    distinct_pdf_paths: list[str]
    metadata_score: int

    @property
    def normalized_title(self) -> str:
        """Return a stable normalized title for grouping."""
        title = (self.title or "").strip().lower()
        title = re.sub(r"\s+", " ", title)
        return title


@dataclass
class CleanupReport:
    """Serializable cleanup outcome."""

    timestamp_local: str
    dry_run: bool
    cas_mrna_collection_id: int
    archive_duplicates_collection_id: int | None
    off_object_collection_id: int | None
    moved_to_duplicates: list[dict]
    moved_to_off_object: list[dict]
    removed_attachment_items: list[dict]
    tagged_items: list[dict]
    split_parent_candidates: list[dict]
    duplicate_groups: list[dict]
    backup_path: str | None = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path(r"C:\Users\cmb17\Zotero\zotero.sqlite"),
        help="Path to live Zotero SQLite database.",
    )
    parser.add_argument(
        "--collection-name",
        default="CAS_mRNA",
        help="Top-level collection to clean.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect actions without writing to Zotero.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("data") / "logs",
        help="Directory for JSON cleanup reports.",
    )
    return parser.parse_args()


def generate_key(existing_keys: set[str]) -> str:
    """Generate a unique Zotero-style 8-character key."""
    while True:
        key = random.choice(KEY_LETTERS) + "".join(random.choices(KEY_CHARS, k=7))
        if key not in existing_keys:
            existing_keys.add(key)
            return key


def normalize_doi(doi: str | None) -> str | None:
    """Return a normalized DOI or ``None``."""
    if not doi:
        return None
    normalized = doi.strip().lower()
    normalized = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", normalized)
    return normalized or None


def now_sql() -> str:
    """Return a Zotero-compatible timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def backup_database(db_path: Path, backup_dir: Path) -> Path:
    """Create a consistent SQLite backup using the sqlite backup API."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{db_path.stem}_cas_mrna_cleanup_{timestamp}.sqlite"

    source = sqlite3.connect(str(db_path))
    target = sqlite3.connect(str(backup_path))
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()
    return backup_path


def get_existing_keys(conn: sqlite3.Connection) -> set[str]:
    """Return existing Zotero keys to avoid collisions."""
    return {row[0] for row in conn.execute("SELECT key FROM items UNION SELECT key FROM collections")}


def get_collection_id(conn: sqlite3.Connection, name: str) -> int:
    """Return the collection ID for a named collection."""
    row = conn.execute(
        "SELECT collectionID FROM collections WHERE collectionName = ?",
        (name,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Collection not found: {name}")
    return int(row[0])


def get_or_create_child_collection(
    conn: sqlite3.Connection,
    parent_collection_id: int,
    child_name: str,
    existing_keys: set[str],
    dry_run: bool,
) -> int | None:
    """Get or create a child collection under CAS_mRNA."""
    row = conn.execute(
        """
        SELECT collectionID
        FROM collections
        WHERE collectionName = ? AND parentCollectionID = ?
        """,
        (child_name, parent_collection_id),
    ).fetchone()
    if row is not None:
        return int(row[0])
    if dry_run:
        return None

    key = generate_key(existing_keys)
    conn.execute(
        """
        INSERT INTO collections (
            collectionName, parentCollectionID, clientDateModified,
            libraryID, key, version, synced
        ) VALUES (?, ?, ?, 1, ?, 0, 0)
        """,
        (child_name, parent_collection_id, now_sql(), key),
    )
    row = conn.execute(
        """
        SELECT collectionID
        FROM collections
        WHERE collectionName = ? AND parentCollectionID = ?
        """,
        (child_name, parent_collection_id),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Failed to create child collection: {child_name}")
    return int(row[0])


def get_or_create_tag(conn: sqlite3.Connection, tag_name: str, dry_run: bool) -> int | None:
    """Get or create a Zotero tag."""
    row = conn.execute("SELECT tagID FROM tags WHERE name = ?", (tag_name,)).fetchone()
    if row is not None:
        return int(row[0])
    if dry_run:
        return None
    conn.execute("INSERT INTO tags (name) VALUES (?)", (tag_name,))
    row = conn.execute("SELECT tagID FROM tags WHERE name = ?", (tag_name,)).fetchone()
    if row is None:
        raise RuntimeError(f"Failed to create tag: {tag_name}")
    return int(row[0])


def add_tag(conn: sqlite3.Connection, item_id: int, tag_name: str, dry_run: bool) -> bool:
    """Attach a tag to an item."""
    tag_id = get_or_create_tag(conn, tag_name, dry_run=dry_run)
    if tag_id is None:
        return False
    if not dry_run:
        conn.execute(
            "INSERT OR IGNORE INTO itemTags (itemID, tagID, type) VALUES (?, ?, 0)",
            (item_id, tag_id),
        )
    return True


def move_item_between_collections(
    conn: sqlite3.Connection,
    item_id: int,
    from_collection_id: int,
    to_collection_id: int | None,
    dry_run: bool,
) -> None:
    """Move an item from one collection to another."""
    if dry_run:
        return
    if to_collection_id is not None:
        conn.execute(
            """
            INSERT OR IGNORE INTO collectionItems (collectionID, itemID, orderIndex)
            VALUES (?, ?, 0)
            """,
            (to_collection_id, item_id),
        )
    conn.execute(
        "DELETE FROM collectionItems WHERE collectionID = ? AND itemID = ?",
        (from_collection_id, item_id),
    )


def fetch_collection_records(conn: sqlite3.Connection, collection_id: int) -> list[CollectionRecord]:
    """Load collection items and lightweight metadata."""
    query = """
    SELECT
        i.itemID,
        i.key,
        it.typeName,
        COALESCE(
            (SELECT idv.value
             FROM itemData id
             JOIN fields f ON id.fieldID = f.fieldID
             JOIN itemDataValues idv ON id.valueID = idv.valueID
             WHERE id.itemID = i.itemID AND f.fieldName = 'title'
             LIMIT 1),
            'Untitled'
        ) AS title,
        (SELECT idv.value
         FROM itemData id
         JOIN fields f ON id.fieldID = f.fieldID
         JOIN itemDataValues idv ON id.valueID = idv.valueID
         WHERE id.itemID = i.itemID AND f.fieldName = 'DOI'
         LIMIT 1) AS doi,
        (SELECT idv.value
         FROM itemData id
         JOIN fields f ON id.fieldID = f.fieldID
         JOIN itemDataValues idv ON id.valueID = idv.valueID
         WHERE id.itemID = i.itemID AND f.fieldName = 'url'
         LIMIT 1) AS url,
        i.dateAdded,
        i.dateModified,
        (SELECT COUNT(*)
         FROM itemAttachments att
         WHERE att.parentItemID = i.itemID AND att.contentType = 'application/pdf'
        ) AS pdf_attachment_count,
        (SELECT COUNT(*)
         FROM itemAttachments att
         WHERE att.parentItemID = i.itemID AND att.contentType = 'text/html'
        ) AS html_attachment_count
    FROM collectionItems ci
    JOIN items i ON ci.itemID = i.itemID
    JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
    WHERE ci.collectionID = ?
    ORDER BY i.itemID
    """
    records: list[CollectionRecord] = []
    conn.row_factory = sqlite3.Row
    for row in conn.execute(query, (collection_id,)):
        pdf_paths = [
            r[0]
            for r in conn.execute(
                """
                SELECT DISTINCT path
                FROM itemAttachments
                WHERE parentItemID = ? AND contentType = 'application/pdf'
                ORDER BY path
                """,
                (row["itemID"],),
            ).fetchall()
        ]
        metadata_score = sum(
            1
            for value in (
                row["title"],
                row["doi"],
                row["url"],
                row["dateAdded"],
                row["dateModified"],
            )
            if value
        )
        records.append(
            CollectionRecord(
                item_id=int(row["itemID"]),
                zotero_key=str(row["key"]),
                item_type=str(row["typeName"]),
                title=str(row["title"]),
                doi=normalize_doi(row["doi"]),
                url=row["url"],
                date_added=str(row["dateAdded"]),
                date_modified=str(row["dateModified"]),
                pdf_attachment_count=int(row["pdf_attachment_count"] or 0),
                html_attachment_count=int(row["html_attachment_count"] or 0),
                distinct_pdf_paths=[str(path) for path in pdf_paths if path],
                metadata_score=metadata_score,
            )
        )
    return records


def record_score(record: CollectionRecord) -> tuple[int, int, int, str]:
    """Rank candidate keepers within a duplicate group."""
    # Higher is better. Favor PDF-backed academic entries, then richer metadata,
    # then most recently added item as the tie-breaker.
    return (
        1 if record.item_type != "webpage" else 0,
        1 if record.pdf_attachment_count > 0 else 0,
        1 if len(record.distinct_pdf_paths) == 1 else 0,
        record.metadata_score,
        record.date_modified,
        record.date_added,
    )


def identify_duplicate_groups(records: Iterable[CollectionRecord]) -> list[dict]:
    """Group exact duplicates by DOI, then by exact normalized title."""
    by_doi: dict[str, list[CollectionRecord]] = {}
    by_title: dict[str, list[CollectionRecord]] = {}
    for record in records:
        if record.item_type == "attachment":
            continue
        if record.doi:
            by_doi.setdefault(record.doi, []).append(record)
        else:
            by_title.setdefault(record.normalized_title, []).append(record)

    groups: list[dict] = []
    for source, mapping in (("doi", by_doi), ("title", by_title)):
        for group_key, members in mapping.items():
            if len(members) < 2:
                continue
            if source == "title":
                has_pdf = any(record.pdf_attachment_count > 0 for record in members)
                has_pdfless = any(record.pdf_attachment_count == 0 for record in members)
                if not (has_pdf and has_pdfless):
                    continue
            ordered = sorted(members, key=record_score, reverse=True)
            groups.append({
                "source": source,
                "group_key": group_key,
                "keeper": ordered[0],
                "duplicates": ordered[1:],
            })
    return groups


def is_off_object(record: CollectionRecord) -> bool:
    """Return whether a record is clearly off-object for CAS_mRNA."""
    if record.doi and record.doi in OFF_OBJECT_DOIS:
        return True
    title = record.normalized_title
    return any(pattern in title for pattern in OFF_OBJECT_TITLE_PATTERNS)


def identify_split_parent_candidates(records: Iterable[CollectionRecord]) -> list[CollectionRecord]:
    """Return parents that contain multiple distinct PDFs under one Zotero item."""
    return [
        record
        for record in records
        if record.item_type != "attachment" and len(record.distinct_pdf_paths) > 1
    ]


def write_report(report: CleanupReport, report_dir: Path) -> Path:
    """Persist cleanup report as JSON."""
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = report_dir / f"cas_mrna_cleanup_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(asdict(report), handle, indent=2, ensure_ascii=False)
    return path


def main() -> int:
    """Run cleanup."""
    args = parse_args()
    timestamp_local = datetime.now().isoformat(timespec="seconds")

    backup_path: Path | None = None
    if not args.dry_run:
        backup_path = backup_database(
            args.db_path,
            args.db_path.parent / "backups",
        )

    conn = sqlite3.connect(str(args.db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        cas_collection_id = get_collection_id(conn, args.collection_name)
        existing_keys = get_existing_keys(conn)

        archive_duplicates_id = get_or_create_child_collection(
            conn,
            cas_collection_id,
            "Archive - Duplicates",
            existing_keys,
            dry_run=args.dry_run,
        )
        off_object_id = get_or_create_child_collection(
            conn,
            cas_collection_id,
            "Off-object",
            existing_keys,
            dry_run=args.dry_run,
        )

        records = fetch_collection_records(conn, cas_collection_id)
        duplicate_groups = identify_duplicate_groups(records)
        split_parent_candidates = identify_split_parent_candidates(records)

        moved_to_duplicates: list[dict] = []
        moved_to_off_object: list[dict] = []
        removed_attachment_items: list[dict] = []
        tagged_items: list[dict] = []

        # Remove literal attachment items from the working collection.
        for record in records:
            if record.item_type != "attachment":
                continue
            move_item_between_collections(
                conn,
                record.item_id,
                cas_collection_id,
                archive_duplicates_id,
                dry_run=args.dry_run,
            )
            removed_attachment_items.append(asdict(record))

        # Move clearly off-object parent items out of CAS_mRNA.
        off_object_ids: set[int] = set()
        for record in records:
            if record.item_type == "attachment" or not is_off_object(record):
                continue
            move_item_between_collections(
                conn,
                record.item_id,
                cas_collection_id,
                off_object_id,
                dry_run=args.dry_run,
            )
            add_tag(conn, record.item_id, OFF_OBJECT_TAG, dry_run=args.dry_run)
            moved_to_off_object.append(asdict(record))
            off_object_ids.add(record.item_id)
            tagged_items.append({"item_id": record.item_id, "tag": OFF_OBJECT_TAG})

        # Archive duplicate parents, keeping the strongest item in CAS_mRNA.
        archived_duplicate_ids: set[int] = set()
        for group in duplicate_groups:
            keeper: CollectionRecord = group["keeper"]
            for duplicate in group["duplicates"]:
                if duplicate.item_id in off_object_ids or duplicate.item_id in archived_duplicate_ids:
                    continue
                move_item_between_collections(
                    conn,
                    duplicate.item_id,
                    cas_collection_id,
                    archive_duplicates_id,
                    dry_run=args.dry_run,
                )
                add_tag(
                    conn,
                    duplicate.item_id,
                    ARCHIVED_DUPLICATE_TAG,
                    dry_run=args.dry_run,
                )
                moved_to_duplicates.append({
                    "duplicate": asdict(duplicate),
                    "keeper": asdict(keeper),
                    "group_key": group["group_key"],
                    "source": group["source"],
                })
                archived_duplicate_ids.add(duplicate.item_id)
                tagged_items.append(
                    {"item_id": duplicate.item_id, "tag": ARCHIVED_DUPLICATE_TAG}
                )

        # Mark known title-collision / attribution-risk records explicitly.
        for record in records:
            if record.zotero_key == BRANSTETTER_COLLISION_KEY:
                if add_tag(
                    conn,
                    record.item_id,
                    BRANSTETTER_COLLISION_TAG,
                    dry_run=args.dry_run,
                ):
                    tagged_items.append(
                        {
                            "item_id": record.item_id,
                            "tag": BRANSTETTER_COLLISION_TAG,
                        }
                    )
            if record in split_parent_candidates:
                if add_tag(
                    conn,
                    record.item_id,
                    NEEDS_PARENT_SPLIT_TAG,
                    dry_run=args.dry_run,
                ):
                    tagged_items.append(
                        {"item_id": record.item_id, "tag": NEEDS_PARENT_SPLIT_TAG}
                    )

        if not args.dry_run:
            conn.commit()
        else:
            conn.rollback()

        report = CleanupReport(
            timestamp_local=timestamp_local,
            dry_run=args.dry_run,
            cas_mrna_collection_id=cas_collection_id,
            archive_duplicates_collection_id=archive_duplicates_id,
            off_object_collection_id=off_object_id,
            moved_to_duplicates=moved_to_duplicates,
            moved_to_off_object=moved_to_off_object,
            removed_attachment_items=removed_attachment_items,
            tagged_items=tagged_items,
            split_parent_candidates=[asdict(record) for record in split_parent_candidates],
            duplicate_groups=[
                {
                    "source": group["source"],
                    "group_key": group["group_key"],
                    "keeper": asdict(group["keeper"]),
                    "duplicates": [asdict(record) for record in group["duplicates"]],
                }
                for group in duplicate_groups
            ],
            backup_path=str(backup_path) if backup_path else None,
        )
        report_path = write_report(report, args.report_dir)
        print(json.dumps({"report_path": str(report_path), **asdict(report)}, indent=2))
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
