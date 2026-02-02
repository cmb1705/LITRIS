"""Parent item creator for Zotero database.

Creates parent items in the Zotero SQLite database for orphan PDF attachments.
Links attachments to their new parent items and adds metadata.
"""

import random
import sqlite3
import string
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.utils.logging_config import get_logger
from src.zotero.metadata_enricher import EnrichedMetadata

logger = get_logger(__name__)


@dataclass
class CreatedParentItem:
    """Record of a created parent item."""

    parent_item_id: int
    parent_key: str
    attachment_item_id: int
    attachment_key: str
    title: str
    item_type: str
    doi: str | None = None
    success: bool = True
    error: str | None = None


class ParentItemCreator:
    """Creates parent items in Zotero database.

    Handles:
    - Generating unique 8-character keys
    - Creating item records
    - Adding metadata fields
    - Adding authors
    - Linking attachments to parents
    - Adding items to collections
    - Adding tags
    """

    # Valid characters for Zotero keys
    KEY_CHARS = string.ascii_uppercase + string.digits

    # Common item type IDs (may vary by Zotero version)
    ITEM_TYPE_NAMES = {
        "journalArticle": "journalArticle",
        "book": "book",
        "bookSection": "bookSection",
        "thesis": "thesis",
        "report": "report",
        "document": "document",
    }

    def __init__(self, db_path: str | Path, dry_run: bool = True):
        """Initialize the creator.

        Args:
            db_path: Path to Zotero SQLite database.
            dry_run: If True, don't commit changes (preview mode).
        """
        self.db_path = Path(db_path)
        self.dry_run = dry_run
        self._conn: sqlite3.Connection | None = None
        self._item_type_cache: dict[str, int] = {}
        self._field_cache: dict[str, int] = {}
        self._creator_type_cache: dict[str, int] = {}
        self._existing_keys: set[str] = set()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, timeout=30.0)
            self._conn.row_factory = sqlite3.Row
            self._load_caches()
        return self._conn

    def _load_caches(self) -> None:
        """Load lookup caches from database."""
        conn = self._conn
        if conn is None:
            return

        # Item types
        cursor = conn.execute("SELECT itemTypeID, typeName FROM itemTypes")
        for row in cursor:
            self._item_type_cache[row["typeName"]] = row["itemTypeID"]

        # Fields
        cursor = conn.execute("SELECT fieldID, fieldName FROM fields")
        for row in cursor:
            self._field_cache[row["fieldName"]] = row["fieldID"]

        # Creator types
        cursor = conn.execute("SELECT creatorTypeID, creatorType FROM creatorTypes")
        for row in cursor:
            self._creator_type_cache[row["creatorType"]] = row["creatorTypeID"]

        # Existing keys
        cursor = conn.execute("SELECT key FROM items")
        for row in cursor:
            self._existing_keys.add(row["key"])

        logger.debug(
            f"Loaded caches: {len(self._item_type_cache)} item types, "
            f"{len(self._field_cache)} fields, {len(self._existing_keys)} keys"
        )

    def _generate_key(self) -> str:
        """Generate a unique 8-character Zotero key."""
        while True:
            key = "".join(random.choices(self.KEY_CHARS, k=8))
            if key not in self._existing_keys:
                self._existing_keys.add(key)
                return key

    def _get_item_type_id(self, type_name: str) -> int:
        """Get item type ID for a type name."""
        if type_name not in self._item_type_cache:
            # Default to document
            type_name = "document"
        return self._item_type_cache.get(type_name, 1)

    def _get_field_id(self, field_name: str) -> int | None:
        """Get field ID for a field name."""
        return self._field_cache.get(field_name)

    def _get_creator_type_id(self, creator_type: str = "author") -> int:
        """Get creator type ID."""
        return self._creator_type_cache.get(creator_type, 1)

    def _insert_item(
        self,
        conn: sqlite3.Connection,
        item_type_id: int,
        key: str,
    ) -> int:
        """Insert a new item record.

        Args:
            conn: Database connection.
            item_type_id: Item type ID.
            key: 8-character unique key.

        Returns:
            New item ID.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor = conn.execute(
            """
            INSERT INTO items (itemTypeID, key, dateAdded, dateModified, clientDateModified, libraryID, version)
            VALUES (?, ?, ?, ?, ?, 1, 0)
            """,
            (item_type_id, key, now, now, now),
        )
        return cursor.lastrowid or 0

    def _insert_field(
        self,
        conn: sqlite3.Connection,
        item_id: int,
        field_name: str,
        value: str,
    ) -> bool:
        """Insert a metadata field for an item.

        Args:
            conn: Database connection.
            item_id: Item ID.
            field_name: Field name (e.g., 'title', 'DOI').
            value: Field value.

        Returns:
            True if successful.
        """
        field_id = self._get_field_id(field_name)
        if field_id is None:
            logger.warning(f"Unknown field: {field_name}")
            return False

        # Check if value already exists
        cursor = conn.execute(
            "SELECT valueID FROM itemDataValues WHERE value = ?",
            (value,),
        )
        row = cursor.fetchone()
        if row:
            value_id = row["valueID"]
        else:
            # Insert new value
            cursor = conn.execute(
                "INSERT INTO itemDataValues (value) VALUES (?)",
                (value,),
            )
            value_id = cursor.lastrowid

        # Insert item data
        conn.execute(
            "INSERT OR REPLACE INTO itemData (itemID, fieldID, valueID) VALUES (?, ?, ?)",
            (item_id, field_id, value_id),
        )
        return True

    def _insert_creator(
        self,
        conn: sqlite3.Connection,
        item_id: int,
        name: str,
        order_index: int,
        creator_type: str = "author",
    ) -> bool:
        """Insert a creator (author) for an item.

        Args:
            conn: Database connection.
            item_id: Item ID.
            name: Creator name.
            order_index: Position in creator list (0-based).
            creator_type: Type of creator (default: 'author').

        Returns:
            True if successful.
        """
        # Parse name into first/last
        first_name = ""
        last_name = name

        if "," in name:
            parts = name.split(",", 1)
            last_name = parts[0].strip()
            first_name = parts[1].strip() if len(parts) > 1 else ""
        elif " " in name:
            parts = name.rsplit(" ", 1)
            first_name = parts[0].strip()
            last_name = parts[1].strip() if len(parts) > 1 else ""

        # Check if creator exists
        cursor = conn.execute(
            "SELECT creatorID FROM creators WHERE firstName = ? AND lastName = ?",
            (first_name, last_name),
        )
        row = cursor.fetchone()
        if row:
            creator_id = row["creatorID"]
        else:
            # Insert new creator
            cursor = conn.execute(
                "INSERT INTO creators (firstName, lastName, fieldMode) VALUES (?, ?, 0)",
                (first_name, last_name),
            )
            creator_id = cursor.lastrowid

        # Get creator type ID
        creator_type_id = self._get_creator_type_id(creator_type)

        # Insert item-creator link
        conn.execute(
            """
            INSERT OR REPLACE INTO itemCreators (itemID, creatorID, creatorTypeID, orderIndex)
            VALUES (?, ?, ?, ?)
            """,
            (item_id, creator_id, creator_type_id, order_index),
        )
        return True

    def _link_attachment(
        self,
        conn: sqlite3.Connection,
        attachment_item_id: int,
        parent_item_id: int,
    ) -> bool:
        """Link an attachment to its parent item.

        Args:
            conn: Database connection.
            attachment_item_id: Attachment item ID.
            parent_item_id: Parent item ID.

        Returns:
            True if successful.
        """
        conn.execute(
            "UPDATE itemAttachments SET parentItemID = ? WHERE itemID = ?",
            (parent_item_id, attachment_item_id),
        )
        return True

    def _add_to_collection(
        self,
        conn: sqlite3.Connection,
        item_id: int,
        collection_id: int,
    ) -> bool:
        """Add an item to a collection.

        Args:
            conn: Database connection.
            item_id: Item ID.
            collection_id: Collection ID.

        Returns:
            True if successful.
        """
        conn.execute(
            "INSERT OR IGNORE INTO collectionItems (collectionID, itemID) VALUES (?, ?)",
            (collection_id, item_id),
        )
        return True

    def _add_tag(
        self,
        conn: sqlite3.Connection,
        item_id: int,
        tag_name: str,
    ) -> bool:
        """Add a tag to an item.

        Args:
            conn: Database connection.
            item_id: Item ID.
            tag_name: Tag name.

        Returns:
            True if successful.
        """
        # Check if tag exists
        cursor = conn.execute(
            "SELECT tagID FROM tags WHERE name = ?",
            (tag_name,),
        )
        row = cursor.fetchone()
        if row:
            tag_id = row["tagID"]
        else:
            # Insert new tag
            cursor = conn.execute(
                "INSERT INTO tags (name) VALUES (?)",
                (tag_name,),
            )
            tag_id = cursor.lastrowid

        # Add tag to item
        conn.execute(
            "INSERT OR IGNORE INTO itemTags (itemID, tagID) VALUES (?, ?)",
            (item_id, tag_id),
        )
        return True

    def _determine_item_type(self, enriched: EnrichedMetadata) -> str:
        """Determine the item type based on metadata.

        Args:
            enriched: Enriched metadata.

        Returns:
            Item type name.
        """
        # Check for ISBN -> book
        if enriched.isbn or enriched.original.isbn:
            return "book"

        # Check for journal
        if enriched.journal or enriched.issn:
            return "journalArticle"

        # Check filename hints
        if enriched.original.pdf_path:
            filename = enriched.original.pdf_path.name.lower()
            if "thesis" in filename or "dissertation" in filename:
                return "thesis"
            if "report" in filename:
                return "report"

        # Default to journal article (most common in academic contexts)
        return "journalArticle"

    def create_parent_item(
        self,
        enriched: EnrichedMetadata,
        collection_id: int | None = None,
        tag: str = "auto-created",
    ) -> CreatedParentItem:
        """Create a parent item for an orphan attachment.

        Args:
            enriched: Enriched metadata for the item.
            collection_id: Collection to add the item to.
            tag: Tag to add to the item.

        Returns:
            CreatedParentItem record.
        """
        conn = self._get_connection()

        attachment_item_id = enriched.original.attachment_item_id
        attachment_key = enriched.original.attachment_key or ""

        if attachment_item_id is None:
            return CreatedParentItem(
                parent_item_id=0,
                parent_key="",
                attachment_item_id=0,
                attachment_key=attachment_key,
                title=enriched.best_title or "",
                item_type="",
                success=False,
                error="No attachment item ID",
            )

        try:
            # Generate key and determine type
            parent_key = self._generate_key()
            item_type = self._determine_item_type(enriched)
            item_type_id = self._get_item_type_id(item_type)

            # Create parent item
            parent_item_id = self._insert_item(conn, item_type_id, parent_key)

            # Add title
            title = enriched.best_title or enriched.original.pdf_path.stem if enriched.original.pdf_path else "Untitled"
            self._insert_field(conn, parent_item_id, "title", title)

            # Add DOI if available
            doi = enriched.best_doi
            if doi:
                self._insert_field(conn, parent_item_id, "DOI", doi)

            # Add other fields
            if enriched.journal:
                self._insert_field(conn, parent_item_id, "publicationTitle", enriched.journal)
            if enriched.best_year:
                self._insert_field(conn, parent_item_id, "date", str(enriched.best_year))
            if enriched.volume:
                self._insert_field(conn, parent_item_id, "volume", enriched.volume)
            if enriched.issue:
                self._insert_field(conn, parent_item_id, "issue", enriched.issue)
            if enriched.pages:
                self._insert_field(conn, parent_item_id, "pages", enriched.pages)
            if enriched.abstract:
                self._insert_field(conn, parent_item_id, "abstractNote", enriched.abstract[:10000])
            if enriched.url:
                self._insert_field(conn, parent_item_id, "url", enriched.url)
            if enriched.issn:
                self._insert_field(conn, parent_item_id, "ISSN", enriched.issn)
            if enriched.isbn or enriched.original.isbn:
                self._insert_field(conn, parent_item_id, "ISBN", enriched.isbn or enriched.original.isbn or "")

            # Add authors
            for i, author in enumerate(enriched.best_authors[:20]):  # Limit to 20 authors
                self._insert_creator(conn, parent_item_id, author, i)

            # Link attachment to parent
            self._link_attachment(conn, attachment_item_id, parent_item_id)

            # Add to collection
            if collection_id:
                self._add_to_collection(conn, parent_item_id, collection_id)

            # Add tag
            if tag:
                self._add_tag(conn, parent_item_id, tag)

            if not self.dry_run:
                conn.commit()
                logger.info(f"Created parent item {parent_key} for attachment {attachment_key}")
            else:
                logger.info(f"[DRY RUN] Would create parent item {parent_key} for attachment {attachment_key}")

            return CreatedParentItem(
                parent_item_id=parent_item_id,
                parent_key=parent_key,
                attachment_item_id=attachment_item_id,
                attachment_key=attachment_key,
                title=title,
                item_type=item_type,
                doi=doi,
                success=True,
            )

        except Exception as e:
            logger.error(f"Failed to create parent item: {e}")
            if not self.dry_run:
                conn.rollback()
            return CreatedParentItem(
                parent_item_id=0,
                parent_key="",
                attachment_item_id=attachment_item_id or 0,
                attachment_key=attachment_key,
                title=enriched.best_title or "",
                item_type="",
                success=False,
                error=str(e),
            )

    def commit(self) -> None:
        """Commit all changes to database."""
        if self._conn and not self.dry_run:
            self._conn.commit()
            logger.info("Committed changes to database")

    def rollback(self) -> None:
        """Rollback all changes."""
        if self._conn:
            self._conn.rollback()
            logger.info("Rolled back changes")

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "ParentItemCreator":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if exc_type is not None:
            self.rollback()
        self.close()
