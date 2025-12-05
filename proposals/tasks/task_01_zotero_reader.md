# Task 01: Zotero Database Reader

**Phase:** 1 (Foundation)
**Priority:** Critical (Blocking)
**Estimated Effort:** 3-4 hours
**Dependencies:** Task 00 (Setup)

---

## Objective

Create a read-only interface to the Zotero SQLite database that extracts paper metadata, author information, collection membership, and resolves PDF file paths.

---

## Prerequisites

- Task 00 completed (configuration system in place)
- Access to Zotero database at configured path
- Understanding of Zotero schema (see technical specification)

---

## Implementation Details

### 01.1 Create Zotero Data Models

**File:** `src/zotero/models.py`

**Purpose:** Pydantic models representing Zotero data structures.

**Models Required:**

#### Author Model

| Field | Type | Description |
|-------|------|-------------|
| first_name | str | Author first name |
| last_name | str | Author last name |
| full_name | str | Computed full name |
| order | int | Position in author list (1 = first author) |
| role | str | author, editor, contributor, translator |

**Computed Fields:**
- `full_name`: Combine first and last name, handle single-name authors

#### Collection Model

| Field | Type | Description |
|-------|------|-------------|
| collection_id | int | Zotero collection ID |
| name | str | Collection name |
| parent_path | list[str] | Ancestor collection names |
| full_path | str | Complete path (e.g., "Parent/Child/Name") |

#### PaperMetadata Model

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| paper_id | str | Yes | Generated UUID |
| zotero_key | str | Yes | 8-character Zotero key |
| zotero_item_id | int | Yes | Zotero internal ID |
| item_type | str | Yes | journalArticle, book, thesis, etc. |
| title | str | Yes | Paper title |
| authors | list[Author] | Yes | Author list (may be empty) |
| publication_year | int | No | Extracted year |
| publication_date | str | No | Full date string |
| journal | str | No | Journal/publication name |
| volume | str | No | Volume number |
| issue | str | No | Issue number |
| pages | str | No | Page range |
| doi | str | No | Digital Object Identifier |
| isbn | str | No | ISBN |
| issn | str | No | ISSN |
| abstract | str | No | Abstract text |
| url | str | No | Source URL |
| collections | list[str] | Yes | Collection names (may be empty) |
| tags | list[str] | Yes | User tags (may be empty) |
| pdf_path | Path | No | Full path to PDF file |
| pdf_attachment_key | str | No | Zotero key for attachment |
| date_added | datetime | Yes | When added to Zotero |
| date_modified | datetime | Yes | Last modification |
| indexed_at | datetime | No | When indexed by this system |

**Validators:**
- Validate `pdf_path` exists if provided
- Parse `publication_date` to extract `publication_year`
- Handle missing optional fields gracefully

---

### 01.2 Create ZoteroDatabase Class

**File:** `src/zotero/database.py`

**Purpose:** SQLite interface for reading Zotero data.

**Class: ZoteroDatabase**

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| db_path | Path | From config | Path to zotero.sqlite |
| storage_path | Path | From config | Path to storage directory |

**Connection Management:**
- Open database in read-only mode using `file:{path}?mode=ro` URI
- Use context manager pattern for connection handling
- Handle database locked errors gracefully
- Set appropriate timeouts for busy database

---

### 01.3 Implement Item Query Methods

**Method:** `get_all_items_with_pdfs()`

**Returns:** `list[PaperMetadata]`

**SQL Logic:**
1. Select from `items` table
2. Join `itemTypes` to filter by type
3. Join `itemAttachments` to find PDF children
4. Filter: `contentType = 'application/pdf'`
5. Filter: `itemTypes.typeName NOT IN ('attachment', 'note', 'annotation')`
6. Return parent item IDs with attachment info

**Query Structure:**
```
SELECT
    parent.itemID,
    parent.key,
    parent.dateAdded,
    parent.dateModified,
    it.typeName,
    att.itemID as attachmentID,
    att.key as attachmentKey,
    att.path as attachmentPath
FROM items parent
JOIN itemTypes it ON parent.itemTypeID = it.itemTypeID
JOIN itemAttachments att ON parent.itemID = att.parentItemID
JOIN items attItem ON att.itemID = attItem.itemID
WHERE att.contentType = 'application/pdf'
  AND it.typeName NOT IN ('attachment', 'note', 'annotation')
```

---

### 01.4 Implement Metadata Extraction

**Method:** `get_item_metadata(item_id: int)`

**Returns:** `dict[str, str]` (field_name -> value)

**SQL Logic:**
```
SELECT f.fieldName, idv.value
FROM itemData id
JOIN fields f ON id.fieldID = f.fieldID
JOIN itemDataValues idv ON id.valueID = idv.valueID
WHERE id.itemID = ?
```

**Field Mapping:**
| Zotero Field | Model Field |
|--------------|-------------|
| title | title |
| abstractNote | abstract |
| date | publication_date |
| publicationTitle | journal |
| volume | volume |
| issue | issue |
| pages | pages |
| DOI | doi |
| ISBN | isbn |
| ISSN | issn |
| url | url |

---

### 01.5 Implement Author Extraction

**Method:** `get_item_authors(item_id: int)`

**Returns:** `list[Author]`

**SQL Logic:**
```
SELECT
    c.firstName,
    c.lastName,
    c.fieldMode,
    ic.orderIndex,
    ct.creatorType
FROM itemCreators ic
JOIN creators c ON ic.creatorID = c.creatorID
JOIN creatorTypes ct ON ic.creatorTypeID = ct.creatorTypeID
WHERE ic.itemID = ?
ORDER BY ic.orderIndex
```

**Field Mode Handling:**
- `fieldMode = 0`: Normal two-field name (first, last)
- `fieldMode = 1`: Single-field name (stored in lastName only)

**Order Index:**
- 0-based in database, convert to 1-based for model

---

### 01.6 Implement Collection Extraction

**Method:** `get_item_collections(item_id: int)`

**Returns:** `list[str]` (collection names)

**SQL Logic:**
```
SELECT c.collectionName, c.parentCollectionID
FROM collectionItems ci
JOIN collections c ON ci.collectionID = c.collectionID
WHERE ci.itemID = ?
```

**Hierarchy Resolution:**
- For each collection, recursively fetch parent names
- Build full path: "Parent/Child/Grandchild"
- Cache collection hierarchy to avoid repeated queries

**Method:** `get_collection_hierarchy()`

**Returns:** `dict[int, Collection]` (collection_id -> Collection)

**Purpose:** Pre-fetch all collections for efficient hierarchy resolution

---

### 01.7 Implement PDF Path Resolution

**Method:** `resolve_pdf_path(attachment_key: str, attachment_path: str)`

**Returns:** `Path | None`

**Logic:**
1. Parse attachment_path format: `storage:{filename}`
2. Extract filename from path
3. Construct full path: `{storage_path}/{attachment_key}/{filename}`
4. Verify file exists
5. Return Path object or None if not found

**Edge Cases:**
- Handle URL attachments (path starts with `http`)
- Handle linked files (path format differs)
- Handle missing files gracefully

---

### 01.8 Implement Tag Extraction

**Method:** `get_item_tags(item_id: int)`

**Returns:** `list[str]`

**SQL Logic:**
```
SELECT t.name
FROM itemTags it
JOIN tags t ON it.tagID = t.tagID
WHERE it.itemID = ?
```

---

### 01.9 Implement Full Paper Assembly

**Method:** `get_paper(item_id: int, attachment_info: dict)`

**Returns:** `PaperMetadata`

**Logic:**
1. Get item metadata
2. Get authors
3. Get collections
4. Get tags
5. Resolve PDF path
6. Generate UUID for paper_id
7. Assemble PaperMetadata model
8. Return validated model

---

### 01.10 Implement Batch Processing

**Method:** `get_all_papers()`

**Returns:** `list[PaperMetadata]`

**Logic:**
1. Pre-fetch collection hierarchy
2. Query all items with PDFs
3. For each item, assemble full paper
4. Track and log any failures
5. Return list of successful papers

**Progress Tracking:**
- Yield papers as generator for memory efficiency
- Or batch with progress callback

---

## Test Scenarios

### T01.1 Database Connection

**Test:** Successfully connect to Zotero database
**Input:** Valid database path from config
**Expected:** Connection opens without error
**Verify:** Can execute simple query

### T01.2 Read-Only Mode

**Test:** Database opened in read-only mode
**Input:** Attempt INSERT statement
**Expected:** OperationalError raised
**Verify:** Error message indicates read-only

### T01.3 Items with PDFs Query

**Test:** Retrieve all items with PDF attachments
**Input:** Database with known paper count
**Expected:** Returns list of correct length
**Verify:** All items have valid attachment info

### T01.4 Metadata Extraction

**Test:** Extract metadata for known item
**Input:** Item ID of paper with complete metadata
**Expected:** All expected fields populated
**Verify:** Title, abstract, DOI match known values

### T01.5 Author Extraction

**Test:** Extract authors with correct ordering
**Input:** Item ID of multi-author paper
**Expected:** Authors in correct order
**Verify:** First author has order=1, names correct

### T01.6 Single-Name Author

**Test:** Handle single-field author names
**Input:** Item with fieldMode=1 author
**Expected:** full_name populated from lastName
**Verify:** No empty first_name errors

### T01.7 Collection Hierarchy

**Test:** Resolve nested collection path
**Input:** Item in child collection with parent
**Expected:** Collection path includes ancestors
**Verify:** Path format is "Parent/Child"

### T01.8 PDF Path Resolution

**Test:** Resolve path to existing PDF
**Input:** Valid attachment key and path
**Expected:** Returns Path object
**Verify:** Path.exists() returns True

### T01.9 Missing PDF Handling

**Test:** Handle missing PDF file gracefully
**Input:** Attachment with non-existent file
**Expected:** pdf_path is None, no exception
**Verify:** Paper metadata still created

### T01.10 Empty Collections

**Test:** Handle item with no collections
**Input:** Item not in any collection
**Expected:** collections is empty list
**Verify:** No exception, valid model

### T01.11 Database Locked

**Test:** Handle locked database gracefully
**Input:** Database locked by running Zotero
**Expected:** Clear error message
**Verify:** Suggests closing Zotero

---

## Caveats and Edge Cases

### Database Locking

- Zotero locks the database when running
- Use `mode=ro` in connection string
- Implement retry with timeout
- Provide clear error message if locked

### Character Encoding

- Zotero stores UTF-8 text
- Some older entries may have encoding issues
- Handle decode errors gracefully
- Log problematic entries

### Large Libraries

- Libraries with 1000+ items need efficient queries
- Use generator pattern to avoid loading all into memory
- Consider batch processing with commits

### Item Types to Skip

- Attachments without parents
- Notes (itemTypeID for 'note')
- Annotations (itemTypeID for 'annotation')
- Standalone attachments

### Missing Metadata

- Many items have incomplete metadata
- Title is required, others optional
- Handle None values throughout

### Date Parsing

- Zotero dates have varying formats
- "2023-05-15", "May 2023", "2023", etc.
- Extract year with regex fallback
- Store original string and parsed year separately

### Linked vs Stored Files

- Stored files: `storage:{filename}`
- Linked files: Full path like `C:\Papers\file.pdf`
- URL attachments: `http://...`
- Handle each type appropriately

### Collection Cycles

- Zotero shouldn't allow cycles, but validate
- Implement max depth limit (10 levels)
- Log warning if max depth reached

### Performance

- Pre-fetch collection hierarchy (one query)
- Consider caching for repeated access
- Use parameterized queries to prevent injection

---

## Acceptance Criteria

- [ ] Can connect to Zotero database in read-only mode
- [ ] Retrieves all items with PDF attachments
- [ ] Extracts complete metadata for each item
- [ ] Correctly orders and formats author names
- [ ] Resolves collection hierarchy paths
- [ ] Resolves PDF file paths correctly
- [ ] Handles missing/incomplete data gracefully
- [ ] Provides clear error for locked database
- [ ] All unit tests pass

---

## Files Created

| File | Status |
|------|--------|
| src/zotero/__init__.py | Task 00 |
| src/zotero/models.py | Pending |
| src/zotero/database.py | Pending |
| tests/test_zotero_reader.py | Pending |

---

*End of Task 01*
