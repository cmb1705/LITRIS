# Task 08: Incremental Updates

**Phase:** 4 (Incremental Updates)
**Priority:** Medium
**Estimated Effort:** 3-4 hours
**Dependencies:** Tasks 01-07 (Full pipeline with robustness)

---

## Objective

Implement change detection from Zotero and incremental index updates, allowing the system to add new papers, update modified papers, and remove deleted papers without full rebuild.

---

## Prerequisites

- Tasks 01-07 completed (full pipeline operational)
- Understanding of Zotero modification tracking
- Existing index with metadata.json

---

## Implementation Details

### 08.1 Create Change Detector

**File:** `src/zotero/change_detector.py`

**Purpose:** Detect changes in Zotero since last index update.

**Class: ChangeDetector**

**Constructor Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| zotero_db | ZoteroDatabase | Database reader |
| index_metadata | IndexMetadata | Last known state |

---

### 08.2 Create Change Report Model

**Model: ChangeReport**

| Field | Type | Description |
|-------|------|-------------|
| detected_at | datetime | When detection ran |
| new_items | list[str] | Zotero keys of new papers |
| modified_items | list[str] | Zotero keys of changed papers |
| deleted_items | list[str] | Paper IDs no longer in Zotero |
| unchanged_items | int | Count of unchanged papers |
| total_in_zotero | int | Total items in Zotero |
| total_in_index | int | Total items in index |

---

### 08.3 Implement New Item Detection

**Method:** `find_new_items()`

**Returns:** `list[str]` (Zotero keys)

**Logic:**
1. Query Zotero for all items with PDFs
2. Get list of zotero_keys already in index
3. Return keys in Zotero but not in index

**Optimization:**
- Use indexed lookup for existing keys
- Store keys in set for O(1) lookup

---

### 08.4 Implement Modified Item Detection

**Method:** `find_modified_items()`

**Returns:** `list[str]` (Zotero keys)

**Logic:**
1. Get last_update timestamp from index metadata
2. Query Zotero for items where dateModified > last_update
3. Filter to items that exist in index
4. Return modified keys

**Date Comparison:**
- Zotero stores timestamps as strings
- Parse to datetime for comparison
- Handle timezone differences

**Modification Triggers:**
- Metadata changes (title, authors, etc.)
- PDF replacement
- Collection changes
- Tag changes

---

### 08.5 Implement Deleted Item Detection

**Method:** `find_deleted_items()`

**Returns:** `list[str]` (paper_ids)

**Logic:**
1. Get all paper_ids from index
2. Get all zotero_keys from Zotero
3. Map paper_ids to zotero_keys
4. Find paper_ids whose zotero_key is no longer in Zotero
5. Return deleted paper_ids

**Considerations:**
- Items removed from Zotero
- Items whose PDF was removed
- Items moved to trash

---

### 08.6 Implement Full Change Detection

**Method:** `detect_changes()`

**Returns:** `ChangeReport`

**Logic:**
1. Find new items
2. Find modified items
3. Find deleted items
4. Calculate unchanged count
5. Build and return ChangeReport

**Performance:**
- Single pass through Zotero items
- Compare against index in memory
- Cache results for subsequent calls

---

### 08.7 Create Update Pipeline

**File:** `scripts/update_index.py`

**Purpose:** Apply detected changes to the index.

**Class: IndexUpdater**

**Constructor Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| zotero_db | ZoteroDatabase | Database reader |
| pdf_extractor | PDFExtractor | Text extraction |
| section_extractor | SectionExtractor | LLM extraction |
| embedding_generator | EmbeddingGenerator | Embeddings |
| vector_store | VectorStore | ChromaDB |
| structured_store | StructuredStore | JSON files |

---

### 08.8 Implement Add New Items

**Method:** `add_items(zotero_keys: list[str])`

**Returns:** `UpdateResult`

**Logic:**
1. For each new zotero_key:
   - Fetch paper metadata from Zotero
   - Extract PDF text
   - Run LLM extraction
   - Generate embeddings
   - Add to vector store
   - Add to structured store
2. Track successes and failures
3. Return result summary

**UpdateResult Model:**

| Field | Type | Description |
|-------|------|-------------|
| action | str | 'add', 'update', 'delete' |
| attempted | int | Items attempted |
| succeeded | int | Items succeeded |
| failed | int | Items failed |
| failures | list[FailedItem] | Failure details |

---

### 08.9 Implement Update Modified Items

**Method:** `update_items(zotero_keys: list[str])`

**Returns:** `UpdateResult`

**Logic:**
1. For each modified zotero_key:
   - Find existing paper_id in index
   - Fetch updated metadata from Zotero
   - Check what changed (metadata only? PDF too?)
   - If PDF changed:
     - Re-extract text
     - Re-run LLM extraction
     - Re-generate embeddings
   - Update vector store (delete old, add new)
   - Update structured store
2. Track results
3. Return summary

**Change Detection Granularity:**

| Change Type | Re-extract PDF | Re-run LLM | Re-embed |
|-------------|----------------|------------|----------|
| Metadata only | No | No | Update metadata |
| PDF replaced | Yes | Yes | Yes |
| Tags only | No | No | Update metadata |
| Collections only | No | No | Update metadata |

---

### 08.10 Implement Delete Items

**Method:** `delete_items(paper_ids: list[str])`

**Returns:** `UpdateResult`

**Logic:**
1. For each paper_id to delete:
   - Remove from vector store (all chunks)
   - Remove from structured store
   - Remove from cache (optional)
2. Track results
3. Return summary

---

### 08.11 Implement Full Update

**Method:** `apply_changes(change_report: ChangeReport)`

**Returns:** `UpdateSummary`

**Logic:**
1. Delete removed items first (frees space)
2. Update modified items
3. Add new items
4. Update metadata with new timestamp
5. Regenerate summary.json
6. Return combined summary

**UpdateSummary Model:**

| Field | Type | Description |
|-------|------|-------------|
| started_at | datetime | Update start time |
| completed_at | datetime | Update end time |
| adds | UpdateResult | Add results |
| updates | UpdateResult | Update results |
| deletes | UpdateResult | Delete results |
| new_total | int | Papers after update |
| errors | list[FailedItem] | All failures |

---

### 08.12 Create Update CLI Script

**File:** `scripts/update_index.py`

**Arguments:**

| Argument | Description |
|----------|-------------|
| `--detect-only` | Show changes without applying |
| `--dry-run` | Simulate update without changes |
| `--force-reprocess` | Re-process even unchanged items |
| `--items KEY1,KEY2` | Update specific items only |
| `--skip-llm` | Update metadata only, skip LLM |
| `--verbose` | Detailed logging |

**Example Usage:**

```bash
# Check for changes
python scripts/update_index.py --detect-only

# Apply all changes
python scripts/update_index.py

# Force reprocess specific items
python scripts/update_index.py --items ABC123,DEF456 --force-reprocess
```

---

### 08.13 Implement State Persistence

**Updates to metadata.json:**

| Field | Purpose |
|-------|---------|
| last_update | Timestamp of last update |
| last_zotero_hash | Hash of Zotero state (optional) |
| update_history | List of recent updates |

**Update History Entry:**

| Field | Type | Description |
|-------|------|-------------|
| timestamp | datetime | When update ran |
| added | int | Papers added |
| updated | int | Papers updated |
| deleted | int | Papers deleted |
| duration_seconds | float | How long it took |

---

### 08.14 Implement Conflict Resolution

**Purpose:** Handle edge cases in updates.

**Conflicts:**

1. **Paper exists in index but not in Zotero:**
   - Treat as deleted
   - Remove from index

2. **Paper modified while processing:**
   - Detect via timestamp
   - Re-queue for next update

3. **Embedding dimension mismatch:**
   - If model changed, need full rebuild
   - Warn user and refuse update

4. **Partial update failure:**
   - Keep successful changes
   - Track failures for retry
   - Maintain consistency

---

## Test Scenarios

### T08.1 New Item Detection

**Test:** Detect newly added paper
**Input:** Add paper to Zotero after last_update
**Expected:** Appears in new_items
**Verify:** zotero_key in change report

### T08.2 Modified Item Detection

**Test:** Detect modified paper
**Input:** Update paper metadata in Zotero
**Expected:** Appears in modified_items
**Verify:** zotero_key in change report

### T08.3 Deleted Item Detection

**Test:** Detect removed paper
**Input:** Delete paper from Zotero
**Expected:** Appears in deleted_items
**Verify:** paper_id in change report

### T08.4 No Changes Detection

**Test:** Detect when nothing changed
**Input:** No Zotero changes since last update
**Expected:** Empty new/modified/deleted lists
**Verify:** unchanged_items equals total

### T08.5 Add New Items

**Test:** Add new paper to index
**Input:** New paper in Zotero
**Expected:** Full pipeline runs
**Verify:** Paper in vector store and JSON

### T08.6 Update Modified Item

**Test:** Update changed paper
**Input:** Modified paper in Zotero
**Expected:** Chunks updated
**Verify:** New metadata in index

### T08.7 Delete Items

**Test:** Remove paper from index
**Input:** Deleted paper
**Expected:** All traces removed
**Verify:** Not in vector store or JSON

### T08.8 Dry Run Mode

**Test:** Preview without changes
**Input:** --dry-run flag
**Expected:** Reports what would happen
**Verify:** No actual changes made

### T08.9 Detect Only Mode

**Test:** Show changes without applying
**Input:** --detect-only flag
**Expected:** ChangeReport displayed
**Verify:** No update applied

### T08.10 Metadata Update Timestamp

**Test:** Update saves timestamp
**Input:** Successful update
**Expected:** last_update updated
**Verify:** Timestamp in metadata.json

### T08.11 Update History

**Test:** History recorded
**Input:** Run update
**Expected:** Entry added to history
**Verify:** update_history has new entry

### T08.12 Partial Failure Handling

**Test:** Some items fail, others succeed
**Input:** Mix of good and bad papers
**Expected:** Successes applied, failures tracked
**Verify:** Good papers in index, failures logged

---

## Caveats and Edge Cases

### Zotero Running During Update

- Zotero may be running and modifying database
- Use read-only connection
- Accept that state may change during update
- Design for eventual consistency

### Timestamp Precision

- Zotero timestamps may have limited precision
- Use >= for date comparisons
- Account for clock skew

### Large Number of Changes

- Many changes may be slow
- Consider batching
- Show progress for large updates
- Allow interruption with checkpoint

### PDF Replaced but Same Name

- Need to detect content change
- Use file hash or modification time
- Don't rely on filename alone

### Metadata Only vs Full Reprocess

- Distinguish lightweight updates
- Avoid unnecessary LLM calls
- But ensure consistency

### Collection Reorganization

- User may reorganize collections
- This modifies many items
- Handle efficiently (metadata only)

### Zotero Sync Timing

- Zotero cloud sync may not be complete
- User should ensure sync before update
- Consider sync status check

### Index Corruption Recovery

- If index gets corrupted
- Full rebuild is the safest option
- Consider backup before update

### Embedding Model Change

- If model changed since last build
- Embeddings incompatible
- Force full rebuild
- Detect via model name in metadata

### Concurrent Updates

- Don't run multiple updates simultaneously
- Use lock file
- Second update should wait or abort

---

## Acceptance Criteria

- [x] Detects new items in Zotero
- [x] Detects modified items since last update
- [x] Detects deleted items
- [x] Adds new items to index
- [x] Updates modified items in index
- [x] Removes deleted items from index
- [x] Dry run mode works correctly
- [x] Detect only mode works correctly
- [x] Updates metadata timestamp
- [x] Records update history
- [x] Handles partial failures
- [x] All unit tests pass

---

## Files Created

| File | Status |
|------|--------|
| src/zotero/change_detector.py | Complete |
| src/indexing/update_state.py | Complete |
| scripts/update_index.py | Complete |
| tests/test_incremental_update.py | Complete (25 tests) |

---

*End of Task 08*
