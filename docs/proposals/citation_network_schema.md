# Citation Network Schema Specification

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0 |
| Created | 2026-01-22 |
| Task | LITRIS-vvx |
| Parent | F1: Citation Network Visualization |

---

## 1. Data Sources

### 1.1 Primary Source: Zotero Relations

Zotero stores explicit user-created relationships in `itemRelations` table.

| Pros | Cons |
|------|------|
| Already in local database | User must create relations manually |
| High confidence links | Sparse coverage |
| No API calls needed | Limited to library scope |

### 1.2 Secondary Source: Extraction-Based References

LLM extractions include `key_references` field with cited work titles.

| Pros | Cons |
|------|------|
| Automatically extracted | Title matching is fuzzy |
| Covers all processed papers | May miss papers outside library |
| Rich context available | Requires deduplication |

### 1.3 Future Source: External APIs (Out of Scope)

OpenAlex, Semantic Scholar, or CrossRef could provide citation data.

| Pros | Cons |
|------|------|
| Comprehensive coverage | Requires API integration |
| Authoritative metadata | Rate limits and quotas |
| DOI-based linking | Network dependency |

**Decision**: Use extraction-based references as primary source, with Zotero relations as supplement. External APIs deferred to future enhancement.

---

## 2. Graph Schema

### 2.1 Node Schema

```json
{
  "id": "string",           // paper_id from LITRIS index
  "type": "paper",          // node type (extensible for authors later)
  "label": "string",        // truncated title for display
  "title": "string",        // full title
  "authors": "string",      // author list
  "year": "integer | null", // publication year
  "collections": ["string"],// Zotero collection names
  "in_library": "boolean",  // true if indexed in LITRIS
  "doi": "string | null",   // DOI if available
  "size": "integer",        // visual size (based on citation count)
  "color": "string"         // hex color (based on type/status)
}
```

### 2.2 Edge Schema

```json
{
  "source": "string",       // citing paper_id
  "target": "string",       // cited paper_id
  "type": "cites",          // edge type
  "confidence": "float",    // 0.0-1.0 match confidence
  "source_type": "string",  // "extraction" | "zotero_relation" | "doi_match"
  "context": "string | null"// citation context if available
}
```

### 2.3 Graph Metadata

```json
{
  "generated": "ISO8601 timestamp",
  "node_count": "integer",
  "edge_count": "integer",
  "index_version": "string",
  "filters_applied": {
    "collections": ["string"] | null,
    "year_range": [min, max] | null
  }
}
```

---

## 3. Node Types and Colors

| Node Type | Color (Light) | Color (Dark) | Criteria |
|-----------|---------------|--------------|----------|
| In-library paper | #4a8c6a | #508c78 | paper_id exists in index |
| External reference | #8c8c8c | #6a6a6a | Referenced but not indexed |
| Focal paper | #e08050 | #e08050 | Selected/searched paper |
| Highly cited | #3a6a9a | #5080b0 | 5+ incoming edges |

---

## 4. Edge Confidence Scoring

| Source | Base Confidence | Adjustments |
|--------|-----------------|-------------|
| DOI exact match | 1.0 | - |
| Zotero relation | 0.95 | - |
| Title exact match | 0.9 | -0.1 if common words only |
| Title fuzzy match (>0.85) | 0.7 | +0.1 if year matches |
| Author + partial title | 0.6 | +0.1 if year matches |

---

## 5. Limitations

1. **Library-scoped**: Only shows citations within indexed papers
2. **Extraction-dependent**: Requires LLM extraction to have completed
3. **Fuzzy matching**: Title-based matching may produce false positives
4. **No external enrichment**: Cannot show citations to papers outside library
5. **Temporal gaps**: Historical citations may be incomplete

---

## 6. Sample Fixture

```json
{
  "metadata": {
    "generated": "2026-01-22T12:00:00Z",
    "node_count": 5,
    "edge_count": 4,
    "index_version": "1.0.0",
    "filters_applied": null
  },
  "nodes": [
    {
      "id": "paper_001",
      "type": "paper",
      "label": "Heterogeneous Graph...",
      "title": "Heterogeneous Graph Transformer",
      "authors": "Hu et al.",
      "year": 2020,
      "collections": ["GNN", "Core Papers"],
      "in_library": true,
      "doi": "10.1145/3366423.3380027",
      "size": 30,
      "color": "#e08050"
    },
    {
      "id": "paper_002",
      "type": "paper",
      "label": "Attention Is All You...",
      "title": "Attention Is All You Need",
      "authors": "Vaswani et al.",
      "year": 2017,
      "collections": ["Transformers"],
      "in_library": true,
      "doi": "10.48550/arXiv.1706.03762",
      "size": 25,
      "color": "#3a6a9a"
    },
    {
      "id": "ext_ref_001",
      "type": "paper",
      "label": "BERT: Pre-training...",
      "title": "BERT: Pre-training of Deep Bidirectional Transformers",
      "authors": "Devlin et al.",
      "year": 2019,
      "collections": [],
      "in_library": false,
      "doi": null,
      "size": 15,
      "color": "#8c8c8c"
    }
  ],
  "edges": [
    {
      "source": "paper_001",
      "target": "paper_002",
      "type": "cites",
      "confidence": 0.95,
      "source_type": "extraction",
      "context": "builds on the transformer architecture"
    },
    {
      "source": "paper_001",
      "target": "ext_ref_001",
      "type": "cites",
      "confidence": 0.7,
      "source_type": "extraction",
      "context": null
    }
  ]
}
```

---

## 7. Implementation Notes

1. **Storage**: Graph stored as JSON in `data/cache/citation_graph.json`
2. **Regeneration**: Rebuild on index update or manual trigger
3. **Filtering**: Support runtime filtering by collection/year without rebuild
4. **Caching**: Cache PyVis HTML output for repeated renders
