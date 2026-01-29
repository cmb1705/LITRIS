# Federated Search Configuration

Federated search enables querying multiple LITRIS indexes simultaneously, merging and ranking results across sources.

## Use Cases

- **Multiple Zotero Libraries**: Search your personal library and a shared lab library together
- **Cross-Group Collaboration**: Combine indexes from different research groups
- **Domain Separation**: Keep separate indexes for different research domains while enabling combined searches
- **Historical Archives**: Include older indexes alongside current research

## Compatibility Requirements

All federated indexes must meet these requirements:

1. **Same Embedding Model**: All indexes must use the same sentence-transformer model (e.g., `all-MiniLM-L6-v2`)
2. **Compatible Schema Versions**: Indexes must have compatible schema versions (same major version)
3. **Valid Index Structure**: Each index must contain:
   - `papers.json` - Paper metadata
   - `extractions.json` - LLM extractions
   - `chroma/` - ChromaDB vector store

## Configuration

Add the `federated` section to your `config.yaml`:

```yaml
federated:
  enabled: true
  merge_strategy: "interleave"
  dedup_threshold: 0.95
  max_results_per_index: 50
  indexes:
    - path: "/path/to/colleague/index"
      label: "Colleague Library"
      enabled: true
      weight: 1.0
    - path: "/path/to/archive/index"
      label: "Historical Archive"
      enabled: true
      weight: 0.8
```

## Configuration Options

| Option | Type | Default | Description |
| ------ | ---- | ------- | ----------- |
| `enabled` | bool | false | Enable federated search |
| `merge_strategy` | string | "interleave" | How to combine results |
| `dedup_threshold` | float | 0.95 | Similarity threshold for deduplication |
| `max_results_per_index` | int | 50 | Max results per index before merging |
| `indexes` | list | [] | Additional indexes to search |

### Merge Strategies

- **interleave**: Round-robin selection by score. Takes the highest-scoring result from any index, then next highest, etc. Best for balanced results across sources.

- **concat**: Primary index results first, then federated indexes in order. Use when primary index should be prioritized.

- **rerank**: Combines scores from all indexes and re-ranks globally. Best when you want unified scoring but may be slower.

### Index Configuration

Each index in the `indexes` list has:

| Option | Type | Default | Description |
| ------ | ---- | ------- | ----------- |
| `path` | string | required | Path to index directory |
| `label` | string | required | Display name in results |
| `enabled` | bool | true | Include in federated searches |
| `weight` | float | 1.0 | Relevance weight (0.0-2.0) |

## Weights

The `weight` parameter affects result scoring:

- **1.0**: Standard weight (equal to primary index)
- **> 1.0**: Boost results from this index
- **< 1.0**: De-prioritize results from this index
- **0.0**: Include but don't rank (only for deduplication)

Example: Give 20% boost to a curated library:

```yaml
indexes:
  - path: "/curated/index"
    label: "Curated Papers"
    weight: 1.2
```

## Deduplication

When the same paper appears in multiple indexes, federated search deduplicates based on:

1. **DOI matching**: Papers with identical DOIs are merged
2. **Title similarity**: Papers with similarity >= `dedup_threshold` are merged
3. **Weight selection**: The version from the highest-weighted index is kept

Set `dedup_threshold`:
- **0.95**: Very strict, only near-identical papers (recommended)
- **0.90**: Allows minor title variations
- **0.80**: More aggressive deduplication (may merge different papers)

## Verifying Index Compatibility

Before adding an index, verify compatibility:

```bash
# Check embedding model in existing index
cat /path/to/index/metadata.json | jq .embedding_model

# Compare with your primary index
cat data/index/metadata.json | jq .embedding_model
```

Both should show the same model (e.g., `"sentence-transformers/all-MiniLM-L6-v2"`).

## Troubleshooting

### "Incompatible embedding model" error

The federated index uses a different embedding model. You must rebuild one of the indexes with the same model.

### Missing or empty results from federated index

1. Check the index path is correct and accessible
2. Verify the index has valid `papers.json` and `chroma/` directory
3. Ensure `enabled: true` for the index

### Slow federated searches

1. Reduce `max_results_per_index` to limit initial retrieval
2. Use `merge_strategy: "interleave"` instead of `"rerank"`
3. Disable unused indexes with `enabled: false`
