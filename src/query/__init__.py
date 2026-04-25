"""Search and query interface module.

The public package exports are lazy to keep import-only paths from loading
ChromaDB and vector-search dependencies before a search engine is created.
"""

__all__ = [
    "EnrichedResult",
    "FederatedResult",
    "FederatedSearchEngine",
    "OutputFormat",
    "SearchEngine",
    "format_brief",
    "format_json",
    "format_markdown",
    "format_paper_detail",
    "format_results",
    "format_summary",
    "save_results",
]

_EXPORTS = {
    "EnrichedResult": "src.query.search",
    "SearchEngine": "src.query.search",
    "FederatedResult": "src.query.federated",
    "FederatedSearchEngine": "src.query.federated",
    "OutputFormat": "src.query.retrieval",
    "format_brief": "src.query.retrieval",
    "format_json": "src.query.retrieval",
    "format_markdown": "src.query.retrieval",
    "format_paper_detail": "src.query.retrieval",
    "format_results": "src.query.retrieval",
    "format_summary": "src.query.retrieval",
    "save_results": "src.query.retrieval",
}


def __getattr__(name: str) -> object:
    """Resolve legacy package exports on first access."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
