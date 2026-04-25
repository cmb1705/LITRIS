"""Zotero database reader module.

Package-level re-exports are lazy so model-only imports do not initialize
change-detection or indexing dependencies.
"""

__all__ = [
    "ChangeDetector",
    "ChangeSet",
    "IndexState",
    "ZoteroDatabase",
    "Author",
    "Collection",
    "PaperMetadata",
]

_EXPORTS = {
    "ChangeDetector": "src.zotero.change_detector",
    "ChangeSet": "src.zotero.change_detector",
    "IndexState": "src.zotero.change_detector",
    "ZoteroDatabase": "src.zotero.database",
    "Author": "src.zotero.models",
    "Collection": "src.zotero.models",
    "PaperMetadata": "src.zotero.models",
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
