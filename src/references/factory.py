"""Factory for creating reference database instances."""

from pathlib import Path
from typing import Any

from src.references.base import BaseReferenceDB, ReferenceProvider


def create_reference_db(
    provider: ReferenceProvider,
    **kwargs: Any,
) -> BaseReferenceDB:
    """Create a reference database instance for the specified provider.

    Args:
        provider: Reference manager provider ('zotero' or 'bibtex').
        **kwargs: Provider-specific configuration.

    For Zotero:
        db_path: Path to zotero.sqlite file.
        storage_path: Path to Zotero storage directory.

    For BibTeX:
        bibtex_path: Path to .bib file.
        pdf_dir: Optional path to directory containing PDFs.

    Returns:
        Configured reference database instance.

    Raises:
        ValueError: If provider is not supported.

    Example:
        # Create Zotero reference database
        db = create_reference_db(
            provider="zotero",
            db_path=Path("~/Zotero/zotero.sqlite"),
            storage_path=Path("~/Zotero/storage"),
        )

        # Create BibTeX reference database
        db = create_reference_db(
            provider="bibtex",
            bibtex_path=Path("references.bib"),
            pdf_dir=Path("papers/"),
        )
    """
    if provider == "zotero":
        from src.references.zotero_adapter import ZoteroReferenceDB

        db_path = kwargs.get("db_path")
        storage_path = kwargs.get("storage_path")

        if not db_path:
            raise ValueError("db_path is required for Zotero provider")
        if not storage_path:
            raise ValueError("storage_path is required for Zotero provider")

        return ZoteroReferenceDB(
            db_path=Path(db_path),
            storage_path=Path(storage_path),
        )

    elif provider == "bibtex":
        from src.references.bibtex_adapter import BibTeXReferenceDB

        bibtex_path = kwargs.get("bibtex_path")
        pdf_dir = kwargs.get("pdf_dir")

        if not bibtex_path:
            raise ValueError("bibtex_path is required for BibTeX provider")

        return BibTeXReferenceDB(
            bibtex_path=Path(bibtex_path),
            pdf_dir=Path(pdf_dir) if pdf_dir else None,
        )

    else:
        raise ValueError(
            f"Unsupported reference provider: {provider}. "
            f"Supported providers: {get_available_providers()}"
        )


def get_available_providers() -> list[str]:
    """Return list of available reference providers."""
    return ["zotero", "bibtex"]
