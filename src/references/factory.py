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
        provider: Reference manager provider.
        **kwargs: Provider-specific configuration.

    For Zotero:
        db_path: Path to zotero.sqlite file.
        storage_path: Path to Zotero storage directory.

    For BibTeX:
        bibtex_path: Path to .bib file.
        pdf_dir: Optional path to directory containing PDFs.

    For PDF Folder:
        folder_path: Path to folder containing PDF files.
        recursive: If True (default), scan subfolders recursively.
        extract_pdf_metadata: If True (default), extract metadata from PDFs.

    For Mendeley:
        db_path: Path to Mendeley SQLite database.
        storage_path: Optional root path for resolving relative file paths.

    For EndNote:
        xml_path: Path to EndNote XML export file.
        pdf_dir: Optional path to directory containing PDFs.

    For Paperpile:
        bibtex_path: Path to BibTeX file exported from Paperpile.
        pdf_dir: Optional path to directory containing PDFs.
        sync_folder: Optional path to Paperpile sync folder (Google Drive).

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

        # Create PDF folder reference database
        db = create_reference_db(
            provider="pdffolder",
            folder_path=Path("papers/"),
            recursive=True,
        )

        # Create Mendeley reference database
        db = create_reference_db(
            provider="mendeley",
            db_path=Path("~/Mendeley Desktop/mendeley.sqlite"),
        )

        # Create EndNote reference database
        db = create_reference_db(
            provider="endnote",
            xml_path=Path("library.xml"),
            pdf_dir=Path("papers/"),
        )

        # Create Paperpile reference database
        db = create_reference_db(
            provider="paperpile",
            bibtex_path=Path("paperpile_export.bib"),
            sync_folder=Path("~/Google Drive/Paperpile"),
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

    elif provider == "pdffolder":
        from src.references.pdffolder_adapter import PDFFolderReferenceDB

        folder_path = kwargs.get("folder_path")
        recursive = kwargs.get("recursive", True)
        extract_pdf_metadata = kwargs.get("extract_pdf_metadata", True)

        if not folder_path:
            raise ValueError("folder_path is required for PDF folder provider")

        return PDFFolderReferenceDB(
            folder_path=Path(folder_path),
            recursive=recursive,
            extract_pdf_metadata=extract_pdf_metadata,
        )

    elif provider == "mendeley":
        from src.references.mendeley_adapter import MendeleyReferenceDB

        db_path = kwargs.get("db_path")
        storage_path = kwargs.get("storage_path")

        if not db_path:
            raise ValueError("db_path is required for Mendeley provider")

        return MendeleyReferenceDB(
            db_path=Path(db_path),
            storage_path=Path(storage_path) if storage_path else None,
        )

    elif provider == "endnote":
        from src.references.endnote_adapter import EndNoteReferenceDB

        xml_path = kwargs.get("xml_path")
        pdf_dir = kwargs.get("pdf_dir")

        if not xml_path:
            raise ValueError("xml_path is required for EndNote provider")

        return EndNoteReferenceDB(
            xml_path=Path(xml_path),
            pdf_dir=Path(pdf_dir) if pdf_dir else None,
        )

    elif provider == "paperpile":
        from src.references.paperpile_adapter import PaperpileReferenceDB

        bibtex_path = kwargs.get("bibtex_path")
        pdf_dir = kwargs.get("pdf_dir")
        sync_folder = kwargs.get("sync_folder")

        if not bibtex_path:
            raise ValueError("bibtex_path is required for Paperpile provider")

        return PaperpileReferenceDB(
            bibtex_path=Path(bibtex_path),
            pdf_dir=Path(pdf_dir) if pdf_dir else None,
            sync_folder=Path(sync_folder) if sync_folder else None,
        )

    else:
        raise ValueError(
            f"Unsupported reference provider: {provider}. "
            f"Supported providers: {get_available_providers()}"
        )


def get_available_providers() -> list[str]:
    """Return list of available reference providers."""
    return ["zotero", "bibtex", "pdffolder", "mendeley", "endnote", "paperpile"]
