"""Tests for reference database interfaces."""

import pytest
from pathlib import Path
from datetime import datetime

from src.references.base import BaseReferenceDB
from src.references.factory import create_reference_db, get_available_providers
from src.zotero.models import Author


class TestBaseReferenceDB:
    """Tests for BaseReferenceDB abstract class."""

    def test_base_db_is_abstract(self):
        """BaseReferenceDB cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseReferenceDB()

    def test_get_available_providers(self):
        """Should return list of providers."""
        providers = BaseReferenceDB.get_available_providers()
        assert "zotero" in providers
        assert "bibtex" in providers

    def test_create_author_helper(self):
        """Should create Author objects."""
        author = BaseReferenceDB.create_author(
            first_name="John",
            last_name="Doe",
            order=1,
            role="author",
        )
        assert isinstance(author, Author)
        assert author.first_name == "John"
        assert author.last_name == "Doe"
        assert author.full_name == "John Doe"


class TestReferenceFactory:
    """Tests for reference database factory."""

    def test_get_available_providers(self):
        """Should return list of available providers."""
        providers = get_available_providers()
        assert isinstance(providers, list)
        assert "zotero" in providers
        assert "bibtex" in providers

    def test_create_invalid_provider(self):
        """Should raise error for invalid provider."""
        with pytest.raises(ValueError, match="Unsupported reference provider"):
            create_reference_db(provider="invalid")

    def test_create_zotero_missing_paths(self):
        """Should raise error when Zotero paths missing."""
        with pytest.raises(ValueError, match="db_path is required"):
            create_reference_db(provider="zotero")

        with pytest.raises(ValueError, match="storage_path is required"):
            create_reference_db(provider="zotero", db_path="/some/path")

    def test_create_bibtex_missing_path(self):
        """Should raise error when BibTeX path missing."""
        with pytest.raises(ValueError, match="bibtex_path is required"):
            create_reference_db(provider="bibtex")


class TestZoteroAdapter:
    """Tests for Zotero reference adapter."""

    def test_provider_property(self, tmp_path):
        """Should return 'zotero' as provider."""
        from src.references.zotero_adapter import ZoteroReferenceDB

        db_path = tmp_path / "zotero.sqlite"
        db_path.touch()
        storage_path = tmp_path / "storage"
        storage_path.mkdir()

        db = ZoteroReferenceDB(db_path, storage_path)
        assert db.provider == "zotero"

    def test_source_path_property(self, tmp_path):
        """Should return database path."""
        from src.references.zotero_adapter import ZoteroReferenceDB

        db_path = tmp_path / "zotero.sqlite"
        db_path.touch()
        storage_path = tmp_path / "storage"
        storage_path.mkdir()

        db = ZoteroReferenceDB(db_path, storage_path)
        assert db.source_path == db_path


class TestBibTeXAdapter:
    """Tests for BibTeX reference adapter."""

    @pytest.fixture
    def sample_bibtex_file(self, tmp_path):
        """Create a sample BibTeX file for testing."""
        bib_content = """
@article{smith2020example,
    author = {Smith, John and Doe, Jane},
    title = {An Example Article Title},
    journal = {Journal of Examples},
    year = {2020},
    volume = {10},
    number = {2},
    pages = {100-120},
    doi = {10.1234/example.2020},
    abstract = {This is an example abstract.},
    keywords = {example, testing, bibtex}
}

@inproceedings{jones2021conference,
    author = {Jones, Alice},
    title = {Conference Paper Example},
    booktitle = {Proceedings of Example Conference},
    year = {2021},
    pages = {50-60}
}

@book{brown2019book,
    author = {Brown, Robert},
    title = {A Book Title},
    publisher = {Example Press},
    year = {2019},
    isbn = {978-0-12345-678-9}
}
"""
        bib_file = tmp_path / "references.bib"
        bib_file.write_text(bib_content, encoding="utf-8")
        return bib_file

    def test_provider_property(self, sample_bibtex_file):
        """Should return 'bibtex' as provider."""
        from src.references.bibtex_adapter import BibTeXReferenceDB

        db = BibTeXReferenceDB(sample_bibtex_file)
        assert db.provider == "bibtex"

    def test_source_path_property(self, sample_bibtex_file):
        """Should return BibTeX file path."""
        from src.references.bibtex_adapter import BibTeXReferenceDB

        db = BibTeXReferenceDB(sample_bibtex_file)
        assert db.source_path == sample_bibtex_file

    def test_get_paper_count(self, sample_bibtex_file):
        """Should count entries correctly."""
        from src.references.bibtex_adapter import BibTeXReferenceDB

        db = BibTeXReferenceDB(sample_bibtex_file)
        assert db.get_paper_count() == 3

    def test_get_all_papers(self, sample_bibtex_file):
        """Should yield all papers."""
        from src.references.bibtex_adapter import BibTeXReferenceDB

        db = BibTeXReferenceDB(sample_bibtex_file)
        papers = list(db.get_all_papers())
        assert len(papers) == 3

    def test_paper_metadata_article(self, sample_bibtex_file):
        """Should parse article metadata correctly."""
        from src.references.bibtex_adapter import BibTeXReferenceDB

        db = BibTeXReferenceDB(sample_bibtex_file)
        paper = db.get_paper_by_key("smith2020example")

        assert paper is not None
        assert paper.title == "An Example Article Title"
        assert paper.item_type == "journalArticle"
        assert paper.publication_year == 2020
        assert paper.journal == "Journal of Examples"
        assert paper.doi == "10.1234/example.2020"
        assert paper.volume == "10"
        assert paper.issue == "2"
        assert paper.pages == "100-120"
        assert "example" in paper.tags
        assert "testing" in paper.tags

    def test_paper_authors(self, sample_bibtex_file):
        """Should parse authors correctly."""
        from src.references.bibtex_adapter import BibTeXReferenceDB

        db = BibTeXReferenceDB(sample_bibtex_file)
        paper = db.get_paper_by_key("smith2020example")

        assert len(paper.authors) == 2
        assert paper.authors[0].last_name == "Smith"
        assert paper.authors[0].first_name == "John"
        assert paper.authors[1].last_name == "Doe"
        assert paper.authors[1].first_name == "Jane"

    def test_paper_type_mapping(self, sample_bibtex_file):
        """Should map BibTeX types to item types."""
        from src.references.bibtex_adapter import BibTeXReferenceDB

        db = BibTeXReferenceDB(sample_bibtex_file)

        article = db.get_paper_by_key("smith2020example")
        assert article.item_type == "journalArticle"

        conference = db.get_paper_by_key("jones2021conference")
        assert conference.item_type == "conferencePaper"

        book = db.get_paper_by_key("brown2019book")
        assert book.item_type == "book"

    def test_get_paper_by_key_not_found(self, sample_bibtex_file):
        """Should return None for unknown key."""
        from src.references.bibtex_adapter import BibTeXReferenceDB

        db = BibTeXReferenceDB(sample_bibtex_file)
        paper = db.get_paper_by_key("nonexistent")
        assert paper is None

    def test_file_not_found(self, tmp_path):
        """Should raise error for missing file."""
        from src.references.bibtex_adapter import BibTeXReferenceDB

        db = BibTeXReferenceDB(tmp_path / "nonexistent.bib")
        with pytest.raises(FileNotFoundError):
            db.get_paper_count()

    def test_pdf_discovery(self, sample_bibtex_file, tmp_path):
        """Should find PDFs in pdf_dir."""
        from src.references.bibtex_adapter import BibTeXReferenceDB

        # Create PDF directory with matching file
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        pdf_file = pdf_dir / "smith2020example.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        db = BibTeXReferenceDB(sample_bibtex_file, pdf_dir=pdf_dir)
        paper = db.get_paper_by_key("smith2020example")

        assert paper.pdf_path == pdf_file

    def test_iterate_papers_with_limit(self, sample_bibtex_file):
        """Should respect limit parameter."""
        from src.references.bibtex_adapter import BibTeXReferenceDB

        db = BibTeXReferenceDB(sample_bibtex_file)
        papers = list(db.iterate_papers(limit=2))
        assert len(papers) == 2

    def test_filter_papers_by_year(self, sample_bibtex_file):
        """Should filter by publication year."""
        from src.references.bibtex_adapter import BibTeXReferenceDB

        db = BibTeXReferenceDB(sample_bibtex_file)

        # Papers from 2020 or later
        papers = list(db.filter_papers(year_min=2020, has_pdf=False))
        assert len(papers) == 2  # smith2020, jones2021

        # Papers before 2020
        papers = list(db.filter_papers(year_max=2019, has_pdf=False))
        assert len(papers) == 1  # brown2019


class TestBibTeXAuthorParsing:
    """Tests for BibTeX author string parsing."""

    @pytest.fixture
    def bibtex_db(self, tmp_path):
        """Create minimal BibTeX database for testing."""
        bib_file = tmp_path / "test.bib"
        bib_file.write_text("@article{test, title={Test}}", encoding="utf-8")

        from src.references.bibtex_adapter import BibTeXReferenceDB
        return BibTeXReferenceDB(bib_file)

    def test_single_author_last_first(self, bibtex_db):
        """Parse 'Last, First' format."""
        authors = bibtex_db._parse_authors("Smith, John")
        assert len(authors) == 1
        assert authors[0].last_name == "Smith"
        assert authors[0].first_name == "John"

    def test_single_author_first_last(self, bibtex_db):
        """Parse 'First Last' format."""
        authors = bibtex_db._parse_authors("John Smith")
        assert len(authors) == 1
        assert authors[0].last_name == "Smith"
        assert authors[0].first_name == "John"

    def test_multiple_authors(self, bibtex_db):
        """Parse multiple authors with 'and'."""
        authors = bibtex_db._parse_authors("Smith, John and Doe, Jane and Brown, Bob")
        assert len(authors) == 3
        assert authors[0].order == 1
        assert authors[1].order == 2
        assert authors[2].order == 3

    def test_single_name(self, bibtex_db):
        """Parse single-word name."""
        authors = bibtex_db._parse_authors("Madonna")
        assert len(authors) == 1
        assert authors[0].last_name == "Madonna"
        assert authors[0].first_name == ""


class TestBibTeXLaTeXCleaning:
    """Tests for LaTeX cleanup in BibTeX."""

    @pytest.fixture
    def bibtex_db(self, tmp_path):
        """Create minimal BibTeX database for testing."""
        bib_file = tmp_path / "test.bib"
        bib_file.write_text("@article{test, title={Test}}", encoding="utf-8")

        from src.references.bibtex_adapter import BibTeXReferenceDB
        return BibTeXReferenceDB(bib_file)

    def test_clean_textbf(self, bibtex_db):
        """Should remove \\textbf."""
        result = bibtex_db._clean_latex("This is \\textbf{bold} text")
        assert result == "This is bold text"

    def test_clean_escaped_chars(self, bibtex_db):
        """Should unescape special characters."""
        result = bibtex_db._clean_latex("R\\&D costs are 10\\%")
        assert result == "R&D costs are 10%"

    def test_clean_multiple(self, bibtex_db):
        """Should handle multiple LaTeX commands."""
        result = bibtex_db._clean_latex("\\textit{italic} and \\textbf{bold}")
        assert result == "italic and bold"
