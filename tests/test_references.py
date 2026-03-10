"""Tests for reference database interfaces."""

from pathlib import Path

import pytest

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


class TestPDFFolderAdapter:
    """Tests for PDF folder reference adapter."""

    @pytest.fixture
    def sample_pdf_folder(self, tmp_path):
        """Create a sample folder with PDF files."""
        # Create test PDFs (minimal valid PDF-like files)
        (tmp_path / "Smith - 2020 - Deep Learning.pdf").write_bytes(b"%PDF-1.4\n%test1\n")
        (tmp_path / "Doe_2021_Neural_Networks.pdf").write_bytes(b"%PDF-1.4\n%test2\n")

        # Create subfolder with PDF
        subfolder = tmp_path / "conference"
        subfolder.mkdir()
        (subfolder / "2022 - Brown - Transformers.pdf").write_bytes(b"%PDF-1.4\n%test3\n")

        return tmp_path

    def test_provider_property(self, sample_pdf_folder):
        """Should return 'pdffolder' as provider."""
        from src.references.pdffolder_adapter import PDFFolderReferenceDB

        db = PDFFolderReferenceDB(sample_pdf_folder)
        assert db.provider == "pdffolder"

    def test_source_path_property(self, sample_pdf_folder):
        """Should return folder path."""
        from src.references.pdffolder_adapter import PDFFolderReferenceDB

        db = PDFFolderReferenceDB(sample_pdf_folder)
        assert db.source_path == sample_pdf_folder

    def test_get_paper_count_recursive(self, sample_pdf_folder):
        """Should count PDFs recursively."""
        from src.references.pdffolder_adapter import PDFFolderReferenceDB

        db = PDFFolderReferenceDB(sample_pdf_folder, recursive=True)
        assert db.get_paper_count() == 3

    def test_get_paper_count_non_recursive(self, sample_pdf_folder):
        """Should count only top-level PDFs."""
        from src.references.pdffolder_adapter import PDFFolderReferenceDB

        db = PDFFolderReferenceDB(sample_pdf_folder, recursive=False)
        assert db.get_paper_count() == 2

    def test_get_all_papers(self, sample_pdf_folder):
        """Should yield all papers."""
        from src.references.pdffolder_adapter import PDFFolderReferenceDB

        db = PDFFolderReferenceDB(sample_pdf_folder)
        papers = list(db.get_all_papers())
        assert len(papers) == 3

    def test_paper_metadata_from_filename(self, sample_pdf_folder):
        """Should parse metadata from filename."""
        from src.references.pdffolder_adapter import PDFFolderReferenceDB

        db = PDFFolderReferenceDB(sample_pdf_folder, extract_pdf_metadata=False)
        paper = db.get_paper_by_key("Smith - 2020 - Deep Learning")

        assert paper is not None
        assert paper.title == "Deep Learning"
        assert paper.publication_year == 2020
        assert len(paper.authors) == 1
        assert paper.authors[0].last_name == "Smith"

    def test_subfolder_as_collection(self, sample_pdf_folder):
        """Should use subfolder name as collection."""
        from src.references.pdffolder_adapter import PDFFolderReferenceDB

        db = PDFFolderReferenceDB(sample_pdf_folder)
        papers = list(db.get_all_papers())

        # Find paper in subfolder
        subfolder_papers = [p for p in papers if p.collections]
        assert len(subfolder_papers) == 1
        assert "conference" in subfolder_papers[0].collections[0]

    def test_get_paper_by_key_not_found(self, sample_pdf_folder):
        """Should return None for unknown key."""
        from src.references.pdffolder_adapter import PDFFolderReferenceDB

        db = PDFFolderReferenceDB(sample_pdf_folder)
        paper = db.get_paper_by_key("nonexistent")
        assert paper is None

    def test_folder_not_found(self, tmp_path):
        """Should raise error for missing folder."""
        from src.references.pdffolder_adapter import PDFFolderReferenceDB

        db = PDFFolderReferenceDB(tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError):
            db.get_paper_count()


class TestPDFFolderFilenameParsing:
    """Tests for PDF folder filename parsing."""

    @pytest.fixture
    def pdffolder_db(self, tmp_path):
        """Create minimal PDF folder database for testing."""
        from src.references.pdffolder_adapter import PDFFolderReferenceDB
        return PDFFolderReferenceDB(tmp_path)

    def test_author_dash_year_dash_title(self, pdffolder_db):
        """Parse 'Author - Year - Title.pdf' format."""
        result = pdffolder_db._parse_filename(Path("Smith - 2020 - Machine Learning.pdf"))
        assert result["authors"] == "Smith"
        assert result["year"] == "2020"
        assert result["title"] == "Machine Learning"

    def test_author_underscore_year_underscore_title(self, pdffolder_db):
        """Parse 'Author_Year_Title.pdf' format."""
        result = pdffolder_db._parse_filename(Path("Smith_2020_Machine_Learning.pdf"))
        assert result["authors"] == "Smith"
        assert result["year"] == "2020"
        assert result["title"] == "Machine Learning"

    def test_year_underscore_author_underscore_title(self, pdffolder_db):
        """Parse 'Year_Author_Title.pdf' format."""
        result = pdffolder_db._parse_filename(Path("2020_Smith_Machine_Learning.pdf"))
        assert result["authors"] == "Smith"
        assert result["year"] == "2020"
        assert result["title"] == "Machine Learning"

    def test_title_only_fallback(self, pdffolder_db):
        """Parse filename with just title."""
        result = pdffolder_db._parse_filename(Path("Just A Title.pdf"))
        assert result["title"] == "Just A Title"
        assert result["authors"] is None
        assert result["year"] is None


class TestPDFFolderAuthorParsing:
    """Tests for PDF folder author string parsing."""

    @pytest.fixture
    def pdffolder_db(self, tmp_path):
        """Create minimal PDF folder database for testing."""
        from src.references.pdffolder_adapter import PDFFolderReferenceDB
        return PDFFolderReferenceDB(tmp_path)

    def test_last_comma_first(self, pdffolder_db):
        """Parse 'Last, First' format."""
        authors = pdffolder_db._parse_authors("Smith, John")
        assert len(authors) == 1
        assert authors[0].last_name == "Smith"
        assert authors[0].first_name == "John"

    def test_first_last(self, pdffolder_db):
        """Parse 'First Last' format."""
        authors = pdffolder_db._parse_authors("John Smith")
        assert len(authors) == 1
        assert authors[0].last_name == "Smith"
        assert authors[0].first_name == "John"

    def test_multiple_authors_and(self, pdffolder_db):
        """Parse multiple authors with 'and'."""
        authors = pdffolder_db._parse_authors("Smith, John and Doe, Jane")
        assert len(authors) == 2
        assert authors[0].last_name == "Smith"
        assert authors[1].last_name == "Doe"

    def test_multiple_authors_semicolon(self, pdffolder_db):
        """Parse multiple authors with semicolon."""
        authors = pdffolder_db._parse_authors("John Smith; Jane Doe")
        assert len(authors) == 2

    def test_et_al_removal(self, pdffolder_db):
        """Should remove 'et al.' from author string."""
        authors = pdffolder_db._parse_authors("Smith et al.")
        assert len(authors) == 1
        assert authors[0].last_name == "Smith"


class TestReferenceFactoryPDFFolder:
    """Tests for reference factory with PDF folder provider."""

    def test_get_available_providers_includes_pdffolder(self):
        """Should include pdffolder in providers."""
        providers = get_available_providers()
        assert "pdffolder" in providers

    def test_create_pdffolder_missing_path(self):
        """Should raise error when folder_path missing."""
        with pytest.raises(ValueError, match="folder_path is required"):
            create_reference_db(provider="pdffolder")

    def test_create_pdffolder_success(self, tmp_path):
        """Should create PDF folder database."""
        (tmp_path / "test.pdf").write_bytes(b"%PDF-1.4\n")

        db = create_reference_db(
            provider="pdffolder",
            folder_path=tmp_path,
            recursive=False,
        )
        assert db.provider == "pdffolder"
        assert db.get_paper_count() == 1
