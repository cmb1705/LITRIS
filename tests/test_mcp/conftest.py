"""Shared fixtures for MCP server tests."""

import json
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def sample_paper_data() -> dict[str, Any]:
    """Sample paper metadata for testing."""
    return {
        "paper_id": "test_paper_001",
        "title": "Graph Neural Networks for Citation Prediction",
        "authors": [
            {"first_name": "John", "last_name": "Smith"},
            {"first_name": "Jane", "last_name": "Doe"},
        ],
        "author_string": "Smith, John; Doe, Jane",
        "publication_year": 2023,
        "publication_date": "2023-06-15",
        "journal": "Nature Machine Intelligence",
        "doi": "10.1038/s42256-023-00001-1",
        "abstract": "This paper presents a novel approach to citation prediction using GNNs.",
        "collections": ["ML Papers", "Citation Analysis"],
        "tags": ["deep learning", "citation networks"],
        "item_type": "journalArticle",
        "pdf_path": "/path/to/paper.pdf",
        "zotero_key": "ABC123XY",
    }


@pytest.fixture
def sample_extraction_data() -> dict[str, Any]:
    """Sample extraction data for testing."""
    return {
        "thesis_statement": "Graph neural networks can effectively predict citation patterns.",
        "research_questions": [
            "How can GNNs model citation networks?",
            "What features are most predictive of citations?",
        ],
        "methodology": {
            "approach": "Machine learning",
            "design": "Experimental",
            "data_sources": ["Web of Science", "Semantic Scholar"],
            "analysis_methods": ["Graph neural networks", "Cross-validation"],
            "sample_size": "100,000 papers",
        },
        "key_findings": [
            {
                "finding": "GNNs outperform baseline methods by 15%",
                "evidence_type": "Quantitative",
                "significance": "Major improvement over state-of-the-art",
            }
        ],
        "conclusions": "GNNs provide a powerful framework for citation prediction.",
        "limitations": [
            "Limited to English-language papers",
            "Requires substantial computational resources",
        ],
        "future_directions": [
            "Extend to multilingual corpora",
            "Incorporate temporal dynamics",
        ],
        "key_claims": [
            {
                "claim": "GNNs capture structural dependencies in citation networks",
                "support_type": "Experimental",
                "page_reference": "p. 5",
            }
        ],
        "contribution_summary": "First application of message-passing GNNs to citation forecasting.",
        "discipline_tags": ["Computer Science", "Bibliometrics", "Machine Learning"],
        "extraction_confidence": 0.92,
    }


@pytest.fixture
def mock_index_data(
    tmp_path: Path, sample_paper_data: dict, sample_extraction_data: dict
) -> Generator[Path, None, None]:
    """Create mock index files for testing."""
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    # Create papers index
    papers_index = {sample_paper_data["paper_id"]: sample_paper_data}
    with open(index_dir / "papers_index.json", "w", encoding="utf-8") as f:
        json.dump(papers_index, f)

    # Create extractions index
    extractions_index = {sample_paper_data["paper_id"]: sample_extraction_data}
    with open(index_dir / "extractions_index.json", "w", encoding="utf-8") as f:
        json.dump(extractions_index, f)

    # Create summary
    summary = {
        "total_papers": 1,
        "total_extractions": 1,
        "papers_by_type": {"journalArticle": 1},
        "papers_by_year": {"2023": 1},
        "papers_by_collection": {"ML Papers": 1, "Citation Analysis": 1},
        "top_disciplines": {"Computer Science": 1},
    }
    with open(index_dir / "index_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f)

    yield index_dir


@pytest.fixture
def mock_search_engine(
    sample_paper_data: dict, sample_extraction_data: dict
) -> Generator[MagicMock, None, None]:
    """Create a mock SearchEngine for testing without real index."""
    mock_engine = MagicMock()

    # Mock search results
    mock_result = MagicMock()
    mock_result.paper_id = sample_paper_data["paper_id"]
    mock_result.title = sample_paper_data["title"]
    mock_result.authors = sample_paper_data["author_string"]
    mock_result.year = sample_paper_data["publication_year"]
    mock_result.collections = sample_paper_data["collections"]
    mock_result.item_type = sample_paper_data["item_type"]
    mock_result.chunk_type = "thesis"
    mock_result.matched_text = sample_extraction_data["thesis_statement"]
    mock_result.score = 0.85
    mock_result.paper_data = sample_paper_data
    mock_result.extraction_data = sample_extraction_data

    mock_engine.search.return_value = [mock_result]
    mock_engine.search_similar_papers.return_value = [mock_result]
    mock_engine.get_paper.return_value = {
        "paper": sample_paper_data,
        "extraction": sample_extraction_data,
    }
    mock_engine.get_summary.return_value = {
        "total_papers": 1,
        "total_extractions": 1,
        "papers_by_type": {"journalArticle": 1},
        "papers_by_year": {"2023": 1},
        "papers_by_collection": {"ML Papers": 1},
        "top_disciplines": {"Computer Science": 1},
        "vector_store": {"total_chunks": 10, "unique_papers": 1},
    }
    mock_engine.get_collections.return_value = ["ML Papers", "Citation Analysis"]

    yield mock_engine


@pytest.fixture
def mock_adapter(mock_search_engine: MagicMock) -> Generator[MagicMock, None, None]:
    """Create a mock LitrisAdapter for testing."""
    with patch("src.mcp.server.get_adapter") as mock_get_adapter:
        mock_adapter = MagicMock()
        mock_adapter.engine = mock_search_engine

        # Configure adapter methods to use mock engine
        mock_adapter.search.return_value = {
            "query": "test query",
            "result_count": 1,
            "results": [
                {
                    "rank": 1,
                    "score": 0.85,
                    "paper_id": "test_paper_001",
                    "title": "Graph Neural Networks for Citation Prediction",
                    "authors": "Smith, John; Doe, Jane",
                    "year": 2023,
                    "collections": ["ML Papers"],
                    "item_type": "journalArticle",
                    "chunk_type": "thesis",
                    "matched_text": "GNNs can predict citations.",
                }
            ],
        }

        mock_adapter.get_paper.return_value = {
            "paper_id": "test_paper_001",
            "found": True,
            "paper": {
                "title": "Graph Neural Networks for Citation Prediction",
                "authors": [{"first_name": "John", "last_name": "Smith"}],
            },
            "extraction": {"thesis_statement": "GNNs predict citations."},
        }

        mock_adapter.find_similar.return_value = {
            "source_paper_id": "test_paper_001",
            "source_title": "Graph Neural Networks",
            "result_count": 0,
            "similar_papers": [],
        }

        mock_adapter.get_summary.return_value = {
            "total_papers": 1,
            "total_extractions": 1,
        }

        mock_adapter.get_collections.return_value = {
            "collections": ["ML Papers"],
            "collection_counts": {"ML Papers": 1},
        }

        mock_get_adapter.return_value = mock_adapter
        yield mock_adapter
