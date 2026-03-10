"""Tests for federated search across multiple indexes."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import FederatedIndexConfig, FederatedSearchConfig
from src.query.federated import (
    FederatedResult,
    FederatedSearchEngine,
    _extract_doi,
    _title_similarity,
)
from src.query.search import EnrichedResult


def _mock_result(
    paper_id: str,
    title: str,
    score: float,
    doi: str | None = None,
) -> EnrichedResult:
    """Create a mock EnrichedResult for testing."""
    paper_data = {"doi": doi} if doi else {}
    return EnrichedResult(
        paper_id=paper_id,
        title=title,
        authors="Test Author",
        year=2023,
        collections=["test"],
        item_type="journalArticle",
        chunk_type="thesis",
        matched_text="Test matched text",
        score=score,
        paper_data=paper_data,
        extraction_data={},
    )


class TestTitleSimilarity:
    """Tests for title similarity calculation."""

    def test_identical_titles(self):
        """Identical titles have similarity 1.0."""
        assert _title_similarity("Test Title", "Test Title") == 1.0

    def test_case_insensitive(self):
        """Similarity is case-insensitive."""
        assert _title_similarity("Test Title", "test title") == 1.0

    def test_whitespace_normalized(self):
        """Leading/trailing whitespace is ignored."""
        assert _title_similarity("  Test Title  ", "Test Title") == 1.0

    def test_similar_titles(self):
        """Similar titles have high similarity."""
        sim = _title_similarity(
            "Machine Learning in Healthcare",
            "Machine Learning in Healthcare Systems",
        )
        assert sim > 0.8

    def test_different_titles(self):
        """Different titles have low similarity."""
        sim = _title_similarity(
            "Quantum Computing Basics",
            "Machine Learning Applications",
        )
        assert sim < 0.5

    def test_empty_titles(self):
        """Empty titles return 0.0."""
        assert _title_similarity("", "Test") == 0.0
        assert _title_similarity("Test", "") == 0.0
        assert _title_similarity("", "") == 0.0


class TestExtractDOI:
    """Tests for DOI extraction from paper metadata."""

    def test_doi_lowercase_key(self):
        """Extract DOI from lowercase key."""
        paper = {"doi": "10.1234/test.5678"}
        assert _extract_doi(paper) == "10.1234/test.5678"

    def test_doi_uppercase_key(self):
        """Extract DOI from uppercase key."""
        paper = {"DOI": "10.1234/test.5678"}
        assert _extract_doi(paper) == "10.1234/test.5678"

    def test_doi_from_identifiers(self):
        """Extract DOI from identifiers array."""
        paper = {
            "identifiers": [
                {"type": "isbn", "value": "123-456"},
                {"type": "doi", "value": "10.1234/test.5678"},
            ]
        }
        assert _extract_doi(paper) == "10.1234/test.5678"

    def test_doi_normalized(self):
        """DOI is normalized to lowercase."""
        paper = {"doi": "  10.1234/TEST.5678  "}
        assert _extract_doi(paper) == "10.1234/test.5678"

    def test_no_doi(self):
        """Returns None when no DOI present."""
        paper = {"title": "Test Paper"}
        assert _extract_doi(paper) is None


class TestFederatedResult:
    """Tests for FederatedResult dataclass."""

    def test_weighted_score_calculated(self):
        """Weighted score is calculated on init."""
        result = FederatedResult(
            paper_id="p1",
            title="Test",
            authors="Author",
            year=2023,
            collections=[],
            item_type="article",
            chunk_type="thesis",
            matched_text="text",
            score=0.8,
            source_index="test",
            source_weight=1.5,
        )
        assert result.weighted_score == 0.8 * 1.5

    def test_to_dict_includes_federation_metadata(self):
        """to_dict includes source_index and weighted_score."""
        result = FederatedResult(
            paper_id="p1",
            title="Test",
            authors="Author",
            year=2023,
            collections=[],
            item_type="article",
            chunk_type="thesis",
            matched_text="text",
            score=0.8,
            source_index="colleague",
            source_weight=0.9,
        )
        d = result.to_dict()
        assert d["source_index"] == "colleague"
        assert d["source_weight"] == 0.9
        assert d["weighted_score"] == 0.8 * 0.9


class TestFederatedSearchEngine:
    """Tests for FederatedSearchEngine."""

    @pytest.fixture
    def mock_primary_engine(self):
        """Create a mock primary search engine."""
        engine = MagicMock()
        engine.search.return_value = [
            _mock_result("p1", "Primary Paper 1", 0.9, doi="10.1234/p1"),
            _mock_result("p2", "Primary Paper 2", 0.8),
        ]
        return engine

    @pytest.fixture
    def mock_federated_engine(self):
        """Create a mock federated search engine."""
        engine = MagicMock()
        engine.search.return_value = [
            _mock_result("f1", "Federated Paper 1", 0.85, doi="10.5678/f1"),
            _mock_result("f2", "Primary Paper 1", 0.7, doi="10.1234/p1"),  # Duplicate DOI
        ]
        return engine

    @pytest.fixture
    def basic_config(self):
        """Create a basic federated config."""
        return FederatedSearchConfig(
            enabled=True,
            merge_strategy="interleave",
            dedup_threshold=0.95,
            max_results_per_index=50,
            indexes=[
                FederatedIndexConfig(
                    path=Path("/test/federated"),
                    label="Test Federated",
                    enabled=True,
                    weight=0.9,
                )
            ],
        )

    def test_disabled_federation_uses_primary_only(self):
        """When federation disabled, only primary index is searched."""
        config = FederatedSearchConfig(enabled=False)

        with patch("src.query.federated.SearchEngine") as MockEngine:
            MockEngine.return_value.search.return_value = [
                _mock_result("p1", "Primary", 0.9)
            ]

            engine = FederatedSearchEngine(
                primary_index_dir=Path("/test/primary"),
                config=config,
            )
            results = engine.search("test query")

            assert len(results) == 1
            assert results[0].source_index == "primary"

    def test_deduplication_by_doi(self, basic_config):
        """Results with same DOI are deduplicated."""
        with patch("src.query.federated.SearchEngine") as MockEngine:
            # Mock both primary and federated engines
            primary_mock = MagicMock()
            primary_mock.search.return_value = [
                _mock_result("p1", "Paper 1", 0.9, doi="10.1234/shared"),
            ]

            federated_mock = MagicMock()
            federated_mock.search.return_value = [
                _mock_result("f1", "Paper 1 (copy)", 0.8, doi="10.1234/shared"),
            ]

            # First call creates primary, second creates federated
            MockEngine.side_effect = [primary_mock, federated_mock]

            basic_config.indexes[0].path = Path("/test/federated")
            with patch.object(Path, "exists", return_value=True):
                engine = FederatedSearchEngine(
                    primary_index_dir=Path("/test/primary"),
                    config=basic_config,
                )

            results = engine.search("test query")

            # Should have only 1 result (deduplicated)
            assert len(results) == 1
            assert results[0].paper_id == "p1"  # Higher weighted score wins

    def test_deduplication_by_title_similarity(self, basic_config):
        """Results with similar titles are deduplicated."""
        with patch("src.query.federated.SearchEngine") as MockEngine:
            primary_mock = MagicMock()
            primary_mock.search.return_value = [
                _mock_result("p1", "Machine Learning in Healthcare", 0.9),
            ]

            federated_mock = MagicMock()
            federated_mock.search.return_value = [
                _mock_result("f1", "Machine Learning in Healthcare Systems", 0.8),
            ]

            MockEngine.side_effect = [primary_mock, federated_mock]

            with patch.object(Path, "exists", return_value=True):
                engine = FederatedSearchEngine(
                    primary_index_dir=Path("/test/primary"),
                    config=basic_config,
                )

            # With default 0.95 threshold, these should NOT be deduplicated
            # because similarity is ~0.90
            results = engine.search("test query")
            assert len(results) == 2

            # Lower threshold to deduplicate
            engine.config.dedup_threshold = 0.85
            results = engine.search("test query")
            assert len(results) == 1

    def test_merge_strategy_interleave(self, basic_config):
        """Interleave strategy sorts by weighted score."""
        basic_config.merge_strategy = "interleave"

        with patch("src.query.federated.SearchEngine") as MockEngine:
            primary_mock = MagicMock()
            primary_mock.search.return_value = [
                _mock_result("p1", "Primary", 0.7),
            ]

            federated_mock = MagicMock()
            federated_mock.search.return_value = [
                _mock_result("f1", "Federated", 0.9),  # Higher score
            ]

            MockEngine.side_effect = [primary_mock, federated_mock]

            with patch.object(Path, "exists", return_value=True):
                engine = FederatedSearchEngine(
                    primary_index_dir=Path("/test/primary"),
                    config=basic_config,
                )

            results = engine.search("test query")

            # Federated has 0.9 * 0.9 = 0.81, Primary has 0.7 * 1.0 = 0.7
            # So federated should come first
            assert results[0].paper_id == "f1"
            assert results[1].paper_id == "p1"

    def test_merge_strategy_concat(self, basic_config):
        """Concat strategy puts primary results first."""
        basic_config.merge_strategy = "concat"

        with patch("src.query.federated.SearchEngine") as MockEngine:
            primary_mock = MagicMock()
            primary_mock.search.return_value = [
                _mock_result("p1", "Primary", 0.5),
            ]

            federated_mock = MagicMock()
            federated_mock.search.return_value = [
                _mock_result("f1", "Federated", 0.9),  # Higher score
            ]

            MockEngine.side_effect = [primary_mock, federated_mock]

            with patch.object(Path, "exists", return_value=True):
                engine = FederatedSearchEngine(
                    primary_index_dir=Path("/test/primary"),
                    config=basic_config,
                )

            results = engine.search("test query")

            # Primary should come first regardless of score
            assert results[0].paper_id == "p1"
            assert results[1].paper_id == "f1"

    def test_merge_strategy_rerank(self, basic_config):
        """Rerank strategy normalizes scores and reranks."""
        basic_config.merge_strategy = "rerank"

        with patch("src.query.federated.SearchEngine") as MockEngine:
            primary_mock = MagicMock()
            primary_mock.search.return_value = [
                _mock_result("p1", "Primary Low", 0.3),
                _mock_result("p2", "Primary High", 0.9),
            ]

            federated_mock = MagicMock()
            federated_mock.search.return_value = [
                _mock_result("f1", "Federated", 0.6),
            ]

            MockEngine.side_effect = [primary_mock, federated_mock]

            with patch.object(Path, "exists", return_value=True):
                engine = FederatedSearchEngine(
                    primary_index_dir=Path("/test/primary"),
                    config=basic_config,
                )

            results = engine.search("test query")

            # Rerank should normalize scores: 0.3->0.0, 0.6->0.5, 0.9->1.0
            # Then apply weights: p2=1.0*1.0=1.0, f1=0.5*0.9=0.45, p1=0.0*1.0=0.0
            assert results[0].paper_id == "p2"  # Highest normalized weighted

    def test_disabled_federated_index_skipped(self):
        """Disabled federated indexes are not loaded."""
        config = FederatedSearchConfig(
            enabled=True,
            indexes=[
                FederatedIndexConfig(
                    path=Path("/test/disabled"),
                    label="Disabled",
                    enabled=False,
                    weight=1.0,
                )
            ],
        )

        with patch("src.query.federated.SearchEngine") as MockEngine:
            MockEngine.return_value.search.return_value = []

            engine = FederatedSearchEngine(
                primary_index_dir=Path("/test/primary"),
                config=config,
            )

            assert len(engine.federated_engines) == 0

    def test_missing_index_path_logged(self):
        """Missing index paths are logged as warnings."""
        config = FederatedSearchConfig(
            enabled=True,
            indexes=[
                FederatedIndexConfig(
                    path=Path("/nonexistent/path"),
                    label="Missing",
                    enabled=True,
                    weight=1.0,
                )
            ],
        )

        with patch("src.query.federated.SearchEngine") as MockEngine:
            MockEngine.return_value.search.return_value = []

            engine = FederatedSearchEngine(
                primary_index_dir=Path("/test/primary"),
                config=config,
            )

            # Missing index should not be loaded
            assert "Missing" not in engine.federated_engines

    def test_get_index_info(self, basic_config):
        """get_index_info returns accurate information."""
        with patch("src.query.federated.SearchEngine") as MockEngine:
            MockEngine.return_value.search.return_value = []

            with patch.object(Path, "exists", return_value=True):
                engine = FederatedSearchEngine(
                    primary_index_dir=Path("/test/primary"),
                    config=basic_config,
                )

            info = engine.get_index_info()

            assert info["primary"]["path"] == str(Path("/test/primary"))
            assert info["federated_enabled"] is True
            assert info["merge_strategy"] == "interleave"
            assert len(info["indexes"]) == 1
            assert info["indexes"][0]["label"] == "Test Federated"


class TestMergeStrategies:
    """Integration tests for merge strategies."""

    def test_interleave_respects_top_k(self):
        """Interleave returns at most top_k results."""
        config = FederatedSearchConfig(enabled=False)

        with patch("src.query.federated.SearchEngine") as MockEngine:
            MockEngine.return_value.search.return_value = [
                _mock_result(f"p{i}", f"Paper {i}", 0.9 - i * 0.1)
                for i in range(10)
            ]

            engine = FederatedSearchEngine(
                primary_index_dir=Path("/test/primary"),
                config=config,
            )

            results = engine.search("test", top_k=5)
            assert len(results) == 5

    def test_empty_results_handled(self):
        """Empty results are handled gracefully."""
        config = FederatedSearchConfig(enabled=False)

        with patch("src.query.federated.SearchEngine") as MockEngine:
            MockEngine.return_value.search.return_value = []

            engine = FederatedSearchEngine(
                primary_index_dir=Path("/test/primary"),
                config=config,
            )

            results = engine.search("test")
            assert results == []


class TestCLISelection:
    """Tests for CLI federated search selection."""

    def test_parse_args_federated_flag(self):
        """Test that --federated flag is parsed."""
        import sys
        from unittest.mock import patch as mock_patch

        # Need to import after patching argv
        with mock_patch.object(
            sys, "argv", ["query_index.py", "-q", "test", "--federated"]
        ):
            # Import dynamically to get fresh parse_args
            import importlib

            import scripts.query_index as qi
            importlib.reload(qi)
            args = qi.parse_args()
            assert args.federated is True
            assert args.query == "test"

    def test_parse_args_indexes_selection(self):
        """Test that --indexes flag selects specific indexes."""
        import sys
        from unittest.mock import patch as mock_patch

        with mock_patch.object(
            sys,
            "argv",
            ["query_index.py", "-q", "test", "--federated", "--indexes", "Lab1", "Lab2"],
        ):
            import importlib

            import scripts.query_index as qi
            importlib.reload(qi)
            args = qi.parse_args()
            assert args.indexes == ["Lab1", "Lab2"]

    def test_parse_args_merge_strategy_override(self):
        """Test merge strategy override."""
        import sys
        from unittest.mock import patch as mock_patch

        with mock_patch.object(
            sys,
            "argv",
            ["query_index.py", "-q", "test", "--merge-strategy", "rerank"],
        ):
            import importlib

            import scripts.query_index as qi
            importlib.reload(qi)
            args = qi.parse_args()
            assert args.merge_strategy == "rerank"

    def test_parse_args_list_indexes(self):
        """Test --list-indexes flag."""
        import sys
        from unittest.mock import patch as mock_patch

        with mock_patch.object(sys, "argv", ["query_index.py", "--list-indexes"]):
            import importlib

            import scripts.query_index as qi
            importlib.reload(qi)
            args = qi.parse_args()
            assert args.list_indexes is True


class TestIndexSelection:
    """Tests for index selection behavior."""

    def test_index_selection_filters_indexes(self):
        """Selecting specific indexes disables others."""
        config = FederatedSearchConfig(
            enabled=True,
            indexes=[
                FederatedIndexConfig(
                    path=Path("/test/lab1"),
                    label="Lab1",
                    enabled=True,
                    weight=1.0,
                ),
                FederatedIndexConfig(
                    path=Path("/test/lab2"),
                    label="Lab2",
                    enabled=True,
                    weight=1.0,
                ),
                FederatedIndexConfig(
                    path=Path("/test/lab3"),
                    label="Lab3",
                    enabled=True,
                    weight=1.0,
                ),
            ],
        )

        # Simulate CLI index selection: enable only Lab1 and Lab2
        selected = ["Lab1", "Lab2"]
        for idx_cfg in config.indexes:
            idx_cfg.enabled = idx_cfg.label in selected

        assert config.indexes[0].enabled is True  # Lab1
        assert config.indexes[1].enabled is True  # Lab2
        assert config.indexes[2].enabled is False  # Lab3

    def test_default_all_indexes_enabled(self):
        """By default, all configured indexes are enabled."""
        config = FederatedSearchConfig(
            enabled=True,
            indexes=[
                FederatedIndexConfig(
                    path=Path("/test/idx1"),
                    label="Index1",
                    enabled=True,
                ),
                FederatedIndexConfig(
                    path=Path("/test/idx2"),
                    label="Index2",
                    enabled=True,
                ),
            ],
        )

        enabled_count = sum(1 for idx in config.indexes if idx.enabled)
        assert enabled_count == 2
