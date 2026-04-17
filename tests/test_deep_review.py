"""Tests for the Deep Review literature synthesis pipeline."""

from unittest.mock import MagicMock, patch

import pytest

from src.query.deep_review import (
    DeepReviewResult,
    PaperReading,
    QAResult,
    deep_review,
    read_papers,
    synthesize,
    verify_citations,
)


def _make_reading(paper_id: str = "p1", title: str = "Paper One") -> PaperReading:
    """Create a test PaperReading."""
    return PaperReading(
        paper_id=paper_id,
        title=title,
        authors="Smith, J.",
        year=2024,
        thesis="Test thesis statement.",
        methodology="Mixed methods approach.",
        key_findings=["Finding 1", "Finding 2"],
        conclusions="Test conclusions.",
        limitations=["Limitation 1"],
        disciplines=["network science"],
    )


class TestPaperReading:
    """Tests for PaperReading dataclass."""

    def test_to_text_includes_all_fields(self):
        reading = _make_reading()
        text = reading.to_text()

        assert "Paper One" in text
        assert "Smith, J." in text
        assert "2024" in text
        assert "Test thesis" in text
        assert "Mixed methods" in text
        assert "Finding 1" in text
        assert "Limitation 1" in text
        assert "network science" in text

    def test_to_text_handles_no_year(self):
        reading = _make_reading()
        reading.year = None
        text = reading.to_text()

        assert "n.d." in text


class TestDeepReviewResult:
    """Tests for DeepReviewResult."""

    def test_to_markdown_basic(self):
        result = DeepReviewResult(
            topic="network analysis",
            papers_discovered=50,
            papers_used=10,
            review_text="## Introduction\n\nThis is the review.",
            qa_result=QAResult(verified=True, issues=[], uncited_papers=[], citation_count=15),
            paper_readings=[_make_reading()],
        )

        md = result.to_markdown()

        assert "# Deep Review: network analysis" in md
        assert "Papers discovered:** 50" in md
        assert "Papers synthesized:** 10" in md
        assert "## Introduction" in md
        assert "Citation Verification: Passed" in md
        assert "Citations found:** 15" in md
        assert "Paper One" in md

    def test_to_markdown_with_qa_issues(self):
        result = DeepReviewResult(
            topic="test",
            papers_discovered=5,
            papers_used=3,
            review_text="Review text.",
            qa_result=QAResult(
                verified=False,
                issues=["Hallucinated citation: (Doe, 2025)"],
                uncited_papers=["p2"],
                citation_count=5,
            ),
            paper_readings=[_make_reading()],
        )

        md = result.to_markdown()

        assert "Issues Found" in md
        assert "Hallucinated citation" in md
        assert "p2" in md

    def test_to_markdown_without_qa(self):
        result = DeepReviewResult(
            topic="test",
            papers_discovered=5,
            papers_used=3,
            review_text="Review text.",
            qa_result=None,
            paper_readings=[_make_reading()],
        )

        md = result.to_markdown()

        assert "Citation Verification" not in md


class TestReadPapers:
    """Tests for Phase 2: paper reading."""

    def test_reads_papers_with_extraction(self):
        mock_result = MagicMock()
        mock_result.paper_id = "p1"
        mock_result.title = "Paper One"
        mock_result.authors = "Smith"
        mock_result.year = 2024

        mock_adapter = MagicMock()
        mock_adapter.get_paper.return_value = {
            "found": True,
            "paper": {"title": "Paper One"},
            "extraction": {
                "q02_thesis": "Main thesis.",
                "q07_methods": "Quantitative",
                "q03_key_claims": "Result A",
                "q04_evidence": "Strong evidence for A",
                "q19_implications": "Summary.",
                "q05_limitations": "Small sample",
                "q17_field": "ML",
            },
        }

        readings = read_papers([mock_result], mock_adapter)

        assert len(readings) == 1
        assert readings[0].thesis == "Main thesis."
        assert readings[0].methodology == "Quantitative"
        # key_findings comes from q03_key_claims and q04_evidence
        assert "Result A" in readings[0].key_findings
        assert "Strong evidence for A" in readings[0].key_findings

    def test_skips_not_found_papers(self):
        mock_result = MagicMock()
        mock_result.paper_id = "missing"

        mock_adapter = MagicMock()
        mock_adapter.get_paper.return_value = {"found": False}

        readings = read_papers([mock_result], mock_adapter)

        assert len(readings) == 0

    def test_handles_empty_extraction(self):
        mock_result = MagicMock()
        mock_result.paper_id = "p1"
        mock_result.title = "Paper"
        mock_result.authors = "Auth"
        mock_result.year = 2024

        mock_adapter = MagicMock()
        mock_adapter.get_paper.return_value = {
            "found": True,
            "extraction": {
                "q07_methods": "Simple method",
            },
        }

        readings = read_papers([mock_result], mock_adapter)

        assert readings[0].methodology == "Simple method"
        assert readings[0].thesis == ""


class TestSynthesize:
    """Tests for Phase 3: LLM synthesis."""

    @patch("src.analysis.llm_factory.create_llm_client")
    def test_generates_review(self, mock_factory):
        mock_client = MagicMock()
        mock_client._call_api.return_value = (
            "## Literature Review\n\nThis is a synthesis.",
            5000,
            3000,
        )
        mock_factory.return_value = mock_client

        readings = [_make_reading(), _make_reading("p2", "Paper Two")]
        result = synthesize("network analysis", readings)

        assert "Literature Review" in result
        # Verify the prompt was called with paper content
        call_args = mock_client._call_api.call_args[0][0]
        assert "Paper One" in call_args
        assert "Paper Two" in call_args
        assert "network analysis" in call_args

    @patch("src.analysis.llm_factory.create_llm_client")
    def test_raises_on_failure(self, mock_factory):
        mock_client = MagicMock()
        mock_client._call_api.side_effect = Exception("API timeout")
        mock_factory.return_value = mock_client

        with pytest.raises(Exception, match="API timeout"):
            synthesize("test", [_make_reading()])


class TestVerifyCitations:
    """Tests for Phase 4: QA verification."""

    @patch("src.analysis.llm_factory.create_llm_client")
    def test_verified_result(self, mock_factory):
        mock_client = MagicMock()
        mock_client._call_api.return_value = (
            '{"verified": true, "issues": [], "uncited_papers": [], "citation_count": 12}',
            200,
            100,
        )
        mock_factory.return_value = mock_client

        result = verify_citations("Review text.", [_make_reading()])

        assert result.verified is True
        assert result.citation_count == 12
        assert result.issues == []

    @patch("src.analysis.llm_factory.create_llm_client")
    def test_issues_found(self, mock_factory):
        mock_client = MagicMock()
        mock_client._call_api.return_value = (
            '{"verified": false, "issues": ["Missing citation for claim X"], '
            '"uncited_papers": ["p2"], "citation_count": 5}',
            200,
            100,
        )
        mock_factory.return_value = mock_client

        result = verify_citations("Review.", [_make_reading()])

        assert result.verified is False
        assert len(result.issues) == 1
        assert "p2" in result.uncited_papers

    @patch("src.analysis.llm_factory.create_llm_client")
    def test_handles_failure(self, mock_factory):
        mock_client = MagicMock()
        mock_client._call_api.side_effect = Exception("API error")
        mock_factory.return_value = mock_client

        result = verify_citations("Review.", [_make_reading()])

        assert result.verified is False
        assert len(result.issues) == 1
        assert "failed" in result.issues[0].lower()

    @patch("src.analysis.llm_factory.create_llm_client")
    def test_handles_markdown_fenced_json(self, mock_factory):
        mock_client = MagicMock()
        mock_client._call_api.return_value = (
            '```json\n{"verified": true, "issues": [], "uncited_papers": [], "citation_count": 8}\n```',
            200,
            100,
        )
        mock_factory.return_value = mock_client

        result = verify_citations("Review.", [_make_reading()])

        assert result.verified is True
        assert result.citation_count == 8


class TestDeepReviewPipeline:
    """Tests for the full deep review pipeline."""

    def _make_engine(self, tmp_path):
        """Create a SearchEngine with mocked components."""
        from src.indexing.structured_store import StructuredStore
        from src.query.search import SearchEngine

        store = StructuredStore(tmp_path)
        store.save_papers(
            [
                {
                    "paper_id": "p1",
                    "title": "Paper One",
                    "author_string": "Smith",
                    "publication_year": 2024,
                    "collections": [],
                    "item_type": "journalArticle",
                },
                {
                    "paper_id": "p2",
                    "title": "Paper Two",
                    "author_string": "Jones",
                    "publication_year": 2023,
                    "collections": [],
                    "item_type": "journalArticle",
                },
            ]
        )
        store.save_extractions(
            {
                "p1": {
                    "paper_id": "p1",
                    "extraction": {
                        "q02_thesis": "Thesis 1",
                        "q07_methods": "Quantitative",
                        "q03_key_claims": "F1",
                        "q04_evidence": "Evidence for F1",
                        "q19_implications": "C1",
                        "q05_limitations": "L1",
                        "q17_field": "DS",
                    },
                },
                "p2": {
                    "paper_id": "p2",
                    "extraction": {
                        "q02_thesis": "Thesis 2",
                        "q07_methods": "Qualitative",
                        "q03_key_claims": "F2",
                        "q04_evidence": "Evidence for F2",
                        "q19_implications": "C2",
                        "q05_limitations": "L2",
                        "q17_field": "ML",
                    },
                },
            }
        )

        engine = SearchEngine.__new__(SearchEngine)
        engine.structured_store = store
        engine.vector_store = MagicMock()
        engine.embedding_generator = MagicMock()

        return engine

    @patch("src.query.deep_review.verify_citations")
    @patch("src.query.deep_review.synthesize")
    @patch("src.query.agentic.analyze_gaps")
    def test_full_pipeline(self, mock_gaps, mock_synth, mock_qa, tmp_path):
        from src.mcp.adapters import LitrisAdapter
        from src.query.agentic import GapAnalysis

        engine = self._make_engine(tmp_path)
        engine.embedding_generator.embed_text.return_value = [0.1] * 384

        from src.indexing.vector_store import SearchResult

        engine.vector_store.search.return_value = [
            SearchResult(
                paper_id="p1",
                chunk_id="c1",
                chunk_type="dim_q02",
                text="text1",
                score=0.9,
                metadata={"title": "Paper One"},
            ),
            SearchResult(
                paper_id="p2",
                chunk_id="c2",
                chunk_type="dim_q02",
                text="text2",
                score=0.8,
                metadata={"title": "Paper Two"},
            ),
        ]

        mock_gaps.return_value = GapAnalysis(gaps=[], follow_up_queries=[])
        mock_synth.return_value = "## Review\n\nSynthesized content."
        mock_qa.return_value = QAResult(
            verified=True, issues=[], uncited_papers=[], citation_count=10
        )

        # Create adapter with mocked engine
        adapter = LitrisAdapter.__new__(LitrisAdapter)
        adapter._engine = engine

        result = deep_review(
            topic="test topic",
            engine=engine,
            adapter=adapter,
            top_k=10,
            max_rounds=1,
            verify=True,
        )

        assert result.topic == "test topic"
        assert result.papers_used == 2
        assert "## Review" in result.review_text
        assert result.qa_result.verified is True
        assert len(result.paper_readings) == 2

    @patch("src.query.agentic.analyze_gaps")
    def test_empty_results(self, mock_gaps, tmp_path):
        from src.mcp.adapters import LitrisAdapter
        from src.query.agentic import GapAnalysis

        engine = self._make_engine(tmp_path)
        engine.embedding_generator.embed_text.return_value = [0.1] * 384
        engine.vector_store.search.return_value = []

        mock_gaps.return_value = GapAnalysis(gaps=[], follow_up_queries=[])

        adapter = LitrisAdapter.__new__(LitrisAdapter)
        adapter._engine = engine

        result = deep_review(
            topic="obscure topic",
            engine=engine,
            adapter=adapter,
            top_k=10,
        )

        assert result.papers_used == 0
        assert "No papers found" in result.review_text

    @patch("src.query.deep_review.synthesize")
    @patch("src.query.agentic.analyze_gaps")
    def test_skip_verification(self, mock_gaps, mock_synth, tmp_path):
        from src.mcp.adapters import LitrisAdapter
        from src.query.agentic import GapAnalysis

        engine = self._make_engine(tmp_path)
        engine.embedding_generator.embed_text.return_value = [0.1] * 384

        from src.indexing.vector_store import SearchResult

        engine.vector_store.search.return_value = [
            SearchResult(
                paper_id="p1",
                chunk_id="c1",
                chunk_type="dim_q02",
                text="text1",
                score=0.9,
                metadata={"title": "Paper One"},
            ),
        ]

        mock_gaps.return_value = GapAnalysis(gaps=[], follow_up_queries=[])
        mock_synth.return_value = "Review content."

        adapter = LitrisAdapter.__new__(LitrisAdapter)
        adapter._engine = engine

        result = deep_review(
            topic="test",
            engine=engine,
            adapter=adapter,
            verify=False,
        )

        assert result.qa_result is None
        assert result.review_text == "Review content."
