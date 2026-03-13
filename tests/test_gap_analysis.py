"""Tests for gap analysis heuristics."""

from pathlib import Path

from src.analysis.gap_detection import (
    GapDetectionConfig,
    _calculate_gap_confidence,
    analyze_gap_report,
    format_gap_report_markdown,
    save_gap_report,
)


def _sample_corpus():
    papers = [
        {
            "paper_id": "p1",
            "title": "Topic A study",
            "abstract": "Qualitative investigation of topic A.",
            "publication_year": 2018,
            "collections": ["core"],
        },
        {
            "paper_id": "p2",
            "title": "Topic A replication",
            "abstract": "Replication study for topic A.",
            "publication_year": 2020,
            "collections": ["core"],
        },
        {
            "paper_id": "p3",
            "title": "Topic B overview",
            "abstract": "Survey of topic B.",
            "publication_year": 2020,
            "collections": ["core"],
        },
    ]
    extractions = {
        "p1": {
            "paper_id": "p1",
            "extraction": {
                "discipline_tags": ["Topic A"],
                "keywords": ["alpha"],
                "methodology": {
                    "approach": "Qualitative",
                    "analysis_methods": ["interviews"],
                },
                "future_directions": ["Explore beta networks"],
            },
        },
        "p2": {
            "paper_id": "p2",
            "extraction": {
                "discipline_tags": ["Topic A"],
                "keywords": ["alpha"],
                "methodology": {
                    "approach": "Quantitative",
                    "analysis_methods": ["regression"],
                },
                "future_directions": ["Explore beta networks"],
            },
        },
        "p3": {
            "paper_id": "p3",
            "extraction": {
                "discipline_tags": ["Topic B"],
                "keywords": ["beta"],
                "methodology": {
                    "approach": "Mixed",
                    "analysis_methods": ["survey"],
                },
                "future_directions": ["Investigate gamma datasets"],
            },
        },
    }
    return papers, extractions


def test_gap_analysis_outputs_expected_sections():
    papers, extractions = _sample_corpus()
    config = GapDetectionConfig(
        max_items=5,
        min_count=2,
        quantile=0.0,
        future_direction_max_coverage=0,
        future_direction_min_mentions=1,
    )
    report = analyze_gap_report(papers, extractions, config)

    topic_labels = {item["label"] for item in report["topics_underrepresented"]}
    assert "topic b" in topic_labels

    method_labels = {item["label"] for item in report["methodologies_underrepresented"]}
    assert any(label.startswith("approach:") for label in method_labels)

    year_gaps = report["year_gaps"]["missing_ranges"]
    assert year_gaps
    assert year_gaps[0]["start"] == 2019

    assert report["future_directions"]


def test_gap_items_have_confidence_scores():
    papers, extractions = _sample_corpus()
    config = GapDetectionConfig(
        max_items=5,
        min_count=1,
        quantile=0.0,
        future_direction_max_coverage=0,
        future_direction_min_mentions=1,
    )
    report = analyze_gap_report(papers, extractions, config)

    # Topic gaps should have confidence
    for item in report["topics_underrepresented"]:
        assert "confidence" in item
        assert 0.0 <= item["confidence"] <= 1.0

    # Methodology gaps should have confidence
    for item in report["methodologies_underrepresented"]:
        assert "confidence" in item
        assert 0.0 <= item["confidence"] <= 1.0

    # Future direction gaps should have confidence
    for item in report["future_directions"]:
        assert "confidence" in item
        assert 0.0 <= item["confidence"] <= 1.0


def test_gap_report_markdown_format():
    papers, extractions = _sample_corpus()
    config = GapDetectionConfig(max_items=3, min_count=1, quantile=0.0)
    report = analyze_gap_report(papers, extractions, config)
    markdown = format_gap_report_markdown(report)
    assert markdown.startswith("# Gap Analysis Report")
    assert "Underrepresented Topics" in markdown


def test_save_gap_report(tmp_path: Path):
    papers, extractions = _sample_corpus()
    config = GapDetectionConfig(max_items=3, min_count=1, quantile=0.0)
    report = analyze_gap_report(papers, extractions, config)
    output_path = save_gap_report(report, tmp_path, "markdown")
    assert output_path.exists()


# --- _calculate_gap_confidence unit tests ---


class TestCalculateGapConfidence:
    """Direct unit tests for the confidence scoring formula."""

    def test_zero_corpus_returns_zero(self):
        """corpus_size=0 short-circuits to 0.0."""
        result = _calculate_gap_confidence(
            count=0, threshold=5, evidence_count=3, corpus_size=0
        )
        assert result == 0.0

    def test_zero_threshold_returns_zero(self):
        """threshold=0 short-circuits to 0.0."""
        result = _calculate_gap_confidence(
            count=0, threshold=0, evidence_count=3, corpus_size=100
        )
        assert result == 0.0

    def test_high_signal_low_count(self):
        """count=0 with high threshold yields maximum signal strength (0.5 weight)."""
        result = _calculate_gap_confidence(
            count=0, threshold=10, evidence_count=3, corpus_size=100
        )
        # signal_strength = 1.0, evidence_quality = 1.0, corpus_factor = 1.0
        # confidence = 0.5*1.0 + 0.3*1.0 + 0.2*1.0 = 1.0
        assert result == 1.0

    def test_count_equals_threshold_zero_signal(self):
        """count==threshold means no underrepresentation signal."""
        result = _calculate_gap_confidence(
            count=5, threshold=5, evidence_count=3, corpus_size=100
        )
        # signal_strength = 1.0 - 5/5 = 0.0
        # confidence = 0.5*0.0 + 0.3*1.0 + 0.2*1.0 = 0.5
        assert result == 0.5

    def test_count_exceeds_threshold(self):
        """count > threshold clamps signal_strength to 0.0 via max()."""
        result = _calculate_gap_confidence(
            count=10, threshold=5, evidence_count=3, corpus_size=100
        )
        # signal_strength = max(0.0, 1.0 - 10/5) = max(0.0, -1.0) = 0.0
        assert result == 0.5

    def test_weighted_components_math(self):
        """Verify the 50/30/20 weighting formula exactly."""
        # count=2, threshold=4: signal = 1 - 2/4 = 0.5
        # evidence_count=1: quality = 1/3 ~= 0.333
        # corpus_size=25: factor = 25/50 = 0.5
        result = _calculate_gap_confidence(
            count=2, threshold=4, evidence_count=1, corpus_size=25
        )
        expected = 0.5 * 0.5 + 0.3 * (1 / 3.0) + 0.2 * 0.5
        assert result == round(min(expected, 1.0), 3)

    def test_evidence_caps_at_three(self):
        """evidence_quality caps at 1.0 when evidence_count >= 3."""
        result_3 = _calculate_gap_confidence(
            count=0, threshold=5, evidence_count=3, corpus_size=100
        )
        result_10 = _calculate_gap_confidence(
            count=0, threshold=5, evidence_count=10, corpus_size=100
        )
        assert result_3 == result_10

    def test_corpus_factor_caps_at_fifty(self):
        """corpus_factor caps at 1.0 when corpus_size >= 50."""
        result_50 = _calculate_gap_confidence(
            count=0, threshold=5, evidence_count=3, corpus_size=50
        )
        result_1000 = _calculate_gap_confidence(
            count=0, threshold=5, evidence_count=3, corpus_size=1000
        )
        assert result_50 == result_1000

    def test_small_corpus_reduces_confidence(self):
        """corpus_size=10 yields lower confidence than corpus_size=100."""
        small = _calculate_gap_confidence(
            count=0, threshold=5, evidence_count=3, corpus_size=10
        )
        large = _calculate_gap_confidence(
            count=0, threshold=5, evidence_count=3, corpus_size=100
        )
        assert small < large

    def test_confidence_never_exceeds_one(self):
        """Even with maximum inputs, confidence does not exceed 1.0."""
        result = _calculate_gap_confidence(
            count=0, threshold=100, evidence_count=100, corpus_size=10000
        )
        assert result <= 1.0


# --- analyze_gap_report integration tests ---


class TestAnalyzeGapReportEdgeCases:
    """Test analyze_gap_report with edge case inputs."""

    def test_empty_corpus(self):
        """Zero papers produces empty gap sections."""
        report = analyze_gap_report(
            [], {}, GapDetectionConfig(min_count=1, quantile=0.0)
        )
        assert report["topics_underrepresented"] == []
        assert report["methodologies_underrepresented"] == []
        assert report["future_directions"] == []
        assert report["corpus"]["papers"] == 0

    def test_collection_filter_restricts_papers(self):
        """Only papers in the specified collection are analyzed."""
        papers, extractions = _sample_corpus()
        # Add a paper in a different collection
        papers.append(
            {
                "paper_id": "p4",
                "title": "Topic C exclusive",
                "abstract": "Exclusive topic.",
                "publication_year": 2021,
                "collections": ["other"],
            }
        )
        extractions["p4"] = {
            "paper_id": "p4",
            "extraction": {
                "discipline_tags": ["Topic C"],
                "keywords": ["gamma"],
                "methodology": {"approach": "Qualitative"},
            },
        }
        report = analyze_gap_report(
            papers,
            extractions,
            GapDetectionConfig(min_count=1, quantile=0.0),
            collections=["core"],
        )
        # p4 should be excluded (only in "other" collection)
        assert report["corpus"]["papers"] == 3
        topic_labels = {
            item["label"] for item in report["topics_underrepresented"]
        }
        assert "topic c" not in topic_labels

    def test_papers_without_years_no_year_gaps(self):
        """Papers missing publication_year do not crash year gap analysis."""
        papers = [
            {"paper_id": "p1", "title": "No year", "collections": []},
            {"paper_id": "p2", "title": "Also no year", "collections": []},
        ]
        extractions = {
            "p1": {"extraction": {"discipline_tags": ["A"]}},
            "p2": {"extraction": {"discipline_tags": ["B"]}},
        }
        report = analyze_gap_report(
            papers,
            extractions,
            GapDetectionConfig(min_count=1, quantile=0.0),
        )
        assert report["year_gaps"]["missing_ranges"] == []
        assert report["year_gaps"]["sparse_years"] == []

    def test_single_year_no_missing_ranges(self):
        """All papers in the same year produces no missing ranges."""
        papers = [
            {
                "paper_id": f"p{i}",
                "title": f"Paper {i}",
                "publication_year": 2020,
                "collections": [],
            }
            for i in range(5)
        ]
        extractions = {
            f"p{i}": {"extraction": {"discipline_tags": [f"T{i}"]}}
            for i in range(5)
        }
        report = analyze_gap_report(
            papers,
            extractions,
            GapDetectionConfig(min_count=1, quantile=0.0),
        )
        assert report["year_gaps"]["missing_ranges"] == []

    def test_future_directions_below_min_mentions_excluded(self):
        """Future directions below min_mentions threshold are excluded."""
        papers = [
            {
                "paper_id": "p1",
                "title": "Paper 1",
                "publication_year": 2020,
                "collections": [],
            },
        ]
        extractions = {
            "p1": {
                "extraction": {
                    "discipline_tags": ["A"],
                    "future_directions": ["Unique direction only mentioned once"],
                },
            },
        }
        # Require at least 2 mentions
        report = analyze_gap_report(
            papers,
            extractions,
            GapDetectionConfig(
                min_count=1,
                quantile=0.0,
                future_direction_min_mentions=2,
            ),
        )
        assert report["future_directions"] == []
