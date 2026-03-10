"""Tests for gap analysis heuristics."""

from pathlib import Path

from src.analysis.gap_detection import (
    GapDetectionConfig,
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
