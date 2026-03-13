"""Tests for coverage scoring module."""

from unittest.mock import MagicMock

from src.analysis.coverage import (
    CORE_ATTRS,
    FLAG_CORE_GAPS,
    FLAG_CRITICAL_GAPS,
    FLAG_PARTIAL_COVERAGE,
    FLAG_SPARSE_COVERAGE,
    QUESTION_ATTRS,
    TOTAL_DIMENSIONS,
    apply_coverage,
    generate_coverage_report,
    score_coverage,
)


def _make_analysis(answered_count: int, skip_core: list[str] | None = None):
    """Create a mock SemanticAnalysis with the given number of answered dimensions.

    Args:
        answered_count: Number of question fields to set as non-None.
        skip_core: List of core attrs to force to None even if within answered_count.
    """
    skip_core = skip_core or []
    mock = MagicMock()
    mock.paper_id = "test-paper-001"

    # Start by setting all to None
    for attr in QUESTION_ATTRS:
        setattr(mock, attr, None)

    # Fill in the first N that aren't in skip_core
    filled = 0
    for attr in QUESTION_ATTRS:
        if attr in skip_core:
            continue
        if filled >= answered_count:
            break
        setattr(mock, attr, f"Answer for {attr}")
        filled += 1

    return mock


class TestScoreCoverage:
    def test_full_coverage(self):
        analysis = _make_analysis(40)
        result = score_coverage(analysis)
        assert result.tier == "full"
        assert result.coverage == 1.0
        assert result.answered == 40
        assert result.flags == []
        assert result.missing_core == []

    def test_full_threshold(self):
        analysis = _make_analysis(34)
        result = score_coverage(analysis)
        assert result.tier == "full"
        assert result.coverage == 34 / 40
        assert FLAG_PARTIAL_COVERAGE not in result.flags

    def test_partial_coverage(self):
        analysis = _make_analysis(30)
        result = score_coverage(analysis)
        assert result.tier == "partial"
        assert FLAG_PARTIAL_COVERAGE in result.flags

    def test_partial_lower_bound(self):
        analysis = _make_analysis(24)
        result = score_coverage(analysis)
        assert result.tier == "partial"
        assert result.coverage == 24 / 40

    def test_sparse_coverage(self):
        analysis = _make_analysis(15)
        result = score_coverage(analysis)
        assert result.tier == "sparse"
        assert FLAG_SPARSE_COVERAGE in result.flags

    def test_sparse_lower_bound(self):
        analysis = _make_analysis(12)
        result = score_coverage(analysis)
        assert result.tier == "sparse"

    def test_critical_coverage(self):
        analysis = _make_analysis(5)
        result = score_coverage(analysis)
        assert result.tier == "critical"
        assert FLAG_CRITICAL_GAPS in result.flags

    def test_zero_coverage(self):
        analysis = _make_analysis(0)
        result = score_coverage(analysis)
        assert result.tier == "critical"
        assert result.coverage == 0.0
        assert result.answered == 0
        assert FLAG_CRITICAL_GAPS in result.flags
        assert FLAG_CORE_GAPS in result.flags
        assert len(result.missing_core) == 5

    def test_core_gaps_detected(self):
        """Missing any core dimension (q01-q05) raises CORE_GAPS."""
        analysis = _make_analysis(35, skip_core=["q01_research_question"])
        result = score_coverage(analysis)
        assert FLAG_CORE_GAPS in result.flags
        assert "q01_research_question" in result.missing_core

    def test_multiple_core_gaps(self):
        skip = ["q01_research_question", "q03_key_claims", "q05_limitations"]
        analysis = _make_analysis(37, skip_core=skip)
        result = score_coverage(analysis)
        assert FLAG_CORE_GAPS in result.flags
        for attr in skip:
            assert attr in result.missing_core

    def test_no_core_gaps_when_all_core_present(self):
        # 10 answered, all core present (first 5 are core)
        analysis = _make_analysis(10)
        result = score_coverage(analysis)
        assert FLAG_CORE_GAPS not in result.flags
        assert result.missing_core == []

    def test_result_to_dict(self):
        analysis = _make_analysis(30)
        result = score_coverage(analysis)
        d = result.to_dict()
        assert d["paper_id"] == "test-paper-001"
        assert d["answered"] == 30
        assert d["total"] == 40
        assert d["coverage"] == round(30 / 40, 4)
        assert d["tier"] == "partial"
        assert isinstance(d["flags"], list)
        assert isinstance(d["missing_core"], list)


class TestApplyCoverage:
    def test_sets_fields(self):
        analysis = _make_analysis(35)
        returned = apply_coverage(analysis)
        assert returned is analysis
        assert analysis.dimension_coverage == 35 / 40
        assert isinstance(analysis.coverage_flags, list)

    def test_full_coverage_no_flags(self):
        analysis = _make_analysis(40)
        apply_coverage(analysis)
        assert analysis.dimension_coverage == 1.0
        assert analysis.coverage_flags == []

    def test_critical_sets_flags(self):
        analysis = _make_analysis(3)
        apply_coverage(analysis)
        assert FLAG_CRITICAL_GAPS in analysis.coverage_flags
        assert FLAG_CORE_GAPS in analysis.coverage_flags


class TestGenerateCoverageReport:
    def test_report_structure(self, tmp_path):
        analyses = [_make_analysis(40), _make_analysis(20), _make_analysis(5)]
        output = tmp_path / "report.json"
        report = generate_coverage_report(analyses, output_path=output)

        assert report["total_papers"] == 3
        assert "average_coverage" in report
        assert "tier_distribution" in report
        assert report["tier_distribution"]["full"] == 1
        assert report["tier_distribution"]["sparse"] == 1
        assert report["tier_distribution"]["critical"] == 1
        assert len(report["papers"]) == 3
        assert output.exists()

    def test_report_file_written(self, tmp_path):
        analyses = [_make_analysis(40)]
        output = tmp_path / "sub" / "report.json"
        generate_coverage_report(analyses, output_path=output)
        assert output.exists()

        import json

        data = json.loads(output.read_text())
        assert data["total_papers"] == 1

    def test_empty_batch(self, tmp_path):
        output = tmp_path / "empty.json"
        report = generate_coverage_report([], output_path=output)
        assert report["total_papers"] == 0
        assert report["average_coverage"] == 0.0

    def test_core_gap_papers_listed(self, tmp_path):
        a1 = _make_analysis(40)
        a2 = _make_analysis(35, skip_core=["q02_thesis"])
        output = tmp_path / "report.json"
        report = generate_coverage_report([a1, a2], output_path=output)
        assert report["core_gap_count"] == 1
        assert "test-paper-001" in report["core_gap_papers"]


class TestConstants:
    def test_total_dimensions(self):
        assert TOTAL_DIMENSIONS == 40

    def test_question_attrs_count(self):
        assert len(QUESTION_ATTRS) == 40

    def test_core_attrs_are_first_five(self):
        assert len(CORE_ATTRS) == 5
        assert CORE_ATTRS[0] == "q01_research_question"
        assert CORE_ATTRS[4] == "q05_limitations"

    def test_all_attrs_start_with_q(self):
        for attr in QUESTION_ATTRS:
            assert attr.startswith("q")
