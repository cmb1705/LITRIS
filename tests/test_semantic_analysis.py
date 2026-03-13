"""Tests for SemanticAnalysis schema."""

import pytest

from src.analysis.schemas import SemanticAnalysis


def _make_analysis(**overrides) -> SemanticAnalysis:
    """Create a SemanticAnalysis with required fields, plus overrides."""
    defaults = {
        "paper_id": "test_id",
        "prompt_version": "2.0.0",
        "extraction_model": "test-model",
        "extracted_at": "2026-01-01T00:00:00Z",
    }
    defaults.update(overrides)
    return SemanticAnalysis(**defaults)


class TestConstruction:
    """Tests for SemanticAnalysis construction."""

    def test_required_fields_only(self):
        """Construction with only required fields succeeds."""
        sa = _make_analysis()
        assert sa.paper_id == "test_id"
        assert sa.prompt_version == "2.0.0"
        assert sa.extraction_model == "test-model"
        assert sa.extracted_at == "2026-01-01T00:00:00Z"

    def test_all_dimensions_default_none(self):
        """All q-fields default to None."""
        sa = _make_analysis()
        for field_name in SemanticAnalysis.DIMENSION_FIELDS:
            assert getattr(sa, field_name) is None

    def test_dimension_coverage_defaults_zero(self):
        """dimension_coverage defaults to 0.0."""
        sa = _make_analysis()
        assert sa.dimension_coverage == 0.0

    def test_coverage_flags_defaults_empty(self):
        """coverage_flags defaults to empty list."""
        sa = _make_analysis()
        assert sa.coverage_flags == []

    def test_set_dimension_fields(self):
        """Setting individual q-fields works."""
        sa = _make_analysis(
            q01_research_question="What is X?",
            q02_thesis="X is Y.",
            q17_field="Computer Science",
        )
        assert sa.q01_research_question == "What is X?"
        assert sa.q02_thesis == "X is Y."
        assert sa.q17_field == "Computer Science"
        # Others remain None
        assert sa.q03_key_claims is None

    def test_missing_required_field_raises(self):
        """Missing required field raises ValidationError."""
        with pytest.raises(Exception):
            SemanticAnalysis(
                prompt_version="2.0.0",
                extraction_model="test-model",
                extracted_at="2026-01-01T00:00:00Z",
            )

    def test_dimension_coverage_bounds(self):
        """dimension_coverage is clamped to 0.0-1.0."""
        sa = _make_analysis(dimension_coverage=0.75)
        assert sa.dimension_coverage == 0.75

        with pytest.raises(Exception):
            _make_analysis(dimension_coverage=1.5)

        with pytest.raises(Exception):
            _make_analysis(dimension_coverage=-0.1)


class TestDimensionFields:
    """Tests for DIMENSION_FIELDS class variable."""

    def test_has_40_fields(self):
        """DIMENSION_FIELDS contains exactly 40 entries."""
        assert len(SemanticAnalysis.DIMENSION_FIELDS) == 40

    def test_fields_are_sequential(self):
        """Fields follow q01 through q40 pattern."""
        for i, field_name in enumerate(SemanticAnalysis.DIMENSION_FIELDS, start=1):
            assert field_name.startswith(f"q{i:02d}_"), (
                f"Field {i} should start with q{i:02d}_, got {field_name}"
            )

    def test_all_fields_exist_on_model(self):
        """Every DIMENSION_FIELDS entry corresponds to a real field."""
        sa = _make_analysis()
        for field_name in SemanticAnalysis.DIMENSION_FIELDS:
            assert hasattr(sa, field_name), f"Missing field: {field_name}"


class TestCoreFields:
    """Tests for CORE_FIELDS class variable."""

    def test_core_fields_count(self):
        """CORE_FIELDS has 5 entries."""
        assert len(SemanticAnalysis.CORE_FIELDS) == 5

    def test_core_fields_are_pass_1(self):
        """CORE_FIELDS are q01 through q05 (Pass 1: Research Core)."""
        expected = [
            "q01_research_question",
            "q02_thesis",
            "q03_key_claims",
            "q04_evidence",
            "q05_limitations",
        ]
        assert SemanticAnalysis.CORE_FIELDS == expected

    def test_core_fields_subset_of_dimension_fields(self):
        """CORE_FIELDS is a subset of DIMENSION_FIELDS."""
        for f in SemanticAnalysis.CORE_FIELDS:
            assert f in SemanticAnalysis.DIMENSION_FIELDS


class TestDimensionGroups:
    """Tests for DIMENSION_GROUPS class variable."""

    def test_six_groups(self):
        """There are exactly 6 dimension groups."""
        assert len(SemanticAnalysis.DIMENSION_GROUPS) == 6

    def test_expected_group_names(self):
        """Groups match expected names."""
        expected = {"research_core", "methodology", "context", "meta", "scholarly", "impact"}
        assert set(SemanticAnalysis.DIMENSION_GROUPS.keys()) == expected

    def test_groups_cover_all_dimensions(self):
        """All 40 dimension fields appear in exactly one group."""
        all_fields_in_groups = []
        for fields in SemanticAnalysis.DIMENSION_GROUPS.values():
            all_fields_in_groups.extend(fields)
        assert sorted(all_fields_in_groups) == sorted(SemanticAnalysis.DIMENSION_FIELDS)

    def test_no_duplicate_fields_across_groups(self):
        """No dimension field appears in more than one group."""
        all_fields = []
        for fields in SemanticAnalysis.DIMENSION_GROUPS.values():
            all_fields.extend(fields)
        assert len(all_fields) == len(set(all_fields))


class TestGetDimensionValue:
    """Tests for get_dimension_value()."""

    def test_returns_set_value(self):
        """Returns the value when a dimension is set."""
        sa = _make_analysis(q02_thesis="X is Y.")
        assert sa.get_dimension_value("q02_thesis") == "X is Y."

    def test_returns_none_for_unset(self):
        """Returns None for unset dimensions."""
        sa = _make_analysis()
        assert sa.get_dimension_value("q02_thesis") is None

    def test_returns_none_for_invalid(self):
        """Returns None for non-existent field names."""
        sa = _make_analysis()
        assert sa.get_dimension_value("nonexistent_field") is None


class TestNonNoneDimensions:
    """Tests for non_none_dimensions()."""

    def test_empty_when_all_none(self):
        """Returns empty dict when no dimensions are set."""
        sa = _make_analysis()
        assert sa.non_none_dimensions() == {}

    def test_returns_only_set_dimensions(self):
        """Returns only non-None dimension fields."""
        sa = _make_analysis(
            q01_research_question="What?",
            q02_thesis="This.",
            q40_policy_recommendations="Do X.",
        )
        result = sa.non_none_dimensions()
        assert len(result) == 3
        assert result["q01_research_question"] == "What?"
        assert result["q02_thesis"] == "This."
        assert result["q40_policy_recommendations"] == "Do X."

    def test_excludes_metadata_fields(self):
        """Does not include paper_id, prompt_version, etc."""
        sa = _make_analysis(q01_research_question="What?")
        result = sa.non_none_dimensions()
        assert "paper_id" not in result
        assert "prompt_version" not in result
        assert "dimension_coverage" not in result


class TestToIndexDict:
    """Tests for to_index_dict()."""

    def test_includes_required_fields(self):
        """Index dict includes paper_id and metadata."""
        sa = _make_analysis()
        d = sa.to_index_dict()
        assert d["paper_id"] == "test_id"
        assert d["prompt_version"] == "2.0.0"
        assert d["extraction_model"] == "test-model"
        assert d["extracted_at"] == "2026-01-01T00:00:00Z"

    def test_includes_dimension_fields(self):
        """Index dict includes dimension values."""
        sa = _make_analysis(q02_thesis="Thesis text")
        d = sa.to_index_dict()
        assert d["q02_thesis"] == "Thesis text"
        assert d["q01_research_question"] is None

    def test_excludes_class_vars(self):
        """Index dict does not include DIMENSION_FIELDS, CORE_FIELDS, DIMENSION_GROUPS."""
        sa = _make_analysis()
        d = sa.to_index_dict()
        assert "DIMENSION_FIELDS" not in d
        assert "CORE_FIELDS" not in d
        assert "DIMENSION_GROUPS" not in d

    def test_includes_coverage_metadata(self):
        """Index dict includes dimension_coverage and coverage_flags."""
        sa = _make_analysis(dimension_coverage=0.5, coverage_flags=["PARTIAL_COVERAGE"])
        d = sa.to_index_dict()
        assert d["dimension_coverage"] == 0.5
        assert d["coverage_flags"] == ["PARTIAL_COVERAGE"]

    def test_roundtrip_serialization(self):
        """Index dict can reconstruct a SemanticAnalysis."""
        original = _make_analysis(
            q01_research_question="Question",
            q02_thesis="Thesis",
            q17_field="CS",
            dimension_coverage=0.075,
        )
        d = original.to_index_dict()
        reconstructed = SemanticAnalysis(**d)
        assert reconstructed.paper_id == original.paper_id
        assert reconstructed.q02_thesis == original.q02_thesis
        assert reconstructed.dimension_coverage == original.dimension_coverage


class TestDimensionCoverage:
    """Tests for dimension_coverage field behavior."""

    def test_zero_coverage(self):
        """Zero coverage when no dimensions set."""
        sa = _make_analysis(dimension_coverage=0.0)
        assert sa.dimension_coverage == 0.0

    def test_full_coverage(self):
        """Full coverage value of 1.0 is valid."""
        sa = _make_analysis(dimension_coverage=1.0)
        assert sa.dimension_coverage == 1.0

    def test_partial_coverage(self):
        """Partial coverage stores correctly."""
        sa = _make_analysis(dimension_coverage=0.325)
        assert sa.dimension_coverage == pytest.approx(0.325)

    def test_coverage_flags_stores_list(self):
        """Coverage flags stores a list of strings."""
        flags = ["PARTIAL_COVERAGE", "CORE_GAPS"]
        sa = _make_analysis(coverage_flags=flags)
        assert sa.coverage_flags == flags
