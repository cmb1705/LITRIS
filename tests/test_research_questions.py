"""Tests for research question generation from gap analysis."""

from src.analysis.research_questions import (
    QuestionScope,
    QuestionStyle,
    ResearchQuestionConfig,
    build_future_direction_prompt,
    build_methodology_gap_prompt,
    build_prompts_from_gap_report,
    build_topic_gap_prompt,
    build_year_gap_prompt,
)


def _sample_gap_report():
    """Create a sample gap analysis report for testing."""
    return {
        "topics_underrepresented": [
            {
                "label": "network resilience",
                "count": 2,
                "evidence": [
                    {"paper_id": "p1", "title": "Network Analysis Methods", "year": 2020},
                    {"paper_id": "p2", "title": "Resilience in Complex Systems", "year": 2021},
                ],
            }
        ],
        "methodologies_underrepresented": [
            {
                "label": "approach: agent-based modeling",
                "count": 1,
                "evidence": [
                    {"paper_id": "p3", "title": "Agent-Based Simulation Study", "year": 2019},
                ],
            }
        ],
        "future_directions": [
            {
                "direction": "Explore cross-domain applications of graph neural networks",
                "mention_count": 3,
                "coverage_count": 0,
                "evidence": [
                    {"paper_id": "p4", "title": "GNN Survey Paper", "year": 2022},
                ],
            }
        ],
        "year_gaps": {
            "min_year": 2015,
            "max_year": 2023,
            "missing_ranges": [{"start": 2016, "end": 2017, "length": 2}],
            "sparse_years": [{"year": 2018, "count": 1}],
        },
    }


class TestResearchQuestionConfig:
    """Tests for configuration options."""

    def test_default_config(self):
        """Default config has sensible values."""
        config = ResearchQuestionConfig()
        assert config.count == 3
        assert config.scope == QuestionScope.MODERATE
        assert config.include_rationale is True
        assert config.styles == []

    def test_custom_config(self):
        """Custom config overrides defaults."""
        config = ResearchQuestionConfig(
            count=5,
            styles=[QuestionStyle.CAUSAL, QuestionStyle.COMPARATIVE],
            scope=QuestionScope.NARROW,
            include_rationale=False,
        )
        assert config.count == 5
        assert QuestionStyle.CAUSAL in config.styles
        assert config.scope == QuestionScope.NARROW
        assert config.include_rationale is False


class TestTopicGapPrompt:
    """Tests for topic gap prompt generation."""

    def test_basic_prompt_structure(self):
        """Prompt contains required elements."""
        gap = {
            "label": "machine learning interpretability",
            "count": 3,
            "evidence": [{"paper_id": "p1", "title": "XAI Survey", "year": 2021}],
        }
        config = ResearchQuestionConfig(count=2)
        prompt = build_topic_gap_prompt(gap, config)

        assert "machine learning interpretability" in prompt
        assert "XAI Survey" in prompt
        assert "2021" in prompt
        assert "2 research question" in prompt
        assert "Quality Guardrails" in prompt

    def test_prompt_includes_scope_instruction(self):
        """Prompt includes scope-specific instruction."""
        gap = {"label": "test", "count": 1, "evidence": []}
        config = ResearchQuestionConfig(scope=QuestionScope.NARROW)
        prompt = build_topic_gap_prompt(gap, config)

        assert "single empirical study" in prompt


class TestMethodologyGapPrompt:
    """Tests for methodology gap prompt generation."""

    def test_methodology_prompt_structure(self):
        """Methodology prompt has correct structure."""
        gap = {
            "label": "approach: mixed methods",
            "count": 2,
            "evidence": [],
        }
        config = ResearchQuestionConfig()
        prompt = build_methodology_gap_prompt(gap, config)

        assert "mixed methods" in prompt
        assert "methodology" in prompt.lower()


class TestFutureDirectionPrompt:
    """Tests for future direction prompt generation."""

    def test_future_direction_includes_coverage(self):
        """Future direction prompt includes coverage metrics."""
        gap = {
            "direction": "Apply neural approaches to citation prediction",
            "mention_count": 5,
            "coverage_count": 1,
            "evidence": [],
        }
        config = ResearchQuestionConfig()
        prompt = build_future_direction_prompt(gap, config)

        assert "Apply neural approaches" in prompt
        assert "5 paper(s)" in prompt
        assert "1 paper(s)" in prompt


class TestYearGapPrompt:
    """Tests for year gap prompt generation."""

    def test_year_gap_prompt_with_missing_range(self):
        """Year gap prompt describes temporal gap."""
        year_gaps = {
            "min_year": 2010,
            "max_year": 2023,
            "missing_ranges": [{"start": 2012, "end": 2014, "length": 3}],
            "sparse_years": [],
        }
        config = ResearchQuestionConfig()
        prompt = build_year_gap_prompt(year_gaps, config)

        assert prompt is not None
        assert "2012-2014" in prompt
        assert "2010" in prompt
        assert "2023" in prompt

    def test_year_gap_returns_none_when_no_gaps(self):
        """Returns None when no missing ranges."""
        year_gaps = {
            "min_year": 2020,
            "max_year": 2023,
            "missing_ranges": [],
            "sparse_years": [],
        }
        config = ResearchQuestionConfig()
        prompt = build_year_gap_prompt(year_gaps, config)

        assert prompt is None


class TestBuildPromptsFromReport:
    """Tests for building all prompts from a gap report."""

    def test_builds_prompts_for_all_gap_types(self):
        """Generates prompts for each gap category."""
        report = _sample_gap_report()
        config = ResearchQuestionConfig(count=2)
        prompts = build_prompts_from_gap_report(report, config)

        types = [p["type"] for p in prompts]
        assert "topic" in types
        assert "methodology" in types
        assert "future_direction" in types
        assert "year_gap" in types

    def test_each_prompt_has_required_fields(self):
        """Each prompt dict has type, gap, and prompt text."""
        report = _sample_gap_report()
        config = ResearchQuestionConfig()
        prompts = build_prompts_from_gap_report(report, config)

        for prompt_dict in prompts:
            assert "type" in prompt_dict
            assert "gap" in prompt_dict
            assert "prompt" in prompt_dict
            assert isinstance(prompt_dict["prompt"], str)
            assert len(prompt_dict["prompt"]) > 100

    def test_empty_report_returns_empty_list(self):
        """Empty gap report produces no prompts."""
        report = {
            "topics_underrepresented": [],
            "methodologies_underrepresented": [],
            "future_directions": [],
            "year_gaps": {"missing_ranges": []},
        }
        config = ResearchQuestionConfig()
        prompts = build_prompts_from_gap_report(report, config)

        assert prompts == []


class TestStyleInstructions:
    """Tests for question style handling."""

    def test_specific_style_in_prompt(self):
        """Specified styles appear in prompt."""
        gap = {"label": "test", "count": 1, "evidence": []}
        config = ResearchQuestionConfig(styles=[QuestionStyle.CAUSAL])
        prompt = build_topic_gap_prompt(gap, config)

        assert "causal" in prompt.lower()

    def test_multiple_styles_listed(self):
        """Multiple allowed styles are listed."""
        gap = {"label": "test", "count": 1, "evidence": []}
        config = ResearchQuestionConfig(
            styles=[QuestionStyle.EXPLORATORY, QuestionStyle.COMPARATIVE]
        )
        prompt = build_topic_gap_prompt(gap, config)

        assert "exploratory" in prompt.lower()
        assert "comparative" in prompt.lower()
