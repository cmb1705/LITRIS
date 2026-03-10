"""Tests for research question generation from gap analysis."""

from src.analysis.research_questions import (
    GeneratedQuestion,
    QuestionScope,
    QuestionStyle,
    ResearchQuestionConfig,
    build_future_direction_prompt,
    build_methodology_gap_prompt,
    build_prompts_from_gap_report,
    build_topic_gap_prompt,
    build_year_gap_prompt,
    deduplicate_questions,
    format_questions_markdown,
    generate_questions_from_prompts,
    parse_llm_response,
    rank_questions,
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


class TestParseLLMResponse:
    """Tests for parsing LLM JSON responses."""

    def test_parses_valid_json(self):
        """Parses well-formed JSON response."""
        response = '''
        {
            "questions": [
                {
                    "question": "How does network topology affect resilience?",
                    "style": "causal",
                    "rationale": "Understanding topology is key."
                }
            ]
        }
        '''
        questions = parse_llm_response(response, "topic", "network resilience")

        assert len(questions) == 1
        assert questions[0].question == "How does network topology affect resilience?"
        assert questions[0].style == "causal"
        assert questions[0].gap_type == "topic"
        assert questions[0].gap_label == "network resilience"

    def test_handles_markdown_wrapped_json(self):
        """Handles JSON wrapped in markdown code blocks."""
        response = '''Here is the output:
        ```json
        {
            "questions": [
                {"question": "What are the key factors?", "style": "exploratory"}
            ]
        }
        ```
        '''
        questions = parse_llm_response(response, "topic", "test")

        assert len(questions) == 1
        assert questions[0].question == "What are the key factors?"

    def test_filters_non_questions(self):
        """Filters entries that don't end with question mark."""
        response = '''
        {
            "questions": [
                {"question": "Valid question?", "style": "exploratory"},
                {"question": "Not a question", "style": "exploratory"}
            ]
        }
        '''
        questions = parse_llm_response(response, "topic", "test")

        assert len(questions) == 1
        assert questions[0].question == "Valid question?"

    def test_handles_invalid_json(self):
        """Returns empty list for invalid JSON."""
        questions = parse_llm_response("not json at all", "topic", "test")
        assert questions == []


class TestDeduplicateQuestions:
    """Tests for question deduplication."""

    def test_removes_similar_questions(self):
        """Removes questions with high similarity."""
        questions = [
            GeneratedQuestion(
                question="How does X affect Y?",
                style="causal",
                gap_type="topic",
                gap_label="test",
            ),
            GeneratedQuestion(
                question="How does X affect Y in the context?",
                style="causal",
                gap_type="topic",
                gap_label="test",
            ),
        ]
        deduplicated, removed = deduplicate_questions(questions, similarity_threshold=0.5)

        assert len(deduplicated) == 1
        assert removed == 1

    def test_keeps_different_questions(self):
        """Keeps questions that are sufficiently different."""
        questions = [
            GeneratedQuestion(
                question="How does network topology affect resilience?",
                style="causal",
                gap_type="topic",
                gap_label="networks",
            ),
            GeneratedQuestion(
                question="What methodologies are used in citation analysis?",
                style="exploratory",
                gap_type="methodology",
                gap_label="citation",
            ),
        ]
        deduplicated, removed = deduplicate_questions(questions)

        assert len(deduplicated) == 2
        assert removed == 0

    def test_handles_empty_list(self):
        """Handles empty input gracefully."""
        deduplicated, removed = deduplicate_questions([])
        assert deduplicated == []
        assert removed == 0


class TestRankQuestions:
    """Tests for question ranking."""

    def test_ranks_by_combined_score(self):
        """Questions are sorted by combined score."""
        questions = [
            GeneratedQuestion(
                question="Short?",
                style="descriptive",
                gap_type="topic",
                gap_label="test",
            ),
            GeneratedQuestion(
                question="How does the implementation of graph neural networks compare to traditional methods for citation prediction tasks?",
                style="causal",
                gap_type="methodology",
                gap_label="GNN",
                rationale="Important comparison",
                methodology_hints=["comparative study"],
            ),
        ]
        ranked = rank_questions(questions)

        # Second question should rank higher (better attributes)
        assert ranked[0].question.startswith("How does")
        assert ranked[0].combined_score > ranked[1].combined_score

    def test_assigns_relevance_scores(self):
        """Questions receive relevance scores."""
        questions = [
            GeneratedQuestion(
                question="What are the key factors influencing network resilience?",
                style="exploratory",
                gap_type="topic",
                gap_label="test",
                rationale="This matters because...",
            ),
        ]
        ranked = rank_questions(questions)

        assert ranked[0].relevance_score > 0
        assert ranked[0].relevance_score <= 1.0

    def test_handles_empty_list(self):
        """Handles empty input."""
        ranked = rank_questions([])
        assert ranked == []


class TestGenerateQuestionsFromPrompts:
    """Tests for the full generation pipeline."""

    def test_generates_and_processes_questions(self):
        """Full pipeline generates, dedupes, and ranks questions."""
        prompts = [
            {
                "type": "topic",
                "gap": {"label": "network resilience"},
                "prompt": "Generate questions about network resilience.",
            }
        ]

        def mock_llm(prompt):
            return '''
            {
                "questions": [
                    {"question": "How does topology affect resilience?", "style": "causal"},
                    {"question": "What factors influence network stability?", "style": "exploratory"}
                ]
            }
            '''

        config = ResearchQuestionConfig()
        result = generate_questions_from_prompts(prompts, mock_llm, config)

        assert result.total_generated == 2
        assert len(result.questions) >= 1
        assert result.duplicates_removed >= 0
        assert result.generation_errors == []

    def test_handles_llm_errors(self):
        """Captures errors from LLM calls."""
        prompts = [
            {
                "type": "topic",
                "gap": {"label": "test"},
                "prompt": "test prompt",
            }
        ]

        def failing_llm(prompt):
            raise ValueError("API error")

        config = ResearchQuestionConfig()
        result = generate_questions_from_prompts(prompts, failing_llm, config)

        assert len(result.generation_errors) == 1
        assert "API error" in result.generation_errors[0]


class TestFormatQuestionsMarkdown:
    """Tests for markdown output formatting."""

    def test_formats_result_as_markdown(self):
        """Formats generation result as readable markdown."""
        from src.analysis.research_questions import GenerationResult

        result = GenerationResult(
            questions=[
                GeneratedQuestion(
                    question="How does X affect Y?",
                    style="causal",
                    gap_type="topic",
                    gap_label="test topic",
                    rationale="Important question",
                    combined_score=0.75,
                )
            ],
            total_generated=2,
            duplicates_removed=1,
            generation_errors=[],
        )
        markdown = format_questions_markdown(result)

        assert "# Generated Research Questions" in markdown
        assert "Total generated:** 2" in markdown
        assert "Duplicates removed:** 1" in markdown
        assert "How does X affect Y?" in markdown
        assert "causal" in markdown
        assert "test topic" in markdown
