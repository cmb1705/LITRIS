"""Research question generation from gap analysis outputs.

This module provides prompt templates and configuration for generating
research questions based on identified literature gaps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QuestionStyle(str, Enum):
    """Research question framing styles."""

    EXPLORATORY = "exploratory"  # What, how, why questions
    COMPARATIVE = "comparative"  # How does X compare to Y
    CAUSAL = "causal"  # What is the effect of X on Y
    DESCRIPTIVE = "descriptive"  # What is the state of X
    EVALUATIVE = "evaluative"  # How effective is X


class QuestionScope(str, Enum):
    """Scope constraints for generated questions."""

    NARROW = "narrow"  # Highly specific, single-study feasible
    MODERATE = "moderate"  # Multi-study or systematic review
    BROAD = "broad"  # Research program or agenda level


@dataclass
class ResearchQuestionConfig:
    """Configuration for research question generation.

    Attributes:
        count: Number of questions to generate per gap category.
        styles: Allowed question styles (empty = all styles).
        scope: Scope constraint for questions.
        include_rationale: Include brief rationale with each question.
        include_methodology_hints: Suggest potential methodological approaches.
        max_tokens: Maximum tokens for LLM response.
        discipline_focus: Optional discipline filter for question framing.
    """

    count: int = 3
    styles: list[QuestionStyle] = field(default_factory=list)
    scope: QuestionScope = QuestionScope.MODERATE
    include_rationale: bool = True
    include_methodology_hints: bool = False
    max_tokens: int = 2000
    discipline_focus: str | None = None


# Guardrails for quality and safety
GUARDRAILS = """
## Quality Guardrails

1. **Specificity**: Questions must be specific enough to guide empirical research.
   - BAD: "What about machine learning?"
   - GOOD: "How do transformer architectures compare to GNNs for citation prediction?"

2. **Novelty**: Questions should address genuine gaps, not well-established areas.
   - Verify the gap claim against the evidence papers provided.

3. **Feasibility**: Questions should be answerable with available methods.
   - Consider data availability and methodological constraints.

4. **Ethical Boundaries**: Avoid questions that:
   - Could enable harmful applications
   - Involve protected populations without clear benefit
   - Require unethical data collection

5. **Scope Alignment**: Match question complexity to the specified scope level.

## Output Validation

Generated questions should:
- Be phrased as actual questions (end with ?)
- Reference specific concepts from the gap analysis
- Not duplicate existing research questions from indexed papers
"""


TOPIC_GAP_TEMPLATE = """Based on the following underrepresented topic in the literature:

Topic: {topic_label}
Occurrence Count: {count}
Evidence Papers:
{evidence}

{style_instruction}

Generate {question_count} research question(s) that could address this gap.
{scope_instruction}
{rationale_instruction}
{methodology_instruction}

{guardrails}

Output Format (JSON):
{{
  "questions": [
    {{
      "question": "The research question text",
      "style": "{style}",
      "rationale": "Why this question matters (if requested)",
      "methodology_hints": ["Potential approaches (if requested)"]
    }}
  ]
}}
"""


METHODOLOGY_GAP_TEMPLATE = """Based on the following underrepresented methodology in the literature:

Methodology: {methodology_label}
Occurrence Count: {count}
Evidence Papers:
{evidence}

{style_instruction}

Generate {question_count} research question(s) exploring applications or variations of this methodology.
{scope_instruction}
{rationale_instruction}
{methodology_instruction}

{guardrails}

Output Format (JSON):
{{
  "questions": [
    {{
      "question": "The research question text",
      "style": "{style}",
      "rationale": "Why this question matters (if requested)",
      "methodology_hints": ["Potential approaches (if requested)"]
    }}
  ]
}}
"""


FUTURE_DIRECTION_TEMPLATE = """Based on the following future research direction identified in multiple papers:

Direction: {direction}
Mentioned by: {mention_count} paper(s)
Current Coverage: {coverage_count} paper(s) addressing this
Evidence Papers:
{evidence}

{style_instruction}

Generate {question_count} specific research question(s) that operationalize this future direction.
{scope_instruction}
{rationale_instruction}
{methodology_instruction}

{guardrails}

Output Format (JSON):
{{
  "questions": [
    {{
      "question": "The research question text",
      "style": "{style}",
      "rationale": "Why this question matters (if requested)",
      "methodology_hints": ["Potential approaches (if requested)"]
    }}
  ]
}}
"""


YEAR_GAP_TEMPLATE = """Based on the following temporal gap in the literature:

Time Period: {period}
Gap Type: {gap_type}
Context: Literature coverage spans {min_year} to {max_year}

{style_instruction}

Generate {question_count} research question(s) that could address why this period is underrepresented
or what developments during this period merit investigation.
{scope_instruction}
{rationale_instruction}
{methodology_instruction}

{guardrails}

Output Format (JSON):
{{
  "questions": [
    {{
      "question": "The research question text",
      "style": "{style}",
      "rationale": "Why this question matters (if requested)",
      "methodology_hints": ["Potential approaches (if requested)"]
    }}
  ]
}}
"""


def _format_evidence(evidence: list[dict]) -> str:
    """Format evidence papers for prompt inclusion."""
    if not evidence:
        return "- No specific evidence papers available"
    lines = []
    for paper in evidence:
        title = paper.get("title", "Unknown")
        year = paper.get("year")
        year_str = f" ({year})" if year else ""
        lines.append(f"- {title}{year_str}")
    return "\n".join(lines)


def _get_style_instruction(config: ResearchQuestionConfig) -> str:
    """Generate style-specific instruction."""
    if not config.styles:
        return "Use any appropriate question framing style."

    style_descriptions = {
        QuestionStyle.EXPLORATORY: "exploratory (what, how, why)",
        QuestionStyle.COMPARATIVE: "comparative (X vs Y)",
        QuestionStyle.CAUSAL: "causal (effect of X on Y)",
        QuestionStyle.DESCRIPTIVE: "descriptive (state of X)",
        QuestionStyle.EVALUATIVE: "evaluative (effectiveness of X)",
    }
    allowed = [style_descriptions[s] for s in config.styles]
    return f"Use one of these question styles: {', '.join(allowed)}."


def _get_scope_instruction(config: ResearchQuestionConfig) -> str:
    """Generate scope-specific instruction."""
    scope_descriptions = {
        QuestionScope.NARROW: "Keep questions narrow and feasible for a single empirical study.",
        QuestionScope.MODERATE: "Frame questions suitable for a multi-study project or systematic review.",
        QuestionScope.BROAD: "Frame questions at the research program or agenda level.",
    }
    return scope_descriptions[config.scope]


def build_topic_gap_prompt(gap: dict, config: ResearchQuestionConfig) -> str:
    """Build prompt for generating questions from a topic gap.

    Args:
        gap: Topic gap dictionary from gap analysis.
        config: Generation configuration.

    Returns:
        Formatted prompt string.
    """
    style = config.styles[0].value if config.styles else "exploratory"
    return TOPIC_GAP_TEMPLATE.format(
        topic_label=gap.get("label", "Unknown"),
        count=gap.get("count", 0),
        evidence=_format_evidence(gap.get("evidence", [])),
        style_instruction=_get_style_instruction(config),
        question_count=config.count,
        scope_instruction=_get_scope_instruction(config),
        rationale_instruction="Include a brief rationale for each question."
        if config.include_rationale
        else "",
        methodology_instruction="Suggest potential methodological approaches."
        if config.include_methodology_hints
        else "",
        guardrails=GUARDRAILS,
        style=style,
    )


def build_methodology_gap_prompt(gap: dict, config: ResearchQuestionConfig) -> str:
    """Build prompt for generating questions from a methodology gap.

    Args:
        gap: Methodology gap dictionary from gap analysis.
        config: Generation configuration.

    Returns:
        Formatted prompt string.
    """
    style = config.styles[0].value if config.styles else "exploratory"
    return METHODOLOGY_GAP_TEMPLATE.format(
        methodology_label=gap.get("label", "Unknown"),
        count=gap.get("count", 0),
        evidence=_format_evidence(gap.get("evidence", [])),
        style_instruction=_get_style_instruction(config),
        question_count=config.count,
        scope_instruction=_get_scope_instruction(config),
        rationale_instruction="Include a brief rationale for each question."
        if config.include_rationale
        else "",
        methodology_instruction="Suggest potential methodological approaches."
        if config.include_methodology_hints
        else "",
        guardrails=GUARDRAILS,
        style=style,
    )


def build_future_direction_prompt(gap: dict, config: ResearchQuestionConfig) -> str:
    """Build prompt for generating questions from a future direction gap.

    Args:
        gap: Future direction gap dictionary from gap analysis.
        config: Generation configuration.

    Returns:
        Formatted prompt string.
    """
    style = config.styles[0].value if config.styles else "exploratory"
    return FUTURE_DIRECTION_TEMPLATE.format(
        direction=gap.get("direction", "Unknown"),
        mention_count=gap.get("mention_count", 0),
        coverage_count=gap.get("coverage_count", 0),
        evidence=_format_evidence(gap.get("evidence", [])),
        style_instruction=_get_style_instruction(config),
        question_count=config.count,
        scope_instruction=_get_scope_instruction(config),
        rationale_instruction="Include a brief rationale for each question."
        if config.include_rationale
        else "",
        methodology_instruction="Suggest potential methodological approaches."
        if config.include_methodology_hints
        else "",
        guardrails=GUARDRAILS,
        style=style,
    )


def build_year_gap_prompt(
    year_gaps: dict, config: ResearchQuestionConfig
) -> str | None:
    """Build prompt for generating questions from temporal gaps.

    Args:
        year_gaps: Year gaps dictionary from gap analysis.
        config: Generation configuration.

    Returns:
        Formatted prompt string, or None if no significant gaps.
    """
    missing = year_gaps.get("missing_ranges", [])
    if not missing:
        return None

    largest_gap = missing[0]  # Already sorted by length
    period = f"{largest_gap['start']}-{largest_gap['end']}"
    gap_type = "missing coverage" if largest_gap["length"] > 1 else "single missing year"

    style = config.styles[0].value if config.styles else "exploratory"
    return YEAR_GAP_TEMPLATE.format(
        period=period,
        gap_type=gap_type,
        min_year=year_gaps.get("min_year", "Unknown"),
        max_year=year_gaps.get("max_year", "Unknown"),
        style_instruction=_get_style_instruction(config),
        question_count=config.count,
        scope_instruction=_get_scope_instruction(config),
        rationale_instruction="Include a brief rationale for each question."
        if config.include_rationale
        else "",
        methodology_instruction="Suggest potential methodological approaches."
        if config.include_methodology_hints
        else "",
        guardrails=GUARDRAILS,
        style=style,
    )


def build_prompts_from_gap_report(
    report: dict, config: ResearchQuestionConfig
) -> list[dict[str, Any]]:
    """Build all prompts from a complete gap analysis report.

    Args:
        report: Gap analysis report dictionary.
        config: Generation configuration.

    Returns:
        List of prompt dictionaries with type and prompt text.
    """
    prompts = []

    for gap in report.get("topics_underrepresented", []):
        prompts.append({
            "type": "topic",
            "gap": gap,
            "prompt": build_topic_gap_prompt(gap, config),
        })

    for gap in report.get("methodologies_underrepresented", []):
        prompts.append({
            "type": "methodology",
            "gap": gap,
            "prompt": build_methodology_gap_prompt(gap, config),
        })

    for gap in report.get("future_directions", []):
        prompts.append({
            "type": "future_direction",
            "gap": gap,
            "prompt": build_future_direction_prompt(gap, config),
        })

    year_gaps = report.get("year_gaps", {})
    year_prompt = build_year_gap_prompt(year_gaps, config)
    if year_prompt:
        prompts.append({
            "type": "year_gap",
            "gap": year_gaps,
            "prompt": year_prompt,
        })

    return prompts


# Generation pipeline components


@dataclass
class GeneratedQuestion:
    """A generated research question with provenance."""

    question: str
    style: str
    gap_type: str
    gap_label: str
    rationale: str | None = None
    methodology_hints: list[str] = field(default_factory=list)
    relevance_score: float = 0.0
    diversity_score: float = 0.0
    combined_score: float = 0.0


@dataclass
class GenerationResult:
    """Result of research question generation."""

    questions: list[GeneratedQuestion]
    total_generated: int
    duplicates_removed: int
    generation_errors: list[str]


def parse_llm_response(
    response_text: str, gap_type: str, gap_label: str
) -> list[GeneratedQuestion]:
    """Parse LLM JSON response into GeneratedQuestion objects.

    Args:
        response_text: Raw LLM response (should be JSON).
        gap_type: Type of gap that spawned this question.
        gap_label: Label/description of the source gap.

    Returns:
        List of parsed GeneratedQuestion objects.
    """
    import json
    import re

    questions = []

    # Try to extract JSON from response (may have markdown wrapping)
    json_match = re.search(r"\{[\s\S]*\}", response_text)
    if not json_match:
        return questions

    try:
        data = json.loads(json_match.group())
        raw_questions = data.get("questions", [])

        for item in raw_questions:
            if not isinstance(item, dict):
                continue
            question_text = item.get("question", "").strip()
            if not question_text or not question_text.endswith("?"):
                continue

            questions.append(
                GeneratedQuestion(
                    question=question_text,
                    style=item.get("style", "exploratory"),
                    gap_type=gap_type,
                    gap_label=gap_label,
                    rationale=item.get("rationale"),
                    methodology_hints=item.get("methodology_hints", []),
                )
            )
    except json.JSONDecodeError:
        pass

    return questions


def _normalize_for_comparison(text: str) -> str:
    """Normalize text for similarity comparison."""
    import re

    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())
    return text


def _jaccard_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard similarity between two texts."""
    words1 = set(_normalize_for_comparison(text1).split())
    words2 = set(_normalize_for_comparison(text2).split())
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)


def deduplicate_questions(
    questions: list[GeneratedQuestion], similarity_threshold: float = 0.7
) -> tuple[list[GeneratedQuestion], int]:
    """Remove duplicate questions based on text similarity.

    Args:
        questions: List of generated questions.
        similarity_threshold: Jaccard similarity above which questions are duplicates.

    Returns:
        Tuple of (deduplicated questions, number removed).
    """
    if not questions:
        return [], 0

    deduplicated = []
    removed = 0

    for q in questions:
        is_duplicate = False
        for existing in deduplicated:
            similarity = _jaccard_similarity(q.question, existing.question)
            if similarity >= similarity_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            deduplicated.append(q)
        else:
            removed += 1

    return deduplicated, removed


def rank_questions(
    questions: list[GeneratedQuestion],
    relevance_weight: float = 0.6,
    diversity_weight: float = 0.4,
) -> list[GeneratedQuestion]:
    """Rank questions by relevance and diversity.

    Relevance is scored by:
    - Question ending with "?" (+0.2)
    - Having rationale (+0.2)
    - Having methodology hints (+0.1)
    - Moderate length (50-200 chars) (+0.2)
    - Style specificity (+0.1-0.3 based on style)

    Diversity is scored by:
    - Unique gap type compared to neighbors
    - Unique style compared to neighbors

    Args:
        questions: List of questions to rank.
        relevance_weight: Weight for relevance score (0-1).
        diversity_weight: Weight for diversity score (0-1).

    Returns:
        Questions sorted by combined score (highest first).
    """
    if not questions:
        return []

    # Score relevance
    for q in questions:
        score = 0.3  # Base score
        if q.question.endswith("?"):
            score += 0.2
        if q.rationale:
            score += 0.2
        if q.methodology_hints:
            score += 0.1
        length = len(q.question)
        if 50 <= length <= 200:
            score += 0.2
        elif 30 <= length <= 250:
            score += 0.1
        # Style bonus
        style_scores = {
            "causal": 0.15,
            "comparative": 0.12,
            "evaluative": 0.10,
            "exploratory": 0.05,
            "descriptive": 0.05,
        }
        score += style_scores.get(q.style, 0.0)
        q.relevance_score = min(score, 1.0)

    # Score diversity (based on variety of gap types and styles)
    gap_types = [q.gap_type for q in questions]
    styles = [q.style for q in questions]
    gap_type_counts = {t: gap_types.count(t) for t in set(gap_types)}
    style_counts = {s: styles.count(s) for s in set(styles)}

    for q in questions:
        # Rarer gap types and styles get higher diversity scores
        gap_rarity = 1.0 / gap_type_counts[q.gap_type]
        style_rarity = 1.0 / style_counts[q.style]
        q.diversity_score = (gap_rarity + style_rarity) / 2

    # Compute combined score
    for q in questions:
        q.combined_score = (
            relevance_weight * q.relevance_score
            + diversity_weight * q.diversity_score
        )

    # Sort by combined score
    return sorted(questions, key=lambda q: q.combined_score, reverse=True)


def generate_questions_from_prompts(
    prompts: list[dict[str, Any]],
    llm_caller: Any,  # Callable that takes prompt and returns response text
    config: ResearchQuestionConfig,
    similarity_threshold: float = 0.7,
) -> GenerationResult:
    """Generate research questions by calling LLM with prompts.

    Args:
        prompts: List of prompt dicts from build_prompts_from_gap_report().
        llm_caller: Callable(prompt: str) -> str that calls the LLM.
        config: Generation configuration.
        similarity_threshold: Threshold for deduplication.

    Returns:
        GenerationResult with ranked, deduplicated questions.
    """
    all_questions: list[GeneratedQuestion] = []
    errors: list[str] = []

    for prompt_dict in prompts:
        prompt_text = prompt_dict["prompt"]
        gap_type = prompt_dict["type"]
        gap = prompt_dict["gap"]

        # Get label based on gap type
        if gap_type == "topic":
            gap_label = gap.get("label", "Unknown topic")
        elif gap_type == "methodology":
            gap_label = gap.get("label", "Unknown methodology")
        elif gap_type == "future_direction":
            gap_label = gap.get("direction", "Unknown direction")
        elif gap_type == "year_gap":
            ranges = gap.get("missing_ranges", [])
            if ranges:
                gap_label = f"Years {ranges[0].get('start')}-{ranges[0].get('end')}"
            else:
                gap_label = "Unknown year gap"
        else:
            gap_label = "Unknown"

        try:
            response_text = llm_caller(prompt_text)
            parsed = parse_llm_response(response_text, gap_type, gap_label)
            all_questions.extend(parsed)
        except Exception as e:
            errors.append(f"Error generating for {gap_type} '{gap_label}': {e}")

    total_generated = len(all_questions)

    # Deduplicate
    deduplicated, removed = deduplicate_questions(all_questions, similarity_threshold)

    # Rank
    ranked = rank_questions(deduplicated)

    return GenerationResult(
        questions=ranked,
        total_generated=total_generated,
        duplicates_removed=removed,
        generation_errors=errors,
    )


def format_questions_markdown(result: GenerationResult) -> str:
    """Format generation result as Markdown.

    Args:
        result: GenerationResult to format.

    Returns:
        Markdown string.
    """
    lines = [
        "# Generated Research Questions",
        "",
        f"**Total generated:** {result.total_generated}",
        f"**Duplicates removed:** {result.duplicates_removed}",
        f"**Final questions:** {len(result.questions)}",
        "",
    ]

    if result.generation_errors:
        lines.extend([
            "## Errors",
            "",
        ])
        for err in result.generation_errors:
            lines.append(f"- {err}")
        lines.append("")

    lines.extend([
        "## Ranked Questions",
        "",
    ])

    for i, q in enumerate(result.questions, 1):
        lines.append(f"### {i}. {q.question}")
        lines.append("")
        lines.append(f"- **Gap Type:** {q.gap_type}")
        lines.append(f"- **Gap Label:** {q.gap_label}")
        lines.append(f"- **Style:** {q.style}")
        lines.append(f"- **Score:** {q.combined_score:.2f}")
        if q.rationale:
            lines.append(f"- **Rationale:** {q.rationale}")
        if q.methodology_hints:
            lines.append(f"- **Methodology Hints:** {', '.join(q.methodology_hints)}")
        lines.append("")

    return "\n".join(lines)
