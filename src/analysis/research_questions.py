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
