"""Multi-round agentic search with gap analysis.

Implements an iterative search loop:
1. Initial query -> vector search
2. LLM analyzes results for topical gaps
3. LLM generates follow-up queries to fill gaps
4. Re-search with follow-up queries and merge results
5. Repeat for up to max_rounds gap-analysis rounds
"""

import json
from dataclasses import dataclass, field

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

_GAP_ANALYSIS_PROMPT = """\
You are a research librarian analyzing search results from an academic literature database.

The user searched for: "{query}"

Here are the papers found so far:
{results_summary}

Analyze these results and identify what is MISSING. Consider:
- Are there important sub-topics or perspectives not represented?
- Are key methodological approaches absent?
- Are there related theoretical frameworks that should appear?
- Is the coverage temporally or disciplinarily skewed?

Return a JSON object with exactly two fields:
- "gaps": A list of 1-3 brief descriptions of what is missing
- "follow_up_queries": A list of 1-3 search queries that would find the missing papers

If the results comprehensively cover the topic, return:
{{"gaps": [], "follow_up_queries": []}}

Return ONLY the JSON object, no other text."""


@dataclass
class GapAnalysis:
    """Result of analyzing search results for topical gaps."""

    gaps: list[str]
    follow_up_queries: list[str]


@dataclass
class AgenticRound:
    """Metadata for one round of agentic search."""

    round_number: int
    queries_used: list[str]
    papers_found: int
    new_papers: int
    gap_analysis: GapAnalysis | None = None


@dataclass
class AgenticSearchResult:
    """Complete result of a multi-round agentic search."""

    original_query: str
    rounds: list[AgenticRound] = field(default_factory=list)
    total_papers: int = 0


def _format_results_for_analysis(results: list[dict]) -> str:
    """Format search results into a concise summary for LLM gap analysis.

    Args:
        results: List of enriched result dicts with title, authors, year, etc.

    Returns:
        Formatted string summarizing found papers.
    """
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "Unknown")
        year = r.get("year", "n/a")
        authors = r.get("authors", "Unknown")
        thesis = ""
        extraction = r.get("extraction", {})
        if extraction:
            thesis = extraction.get("q02_thesis", "")
        disciplines = [extraction.get("q17_field", "")] if extraction and extraction.get("q17_field") else []

        line = f"{i}. [{year}] {title} ({authors})"
        if thesis:
            line += f"\n   Thesis: {thesis[:200]}"
        if disciplines:
            line += f"\n   Disciplines: {', '.join(disciplines[:5])}"
        lines.append(line)

    return "\n".join(lines) if lines else "(no results found)"


def analyze_gaps(
    query: str,
    results: list[dict],
    provider: str = "anthropic",
    model: str | None = None,
) -> GapAnalysis:
    """Analyze search results for topical gaps and generate follow-up queries.

    Args:
        query: The original search query.
        results: Formatted search results to analyze.
        provider: LLM provider for gap analysis.
        model: LLM model. Defaults to provider's default.

    Returns:
        GapAnalysis with identified gaps and follow-up queries.
    """
    from src.analysis.llm_factory import create_llm_client

    results_summary = _format_results_for_analysis(results)
    prompt = _GAP_ANALYSIS_PROMPT.format(
        query=query,
        results_summary=results_summary,
    )

    client = create_llm_client(
        provider=provider,
        mode="api",
        model=model,
        max_tokens=1024,
        timeout=30,
    )

    try:
        response_text, _, _ = client._call_api(prompt)

        # Strip markdown code fences if present
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            logger.warning("Gap analysis returned non-dict, skipping")
            return GapAnalysis(gaps=[], follow_up_queries=[])

        gaps = parsed.get("gaps", [])
        follow_ups = parsed.get("follow_up_queries", [])

        # Validate types
        gaps = [str(g) for g in gaps if g][:3]
        follow_ups = [str(q) for q in follow_ups if q][:3]

        return GapAnalysis(gaps=gaps, follow_up_queries=follow_ups)

    except Exception as e:
        logger.warning(f"Gap analysis failed: {e}. No follow-up queries.")
        return GapAnalysis(gaps=[], follow_up_queries=[])
