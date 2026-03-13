"""RAPTOR hierarchical summarization for multi-granularity search.

Generates 2 summary layers per paper from SemanticAnalysis q-field data:
1. Paper overview (~100-150 words): Abstract-level synthesis
2. Core contribution (1 sentence): Single-sentence distillation

These layers are embedded as separate chunks, enabling search to match
at different levels of abstraction -- a broad topic scan hits the core
contribution, while a more detailed query hits the paper overview.

Inspired by RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval).
"""

import json
from dataclasses import dataclass

from src.analysis.schemas import SemanticAnalysis
from src.utils.logging_config import get_logger
from src.zotero.models import PaperMetadata

logger = get_logger(__name__)

_RAPTOR_PROMPT = """\
You are a research librarian creating multi-level summaries of academic papers.

Given the semantic analysis below, generate exactly 2 summary layers:

**Paper**: {title} ({year})
**Authors**: {authors}

**Semantic Analysis**:
- Thesis: {thesis}
- Contribution: {contribution}
- Framework: {framework}
- Research Question: {research_question}
- Methods: {methods}
- Key Claims: {key_claims}
- Evidence: {evidence}
- Implications: {implications}
- Limitations: {limitations}
- Future Work: {future_work}
- Field: {field}
- Paradigm: {paradigm}

Generate a JSON object with exactly two fields:

1. "paper_overview": A concise 100-150 word abstract-level overview capturing the \
paper's purpose, approach, and main contribution. Similar in style to a journal abstract.

2. "core_contribution": A single sentence (max 40 words) capturing the paper's most \
important contribution or finding.

Return ONLY the JSON object, no other text."""


@dataclass
class RaptorSummaries:
    """Two-layer RAPTOR summaries for a single paper."""

    paper_id: str
    paper_overview: str
    core_contribution: str


def _format_analysis_for_prompt(
    paper: PaperMetadata,
    analysis: SemanticAnalysis,
) -> dict[str, str]:
    """Format SemanticAnalysis fields into prompt template values.

    Args:
        paper: Paper metadata.
        analysis: SemanticAnalysis result.

    Returns:
        Dict of template field names to formatted values.
    """
    return {
        "title": paper.title or "Unknown",
        "year": str(paper.publication_year or "n/a"),
        "authors": paper.author_string or "Unknown",
        "thesis": analysis.q02_thesis or "(not extracted)",
        "contribution": analysis.q22_contribution or "(not extracted)",
        "framework": analysis.q10_framework or "(not specified)",
        "research_question": analysis.q01_research_question or "(none)",
        "methods": analysis.q07_methods or "(not extracted)",
        "key_claims": analysis.q03_key_claims or "(not extracted)",
        "evidence": analysis.q04_evidence or "(not extracted)",
        "implications": analysis.q19_implications or "(not extracted)",
        "limitations": analysis.q05_limitations or "(none noted)",
        "future_work": analysis.q20_future_work or "(none noted)",
        "field": analysis.q17_field or "(not specified)",
        "paradigm": analysis.q06_paradigm or "(not specified)",
    }


def generate_raptor_summaries(
    paper: PaperMetadata,
    analysis: SemanticAnalysis,
    provider: str = "anthropic",
    model: str | None = None,
) -> RaptorSummaries | None:
    """Generate 2-layer RAPTOR summaries for a paper using LLM.

    Args:
        paper: Paper metadata.
        analysis: SemanticAnalysis for the paper.
        provider: LLM provider for summarization.
        model: LLM model. Defaults to provider's default.

    Returns:
        RaptorSummaries with 2 layers, or None if generation fails.
    """
    from src.analysis.llm_factory import create_llm_client

    template_values = _format_analysis_for_prompt(paper, analysis)
    prompt = _RAPTOR_PROMPT.format(**template_values)

    client = create_llm_client(
        provider=provider,
        mode="api",
        model=model,
        max_tokens=2048,
        timeout=60,
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
            logger.warning(f"RAPTOR response not a dict for {paper.paper_id}")
            return None

        paper_overview = parsed.get("paper_overview", "")
        core_contribution = parsed.get("core_contribution", "")

        if not paper_overview or not core_contribution:
            logger.warning(f"RAPTOR response missing fields for {paper.paper_id}")
            return None

        return RaptorSummaries(
            paper_id=paper.paper_id,
            paper_overview=str(paper_overview),
            core_contribution=str(core_contribution),
        )

    except json.JSONDecodeError as e:
        logger.warning(f"RAPTOR JSON parse error for {paper.paper_id}: {e}")
        return None
    except Exception as e:
        logger.warning(f"RAPTOR generation failed for {paper.paper_id}: {e}")
        return None


def generate_raptor_batch(
    papers: list[PaperMetadata],
    analyses: dict[str, SemanticAnalysis],
    provider: str = "anthropic",
    model: str | None = None,
) -> dict[str, RaptorSummaries]:
    """Generate RAPTOR summaries for a batch of papers.

    Args:
        papers: List of paper metadata.
        analyses: Dict of paper_id -> SemanticAnalysis.
        provider: LLM provider.
        model: LLM model.

    Returns:
        Dict of paper_id -> RaptorSummaries (only successful generations).
    """
    results = {}
    total = len(papers)

    for i, paper in enumerate(papers):
        if paper.paper_id not in analyses:
            continue

        logger.info(f"RAPTOR [{i + 1}/{total}] Generating summaries for: {paper.title}")
        summaries = generate_raptor_summaries(
            paper=paper,
            analysis=analyses[paper.paper_id],
            provider=provider,
            model=model,
        )
        if summaries:
            results[paper.paper_id] = summaries
        else:
            logger.warning(f"RAPTOR [{i + 1}/{total}] Failed for: {paper.title}")

    logger.info(f"RAPTOR complete: {len(results)}/{total} papers summarized")
    return results
