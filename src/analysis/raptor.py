"""RAPTOR hierarchical summarization for multi-granularity search.

Generates 3 summary layers per paper from existing LLM extraction data:
1. Section summary (~300-500 words): Coherent narrative covering all extraction fields
2. Paper overview (~100-150 words): Abstract-level synthesis
3. Core contribution (1 sentence): Single-sentence distillation

These layers are embedded as separate chunks, enabling search to match
at different levels of abstraction -- a detailed methodology query hits
the section summary, while a broad topic scan hits the core contribution.

Inspired by RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval).
"""

import json
from dataclasses import dataclass

from src.analysis.schemas import PaperExtraction
from src.utils.logging_config import get_logger
from src.zotero.models import PaperMetadata

logger = get_logger(__name__)

_RAPTOR_PROMPT = """\
You are a research librarian creating multi-level summaries of academic papers.

Given the structured extraction below, generate exactly 3 summary layers:

**Paper**: {title} ({year})
**Authors**: {authors}

**Extraction Data**:
- Thesis: {thesis}
- Contribution: {contribution}
- Theoretical Framework: {framework}
- Research Questions: {research_questions}
- Methodology: {methodology}
- Key Findings: {findings}
- Conclusions: {conclusions}
- Limitations: {limitations}
- Future Directions: {future_directions}

Generate a JSON object with exactly three fields:

1. "section_summary": A coherent 300-500 word narrative synthesizing ALL extraction \
fields into a detailed summary. Cover the research problem, methodology, findings, \
contributions, and limitations in flowing prose. Do not use bullet points.

2. "paper_overview": A concise 100-150 word abstract-level overview capturing the \
paper's purpose, approach, and main contribution. Similar in style to a journal abstract.

3. "core_contribution": A single sentence (max 40 words) capturing the paper's most \
important contribution or finding.

Return ONLY the JSON object, no other text."""


@dataclass
class RaptorSummaries:
    """Three-layer RAPTOR summaries for a single paper."""

    paper_id: str
    section_summary: str
    paper_overview: str
    core_contribution: str


def _format_extraction_for_prompt(
    paper: PaperMetadata,
    extraction: PaperExtraction,
) -> dict[str, str]:
    """Format extraction fields into prompt template values.

    Args:
        paper: Paper metadata.
        extraction: LLM extraction result.

    Returns:
        Dict of template field names to formatted values.
    """
    # Methodology
    method_parts = []
    if extraction.methodology:
        m = extraction.methodology
        if m.approach:
            method_parts.append(f"Approach: {m.approach}")
        if m.design:
            method_parts.append(f"Design: {m.design}")
        if m.data_sources:
            method_parts.append(f"Data sources: {', '.join(m.data_sources)}")
        if m.analysis_methods:
            method_parts.append(f"Analysis: {', '.join(m.analysis_methods)}")
        if m.sample_size:
            method_parts.append(f"Sample: {m.sample_size}")

    # Findings
    findings_parts = []
    if extraction.key_findings:
        findings_parts = [f.finding for f in extraction.key_findings]

    return {
        "title": paper.title or "Unknown",
        "year": str(paper.publication_year or "n/a"),
        "authors": paper.author_string or "Unknown",
        "thesis": extraction.thesis_statement or "(not extracted)",
        "contribution": extraction.contribution_summary or "(not extracted)",
        "framework": extraction.theoretical_framework or "(not specified)",
        "research_questions": "; ".join(extraction.research_questions) if extraction.research_questions else "(none)",
        "methodology": "; ".join(method_parts) if method_parts else "(not extracted)",
        "findings": "; ".join(findings_parts) if findings_parts else "(not extracted)",
        "conclusions": extraction.conclusions or "(not extracted)",
        "limitations": "; ".join(extraction.limitations) if extraction.limitations else "(none noted)",
        "future_directions": "; ".join(extraction.future_directions) if extraction.future_directions else "(none noted)",
    }


def generate_raptor_summaries(
    paper: PaperMetadata,
    extraction: PaperExtraction,
    provider: str = "anthropic",
    model: str | None = None,
) -> RaptorSummaries | None:
    """Generate 3-layer RAPTOR summaries for a paper using LLM.

    Args:
        paper: Paper metadata.
        extraction: LLM extraction for the paper.
        provider: LLM provider for summarization.
        model: LLM model. Defaults to provider's default.

    Returns:
        RaptorSummaries with 3 layers, or None if generation fails.
    """
    from src.analysis.llm_factory import create_llm_client

    template_values = _format_extraction_for_prompt(paper, extraction)
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

        section_summary = parsed.get("section_summary", "")
        paper_overview = parsed.get("paper_overview", "")
        core_contribution = parsed.get("core_contribution", "")

        if not section_summary or not paper_overview or not core_contribution:
            logger.warning(f"RAPTOR response missing fields for {paper.paper_id}")
            return None

        return RaptorSummaries(
            paper_id=paper.paper_id,
            section_summary=str(section_summary),
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
    extractions: dict[str, PaperExtraction],
    provider: str = "anthropic",
    model: str | None = None,
) -> dict[str, RaptorSummaries]:
    """Generate RAPTOR summaries for a batch of papers.

    Args:
        papers: List of paper metadata.
        extractions: Dict of paper_id -> PaperExtraction.
        provider: LLM provider.
        model: LLM model.

    Returns:
        Dict of paper_id -> RaptorSummaries (only successful generations).
    """
    results = {}
    total = len(papers)

    for i, paper in enumerate(papers):
        if paper.paper_id not in extractions:
            continue

        logger.info(f"RAPTOR [{i + 1}/{total}] Generating summaries for: {paper.title}")
        summaries = generate_raptor_summaries(
            paper=paper,
            extraction=extractions[paper.paper_id],
            provider=provider,
            model=model,
        )
        if summaries:
            results[paper.paper_id] = summaries
        else:
            logger.warning(f"RAPTOR [{i + 1}/{total}] Failed for: {paper.title}")

    logger.info(f"RAPTOR complete: {len(results)}/{total} papers summarized")
    return results
