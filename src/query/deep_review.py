"""Deep Review literature synthesis pipeline.

4-phase pipeline:
1. Discovery -- multi-round agentic search for comprehensive paper retrieval
2. Reading -- load full extractions for discovered papers
3. Synthesis -- LLM generates integrated literature review
4. QA Verification -- cross-reference citations against source papers
"""

import json
from dataclasses import dataclass, field
from datetime import datetime

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

_SYNTHESIS_PROMPT = """\
You are an expert academic researcher writing an integrated literature review.

TOPIC: {topic}

Below are structured extractions from {n_papers} relevant papers. Write a \
comprehensive, integrated literature review of 3000-5000 words that:

1. Opens with a clear framing of the research landscape
2. Organizes findings thematically (NOT paper-by-paper)
3. Identifies areas of consensus and tension between studies
4. Notes methodological patterns and limitations
5. Highlights gaps and future research directions
6. Uses inline citations in (Author, Year) format

IMPORTANT RULES:
- Synthesize across papers, do not merely summarize each one
- Every factual claim must cite the source paper(s)
- Note when papers disagree or offer competing explanations
- Identify methodological trends (qualitative vs quantitative, etc.)
- End with a clear summary of the state of knowledge and open questions

PAPER EXTRACTIONS:
{papers_text}

Write the literature review now. Use markdown formatting with section headers."""

_QA_PROMPT = """\
You are a citation verification assistant. Review this literature review and \
check that every citation matches the source papers provided.

LITERATURE REVIEW:
{review_text}

SOURCE PAPERS (author, year, title, thesis):
{papers_reference}

Check for:
1. Citations that do not match any source paper (hallucinated citations)
2. Claims attributed to the wrong paper
3. Factual statements without any citation
4. Papers in the source list that are not cited but should be

Return a JSON object with:
- "verified": true/false (true if no issues found)
- "issues": list of issue descriptions (empty if verified)
- "uncited_papers": list of paper IDs that should probably be cited
- "citation_count": number of inline citations found

Return ONLY the JSON object."""


@dataclass
class PaperReading:
    """Extracted content for one paper used in synthesis."""

    paper_id: str
    title: str
    authors: str
    year: int | None
    thesis: str
    methodology: str
    key_findings: list[str]
    conclusions: str
    limitations: list[str]
    disciplines: list[str]

    def to_text(self) -> str:
        """Format as text block for the synthesis prompt."""
        findings_str = "\n".join(f"  - {f}" for f in self.key_findings[:8])
        limitations_str = "\n".join(f"  - {lim}" for lim in self.limitations[:5])

        return (
            f"--- {self.title} ({self.authors}, {self.year or 'n.d.'}) ---\n"
            f"Thesis: {self.thesis}\n"
            f"Methodology: {self.methodology}\n"
            f"Key Findings:\n{findings_str}\n"
            f"Conclusions: {self.conclusions}\n"
            f"Limitations:\n{limitations_str}\n"
            f"Disciplines: {', '.join(self.disciplines[:5])}\n"
        )


@dataclass
class QAResult:
    """Result of citation verification."""

    verified: bool
    issues: list[str]
    uncited_papers: list[str]
    citation_count: int


@dataclass
class DeepReviewResult:
    """Complete result of the deep review pipeline."""

    topic: str
    papers_discovered: int
    papers_used: int
    review_text: str
    qa_result: QAResult | None
    paper_readings: list[PaperReading]
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_markdown(self) -> str:
        """Format the complete review as a markdown document."""
        header = (
            f"# Deep Review: {self.topic}\n\n"
            f"**Generated:** {self.generated_at}\n"
            f"**Papers discovered:** {self.papers_discovered}\n"
            f"**Papers synthesized:** {self.papers_used}\n\n"
        )

        qa_section = ""
        if self.qa_result:
            status = "Passed" if self.qa_result.verified else "Issues Found"
            qa_section = (
                f"\n\n---\n\n## Citation Verification: {status}\n\n"
                f"**Citations found:** {self.qa_result.citation_count}\n\n"
            )
            if self.qa_result.issues:
                qa_section += "**Issues:**\n"
                for issue in self.qa_result.issues:
                    qa_section += f"- {issue}\n"
            if self.qa_result.uncited_papers:
                qa_section += "\n**Uncited papers that may be relevant:**\n"
                for pid in self.qa_result.uncited_papers:
                    qa_section += f"- {pid}\n"

        sources = "\n\n---\n\n## Source Papers\n\n"
        for p in self.paper_readings:
            sources += (
                f"- **{p.title}** ({p.authors}, {p.year or 'n.d.'})"
                f" [ID: {p.paper_id}]\n"
            )

        return header + self.review_text + qa_section + sources


def read_papers(
    paper_results: list,
    adapter,
) -> list[PaperReading]:
    """Phase 2: Read full extractions for discovered papers.

    Args:
        paper_results: EnrichedResult objects from search.
        adapter: LitrisAdapter or SearchEngine with get_paper().

    Returns:
        List of PaperReading objects with structured content.
    """
    readings = []

    for result in paper_results:
        paper_data = adapter.get_paper(result.paper_id)
        if not paper_data or not paper_data.get("found", True):
            continue

        extraction = paper_data.get("extraction", {})
        if not extraction:
            # Use what we have from the search result
            extraction = getattr(result, "extraction_data", {}) or {}
            # Unwrap nested extraction
            if "extraction" in extraction:
                extraction = extraction["extraction"]

        readings.append(PaperReading(
            paper_id=result.paper_id,
            title=result.title,
            authors=result.authors,
            year=result.year,
            thesis=extraction.get("q02_thesis", ""),
            methodology=extraction.get("q07_methods", ""),
            key_findings=[
                claim for claim in [
                    extraction.get("q03_key_claims", ""),
                    extraction.get("q04_evidence", ""),
                ] if claim
            ],
            conclusions=extraction.get("q19_implications", ""),
            limitations=[lim for lim in [extraction.get("q05_limitations", "")] if lim],
            disciplines=[f for f in [extraction.get("q17_field", "")] if f],
        ))

    logger.info(f"Read {len(readings)} papers for synthesis")
    return readings


def synthesize(
    topic: str,
    readings: list[PaperReading],
    provider: str = "anthropic",
    model: str | None = None,
) -> str:
    """Phase 3: Generate integrated literature review via LLM.

    Args:
        topic: Research topic for the review.
        readings: Structured paper readings.
        provider: LLM provider.
        model: LLM model (defaults to provider's default).

    Returns:
        Literature review text in markdown format.
    """
    from src.analysis.llm_factory import create_llm_client

    papers_text = "\n\n".join(r.to_text() for r in readings)

    prompt = _SYNTHESIS_PROMPT.format(
        topic=topic,
        n_papers=len(readings),
        papers_text=papers_text,
    )

    client = create_llm_client(
        provider=provider,
        mode="api",
        model=model,
        max_tokens=8192,
        timeout=120,
    )

    try:
        response_text, _, _ = client._call_api(prompt)
        return response_text.strip()
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        raise


def verify_citations(
    review_text: str,
    readings: list[PaperReading],
    provider: str = "anthropic",
    model: str | None = None,
) -> QAResult:
    """Phase 4: QA verification of citations in the review.

    Args:
        review_text: Generated literature review.
        readings: Source paper readings for cross-reference.
        provider: LLM provider.
        model: LLM model.

    Returns:
        QAResult with verification status and issues.
    """
    from src.analysis.llm_factory import create_llm_client

    papers_reference = "\n".join(
        f"- [{r.paper_id}] {r.authors} ({r.year or 'n.d.'}): "
        f"{r.title}. Thesis: {r.thesis[:150]}"
        for r in readings
    )

    prompt = _QA_PROMPT.format(
        review_text=review_text,
        papers_reference=papers_reference,
    )

    client = create_llm_client(
        provider=provider,
        mode="api",
        model=model,
        max_tokens=2048,
        timeout=60,
    )

    try:
        response_text, _, _ = client._call_api(prompt)

        # Parse JSON response
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        parsed = json.loads(text)

        return QAResult(
            verified=bool(parsed.get("verified", False)),
            issues=[str(i) for i in parsed.get("issues", [])],
            uncited_papers=[str(p) for p in parsed.get("uncited_papers", [])],
            citation_count=int(parsed.get("citation_count", 0)),
        )

    except Exception as e:
        logger.warning(f"QA verification failed: {e}")
        return QAResult(
            verified=False,
            issues=[f"QA verification failed: {e}"],
            uncited_papers=[],
            citation_count=0,
        )


def deep_review(
    topic: str,
    engine,
    adapter,
    top_k: int = 20,
    max_rounds: int = 2,
    verify: bool = True,
    provider: str = "anthropic",
    model: str | None = None,
) -> DeepReviewResult:
    """Execute the full deep review pipeline.

    Args:
        topic: Research topic or question for the review.
        engine: SearchEngine instance.
        adapter: LitrisAdapter instance.
        top_k: Number of papers to include in synthesis.
        max_rounds: Max gap-analysis rounds for discovery.
        verify: Whether to run QA verification (Phase 4).
        provider: LLM provider for synthesis and QA.
        model: LLM model.

    Returns:
        DeepReviewResult with the complete literature review.
    """
    logger.info(f"Starting deep review: '{topic[:50]}...'")

    # Phase 1: Discovery via agentic search
    logger.info("Phase 1: Discovery")
    results, metadata = engine.search_agentic(
        query=topic,
        top_k=top_k,
        max_rounds=max_rounds,
        include_extraction=True,
        provider=provider,
        model=model,
    )

    if not results:
        return DeepReviewResult(
            topic=topic,
            papers_discovered=0,
            papers_used=0,
            review_text="No papers found for this topic.",
            qa_result=None,
            paper_readings=[],
        )

    # Phase 2: Reading
    logger.info(f"Phase 2: Reading {len(results)} papers")
    readings = read_papers(results, adapter)

    if not readings:
        return DeepReviewResult(
            topic=topic,
            papers_discovered=metadata.total_papers,
            papers_used=0,
            review_text="Papers found but no extractions available.",
            qa_result=None,
            paper_readings=[],
        )

    # Phase 3: Synthesis
    logger.info(f"Phase 3: Synthesizing {len(readings)} papers")
    review_text = synthesize(topic, readings, provider=provider, model=model)

    # Phase 4: QA Verification
    qa_result = None
    if verify:
        logger.info("Phase 4: QA Verification")
        qa_result = verify_citations(
            review_text, readings, provider=provider, model=model
        )

    result = DeepReviewResult(
        topic=topic,
        papers_discovered=metadata.total_papers,
        papers_used=len(readings),
        review_text=review_text,
        qa_result=qa_result,
        paper_readings=readings,
    )

    logger.info(
        f"Deep review complete: {result.papers_used} papers synthesized, "
        f"QA {'passed' if qa_result and qa_result.verified else 'skipped/failed'}"
    )
    return result
