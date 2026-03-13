"""Research digest generation for newly indexed papers.

Monitors the index for newly added papers and generates narrative
research summaries suitable for email or chatbot delivery.

Inspired by Minty yesterday-in-ai daemon.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.utils.file_utils import safe_read_json, safe_write_json

logger = logging.getLogger(__name__)

# State file tracks which papers have been included in digests
_DEFAULT_STATE_FILE = "digest_state.json"


@dataclass
class DigestConfig:
    """Configuration for research digest generation."""

    max_papers: int = 10
    include_methodology: bool = True
    include_key_findings: bool = True
    include_relevance_note: bool = True
    state_file: str = _DEFAULT_STATE_FILE


@dataclass
class PaperHighlight:
    """Summary highlight for a single paper."""

    paper_id: str
    title: str
    authors: str
    year: int | None
    collections: list[str]
    summary: str
    methodology: str | None = None
    key_findings: list[str] = field(default_factory=list)
    relevance_note: str | None = None


@dataclass
class ResearchDigest:
    """A research digest containing paper highlights."""

    generated_at: str
    period_start: str | None
    period_end: str
    new_paper_count: int
    highlights: list[PaperHighlight]
    narrative: str = ""


def _load_digest_state(state_path: Path) -> dict:
    """Load the digest state tracking which papers have been processed."""
    state = safe_read_json(state_path, default=None)
    if state is None:
        return {"processed_ids": [], "last_run": None}
    return state


def _save_digest_state(state_path: Path, state: dict) -> bool:
    """Save the digest state. Returns True on success."""
    return safe_write_json(state_path, state)


def find_new_papers(
    index_dir: Path,
    state_path: Path,
    _state: dict | None = None,
) -> list[dict]:
    """Find papers that haven't been included in a digest yet.

    Args:
        index_dir: Path to the index directory.
        state_path: Path to the digest state file.
        _state: Pre-loaded state dict (internal use to avoid double load).

    Returns:
        List of new paper metadata dictionaries.
    """
    if _state is None:
        _state = _load_digest_state(state_path)
    processed = set(_state.get("processed_ids", []))

    papers_data = safe_read_json(index_dir / "papers.json", default={"papers": []})
    if isinstance(papers_data, dict) and "papers" in papers_data:
        papers = papers_data["papers"]
    elif isinstance(papers_data, list):
        papers = papers_data
    else:
        papers = []

    new_papers = [
        p for p in papers
        if p.get("paper_id") and p["paper_id"] not in processed
    ]

    return new_papers


def build_paper_highlight(
    paper: dict,
    extraction: dict | None,
    config: DigestConfig,
) -> PaperHighlight:
    """Build a highlight summary for a single paper.

    Args:
        paper: Paper metadata dictionary.
        extraction: Optional extraction dictionary for the paper.
        config: Digest configuration.

    Returns:
        PaperHighlight with summary information.
    """
    ext_data = {}
    if extraction:
        ext_data = extraction.get("extraction", extraction)

    # Build summary from q02_thesis or abstract
    summary = (
        ext_data.get("q02_thesis")
        or ext_data.get("q22_contribution")
        or paper.get("abstract", "")
        or "No summary available."
    )

    methodology = None
    if config.include_methodology:
        methodology = ext_data.get("q07_methods") or None

    key_findings = []
    if config.include_key_findings:
        # q04_evidence is prose; include as a single finding entry if present
        evidence = ext_data.get("q04_evidence")
        if evidence:
            key_findings.append(evidence)

    return PaperHighlight(
        paper_id=paper.get("paper_id", ""),
        title=paper.get("title", "Unknown"),
        authors=paper.get("authors", "Unknown"),
        year=paper.get("publication_year"),
        collections=paper.get("collections", []) or [],
        summary=summary[:500],
        methodology=methodology,
        key_findings=key_findings,
    )


def generate_digest(
    index_dir: Path,
    config: DigestConfig | None = None,
    mark_processed: bool = True,
) -> ResearchDigest:
    """Generate a research digest from newly indexed papers.

    Args:
        index_dir: Path to the index directory.
        config: Optional digest configuration.
        mark_processed: If True, update state to mark papers as processed.

    Returns:
        ResearchDigest with highlights for new papers.
    """
    if config is None:
        config = DigestConfig()

    state_path = index_dir / config.state_file
    state = _load_digest_state(state_path)

    new_papers = find_new_papers(index_dir, state_path, _state=state)

    # Sort by year (newest first), limit to max_papers
    new_papers.sort(
        key=lambda p: p.get("publication_year") or 0, reverse=True
    )
    selected = new_papers[: config.max_papers]

    # Load extractions
    ext_data = safe_read_json(index_dir / "semantic_analyses.json", default={})
    if isinstance(ext_data, dict) and "extractions" in ext_data:
        extractions = ext_data["extractions"]
    elif isinstance(ext_data, dict):
        extractions = ext_data
    else:
        extractions = {}

    highlights = []
    for paper in selected:
        pid = paper.get("paper_id", "")
        extraction = extractions.get(pid)
        highlight = build_paper_highlight(paper, extraction, config)
        highlights.append(highlight)

    now = datetime.now().isoformat()
    digest = ResearchDigest(
        generated_at=now,
        period_start=state.get("last_run"),
        period_end=now,
        new_paper_count=len(new_papers),
        highlights=highlights,
    )

    if mark_processed and new_papers:
        processed = set(state.get("processed_ids", []))
        # Mark ALL new papers as processed, not just selected ones,
        # to prevent paper starvation where older papers are perpetually
        # bumped by newer arrivals.
        for paper in new_papers:
            pid = paper.get("paper_id")
            if pid:
                processed.add(pid)
        state["processed_ids"] = sorted(processed)
        state["last_run"] = now
        if not _save_digest_state(state_path, state):
            logger.error(f"Failed to save digest state to {state_path}")

    return digest


def format_digest_markdown(digest: ResearchDigest) -> str:
    """Format a research digest as Markdown.

    Args:
        digest: ResearchDigest to format.

    Returns:
        Markdown string.
    """
    lines = [
        "# Research Digest",
        "",
        f"**Generated:** {digest.generated_at}",
        f"**New papers:** {digest.new_paper_count}",
    ]
    if digest.period_start:
        lines.append(f"**Period:** {digest.period_start} to {digest.period_end}")
    lines.append("")

    if not digest.highlights:
        lines.append("*No new papers to report.*")
        return "\n".join(lines)

    if digest.narrative:
        lines.extend([digest.narrative, ""])

    lines.append("## Paper Highlights")
    lines.append("")

    for i, h in enumerate(digest.highlights, 1):
        year_str = f" ({h.year})" if h.year else ""
        lines.append(f"### {i}. {h.title}{year_str}")
        lines.append(f"*{h.authors}*")
        if h.collections:
            lines.append(f"Collections: {', '.join(h.collections)}")
        lines.append("")
        lines.append(h.summary)
        lines.append("")

        if h.methodology:
            lines.append(f"**Methodology:** {h.methodology}")
            lines.append("")

        if h.key_findings:
            lines.append("**Key findings:**")
            for finding in h.key_findings:
                lines.append(f"- {finding}")
            lines.append("")

        if h.relevance_note:
            lines.append(f"*Relevance: {h.relevance_note}*")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def format_digest_json(digest: ResearchDigest) -> str:
    """Format a research digest as JSON."""
    data = {
        "generated_at": digest.generated_at,
        "period_start": digest.period_start,
        "period_end": digest.period_end,
        "new_paper_count": digest.new_paper_count,
        "narrative": digest.narrative,
        "highlights": [
            {
                "paper_id": h.paper_id,
                "title": h.title,
                "authors": h.authors,
                "year": h.year,
                "collections": h.collections,
                "summary": h.summary,
                "methodology": h.methodology,
                "key_findings": h.key_findings,
                "relevance_note": h.relevance_note,
            }
            for h in digest.highlights
        ],
    }
    return json.dumps(data, indent=2)
