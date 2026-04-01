"""Shared RAPTOR generation helpers for indexing workflows."""

from __future__ import annotations

from pathlib import Path

from src.analysis.dimensions import get_dimension_value
from src.analysis.raptor import RaptorSummaries, generate_raptor_summaries
from src.analysis.schemas import SemanticAnalysis
from src.utils.file_utils import safe_read_json, safe_write_json
from src.utils.logging_config import get_logger
from src.zotero.models import Author, PaperMetadata

logger = get_logger(__name__)

RAPTOR_SCHEMA_VERSION = "1.0.0"
RAPTOR_CACHE_FILENAME = "raptor_summaries.json"


def _synthesize_overview(analysis: SemanticAnalysis) -> str:
    """Synthesize a concise overview from semantic dimensions without an LLM."""
    parts = []
    research_question = get_dimension_value(analysis, "research_question")
    thesis = get_dimension_value(analysis, "thesis")
    methods = get_dimension_value(analysis, "methods")
    contribution = get_dimension_value(analysis, "contribution")
    implications = get_dimension_value(analysis, "implications")
    field_value = get_dimension_value(analysis, "field")

    if research_question:
        parts.append(research_question.strip().rstrip(".") + ".")
    elif thesis:
        parts.append(thesis.strip().rstrip(".") + ".")

    if methods:
        sentences = methods.split(". ")
        parts.append(". ".join(sentences[:2]).strip().rstrip(".") + ".")
    if contribution:
        sentences = contribution.split(". ")
        parts.append(". ".join(sentences[:2]).strip().rstrip(".") + ".")
    if implications:
        sentences = implications.split(". ")
        parts.append(sentences[0].strip().rstrip(".") + ".")
    if field_value:
        parts.append(f"Field: {field_value.strip()}.")

    overview = " ".join(parts)
    words = overview.split()
    if len(words) > 180:
        overview = " ".join(words[:170]) + "..."
    return overview


def _synthesize_core(analysis: SemanticAnalysis) -> str:
    """Synthesize a one-sentence core contribution without an LLM."""
    contribution = get_dimension_value(analysis, "contribution")
    thesis = get_dimension_value(analysis, "thesis")
    field_value = get_dimension_value(analysis, "field")

    if contribution:
        first_sentence = contribution.split(". ")[0].strip().rstrip(".") + "."
        words = first_sentence.split()
        if len(words) > 45:
            first_sentence = " ".join(words[:40]) + "..."
        return first_sentence
    if thesis:
        first_sentence = thesis.split(". ")[0].strip().rstrip(".") + "."
        words = first_sentence.split()
        if len(words) > 45:
            first_sentence = " ".join(words[:40]) + "..."
        return first_sentence
    return f"Study in {field_value or 'unspecified field'}."


def load_raptor_cache(index_dir: Path) -> dict[str, dict]:
    """Load cached RAPTOR summaries."""
    path = index_dir / RAPTOR_CACHE_FILENAME
    data = safe_read_json(path, default={})
    if isinstance(data, dict) and "summaries" in data:
        return data["summaries"]
    return {}


def save_raptor_cache(index_dir: Path, summaries: dict[str, dict], mode: str) -> None:
    """Save RAPTOR summaries to the index directory cache."""
    safe_write_json(
        index_dir / RAPTOR_CACHE_FILENAME,
        {
            "schema_version": RAPTOR_SCHEMA_VERSION,
            "generated_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
            "mode": mode,
            "count": len(summaries),
            "summaries": summaries,
        },
    )


def _paper_from_index_dict(paper_id: str, paper_dict: dict) -> PaperMetadata:
    authors = []
    for author_data in paper_dict.get("authors", []):
        if isinstance(author_data, dict):
            authors.append(Author(**author_data))
    return PaperMetadata(
        paper_id=paper_id,
        zotero_key=paper_dict.get("zotero_key", paper_id.split("_")[0]),
        zotero_item_id=paper_dict.get("zotero_item_id", 0),
        item_type=paper_dict.get("item_type", "journalArticle"),
        title=paper_dict.get("title", "Unknown"),
        authors=authors,
        publication_year=paper_dict.get("publication_year"),
        publication_date=paper_dict.get("publication_date"),
        journal=paper_dict.get("journal"),
        doi=paper_dict.get("doi"),
        abstract=paper_dict.get("abstract"),
        url=paper_dict.get("url"),
        collections=paper_dict.get("collections", []),
        tags=paper_dict.get("tags", []),
        pdf_path=paper_dict.get("pdf_path"),
        pdf_attachment_key=paper_dict.get("pdf_attachment_key"),
        date_added=paper_dict.get("date_added") or "2020-01-01T00:00:00",
        date_modified=paper_dict.get("date_modified") or "2020-01-01T00:00:00",
    )


def generate_scoped_raptor_summaries(
    papers: list[PaperMetadata],
    extractions: dict[str, SemanticAnalysis],
    index_dir: Path,
    mode: str = "template",
    provider: str = "google",
    model: str | None = None,
    force: bool = False,
) -> dict[str, RaptorSummaries]:
    """Generate RAPTOR summaries for the provided paper scope."""
    cached = {} if force else load_raptor_cache(index_dir)
    results: dict[str, RaptorSummaries] = {}
    to_generate: list[PaperMetadata] = []

    for paper in papers:
        if paper.paper_id not in extractions:
            continue
        cache_entry = cached.get(paper.paper_id)
        if cache_entry and not force:
            results[paper.paper_id] = RaptorSummaries(
                paper_id=paper.paper_id,
                paper_overview=cache_entry.get("paper_overview", ""),
                core_contribution=cache_entry.get("core_contribution", ""),
            )
            continue
        to_generate.append(paper)

    for paper in to_generate:
        analysis = extractions[paper.paper_id]
        if mode == "llm":
            raptor = generate_raptor_summaries(
                paper=paper,
                analysis=analysis,
                provider=provider,
                model=model,
            )
            if raptor is None:
                raptor = RaptorSummaries(
                    paper_id=paper.paper_id,
                    paper_overview=_synthesize_overview(analysis),
                    core_contribution=_synthesize_core(analysis),
                )
        else:
            raptor = RaptorSummaries(
                paper_id=paper.paper_id,
                paper_overview=_synthesize_overview(analysis),
                core_contribution=_synthesize_core(analysis),
            )
        results[paper.paper_id] = raptor

    merged_cache = dict(cached)
    for paper_id, summary in results.items():
        merged_cache[paper_id] = {
            "paper_overview": summary.paper_overview,
            "core_contribution": summary.core_contribution,
        }
    save_raptor_cache(index_dir, merged_cache, mode)
    return results


def rebuild_raptor_cache_from_store(
    index_dir: Path,
    paper_dicts: dict[str, dict],
    extractions: dict[str, SemanticAnalysis],
    mode: str = "template",
    provider: str = "google",
    model: str | None = None,
    force: bool = False,
) -> dict[str, RaptorSummaries]:
    """Rebuild RAPTOR summaries for all papers present in the structured store."""
    papers = [
        _paper_from_index_dict(paper_id, paper_dict)
        for paper_id, paper_dict in paper_dicts.items()
        if paper_id in extractions
    ]
    return generate_scoped_raptor_summaries(
        papers=papers,
        extractions=extractions,
        index_dir=index_dir,
        mode=mode,
        provider=provider,
        model=model,
        force=force,
    )


def prune_raptor_cache(index_dir: Path, valid_paper_ids: set[str]) -> None:
    """Remove deleted papers from RAPTOR cache."""
    cached = load_raptor_cache(index_dir)
    pruned = {
        paper_id: value for paper_id, value in cached.items() if paper_id in valid_paper_ids
    }
    if pruned != cached:
        save_raptor_cache(index_dir, pruned, mode="template")
