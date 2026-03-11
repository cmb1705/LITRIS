"""Citation graph generation from LITRIS index data.

Builds a citation network by matching paper titles and DOIs across
the indexed corpus. Uses extraction text (thesis, findings, claims)
to find references to other papers in the library.

See docs/proposals/citation_network_schema.md for the schema spec.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.utils.file_utils import safe_read_json, safe_write_json

logger = logging.getLogger(__name__)

# Minimum title length to avoid false positives on short titles
_MIN_TITLE_LENGTH = 15

# Words that are too common to be meaningful in title matching
_COMMON_TITLE_WORDS = {
    "the", "and", "for", "from", "with", "that", "this", "are",
    "was", "were", "been", "have", "has", "had", "not", "but",
    "can", "will", "may", "use", "its", "our", "all", "new",
}


@dataclass
class GraphConfig:
    """Configuration for citation graph generation."""

    min_title_length: int = _MIN_TITLE_LENGTH
    fuzzy_threshold: float = 0.85
    max_label_length: int = 50
    collections_filter: list[str] | None = None
    year_range: tuple[int, int] | None = None


@dataclass
class GraphNode:
    """A node in the citation graph."""

    id: str
    node_type: str = "paper"
    label: str = ""
    title: str = ""
    authors: str = ""
    year: int | None = None
    collections: list[str] = field(default_factory=list)
    in_library: bool = True
    doi: str | None = None
    size: int = 10
    color: str = "#4a8c6a"


@dataclass
class GraphEdge:
    """An edge in the citation graph."""

    source: str
    target: str
    edge_type: str = "cites"
    confidence: float = 0.0
    source_type: str = "extraction"
    context: str | None = None


def _normalize_title(title: str) -> str:
    """Normalize a title for matching."""
    title = title.lower().strip()
    title = re.sub(r"[^\w\s]", "", title)
    title = " ".join(title.split())
    return title


def _tokenize_title(title: str) -> set[str]:
    """Tokenize a title into significant words."""
    normalized = _normalize_title(title)
    words = normalized.split()
    return {w for w in words if len(w) > 2 and w not in _COMMON_TITLE_WORDS}


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def _extract_text_fields(extraction: dict) -> str:
    """Extract all searchable text from an extraction."""
    ext_data = extraction.get("extraction", extraction)
    parts = []

    for field_name in ("thesis_statement", "conclusions", "contribution_summary",
                       "theoretical_framework"):
        value = ext_data.get(field_name)
        if value:
            parts.append(str(value))

    for list_field in ("research_questions", "limitations", "future_directions"):
        items = ext_data.get(list_field) or []
        parts.extend(str(item) for item in items)

    findings = ext_data.get("key_findings") or []
    for finding in findings:
        if isinstance(finding, dict):
            parts.append(finding.get("finding", ""))
        else:
            parts.append(str(finding))

    claims = ext_data.get("key_claims") or []
    for claim in claims:
        if isinstance(claim, dict):
            parts.append(claim.get("claim", ""))
        else:
            parts.append(str(claim))

    return " ".join(parts)


def _build_title_index(papers: list[dict], min_title_length: int = _MIN_TITLE_LENGTH) -> dict[str, dict]:
    """Build a lookup of normalized title tokens to papers."""
    index: dict[str, dict] = {}
    for paper in papers:
        paper_id = paper.get("paper_id")
        title = paper.get("title") or ""
        if not paper_id or not title or len(title) < min_title_length:
            continue
        tokens = _tokenize_title(title)
        if len(tokens) >= 3:
            index[paper_id] = {
                "paper": paper,
                "tokens": tokens,
                "normalized": _normalize_title(title),
            }
    return index


def _find_title_matches(
    text: str,
    source_id: str,
    title_index: dict[str, dict],
    threshold: float,
) -> list[tuple[str, float, str | None]]:
    """Find papers referenced by title in text.

    Returns list of (target_paper_id, confidence, context_snippet).
    """
    matches = []
    text_lower = text.lower()
    text_tokens = set(re.findall(r"\b\w{3,}\b", text_lower)) - _COMMON_TITLE_WORDS

    for paper_id, info in title_index.items():
        if paper_id == source_id:
            continue

        title_tokens = info["tokens"]
        # Quick check: do enough title tokens appear in the text?
        overlap = title_tokens & text_tokens
        if len(overlap) < min(3, len(title_tokens)):
            continue

        similarity = _jaccard_similarity(title_tokens, text_tokens)
        if similarity >= threshold:
            # Find context around the match
            normalized_title = info["normalized"]
            context = None
            title_words = normalized_title.split()[:4]
            if title_words:
                pattern = r"[^.]*" + re.escape(title_words[0]) + r"[^.]*\."
                ctx_match = re.search(pattern, text, re.IGNORECASE)
                if ctx_match:
                    context = ctx_match.group().strip()[:200]

            # Adjust confidence based on match quality
            base_confidence = 0.7
            if similarity > 0.9:
                base_confidence = 0.9

            matches.append((paper_id, base_confidence, context))

    return matches


def _find_doi_matches(
    papers: list[dict],
    extractions: dict[str, dict],
) -> list[tuple[str, str, float]]:
    """Find DOI-based citation links.

    Returns list of (source_id, target_id, confidence).
    """
    doi_to_id: dict[str, str] = {}
    for paper in papers:
        doi = paper.get("doi")
        paper_id = paper.get("paper_id")
        if doi and paper_id:
            doi_to_id[doi.lower()] = paper_id

    matches = []
    doi_pattern = re.compile(r"10\.\d{4,}/[^\s,;\"')\]]+")

    for paper in papers:
        source_id = paper.get("paper_id")
        if not source_id:
            continue

        extraction = extractions.get(source_id, {})
        text = _extract_text_fields(extraction)
        found_dois = doi_pattern.findall(text)

        for doi in found_dois:
            doi_clean = doi.lower().rstrip(".")
            target_id = doi_to_id.get(doi_clean)
            if target_id and target_id != source_id:
                matches.append((source_id, target_id, 1.0))

    return matches


def build_citation_graph(
    papers: list[dict],
    extractions: dict[str, dict],
    config: GraphConfig | None = None,
) -> dict:
    """Build a citation graph from papers and extractions.

    Args:
        papers: List of paper metadata dictionaries.
        extractions: Mapping of paper_id to extraction dictionaries.
        config: Optional configuration.

    Returns:
        Citation graph dictionary matching the schema spec.
    """
    if config is None:
        config = GraphConfig()

    # Filter papers by collection/year if configured
    filtered_papers = _filter_papers(papers, config)
    title_index = _build_title_index(filtered_papers, config.min_title_length)

    # Build nodes
    nodes: dict[str, GraphNode] = {}
    for paper in filtered_papers:
        paper_id = paper.get("paper_id")
        if not paper_id:
            continue
        title = paper.get("title") or "Unknown"
        nodes[paper_id] = GraphNode(
            id=paper_id,
            label=title[:config.max_label_length] + ("..." if len(title) > config.max_label_length else ""),
            title=title,
            authors=paper.get("authors") or "",
            year=paper.get("publication_year"),
            collections=list(paper.get("collections") or []),
            in_library=True,
            doi=paper.get("doi"),
        )

    # Find edges
    edges: list[GraphEdge] = []
    edge_set: set[tuple[str, str]] = set()

    # DOI-based matches (highest confidence)
    doi_matches = _find_doi_matches(filtered_papers, extractions)
    for source_id, target_id, confidence in doi_matches:
        key = (source_id, target_id)
        if key not in edge_set:
            edge_set.add(key)
            edges.append(GraphEdge(
                source=source_id,
                target=target_id,
                confidence=confidence,
                source_type="doi_match",
            ))

    # Title-based matches
    for paper in filtered_papers:
        paper_id = paper.get("paper_id")
        if not paper_id:
            continue
        extraction = extractions.get(paper_id, {})
        text = _extract_text_fields(extraction)
        if not text.strip():
            continue

        title_matches = _find_title_matches(
            text, paper_id, title_index, config.fuzzy_threshold
        )
        for target_id, confidence, context in title_matches:
            key = (paper_id, target_id)
            if key not in edge_set:
                edge_set.add(key)
                edges.append(GraphEdge(
                    source=paper_id,
                    target=target_id,
                    confidence=confidence,
                    source_type="extraction",
                    context=context,
                ))

    # Update node sizes based on in-degree (citation count)
    in_degree: dict[str, int] = defaultdict(int)
    for edge in edges:
        in_degree[edge.target] += 1

    for node_id, node in nodes.items():
        citations = in_degree.get(node_id, 0)
        node.size = 10 + citations * 5
        if citations >= 5:
            node.color = "#3a6a9a"  # Highly cited

    # Build output
    graph = {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "papers_analyzed": len(filtered_papers),
            "extractions_used": sum(
                1 for p in filtered_papers
                if p.get("paper_id") in extractions
            ),
            "filters_applied": {
                "collections": config.collections_filter,
                "year_range": list(config.year_range) if config.year_range else None,
            },
        },
        "nodes": [
            {
                "id": n.id,
                "type": n.node_type,
                "label": n.label,
                "title": n.title,
                "authors": n.authors,
                "year": n.year,
                "collections": n.collections,
                "in_library": n.in_library,
                "doi": n.doi,
                "size": n.size,
                "color": n.color,
            }
            for n in nodes.values()
        ],
        "edges": [
            {
                "source": e.source,
                "target": e.target,
                "type": e.edge_type,
                "confidence": e.confidence,
                "source_type": e.source_type,
                "context": e.context,
            }
            for e in edges
        ],
    }

    return graph


def _filter_papers(papers: list[dict], config: GraphConfig) -> list[dict]:
    """Filter papers by collection and year range."""
    filtered = papers

    if config.collections_filter:
        cols = set(config.collections_filter)
        filtered = [
            p for p in filtered
            if cols & set(p.get("collections", []) or [])
        ]

    if config.year_range:
        min_year, max_year = config.year_range
        filtered = [
            p for p in filtered
            if p.get("publication_year") and min_year <= p["publication_year"] <= max_year
        ]

    return filtered


def load_and_build_graph(
    index_dir: Path,
    config: GraphConfig | None = None,
) -> dict:
    """Load index data and build citation graph.

    Args:
        index_dir: Path to the index directory.
        config: Optional graph configuration.

    Returns:
        Citation graph dictionary.
    """
    papers_data = safe_read_json(index_dir / "papers.json", default={"papers": []})
    if isinstance(papers_data, dict) and "papers" in papers_data:
        papers = papers_data["papers"]
    elif isinstance(papers_data, list):
        papers = papers_data
    else:
        papers = []

    extractions_data = safe_read_json(index_dir / "extractions.json", default={})
    if isinstance(extractions_data, dict) and "extractions" in extractions_data:
        extractions = extractions_data["extractions"]
    elif isinstance(extractions_data, dict):
        extractions = extractions_data
    else:
        extractions = {}

    return build_citation_graph(papers, extractions, config)


def save_citation_graph(graph: dict, output_path: Path) -> Path:
    """Save citation graph to JSON file.

    Args:
        graph: Citation graph dictionary.
        output_path: Path to save the graph.

    Returns:
        Path to the saved file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_write_json(output_path, graph)
    logger.info(
        "Saved citation graph: %d nodes, %d edges -> %s",
        graph["metadata"]["node_count"],
        graph["metadata"]["edge_count"],
        output_path,
    )
    return output_path
