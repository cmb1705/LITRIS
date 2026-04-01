"""Citation graph generation from LITRIS index data.

Builds a citation network by matching paper titles and DOIs across
the indexed corpus. Uses SemanticAnalysis dimension text (q01-q40
fields) to find references to other papers in the library.

See docs/proposals/citation_network_schema.md for the schema spec.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.analysis.dimensions import get_dimension_map, get_dimension_value
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
    """Extract all searchable text from a semantic analysis.

    Concatenates all non-None q-fields into a single search string.
    """
    parts = [
        str(value)
        for value in get_dimension_map(extraction).values()
        if value
    ]
    return " ".join(parts)


_CHAPTER_PATTERN = re.compile(
    r"(?:chapter|part)\s+(?:\d+|[ivxlcdm]+)",
    re.IGNORECASE,
)


def _find_part_of_relationships(
    papers: list[dict],
    extractions: dict[str, dict] | None = None,
) -> list[tuple[str, str, str]]:
    """Detect hierarchical relationships between documents.

    Returns (child_id, parent_id, relationship_subtype) tuples.
    Subtypes: "chapter_in_book", "reading_in_syllabus", "section_in_report".
    """
    if extractions is None:
        extractions = {}

    results: list[tuple[str, str, str]] = []
    used_children: set[str] = set()

    # Build lookup by zotero_key
    by_zotero_key: dict[str, list[dict]] = defaultdict(list)
    for paper in papers:
        zk = paper.get("zotero_key")
        if zk:
            by_zotero_key[zk].append(paper)

    # Heuristic 1: bookSection -> book via shared zotero_key
    for _zk, group in by_zotero_key.items():
        if len(group) < 2:
            continue
        books = [p for p in group if p.get("item_type") == "book"]
        sections = [p for p in group if p.get("item_type") == "bookSection"]
        for section in sections:
            for book in books:
                child_id = section.get("paper_id")
                parent_id = book.get("paper_id")
                if child_id and parent_id and child_id != parent_id:
                    results.append((child_id, parent_id, "chapter_in_book"))
                    used_children.add(child_id)

    # Heuristic 2: Shared title with distinct content -> syllabus bundle
    title_groups: dict[str, list[dict]] = defaultdict(list)
    for paper in papers:
        title = paper.get("title") or ""
        normalized = _normalize_title(title)
        if normalized and normalized != "untitled":
            title_groups[normalized].append(paper)

    for _normalized_title, group in title_groups.items():
        if len(group) < 3:
            continue
        # Check for distinct content via q02_thesis
        statements: set[str] = set()
        for p in group:
            pid = p.get("paper_id")
            if not pid:
                continue
            ext = extractions.get(pid, {})
            thesis = get_dimension_value(ext, "thesis") or ""
            if thesis.strip():
                statements.add(thesis.strip())
        if len(statements) < 2:
            continue
        # Create virtual parent node ID
        virtual_parent_id = f"virtual_collection_{_normalized_title[:40].replace(' ', '_')}"
        for p in group:
            child_id = p.get("paper_id")
            if child_id and child_id not in used_children:
                results.append((child_id, virtual_parent_id, "reading_in_syllabus"))
                used_children.add(child_id)

    # Heuristic 3: Chapter numbering in title
    for paper in papers:
        child_id = paper.get("paper_id")
        if not child_id or child_id in used_children:
            continue
        title = paper.get("title") or ""
        if not _CHAPTER_PATTERN.search(title):
            continue
        # Match to parent book by same authors + similar title prefix
        paper_authors = _normalize_title(paper.get("authors") or "")
        if not paper_authors:
            continue
        title_prefix = _normalize_title(title).split()[:3]
        for candidate in papers:
            cand_id = candidate.get("paper_id")
            if not cand_id or cand_id == child_id:
                continue
            cand_authors = _normalize_title(candidate.get("authors") or "")
            if cand_authors != paper_authors:
                continue
            cand_title_words = _normalize_title(candidate.get("title") or "").split()
            # Check if candidate title shares prefix and doesn't have chapter pattern
            if (
                len(cand_title_words) >= 3
                and cand_title_words[:3] == title_prefix
                and not _CHAPTER_PATTERN.search(candidate.get("title") or "")
            ):
                results.append((child_id, cand_id, "chapter_in_book"))
                used_children.add(child_id)
                break

    return results


def _deduplicate_papers(
    papers: list[dict],
    part_of_pairs: set[tuple[str, str]] | None = None,
) -> tuple[list[dict], dict[str, str]]:
    """Merge duplicate papers, keeping the one with richest metadata.

    Args:
        papers: List of paper metadata dictionaries.
        part_of_pairs: Set of (child_id, parent_id) part-of relationships.
            Papers in part_of relationships are excluded from dedup.

    Returns:
        Tuple of (deduplicated papers list, redirect map {removed_id: surviving_id}).
    """
    if part_of_pairs is None:
        part_of_pairs = set()

    # IDs involved in part_of relationships
    part_of_ids: set[str] = set()
    for child_id, parent_id in part_of_pairs:
        part_of_ids.add(child_id)
        part_of_ids.add(parent_id)

    # Group papers by normalized title
    title_groups: dict[str, list[dict]] = defaultdict(list)
    for paper in papers:
        title = paper.get("title") or ""
        normalized = _normalize_title(title)
        if not normalized or normalized == "untitled":
            continue
        title_groups[normalized].append(paper)

    # Papers not grouped (empty/untitled titles) pass through
    grouped_ids: set[str] = set()
    for group in title_groups.values():
        for p in group:
            pid = p.get("paper_id")
            if pid:
                grouped_ids.add(pid)

    redirect_map: dict[str, str] = {}
    surviving_papers: dict[str, dict] = {}

    for _normalized_title, group in title_groups.items():
        if len(group) < 2:
            # No duplicates
            pid = group[0].get("paper_id")
            if pid:
                surviving_papers[pid] = group[0]
            continue

        # Skip if any member is in a part_of relationship
        group_ids = {p.get("paper_id") for p in group if p.get("paper_id")}
        if group_ids & part_of_ids:
            for p in group:
                pid = p.get("paper_id")
                if pid:
                    surviving_papers[pid] = p
            continue

        # Score each paper: +2 DOI, +2 authors, +1 year, +1 abstract, +1 per collection
        def _score(p: dict) -> int:
            s = 0
            if p.get("doi"):
                s += 2
            if p.get("authors"):
                s += 2
            if p.get("publication_year"):
                s += 1
            if p.get("abstract"):
                s += 1
            s += len(p.get("collections") or [])
            return s

        scored = sorted(group, key=_score, reverse=True)
        survivor = scored[0].copy()
        survivor_id = survivor.get("paper_id")

        # Union all collections into the survivor
        all_collections: list[str] = []
        seen_collections: set[str] = set()
        for p in scored:
            for col in p.get("collections") or []:
                if col not in seen_collections:
                    seen_collections.add(col)
                    all_collections.append(col)
        survivor["collections"] = all_collections

        surviving_papers[survivor_id] = survivor

        # Map removed papers to the survivor
        for p in scored[1:]:
            removed_id = p.get("paper_id")
            if removed_id and removed_id != survivor_id:
                redirect_map[removed_id] = survivor_id

    # Add papers that weren't grouped (no title or "untitled")
    result = list(surviving_papers.values())
    for paper in papers:
        pid = paper.get("paper_id")
        if pid and pid not in grouped_ids:
            result.append(paper)

    logger.info(
        "Deduplication: %d papers -> %d (%d duplicates merged)",
        len(papers), len(result), len(redirect_map),
    )
    return result, redirect_map


def _redirect_edges(
    edges: list[GraphEdge],
    redirect_map: dict[str, str],
) -> list[GraphEdge]:
    """Redirect edges after deduplication, removing self-references and duplicates."""
    redirected: list[GraphEdge] = []
    seen: set[tuple[str, str]] = set()

    for edge in edges:
        source = redirect_map.get(edge.source, edge.source)
        target = redirect_map.get(edge.target, edge.target)

        # Skip self-references created by merging
        if source == target:
            continue

        key = (source, target)
        if key in seen:
            continue
        seen.add(key)

        redirected.append(GraphEdge(
            source=source,
            target=target,
            edge_type=edge.edge_type,
            confidence=edge.confidence,
            source_type=edge.source_type,
            context=edge.context,
        ))

    return redirected


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
        if not title_tokens:
            continue

        # Containment: what fraction of title tokens appear in the text?
        overlap = title_tokens & text_tokens
        if len(overlap) < min(3, len(title_tokens)):
            continue

        containment = len(overlap) / len(title_tokens)
        if containment >= threshold:
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
            if containment > 0.9:
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


def _find_reference_matches(
    papers: list[dict],
    extractions: dict[str, dict],
    title_index: dict[str, dict],
    fuzzy_threshold: float,
) -> list[tuple[str, str, float, str]]:
    """Find citation links from parsed reference lists.

    Matches references against the paper index by DOI (confidence 1.0)
    or title fuzzy match (confidence 0.95).

    Returns list of (source_id, target_id, confidence, source_type).
    """
    # Build DOI lookup
    doi_to_id: dict[str, str] = {}
    for paper in papers:
        doi = paper.get("doi")
        paper_id = paper.get("paper_id")
        if doi and paper_id:
            doi_to_id[doi.lower().rstrip(".")] = paper_id

    matches = []

    for paper in papers:
        source_id = paper.get("paper_id")
        if not source_id:
            continue

        extraction = extractions.get(source_id, {})
        ext_data = extraction.get("extraction", extraction)
        reference_list = ext_data.get("reference_list") or []

        for ref in reference_list:
            if not isinstance(ref, dict):
                continue

            # Try DOI match first (highest confidence)
            ref_doi = ref.get("parsed_doi")
            if ref_doi:
                doi_clean = ref_doi.lower().rstrip(".")
                target_id = doi_to_id.get(doi_clean)
                if target_id and target_id != source_id:
                    matches.append((source_id, target_id, 1.0, "reference_doi_match"))
                    continue

            # Try title fuzzy match
            ref_title = ref.get("parsed_title")
            if not ref_title or len(ref_title) < _MIN_TITLE_LENGTH:
                continue

            ref_tokens = _tokenize_title(ref_title)
            if len(ref_tokens) < 3:
                continue

            for paper_id, info in title_index.items():
                if paper_id == source_id:
                    continue

                similarity = _jaccard_similarity(ref_tokens, info["tokens"])
                if similarity >= max(fuzzy_threshold, 0.85):
                    matches.append((source_id, paper_id, 0.95, "reference_title_match"))
                    break  # One match per reference entry

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

    # Detect part_of relationships before dedup
    part_of_triples = _find_part_of_relationships(filtered_papers, extractions)
    part_of_pairs: set[tuple[str, str]] = {(c, p) for c, p, _ in part_of_triples}

    # Deduplicate papers (excluding part_of members)
    filtered_papers, redirect_map = _deduplicate_papers(filtered_papers, part_of_pairs)

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

    # Add virtual parent nodes for syllabus bundles
    virtual_parents: set[str] = set()
    for _child_id, parent_id, subtype in part_of_triples:
        if subtype == "reading_in_syllabus" and parent_id not in nodes:
            virtual_parents.add(parent_id)
            # Derive label from the virtual parent ID
            label = parent_id.replace("virtual_collection_", "").replace("_", " ")
            nodes[parent_id] = GraphNode(
                id=parent_id,
                node_type="collection",
                label=label[:config.max_label_length],
                title=label,
                in_library=False,
                color="#8a6a3a",
            )

    # Build part_of edges
    part_of_edges: list[GraphEdge] = []
    for child_id, parent_id, subtype in part_of_triples:
        if child_id in nodes or child_id in redirect_map:
            resolved_child = redirect_map.get(child_id, child_id)
            resolved_parent = redirect_map.get(parent_id, parent_id)
            if resolved_child != resolved_parent:
                part_of_edges.append(GraphEdge(
                    source=resolved_child,
                    target=resolved_parent,
                    edge_type="part_of",
                    confidence=1.0,
                    source_type="metadata_match",
                    context=subtype,
                ))

    # Find citation edges
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

    # Reference-list-based matches (higher confidence than title matching)
    ref_matches = _find_reference_matches(
        filtered_papers, extractions, title_index, config.fuzzy_threshold
    )
    for source_id, target_id, confidence, source_type in ref_matches:
        key = (source_id, target_id)
        if key not in edge_set:
            edge_set.add(key)
            edges.append(GraphEdge(
                source=source_id,
                target=target_id,
                confidence=confidence,
                source_type=source_type,
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

    # Redirect edges for deduplicated papers
    if redirect_map:
        edges = _redirect_edges(edges, redirect_map)

    # Combine citation edges and part_of edges
    all_edges = edges + part_of_edges

    # Update node sizes based on in-degree (citation count, excluding part_of)
    in_degree: dict[str, int] = defaultdict(int)
    for edge in all_edges:
        if edge.edge_type != "part_of":
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
            "edge_count": len(all_edges),
            "papers_analyzed": len(filtered_papers),
            "duplicates_merged": len(redirect_map),
            "part_of_count": len(part_of_edges),
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
            for e in all_edges
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

    extractions_data = safe_read_json(index_dir / "semantic_analyses.json", default={})
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
