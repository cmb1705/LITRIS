"""Tests for citation graph generation pipeline."""

from pathlib import Path

from src.analysis.citation_graph import (
    GraphConfig,
    _deduplicate_papers,
    _find_part_of_relationships,
    _find_reference_matches,
    _redirect_edges,
    GraphEdge,
    build_citation_graph,
    save_citation_graph,
)


def _sample_corpus():
    """Build a test corpus with cross-references."""
    papers = [
        {
            "paper_id": "p1",
            "title": "Heterogeneous Graph Transformer for Web-Scale Recommender Systems",
            "authors": "Hu et al.",
            "publication_year": 2020,
            "collections": ["GNN"],
            "doi": "10.1145/3366423.3380027",
        },
        {
            "paper_id": "p2",
            "title": "Attention Is All You Need for Sequence Modeling",
            "authors": "Vaswani et al.",
            "publication_year": 2017,
            "collections": ["Transformers"],
            "doi": "10.48550/arXiv.1706.03762",
        },
        {
            "paper_id": "p3",
            "title": "Citation Network Analysis for Research Front Detection",
            "authors": "Smith et al.",
            "publication_year": 2022,
            "collections": ["Scientometrics"],
            "doi": None,
        },
        {
            "paper_id": "p4",
            "title": "Short Title",  # Too short for matching
            "authors": "Jones",
            "publication_year": 2021,
            "collections": ["Other"],
            "doi": None,
        },
    ]

    extractions = {
        "p1": {
            "extraction": {
                "thesis_statement": "We propose a heterogeneous graph approach.",
                "key_claims": [
                    {"claim": "Building on Attention Is All You Need for Sequence Modeling."}
                ],
                "future_directions": ["Apply to citation network analysis for research front detection."],
            }
        },
        "p3": {
            "extraction": {
                "thesis_statement": "Citation networks reveal research fronts.",
                "key_claims": [
                    {"claim": "Uses heterogeneous graph transformer for web-scale recommender systems approach (doi: 10.1145/3366423.3380027)."}
                ],
                "future_directions": [],
            }
        },
    }

    return papers, extractions


def test_basic_graph_generation():
    """Graph is generated with correct structure."""
    papers, extractions = _sample_corpus()
    graph = build_citation_graph(papers, extractions)

    assert "metadata" in graph
    assert "nodes" in graph
    assert "edges" in graph
    assert graph["metadata"]["node_count"] > 0


def test_nodes_have_required_fields():
    """All nodes have required schema fields."""
    papers, extractions = _sample_corpus()
    graph = build_citation_graph(papers, extractions)

    required = {"id", "type", "label", "title", "authors", "year",
                "collections", "in_library", "doi", "size", "color"}
    for node in graph["nodes"]:
        assert required <= set(node.keys()), f"Missing fields in node {node['id']}"


def test_edges_have_required_fields():
    """All edges have required schema fields."""
    papers, extractions = _sample_corpus()
    graph = build_citation_graph(papers, extractions)

    required = {"source", "target", "type", "confidence", "source_type", "context"}
    for edge in graph["edges"]:
        assert required <= set(edge.keys()), f"Missing fields in edge"


def test_title_matching_finds_references():
    """Title-based matching detects cross-references in extraction text."""
    papers, extractions = _sample_corpus()
    graph = build_citation_graph(papers, extractions)

    edge_pairs = {(e["source"], e["target"]) for e in graph["edges"]}
    # p1 references p2 ("Attention Is All You Need for Sequence Modeling")
    assert ("p1", "p2") in edge_pairs


def test_doi_matching():
    """DOI references in text create high-confidence edges."""
    papers, extractions = _sample_corpus()
    graph = build_citation_graph(papers, extractions)

    doi_edges = [e for e in graph["edges"] if e["source_type"] == "doi_match"]
    # p3 references p1's DOI
    assert any(e["source"] == "p3" and e["target"] == "p1" for e in doi_edges)
    # DOI matches have confidence 1.0
    for e in doi_edges:
        assert e["confidence"] == 1.0


def test_no_self_references():
    """A paper should not cite itself."""
    papers, extractions = _sample_corpus()
    graph = build_citation_graph(papers, extractions)

    for edge in graph["edges"]:
        assert edge["source"] != edge["target"]


def test_short_titles_excluded():
    """Papers with very short titles are excluded from title matching."""
    papers, extractions = _sample_corpus()
    graph = build_citation_graph(papers, extractions)

    # p4 "Short Title" should not appear as a target from title matching
    title_match_targets = {
        e["target"] for e in graph["edges"]
        if e["source_type"] == "extraction"
    }
    assert "p4" not in title_match_targets


def test_collection_filter():
    """Collection filter limits graph to specified collections."""
    papers, extractions = _sample_corpus()
    config = GraphConfig(collections_filter=["GNN"])
    graph = build_citation_graph(papers, extractions, config)

    # Only GNN papers should be nodes
    for node in graph["nodes"]:
        assert "GNN" in node["collections"]


def test_year_range_filter():
    """Year range filter limits graph to papers within range."""
    papers, extractions = _sample_corpus()
    config = GraphConfig(year_range=(2020, 2025))
    graph = build_citation_graph(papers, extractions, config)

    for node in graph["nodes"]:
        assert node["year"] is None or 2020 <= node["year"] <= 2025


def test_highly_cited_node_color():
    """Nodes with 5+ incoming edges get the highly-cited color."""
    # Create a corpus where one paper is cited by many
    papers = [
        {"paper_id": f"p{i}", "title": f"Paper number {i} with a long enough title for matching", "publication_year": 2020}
        for i in range(8)
    ]
    papers[0]["title"] = "The Foundational Work on Graph Neural Network Architectures"

    extractions = {}
    for i in range(1, 8):
        extractions[f"p{i}"] = {
            "extraction": {
                "thesis_statement": f"Building on the foundational work on graph neural network architectures.",
            }
        }

    graph = build_citation_graph(papers, extractions)

    p0_node = next((n for n in graph["nodes"] if n["id"] == "p0"), None)
    assert p0_node is not None
    assert p0_node["color"] == "#3a6a9a"


def test_deterministic_output():
    """Same inputs produce the same graph (except timestamp)."""
    papers, extractions = _sample_corpus()
    graph1 = build_citation_graph(papers, extractions)
    graph2 = build_citation_graph(papers, extractions)

    assert graph1["nodes"] == graph2["nodes"]
    assert graph1["edges"] == graph2["edges"]


def test_empty_corpus():
    """Empty corpus produces empty graph."""
    graph = build_citation_graph([], {})
    assert graph["metadata"]["node_count"] == 0
    assert graph["metadata"]["edge_count"] == 0
    assert graph["nodes"] == []
    assert graph["edges"] == []


def test_save_citation_graph(tmp_path: Path):
    """Graph is saved to JSON file."""
    papers, extractions = _sample_corpus()
    graph = build_citation_graph(papers, extractions)
    output_path = tmp_path / "citation_graph.json"
    result = save_citation_graph(graph, output_path)
    assert result.exists()


# --- Deduplication tests ---


def _duplicate_corpus():
    """Build a corpus with duplicate papers for dedup testing."""
    return [
        {
            "paper_id": "d1",
            "title": "Heterogeneous Graph Transformer for Web-Scale Recommender Systems",
            "authors": "Hu et al.",
            "publication_year": 2020,
            "collections": ["GNN"],
            "doi": "10.1145/3366423.3380027",
        },
        {
            "paper_id": "d2",
            "title": "Heterogeneous Graph Transformer for Web-Scale Recommender Systems",
            "authors": "",
            "publication_year": None,
            "collections": ["Deep Learning"],
            "doi": None,
        },
        {
            "paper_id": "d3",
            "title": "Citation Network Analysis for Research Front Detection",
            "authors": "Smith et al.",
            "publication_year": 2022,
            "collections": ["Scientometrics"],
            "doi": None,
        },
    ]


def test_exact_duplicates_merged():
    """Two papers with same title and authors are merged to one node."""
    papers = _duplicate_corpus()
    deduped, redirect_map = _deduplicate_papers(papers)
    ids = {p["paper_id"] for p in deduped}
    # d1 and d2 have same title, one should be removed
    assert len(redirect_map) == 1
    assert "d2" in redirect_map
    assert redirect_map["d2"] == "d1"
    assert "d1" in ids
    assert "d2" not in ids


def test_richer_metadata_survives():
    """Paper with DOI and authors beats paper without."""
    papers = _duplicate_corpus()
    deduped, redirect_map = _deduplicate_papers(papers)
    survivor = next(p for p in deduped if p["paper_id"] == "d1")
    assert survivor["doi"] == "10.1145/3366423.3380027"
    assert survivor["authors"] == "Hu et al."


def test_collections_unioned():
    """Surviving node has collections from all merged duplicates."""
    papers = _duplicate_corpus()
    deduped, _ = _deduplicate_papers(papers)
    survivor = next(p for p in deduped if p["paper_id"] == "d1")
    assert "GNN" in survivor["collections"]
    assert "Deep Learning" in survivor["collections"]


def test_part_of_siblings_not_merged():
    """Papers in part_of relationships are excluded from dedup."""
    papers = [
        {"paper_id": "po1", "title": "PAF540 Advanced Research Methods Syllabus", "collections": ["Syllabi"]},
        {"paper_id": "po2", "title": "PAF540 Advanced Research Methods Syllabus", "collections": ["Syllabi"]},
    ]
    part_of_pairs = {("po1", "parent1"), ("po2", "parent1")}
    deduped, redirect_map = _deduplicate_papers(papers, part_of_pairs)
    assert len(redirect_map) == 0
    ids = {p["paper_id"] for p in deduped}
    assert "po1" in ids and "po2" in ids


def test_untitled_papers_not_merged():
    """Papers with 'Untitled' title are not merged together."""
    papers = [
        {"paper_id": "u1", "title": "Untitled", "collections": ["A"]},
        {"paper_id": "u2", "title": "Untitled", "collections": ["B"]},
    ]
    deduped, redirect_map = _deduplicate_papers(papers)
    assert len(redirect_map) == 0
    ids = {p["paper_id"] for p in deduped}
    assert "u1" in ids and "u2" in ids


def test_redirected_edges_no_self_reference():
    """No source==target after redirect."""
    edges = [
        GraphEdge(source="d1", target="d2", confidence=0.8),
        GraphEdge(source="d2", target="d3", confidence=0.7),
        GraphEdge(source="d3", target="d1", confidence=0.6),
    ]
    redirect_map = {"d2": "d1"}
    redirected = _redirect_edges(edges, redirect_map)
    for e in redirected:
        assert e.source != e.target, f"Self-reference: {e.source} -> {e.target}"
    # d1->d2 becomes d1->d1 (self-ref, removed)
    # d2->d3 becomes d1->d3
    # d3->d1 stays
    assert len(redirected) == 2


def test_dedup_metadata_count():
    """metadata.duplicates_merged reports accurate count."""
    papers = _duplicate_corpus()
    extractions = {}
    graph = build_citation_graph(papers, extractions)
    assert graph["metadata"]["duplicates_merged"] == 1


# --- Part-of relationship tests ---


def test_book_section_creates_part_of_edge():
    """bookSection item + matching book item -> part_of edge."""
    papers = [
        {
            "paper_id": "book1",
            "title": "Introduction to Network Science",
            "item_type": "book",
            "zotero_key": "ZK100",
            "authors": "Barabasi",
            "publication_year": 2016,
        },
        {
            "paper_id": "ch1",
            "title": "Scale-Free Networks",
            "item_type": "bookSection",
            "zotero_key": "ZK100",
            "authors": "Barabasi",
            "publication_year": 2016,
        },
    ]
    triples = _find_part_of_relationships(papers)
    assert len(triples) == 1
    child_id, parent_id, subtype = triples[0]
    assert child_id == "ch1"
    assert parent_id == "book1"
    assert subtype == "chapter_in_book"

    # Also verify via full graph build
    graph = build_citation_graph(papers, {})
    part_of_edges = [e for e in graph["edges"] if e["type"] == "part_of"]
    assert len(part_of_edges) == 1
    assert part_of_edges[0]["source"] == "ch1"
    assert part_of_edges[0]["target"] == "book1"
    assert part_of_edges[0]["source_type"] == "metadata_match"


def test_syllabus_bundle_detection():
    """3+ papers with same title, different content -> virtual parent + part_of edges."""
    papers = [
        {"paper_id": f"paf{i}", "title": "PAF540 Advanced Research Methods", "publication_year": 2020}
        for i in range(4)
    ]
    extractions = {
        "paf0": {"extraction": {"thesis_statement": "Network analysis is fundamental."}},
        "paf1": {"extraction": {"thesis_statement": "Clustering improves outcomes."}},
        "paf2": {"extraction": {"thesis_statement": "Optimization under constraints."}},
        "paf3": {"extraction": {"thesis_statement": "Policy analysis frameworks."}},
    }
    graph = build_citation_graph(papers, extractions)

    part_of_edges = [e for e in graph["edges"] if e["type"] == "part_of"]
    assert len(part_of_edges) == 4

    # Virtual parent node exists with collection type
    virtual_nodes = [n for n in graph["nodes"] if n["type"] == "collection"]
    assert len(virtual_nodes) == 1
    assert virtual_nodes[0]["in_library"] is False
    assert virtual_nodes[0]["color"] == "#8a6a3a"

    assert graph["metadata"]["part_of_count"] == 4


def test_part_of_not_counted_as_citation():
    """part_of edges should not inflate citation in-degree counts."""
    papers = [
        {
            "paper_id": "book1",
            "title": "Introduction to Network Science",
            "item_type": "book",
            "zotero_key": "ZK100",
            "authors": "Barabasi",
            "publication_year": 2016,
        },
        {
            "paper_id": "ch1",
            "title": "Scale-Free Networks Chapter One",
            "item_type": "bookSection",
            "zotero_key": "ZK100",
            "authors": "Barabasi",
            "publication_year": 2016,
        },
    ]
    graph = build_citation_graph(papers, {})

    # book1 is a part_of target but should not get inflated size
    book_node = next(n for n in graph["nodes"] if n["id"] == "book1")
    assert book_node["size"] == 10  # Base size, no citation boost


def test_chapter_numbering_detection():
    """Title with 'Chapter 7' pattern detected as part_of."""
    papers = [
        {
            "paper_id": "parent_book",
            "title": "Advanced Methods in Policy Analysis",
            "item_type": "book",
            "authors": "Smith, J.",
            "publication_year": 2020,
        },
        {
            "paper_id": "ch7",
            "title": "Advanced Methods in Policy Analysis Chapter 7",
            "authors": "Smith, J.",
            "publication_year": 2020,
        },
    ]
    triples = _find_part_of_relationships(papers)
    assert len(triples) >= 1
    child_ids = {t[0] for t in triples}
    assert "ch7" in child_ids
    parent_ids = {t[1] for t in triples}
    assert "parent_book" in parent_ids


# --- Reference list matching tests ---


def _reference_corpus():
    """Build a corpus with reference lists for testing."""
    papers = [
        {
            "paper_id": "r1",
            "title": "Deep Learning for Natural Language Processing Tasks",
            "authors": "Author A",
            "publication_year": 2021,
            "collections": ["NLP"],
            "doi": "10.1234/nlp.2021.001",
        },
        {
            "paper_id": "r2",
            "title": "Attention Is All You Need for Sequence Modeling",
            "authors": "Vaswani et al.",
            "publication_year": 2017,
            "collections": ["Transformers"],
            "doi": "10.48550/arXiv.1706.03762",
        },
        {
            "paper_id": "r3",
            "title": "Graph Neural Networks for Citation Analysis Research",
            "authors": "Smith et al.",
            "publication_year": 2022,
            "collections": ["GNN"],
            "doi": "10.5678/gnn.2022.042",
        },
    ]

    extractions = {
        "r1": {
            "extraction": {
                "thesis_statement": "We apply deep learning to NLP.",
                "reference_list": [
                    {
                        "raw_text": "Vaswani et al. (2017). Attention Is All You Need for Sequence Modeling. arXiv:1706.03762.",
                        "parsed_title": "Attention Is All You Need for Sequence Modeling",
                        "parsed_authors": "Vaswani et al.",
                        "parsed_year": 2017,
                        "parsed_doi": "10.48550/arXiv.1706.03762",
                    },
                    {
                        "raw_text": "Smith et al. (2022). Graph Neural Networks for Citation Analysis Research.",
                        "parsed_title": "Graph Neural Networks for Citation Analysis Research",
                        "parsed_authors": "Smith et al.",
                        "parsed_year": 2022,
                        "parsed_doi": None,
                    },
                ],
            }
        },
    }

    return papers, extractions


def test_reference_doi_match():
    """Reference with DOI matching another paper creates confidence-1.0 edge."""
    papers, extractions = _reference_corpus()
    graph = build_citation_graph(papers, extractions)

    doi_ref_edges = [
        e for e in graph["edges"]
        if e["source_type"] == "reference_doi_match"
    ]
    assert any(
        e["source"] == "r1" and e["target"] == "r2" for e in doi_ref_edges
    )
    for e in doi_ref_edges:
        assert e["confidence"] == 1.0


def test_reference_title_match():
    """Reference title matching a paper creates confidence-0.95 edge."""
    papers, extractions = _reference_corpus()
    graph = build_citation_graph(papers, extractions)

    title_ref_edges = [
        e for e in graph["edges"]
        if e["source_type"] == "reference_title_match"
    ]
    assert any(
        e["source"] == "r1" and e["target"] == "r3" for e in title_ref_edges
    )
    for e in title_ref_edges:
        assert e["confidence"] == 0.95


def test_reference_no_self_cite():
    """Paper's own DOI in references does not create a self-edge."""
    papers = [
        {
            "paper_id": "s1",
            "title": "Some Paper About Self-Referencing in Academic Literature",
            "authors": "Author X",
            "publication_year": 2023,
            "doi": "10.9999/self.2023",
        },
    ]
    extractions = {
        "s1": {
            "extraction": {
                "thesis_statement": "Self-referencing study.",
                "reference_list": [
                    {
                        "raw_text": "Author X (2023). Some Paper About Self-Referencing...",
                        "parsed_title": "Some Paper About Self-Referencing in Academic Literature",
                        "parsed_authors": "Author X",
                        "parsed_year": 2023,
                        "parsed_doi": "10.9999/self.2023",
                    },
                ],
            }
        },
    }
    graph = build_citation_graph(papers, extractions)
    for edge in graph["edges"]:
        assert edge["source"] != edge["target"], "Self-citation edge found"


def test_reference_dedup_with_title_match():
    """Reference DOI match takes precedence; no duplicate edge from title match."""
    papers, extractions = _reference_corpus()
    graph = build_citation_graph(papers, extractions)

    # r1 -> r2 should appear only once (via DOI match, not also title match)
    r1_to_r2 = [
        e for e in graph["edges"]
        if e["source"] == "r1" and e["target"] == "r2"
    ]
    assert len(r1_to_r2) == 1
    assert r1_to_r2[0]["source_type"] == "reference_doi_match"
