"""Tests for research digest generation."""

import json
from pathlib import Path

from src.analysis.research_digest import (
    DigestConfig,
    build_paper_highlight,
    find_new_papers,
    format_digest_json,
    format_digest_markdown,
    generate_digest,
)


def _write_index(tmp_path, papers, extractions=None):
    """Write test index files."""
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "papers.json").write_text(json.dumps({"papers": papers}))
    if extractions is not None:
        (index_dir / "extractions.json").write_text(json.dumps(extractions))
    return index_dir


def _sample_papers():
    return [
        {
            "paper_id": "p1",
            "title": "Deep Learning for NLP",
            "authors": "Smith et al.",
            "publication_year": 2023,
            "collections": ["AI"],
            "abstract": "A survey of deep learning approaches for NLP tasks.",
        },
        {
            "paper_id": "p2",
            "title": "Graph Neural Networks",
            "authors": "Jones et al.",
            "publication_year": 2022,
            "collections": ["AI", "GNN"],
        },
        {
            "paper_id": "p3",
            "title": "Climate Modeling",
            "authors": "Lee et al.",
            "publication_year": 2024,
            "collections": ["Climate"],
        },
    ]


def _sample_extractions():
    return {
        "p1": {
            "extraction": {
                "thesis_statement": "Deep learning has transformed NLP.",
                "methodology": {
                    "approach": "Survey",
                    "analysis_methods": ["literature review", "meta-analysis"],
                },
                "key_findings": [
                    {"finding": "Transformers dominate modern NLP."},
                    {"finding": "Pre-training improves downstream tasks."},
                ],
            }
        },
        "p2": {
            "extraction": {
                "thesis_statement": "GNNs generalize neural networks to graphs.",
                "key_claims": [{"claim": "Message passing is key."}],
            }
        },
    }


def test_find_new_papers_all_new(tmp_path):
    """All papers are new when no state file exists."""
    index_dir = _write_index(tmp_path, _sample_papers())
    new = find_new_papers(index_dir, index_dir / "digest_state.json")
    assert len(new) == 3


def test_find_new_papers_with_state(tmp_path):
    """Already-processed papers are excluded."""
    index_dir = _write_index(tmp_path, _sample_papers())
    state_path = index_dir / "digest_state.json"
    state_path.write_text(json.dumps({"processed_ids": ["p1", "p2"], "last_run": "2024-01-01"}))
    new = find_new_papers(index_dir, state_path)
    assert len(new) == 1
    assert new[0]["paper_id"] == "p3"


def test_build_paper_highlight_with_extraction():
    """Highlight includes extraction data."""
    paper = _sample_papers()[0]
    extraction = _sample_extractions()["p1"]
    config = DigestConfig()
    highlight = build_paper_highlight(paper, extraction, config)

    assert highlight.paper_id == "p1"
    assert highlight.title == "Deep Learning for NLP"
    assert "transformed NLP" in highlight.summary
    assert highlight.methodology is not None
    assert "Survey" in highlight.methodology
    assert len(highlight.key_findings) == 2


def test_build_paper_highlight_no_extraction():
    """Highlight falls back to abstract without extraction."""
    paper = _sample_papers()[0]
    config = DigestConfig()
    highlight = build_paper_highlight(paper, None, config)
    assert "survey of deep learning" in highlight.summary


def test_generate_digest(tmp_path):
    """Full digest generation pipeline."""
    index_dir = _write_index(tmp_path, _sample_papers(), _sample_extractions())
    config = DigestConfig(max_papers=2)
    digest = generate_digest(index_dir, config, mark_processed=False)

    assert digest.new_paper_count == 3
    assert len(digest.highlights) == 2
    # Should be sorted by year descending
    assert digest.highlights[0].year >= digest.highlights[1].year


def test_generate_digest_marks_processed(tmp_path):
    """Digest marks ALL new papers as processed to prevent starvation."""
    index_dir = _write_index(tmp_path, _sample_papers(), _sample_extractions())
    config = DigestConfig(max_papers=2)

    # First run: 3 new papers, highlights limited to 2, but all 3 marked processed
    digest1 = generate_digest(index_dir, config, mark_processed=True)
    assert digest1.new_paper_count == 3
    assert len(digest1.highlights) == 2

    # Second run: all papers already processed, none new
    digest2 = generate_digest(index_dir, config, mark_processed=True)
    assert digest2.new_paper_count == 0
    assert len(digest2.highlights) == 0


def test_format_digest_markdown():
    """Markdown format includes paper highlights."""
    index_dir_stub = Path("/tmp/stub")
    papers = _sample_papers()[:1]
    extraction = _sample_extractions()["p1"]
    config = DigestConfig()
    highlight = build_paper_highlight(papers[0], extraction, config)

    from src.analysis.research_digest import ResearchDigest

    digest = ResearchDigest(
        generated_at="2024-01-01",
        period_start=None,
        period_end="2024-01-01",
        new_paper_count=1,
        highlights=[highlight],
    )
    md = format_digest_markdown(digest)
    assert "# Research Digest" in md
    assert "Deep Learning for NLP" in md
    assert "Survey" in md


def test_format_digest_json():
    """JSON format is valid and contains highlights."""
    from src.analysis.research_digest import ResearchDigest

    papers = _sample_papers()[:1]
    config = DigestConfig()
    highlight = build_paper_highlight(papers[0], None, config)

    digest = ResearchDigest(
        generated_at="2024-01-01",
        period_start=None,
        period_end="2024-01-01",
        new_paper_count=1,
        highlights=[highlight],
    )
    json_str = format_digest_json(digest)
    data = json.loads(json_str)
    assert data["new_paper_count"] == 1
    assert len(data["highlights"]) == 1
    assert data["highlights"][0]["title"] == "Deep Learning for NLP"


def test_empty_index(tmp_path):
    """Empty index produces empty digest."""
    index_dir = _write_index(tmp_path, [])
    digest = generate_digest(index_dir, mark_processed=False)
    assert digest.new_paper_count == 0
    assert len(digest.highlights) == 0
