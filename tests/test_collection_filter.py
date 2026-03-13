"""Tests for collection filtering in build_index.py."""


def _filter_by_collection(papers: list[dict], collection: str) -> list[dict]:
    """Replicate the collection filter logic from build_index.py (lines 713-715)."""
    return [
        p
        for p in papers
        if any(collection in c for c in p.get("collections", []))
    ]


class TestCollectionFilter:
    """Tests for --collection substring matching."""

    def test_substring_match(self):
        """'ML' matches papers in 'ML Papers' collection."""
        papers = [
            {"paper_id": "p1", "collections": ["ML Papers"]},
            {"paper_id": "p2", "collections": ["Deep Learning"]},
        ]
        result = _filter_by_collection(papers, "ML")
        assert len(result) == 1
        assert result[0]["paper_id"] == "p1"

    def test_exact_match(self):
        """Exact collection name matches."""
        papers = [
            {"paper_id": "p1", "collections": ["Network Science"]},
            {"paper_id": "p2", "collections": ["Other"]},
        ]
        result = _filter_by_collection(papers, "Network Science")
        assert len(result) == 1
        assert result[0]["paper_id"] == "p1"

    def test_no_match_returns_empty(self):
        """No matching collections returns empty list."""
        papers = [
            {"paper_id": "p1", "collections": ["Biology"]},
            {"paper_id": "p2", "collections": ["Chemistry"]},
        ]
        result = _filter_by_collection(papers, "Physics")
        assert result == []

    def test_empty_collections_excluded(self):
        """Papers with empty collections list are excluded."""
        papers = [
            {"paper_id": "p1", "collections": []},
            {"paper_id": "p2", "collections": ["Target"]},
        ]
        result = _filter_by_collection(papers, "Target")
        assert len(result) == 1
        assert result[0]["paper_id"] == "p2"

    def test_multiple_collections_any_match(self):
        """Paper with multiple collections matches if any contains the substring."""
        papers = [
            {"paper_id": "p1", "collections": ["Other", "ML Papers", "Archive"]},
        ]
        result = _filter_by_collection(papers, "ML")
        assert len(result) == 1

    def test_case_sensitive(self):
        """Filter is case-sensitive (matches build_index.py behavior)."""
        papers = [
            {"paper_id": "p1", "collections": ["ml papers"]},
            {"paper_id": "p2", "collections": ["ML Papers"]},
        ]
        result = _filter_by_collection(papers, "ML")
        assert len(result) == 1
        assert result[0]["paper_id"] == "p2"
