from scripts.dimensions import _checkpoint_can_reconcile_corpus, _paper_from_index_dict


def test_paper_from_index_dict_preserves_source_metadata(tmp_path):
    """Webpage-backed papers must retain local source metadata from papers.json."""

    html_path = tmp_path / "paper.html"
    html_path.write_text("<html><body>test</body></html>", encoding="utf-8")

    paper = _paper_from_index_dict(
        "ABCD1234_EFGH5678",
        {
            "zotero_key": "ABCD1234",
            "zotero_item_id": 123,
            "item_type": "webpage",
            "title": "HTML source test",
            "authors": [],
            "collections": [],
            "tags": [],
            "pdf_path": None,
            "pdf_attachment_key": None,
            "source_path": str(html_path),
            "source_attachment_key": "EFGH5678",
            "source_media_type": "text/html",
            "date_added": "2026-01-01T00:00:00",
            "date_modified": "2026-01-02T00:00:00",
        },
    )

    assert paper.pdf_path is None
    assert paper.source_path == html_path
    assert paper.source_attachment_key == "EFGH5678"
    assert paper.source_media_type == "text/html"
    assert paper.has_extractable_source


def test_checkpoint_can_reconcile_corpus_for_same_full_scope_request():
    """Full-corpus checkpoints may resume after the corpus changes."""

    state_metadata = {
        "paper_ids": ["A", "B"],
        "old_profile_id": "legacy_semantic_v1",
        "old_profile_fingerprint": "old",
        "target_profile_id": "stp_cas_v1",
        "target_profile_fingerprint": "new",
        "changed_sections": ["impact", "scholarly"],
        "reextract_sections": ["impact", "scholarly"],
        "partial_scope": False,
        "refresh_text": False,
        "fulltext_only": False,
    }
    request_metadata = {
        **state_metadata,
        "paper_ids": ["A", "B", "C"],
    }

    assert _checkpoint_can_reconcile_corpus(state_metadata, request_metadata)


def test_checkpoint_can_reconcile_corpus_rejects_partial_scope():
    """Subset backfills must continue to require an exact checkpoint match."""

    state_metadata = {
        "paper_ids": ["A", "B"],
        "old_profile_id": "legacy_semantic_v1",
        "old_profile_fingerprint": "old",
        "target_profile_id": "stp_cas_v1",
        "target_profile_fingerprint": "new",
        "changed_sections": ["impact", "scholarly"],
        "reextract_sections": ["impact", "scholarly"],
        "partial_scope": True,
        "refresh_text": False,
        "fulltext_only": False,
    }
    request_metadata = {
        **state_metadata,
        "paper_ids": ["A", "B", "C"],
    }

    assert not _checkpoint_can_reconcile_corpus(state_metadata, request_metadata)


def test_checkpoint_can_reconcile_corpus_allows_added_and_removed_ids():
    """Full-scope checkpoint reuse tolerates current-corpus churn."""

    state_metadata = {
        "paper_ids": ["A", "B", "C"],
        "old_profile_id": "legacy_semantic_v1",
        "old_profile_fingerprint": "old",
        "target_profile_id": "stp_cas_v1",
        "target_profile_fingerprint": "new",
        "changed_sections": ["impact", "scholarly"],
        "reextract_sections": ["impact", "scholarly"],
        "partial_scope": False,
        "refresh_text": False,
        "fulltext_only": False,
    }
    request_metadata = {
        **state_metadata,
        "paper_ids": ["B", "C", "D"],
    }

    assert _checkpoint_can_reconcile_corpus(state_metadata, request_metadata)
