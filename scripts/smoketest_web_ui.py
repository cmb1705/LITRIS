#!/usr/bin/env python
"""Smoketest for the LITRIS web UI components."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

STREAMLIT_AVAILABLE = False
try:
    import streamlit
    STREAMLIT_AVAILABLE = True
except ImportError:
    pass


def test_imports() -> bool:
    """Test that all required imports work."""
    print("=" * 60)
    print("Testing imports")
    print("=" * 60)

    if STREAMLIT_AVAILABLE:
        print("  streamlit: OK")
    else:
        print("  streamlit: NOT INSTALLED (pip install streamlit)")
        print("  Skipping streamlit-dependent tests...")
        return True  # Don't fail, just note it's not installed

    try:
        from src.query.search import SearchEngine, EnrichedResult
        print("  SearchEngine: OK")
    except ImportError as e:
        print(f"  SearchEngine: FAIL ({e})")
        return False

    try:
        from src.query.retrieval import (
            format_results,
            format_paper_detail,
            format_summary,
            save_results,
        )
        print("  retrieval functions: OK")
    except ImportError as e:
        print(f"  retrieval functions: FAIL ({e})")
        return False

    try:
        from src.indexing.embeddings import CHUNK_TYPES
        print("  CHUNK_TYPES: OK")
    except ImportError as e:
        print(f"  CHUNK_TYPES: FAIL ({e})")
        return False

    return True


def test_helper_functions() -> bool:
    """Test web UI helper functions."""
    print("\n" + "=" * 60)
    print("Testing helper functions")
    print("=" * 60)

    if not STREAMLIT_AVAILABLE:
        print("  SKIPPED (streamlit not installed)")
        return True

    # Import the web_ui module
    try:
        from scripts.web_ui import (
            sanitize_csv_field,
            escape_bibtex,
            highlight_query_terms,
            results_to_csv,
            results_to_bibtex,
            check_index_exists,
        )
        print("  imports: OK")
    except ImportError as e:
        print(f"  imports: FAIL ({e})")
        return False

    # Test sanitize_csv_field
    assert sanitize_csv_field("=SUM(A1)") == "'=SUM(A1)", "CSV injection not sanitized"
    assert sanitize_csv_field("normal text") == "normal text", "Normal text altered"
    assert sanitize_csv_field("") == "", "Empty string not handled"
    print("  sanitize_csv_field: OK")

    # Test highlight_query_terms
    highlighted = highlight_query_terms("This is a test string", "test")
    assert "<mark" in highlighted, "Highlight mark not found"
    assert "test" in highlighted.lower(), "Query term not in output"
    # Test that HTML is escaped
    escaped = highlight_query_terms("<script>alert</script>", "alert")
    assert "&lt;script&gt;" in escaped, "HTML not escaped"
    print("  highlight_query_terms: OK")

    # Test escape_bibtex
    escaped = escape_bibtex("a & b")
    assert "\\&" in escaped, "BibTeX & not escaped properly"
    assert escape_bibtex("") == "", "Empty string not handled"
    print("  escape_bibtex: OK")

    # Test check_index_exists
    exists, path, has_embeddings = check_index_exists()
    print(f"  check_index_exists: OK (exists={exists}, path={path}, embeddings={has_embeddings})")

    return True


def test_mock_results() -> bool:
    """Test CSV and BibTeX conversion with mock data."""
    print("\n" + "=" * 60)
    print("Testing result conversions")
    print("=" * 60)

    if not STREAMLIT_AVAILABLE:
        print("  SKIPPED (streamlit not installed)")
        return True

    from scripts.web_ui import results_to_csv, results_to_bibtex
    from src.query.search import EnrichedResult

    # Create mock result
    mock_result = EnrichedResult(
        paper_id="test-001",
        title="Test Paper Title",
        authors="Smith, John and Doe, Jane",
        year=2022,
        collections=["Test Collection"],
        item_type="journalArticle",
        chunk_type="thesis",
        matched_text="This is the matched text from the paper.",
        score=0.85,
        paper_data={
            "journal": "Test Journal",
            "doi": "10.1000/test",
            "abstract": "Test abstract.",
        },
        extraction_data={},
    )

    # Test CSV conversion
    csv_output = results_to_csv([mock_result])
    assert "test-001" in csv_output, "Paper ID not in CSV"
    assert "Test Paper Title" in csv_output, "Title not in CSV"
    assert "Smith" in csv_output, "Author not in CSV"
    print("  results_to_csv: OK")

    # Test BibTeX conversion
    bib_output = results_to_bibtex([mock_result])
    assert "@article" in bib_output, "Article type not in BibTeX"
    assert "Test Paper Title" in bib_output, "Title not in BibTeX"
    assert "2022" in bib_output, "Year not in BibTeX"
    print("  results_to_bibtex: OK")

    return True


def test_ui_layout_functions() -> bool:
    """Test that UI layout functions are defined."""
    print("\n" + "=" * 60)
    print("Testing UI layout functions")
    print("=" * 60)

    if not STREAMLIT_AVAILABLE:
        print("  SKIPPED (streamlit not installed)")
        return True

    try:
        from scripts.web_ui import (
            inject_styles,
            render_header,
            render_index_summary,
            render_no_index_message,
            render_build_controls,
            render_active_filters,
            load_filter_options,
            execute_search,
            resolve_detail_markdown,
            save_export,
            find_similar_papers,
            build_similarity_network,
            render_similarity_network,
            PYVIS_AVAILABLE,
            main,
        )
        print("  All UI functions defined: OK")
        print(f"  PYVIS_AVAILABLE: {PYVIS_AVAILABLE}")
        return True
    except ImportError as e:
        print(f"  UI functions: FAIL ({e})")
        return False


def main_test() -> int:
    """Run all smoketests."""
    print("LITRIS Web UI Smoketest")
    print("=" * 60)

    results = {
        "imports": test_imports(),
        "helpers": test_helper_functions(),
        "conversions": test_mock_results(),
        "ui_functions": test_ui_layout_functions(),
    }

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for test, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test}: {status}")

    all_passed = all(results.values())
    print("\nWeb UI components ready!" if all_passed else "\nSome tests failed.")

    # Print usage info
    if all_passed:
        print("\nTo launch the web UI:")
        print("  streamlit run scripts/web_ui.py")
        print("\nOr with custom port:")
        print("  streamlit run scripts/web_ui.py --server.port 8501")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main_test())
