#!/usr/bin/env python
"""Smoketest for EndNote XML reference adapter."""

import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<xml>
  <records>
    <record>
      <rec-number>1</rec-number>
      <ref-type name="Journal Article">17</ref-type>
      <titles>
        <title>Test Paper</title>
        <secondary-title>Journal of Testing</secondary-title>
      </titles>
      <contributors>
        <authors>
          <author>Smith, Alice</author>
          <author>Jones, Bob</author>
        </authors>
      </contributors>
      <dates>
        <year>2022</year>
      </dates>
      <periodical>
        <full-title>Journal of Testing</full-title>
      </periodical>
      <volume>10</volume>
      <number>2</number>
      <pages>100-110</pages>
      <abstract>Test abstract.</abstract>
      <electronic-resource-num>10.1000/test</electronic-resource-num>
      <isbn>978-0-00000-000-0</isbn>
      <urls>
        <related-urls>
          <url>https://example.com/paper</url>
        </related-urls>
      </urls>
      <keywords>
        <keyword>smoketest</keyword>
        <keyword>xml</keyword>
      </keywords>
    </record>
  </records>
</xml>
"""


def write_sample(xml_path: Path) -> None:
    """Write a minimal EndNote XML export file."""
    xml_path.write_text(SAMPLE_XML, encoding="utf-8")


def test_adapter() -> bool:
    """Test basic adapter functionality."""
    print("=" * 60)
    print("Testing EndNote Adapter")
    print("=" * 60)

    from src.references.endnote_adapter import EndNoteReferenceDB

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        xml_path = tmp_path / "library.xml"
        write_sample(xml_path)

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        pdf_path = pdf_dir / "1.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%test\n")

        adapter = EndNoteReferenceDB(xml_path, pdf_dir=pdf_dir)

        count = adapter.get_paper_count()
        print(f"  Paper count: {count}")
        count_passed = count == 1

        papers = list(adapter.get_all_papers())
        paper = papers[0] if papers else None
        print(f"  Papers loaded: {len(papers)}")
        paper_passed = paper is not None and paper.title == "Test Paper"

        type_passed = paper and paper.item_type == "journalArticle"
        author_passed = paper and len(paper.authors) == 2
        tag_passed = paper and "smoketest" in paper.tags
        doi_passed = paper and paper.doi == "10.1000/test"
        url_passed = paper and paper.url == "https://example.com/paper"
        pdf_passed = paper and paper.pdf_path == pdf_path

        print(f"  Item type: {'PASS' if type_passed else 'FAIL'}")
        print(f"  Authors parsed: {'PASS' if author_passed else 'FAIL'}")
        print(f"  Tags parsed: {'PASS' if tag_passed else 'FAIL'}")
        print(f"  DOI parsed: {'PASS' if doi_passed else 'FAIL'}")
        print(f"  URL parsed: {'PASS' if url_passed else 'FAIL'}")
        print(f"  PDF matched: {'PASS' if pdf_passed else 'FAIL'}")

        by_key = adapter.get_paper_by_key("1")
        key_passed = by_key is not None and by_key.title == "Test Paper"
        print(f"  Get by record number: {'PASS' if key_passed else 'FAIL'}")

        return all(
            [
                count_passed,
                paper_passed,
                type_passed,
                author_passed,
                tag_passed,
                doi_passed,
                url_passed,
                pdf_passed,
                key_passed,
            ]
        )


def test_factory() -> bool:
    """Test factory integration."""
    print("\n" + "=" * 60)
    print("Testing Factory Integration")
    print("=" * 60)

    from src.references.factory import create_reference_db, get_available_providers

    providers = get_available_providers()
    print(f"  Available providers: {providers}")
    endnote_available = "endnote" in providers

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        xml_path = tmp_path / "library.xml"
        write_sample(xml_path)

        try:
            db = create_reference_db(
                provider="endnote",
                xml_path=xml_path,
            )
            factory_works = db.provider == "endnote"
            print(f"  Factory created: {db.provider}")
        except Exception as e:
            print(f"  Factory error: {e}")
            factory_works = False

    return endnote_available and factory_works


def main() -> int:
    print("LITRIS EndNote Adapter Smoketest")
    print("=" * 60)

    results = {
        "adapter": test_adapter(),
        "factory": test_factory(),
    }

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for test, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test}: {status}")

    all_passed = all(results.values())
    print("\nEndNote adapter is ready!" if all_passed else "\nSome tests failed.")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
