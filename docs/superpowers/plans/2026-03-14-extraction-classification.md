# Extraction Cascade, Classification Index, and Build Gating -- Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the existing dead extraction cascade into the active pipeline, add a classification pre-pass to filter non-academic content, and gate `build_index` on classification results.

**Architecture:** Three features layered bottom-up. (1) The extraction cascade (`cascade.py`, already implemented) gets a new Companion tier and is wired into `SectionExtractor` to replace direct `PDFExtractor` calls. (2) A classification store persists per-paper document type and extractability to `classification_index.json`. (3) `build_index.py` loads the classification index to skip non-academic papers before expensive LLM extraction.

**Tech Stack:** Python 3.10+, Pydantic, PyMuPDF, marker-pdf (optional), existing `document_classifier.py`

**Spec:** `docs/superpowers/specs/2026-03-14-extraction-classification-design.md`

---

## Chunk 1: Foundation -- Config, Schema, and Cascade Dataclass Changes

### Task 1: Add cascade config fields to ProcessingConfig

**Files:**
- Modify: `src/config.py:232-244` (ProcessingConfig class)
- Modify: `config.yaml:49-71` (processing section)

- [ ] **Step 1: Write the failing test**

Create `tests/test_cascade_config.py`:

```python
"""Tests for cascade configuration fields in ProcessingConfig."""

from pathlib import Path

from src.config import ProcessingConfig


def test_processing_config_cascade_defaults():
    """ProcessingConfig has cascade fields with correct defaults."""
    config = ProcessingConfig()
    assert config.cascade_enabled is True
    assert config.companion_dir is None
    assert config.arxiv_enabled is True
    assert config.marker_enabled is True


def test_processing_config_cascade_overrides():
    """Cascade fields can be set via constructor."""
    config = ProcessingConfig(
        cascade_enabled=False,
        companion_dir=Path("/tmp/companions"),
        arxiv_enabled=False,
        marker_enabled=False,
    )
    assert config.cascade_enabled is False
    assert config.companion_dir == Path("/tmp/companions")
    assert config.arxiv_enabled is False
    assert config.marker_enabled is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cascade_config.py -v`
Expected: FAIL -- `ProcessingConfig` does not have these fields yet.

- [ ] **Step 3: Add fields to ProcessingConfig**

In `src/config.py`, add to the `ProcessingConfig` class (after line 244):

```python
    cascade_enabled: bool = True
    companion_dir: Path | None = None
    arxiv_enabled: bool = True
    marker_enabled: bool = True
```

- [ ] **Step 4: Add defaults to config.yaml**

In `config.yaml`, add under the `processing:` section (after `min_section_hits: 0`, before `classification:`):

```yaml
  cascade_enabled: true
  companion_dir: null
  arxiv_enabled: true
  marker_enabled: true
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_cascade_config.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/config.py config.yaml tests/test_cascade_config.py
git commit -m "feat: add cascade config fields to ProcessingConfig"
```

---

### Task 2: Add extraction_method field to ExtractionResult

**Files:**
- Modify: `src/analysis/schemas.py:505-546` (ExtractionResult class)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cascade_config.py`:

```python
from src.analysis.schemas import ExtractionResult


def test_extraction_result_has_extraction_method():
    """ExtractionResult accepts optional extraction_method field."""
    result = ExtractionResult(
        paper_id="test-001",
        success=True,
        extraction_method="companion",
    )
    assert result.extraction_method == "companion"


def test_extraction_result_extraction_method_default_none():
    """ExtractionResult defaults extraction_method to None."""
    result = ExtractionResult(paper_id="test-002", success=False)
    assert result.extraction_method is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cascade_config.py::test_extraction_result_has_extraction_method -v`
Expected: FAIL -- `ExtractionResult` does not accept `extraction_method`.

- [ ] **Step 3: Add field to ExtractionResult**

In `src/analysis/schemas.py`, add after the `output_tokens` field (around line 545):

```python
    extraction_method: str | None = Field(
        default=None,
        description="Cascade tier that produced the text (e.g. companion, marker, pymupdf).",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cascade_config.py -v`
Expected: PASS (all 4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/analysis/schemas.py tests/test_cascade_config.py
git commit -m "feat: add extraction_method field to ExtractionResult"
```

---

### Task 3: Add is_markdown field and companion method to CascadeResult

**Files:**
- Modify: `src/extraction/cascade.py:32-52` (CascadeMethod, CascadeResult)

- [ ] **Step 1: Write the failing test**

Create `tests/test_cascade_wiring.py`:

```python
"""Tests for extraction cascade wiring and companion tier."""

from src.extraction.cascade import CascadeMethod, CascadeResult


def test_cascade_result_has_is_markdown():
    """CascadeResult includes is_markdown field defaulting to False."""
    result = CascadeResult(
        text="hello world " * 50,
        method="pymupdf",
        word_count=100,
        tiers_attempted=["pymupdf"],
    )
    assert result.is_markdown is False


def test_cascade_result_is_markdown_true():
    """CascadeResult can set is_markdown=True."""
    result = CascadeResult(
        text="# Title\n\nBody",
        method="companion",
        word_count=2,
        tiers_attempted=["companion"],
        is_markdown=True,
    )
    assert result.is_markdown is True
    assert result.method == "companion"


def test_companion_is_valid_cascade_method():
    """'companion' is a valid CascadeMethod literal value."""
    # This validates at type-check time, but we can verify at runtime
    # by constructing a CascadeResult with method="companion"
    result = CascadeResult(
        text="test",
        method="companion",
        word_count=1,
        tiers_attempted=["companion"],
        is_markdown=True,
    )
    assert result.method == "companion"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cascade_wiring.py -v`
Expected: FAIL -- `CascadeResult` has no `is_markdown` field, `"companion"` not in `CascadeMethod`.

- [ ] **Step 3: Update CascadeMethod and CascadeResult**

In `src/extraction/cascade.py`:

Replace the `CascadeMethod` literal (lines 32-39):

```python
CascadeMethod = Literal[
    "companion",
    "arxiv_html",
    "ar5iv",
    "marker",
    "pymupdf",
    "ocr",
    "hybrid",
]
```

Add `is_markdown` to `CascadeResult` (after line 52):

```python
@dataclass
class CascadeResult:
    """Result of cascade extraction with provenance."""

    text: str
    method: CascadeMethod
    word_count: int
    tiers_attempted: list[str]
    is_markdown: bool = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cascade_wiring.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/extraction/cascade.py tests/test_cascade_wiring.py
git commit -m "feat: add companion method and is_markdown to CascadeResult"
```

---

### Task 4: Install marker-pdf dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add marker-pdf to requirements.txt**

Add at the end of the file, after the last dependency:

```
marker-pdf>=1.0,<2.0  # optional: ML-based PDF to markdown
```

- [ ] **Step 2: Install**

Run: `pip install marker-pdf`

Note: This is optional. If it fails to install (GPU dependencies, etc.), that is acceptable -- the cascade skips marker when unavailable.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "feat: add marker-pdf optional dependency"
```

---

## Chunk 2: Cascade Wiring -- Companion Tier, TextCleaner, SectionExtractor

### Task 5: Implement Companion tier in ExtractionCascade

**Files:**
- Modify: `src/extraction/cascade.py:63-69` (constructor), `cascade.py:89-179` (extract_text)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cascade_wiring.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.extraction.cascade import ExtractionCascade
from src.extraction.pdf_extractor import PDFExtractor


def test_companion_tier_finds_md_file(tmp_path):
    """Companion tier reads .md file alongside PDF."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_text("dummy pdf")
    md = tmp_path / "paper.md"
    md.write_text("# Title\n\n" + "Body text. " * 50)

    pdf_extractor = MagicMock(spec=PDFExtractor)
    cascade = ExtractionCascade(pdf_extractor, companion_dir=None)

    result = cascade.extract_text(pdf)
    assert result.method == "companion"
    assert result.is_markdown is True
    assert "Body text" in result.text
    # PDFExtractor should NOT have been called
    pdf_extractor.extract_text_with_method.assert_not_called()


def test_companion_tier_checks_companion_dir(tmp_path):
    """Companion tier checks configured companion_dir."""
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    pdf = pdf_dir / "paper.pdf"
    pdf.write_text("dummy pdf")

    comp_dir = tmp_path / "companions"
    comp_dir.mkdir()
    md = comp_dir / "paper.md"
    md.write_text("# Title\n\n" + "Content here. " * 50)

    pdf_extractor = MagicMock(spec=PDFExtractor)
    cascade = ExtractionCascade(pdf_extractor, companion_dir=comp_dir)

    result = cascade.extract_text(pdf)
    assert result.method == "companion"
    assert result.is_markdown is True


def test_companion_tier_skips_insufficient_text(tmp_path):
    """Companion tier falls through when .md has too few words."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_text("dummy pdf")
    md = tmp_path / "paper.md"
    md.write_text("Short.")  # Below min_words threshold

    pdf_extractor = MagicMock(spec=PDFExtractor)
    pdf_extractor.extract_text_with_method.return_value = (
        "Fallback text. " * 50,
        "pymupdf",
    )

    cascade = ExtractionCascade(pdf_extractor, companion_dir=None)
    result = cascade.extract_text(pdf)
    assert result.method == "pymupdf"
    assert "companion" in result.tiers_attempted


def test_companion_tier_not_found_falls_through(tmp_path):
    """No companion .md file -- cascade falls through to next tier."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_text("dummy pdf")
    # No .md file

    pdf_extractor = MagicMock(spec=PDFExtractor)
    pdf_extractor.extract_text_with_method.return_value = (
        "Extracted text. " * 50,
        "pymupdf",
    )

    cascade = ExtractionCascade(
        pdf_extractor, enable_arxiv=False, enable_marker=False,
    )
    result = cascade.extract_text(pdf)
    assert result.method == "pymupdf"
    assert "companion" not in result.tiers_attempted
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cascade_wiring.py -k "companion_tier" -v`
Expected: FAIL -- `ExtractionCascade.__init__` does not accept `companion_dir`.

- [ ] **Step 3: Implement companion tier**

In `src/extraction/cascade.py`, modify `__init__` (line 63):

```python
    def __init__(
        self,
        pdf_extractor: PDFExtractor,
        enable_arxiv: bool = True,
        enable_marker: bool = True,
        arxiv_timeout: int = 30,
        min_words: int = MIN_EXTRACTION_WORDS,
        companion_dir: Path | None = None,
    ):
```

Add `self.companion_dir = companion_dir` after line 84.

In `extract_text()`, add companion tier BEFORE the arXiv detection block (before line 112). Insert after `tiers_attempted: list[str] = []` (line 110):

```python
        # Tier 0 (Priority 1): Companion .md file
        companion_path = self._find_companion(pdf_path)
        if companion_path:
            tiers_attempted.append("companion")
            try:
                text = companion_path.read_text(encoding="utf-8")
                if self._is_sufficient(text):
                    logger.info(f"Companion file found: {companion_path.name}")
                    return CascadeResult(
                        text=text,
                        method="companion",
                        word_count=len(text.split()),
                        tiers_attempted=tiers_attempted,
                        is_markdown=True,
                    )
                else:
                    logger.debug(
                        f"Companion file {companion_path.name} has insufficient text"
                    )
            except OSError as e:
                logger.warning(f"Failed to read companion file: {e}")
```

Add `is_markdown=True` to the Marker tier return (around the existing line 150):

```python
                return CascadeResult(
                    text=text,
                    method="marker",
                    word_count=len(text.split()),
                    tiers_attempted=tiers_attempted,
                    is_markdown=True,
                )
```

Add the `_find_companion` helper method at the end of the class:

```python
    def _find_companion(self, pdf_path: Path) -> Path | None:
        """Find a companion .md file for the given PDF.

        Checks:
        1. Same directory as PDF with .md extension
        2. companion_dir (if configured) with same stem

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Path to companion .md file, or None if not found.
        """
        # Check alongside PDF
        md_path = pdf_path.with_suffix(".md")
        if md_path.is_file():
            return md_path

        # Check companion directory
        if self.companion_dir and self.companion_dir.is_dir():
            md_path = self.companion_dir / f"{pdf_path.stem}.md"
            if md_path.is_file():
                return md_path

        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cascade_wiring.py -v`
Expected: PASS (all 7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/extraction/cascade.py tests/test_cascade_wiring.py
git commit -m "feat: add companion tier to ExtractionCascade"
```

---

### Task 6: Add preserve_markdown parameter to TextCleaner

**Files:**
- Modify: `src/extraction/text_cleaner.py:53-86` (clean method)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_text_cleaner_markdown.py`:

```python
"""Tests for TextCleaner markdown preservation."""

from src.extraction.text_cleaner import TextCleaner


def test_clean_preserves_short_lines_in_markdown():
    """preserve_markdown=True keeps short lines (table rows, list items)."""
    cleaner = TextCleaner()
    text = "# Title\n\n- item 1\n- item 2\n- item 3\n\n| A | B |\n|---|---|\n| 1 | 2 |"
    result = cleaner.clean(text, preserve_markdown=True)
    assert "- item 1" in result
    assert "| A | B |" in result


def test_clean_removes_short_lines_by_default():
    """Default clean() removes short lines as artifacts."""
    cleaner = TextCleaner()
    text = "This is a long enough line that should be kept.\n- short\n- tiny"
    result = cleaner.clean(text)
    assert "- short" not in result  # Below min_line_length=10


def test_clean_preserves_markdown_keeps_hyphen_fix():
    """preserve_markdown still fixes hyphenated line breaks."""
    cleaner = TextCleaner()
    text = "hyphen-\nated word"
    result = cleaner.clean(text, preserve_markdown=True)
    assert "hyphenated" in result


def test_clean_preserves_markdown_skips_header_footer():
    """preserve_markdown skips HEADER_FOOTER regex removal."""
    cleaner = TextCleaner()
    # This would match HEADER_FOOTER pattern in normal mode
    text = "# Introduction\n\nSome body text that is long enough to keep.\n\npage 1"
    result_normal = cleaner.clean(text, preserve_markdown=False)
    result_md = cleaner.clean(text, preserve_markdown=True)
    # In markdown mode, "page 1" line should still be there
    # (HEADER_FOOTER regex could match it but is skipped)
    assert "page 1" in result_md
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_text_cleaner_markdown.py -v`
Expected: FAIL -- `clean()` does not accept `preserve_markdown` parameter.

- [ ] **Step 3: Add preserve_markdown parameter**

In `src/extraction/text_cleaner.py`, modify the `clean` method signature (line 53):

```python
    def clean(self, text: str, preserve_markdown: bool = False) -> str:
```

Modify the cleaning logic (lines 62-86). Replace the body of `clean` with:

```python
        if not text:
            return ""

        # Fix hyphenated line breaks (safe for all content types)
        if self.fix_hyphenation:
            text = self.HYPHEN_LINEBREAK.sub(r"\1\2", text)

        # Remove page numbers and headers/footers
        if self.remove_headers_footers:
            text = self.PAGE_NUMBERS.sub("", text)
            if not preserve_markdown:
                # HEADER_FOOTER regex can destroy markdown headers -- skip in md mode
                text = self.HEADER_FOOTER.sub("", text)

        # Normalize whitespace
        text = self.MULTIPLE_SPACES.sub(" ", text)
        text = self.MULTIPLE_NEWLINES.sub("\n\n", text)

        if not preserve_markdown:
            # Filter short lines (destroys markdown list items, table rows)
            lines = text.split("\n")
            lines = [
                line for line in lines
                if len(line.strip()) >= self.min_line_length or not line.strip()
            ]
            text = "\n".join(lines)

        return text.strip()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_text_cleaner_markdown.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run existing text_cleaner tests for regression**

Run: `python -m pytest tests/ -k "text_cleaner or cleaner" -v`
Expected: All existing tests still pass (the default `preserve_markdown=False` preserves old behavior).

- [ ] **Step 6: Commit**

```bash
git add src/extraction/text_cleaner.py tests/test_text_cleaner_markdown.py
git commit -m "feat: add preserve_markdown parameter to TextCleaner.clean()"
```

---

### Task 7: Wire ExtractionCascade into SectionExtractor

**Files:**
- Modify: `src/analysis/section_extractor.py:220-300` (\_\_init\_\_), `section_extractor.py:338-340` (extract_paper)

This is the critical wiring change. SectionExtractor currently calls `PDFExtractor.extract_text_with_method()` directly. We replace this with `ExtractionCascade.extract_text()` when `cascade_enabled=True`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cascade_wiring.py`:

```python
from src.analysis.section_extractor import SectionExtractor


@patch("src.analysis.section_extractor.create_llm_client")
def test_section_extractor_creates_cascade_when_enabled(mock_factory):
    """SectionExtractor creates ExtractionCascade when cascade_enabled."""
    mock_factory.return_value = MagicMock()
    extractor = SectionExtractor(cascade_enabled=True)
    assert hasattr(extractor, "cascade")
    assert extractor.cascade is not None


@patch("src.analysis.section_extractor.create_llm_client")
def test_section_extractor_no_cascade_when_disabled(mock_factory):
    """SectionExtractor uses PDFExtractor directly when cascade disabled."""
    mock_factory.return_value = MagicMock()
    extractor = SectionExtractor(cascade_enabled=False)
    assert extractor.cascade is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cascade_wiring.py -k "section_extractor" -v`
Expected: FAIL -- `SectionExtractor.__init__` does not accept `cascade_enabled`.

- [ ] **Step 3: Wire the cascade into SectionExtractor**

In `src/analysis/section_extractor.py`:

Add import at top (after existing extraction imports):
```python
from src.extraction.cascade import ExtractionCascade
```

Add parameters to `__init__` signature (after `effort: str | None = None`):
```python
        cascade_enabled: bool = True,
        companion_dir: Path | None = None,
        arxiv_enabled: bool = True,
        marker_enabled: bool = True,
```

After `self.text_cleaner = TextCleaner()` (line 268), add:
```python
        # Extraction cascade (wraps pdf_extractor with higher-quality tiers)
        self.cascade: ExtractionCascade | None = None
        if cascade_enabled:
            self.cascade = ExtractionCascade(
                pdf_extractor=self.pdf_extractor,
                enable_arxiv=arxiv_enabled,
                enable_marker=marker_enabled,
                min_words=min_text_length,
                companion_dir=companion_dir,
            )
```

In `extract_paper()`, replace the PDF extraction block (lines 337-347). Find the existing:
```python
            text, method = self.pdf_extractor.extract_text_with_method(
                paper.pdf_path
            )
```

Replace with:
```python
            cascade_result = None
            if self.cascade:
                from src.extraction.cascade import CascadeResult
                cascade_result = self.cascade.extract_text(
                    paper.pdf_path,
                    doi=getattr(paper, "doi", None),
                    url=getattr(paper, "url", None),
                )
                text = cascade_result.text
                method = cascade_result.method
            else:
                text, method = self.pdf_extractor.extract_text_with_method(
                    paper.pdf_path
                )
```

Find the `text = self.text_cleaner.clean(text)` call (line 382) and replace with:
```python
            preserve_md = cascade_result.is_markdown if cascade_result else False
            text = self.text_cleaner.clean(text, preserve_markdown=preserve_md)
```

Find where `ExtractionResult` is constructed for success cases in `extract_paper()` and add `extraction_method`:
```python
                extraction_method=cascade_result.method if cascade_result else method,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cascade_wiring.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Run full test suite regression**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All existing tests pass. The cascade defaults to enabled but should not break anything since it falls through to PyMuPDF (same as before).

- [ ] **Step 6: Commit**

```bash
git add src/analysis/section_extractor.py tests/test_cascade_wiring.py
git commit -m "feat: wire ExtractionCascade into SectionExtractor"
```

---

## Chunk 3: Classification Store

### Task 8: Build ClassificationStore module

**Files:**
- Create: `src/analysis/classification_store.py`
- Test: `tests/test_classification_store.py`

This module manages the `classification_index.json` file. It uses the existing `classify()` function from `document_classifier.py` -- no new classification logic.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_classification_store.py`:

```python
"""Tests for ClassificationStore -- persistent classification index."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from src.analysis.classification_store import (
    ClassificationIndex,
    ClassificationRecord,
    ClassificationStore,
)


@pytest.fixture
def store(tmp_path):
    """ClassificationStore pointed at a temp directory."""
    return ClassificationStore(index_dir=tmp_path)


@pytest.fixture
def sample_record():
    """A sample ClassificationRecord."""
    return ClassificationRecord(
        title="Test Paper",
        item_type="journalArticle",
        document_type="RESEARCH_PAPER",
        confidence=0.90,
        tier=1,
        reasons=["item_type=journalArticle"],
        extractable=True,
        word_count=5000,
        page_count=10,
        section_markers=5,
        classified_at=datetime.now().isoformat(),
    )


class TestClassificationRecord:
    def test_extractable_true_for_academic(self, sample_record):
        assert sample_record.extractable is True

    def test_extractable_false_for_non_academic(self):
        rec = ClassificationRecord(
            title="Course Slides",
            item_type="presentation",
            document_type="NON_ACADEMIC",
            confidence=0.95,
            tier=1,
            reasons=["item_type=presentation"],
            extractable=False,
            word_count=200,
            page_count=30,
            section_markers=0,
            classified_at=datetime.now().isoformat(),
        )
        assert rec.extractable is False


class TestClassificationStoreIO:
    def test_load_returns_empty_when_no_file(self, store):
        index = store.load()
        assert index.papers == {}
        assert index.schema_version == "1.0.0"

    def test_save_and_load_roundtrip(self, store, sample_record):
        index = ClassificationIndex()
        index.papers["paper-001"] = sample_record
        store.save(index)

        loaded = store.load()
        assert "paper-001" in loaded.papers
        assert loaded.papers["paper-001"].document_type == "RESEARCH_PAPER"
        assert loaded.stats["total"] == 1

    def test_save_recomputes_stats(self, store, sample_record):
        index = ClassificationIndex()
        index.papers["p1"] = sample_record
        non_academic = ClassificationRecord(
            title="Slides",
            item_type="presentation",
            document_type="NON_ACADEMIC",
            confidence=0.95,
            tier=1,
            reasons=[],
            extractable=False,
            word_count=100,
            page_count=1,
            section_markers=0,
            classified_at=datetime.now().isoformat(),
        )
        index.papers["p2"] = non_academic
        store.save(index)

        loaded = store.load()
        assert loaded.stats["total"] == 2
        assert loaded.stats["extractable_count"] == 1
        assert loaded.stats["non_extractable_count"] == 1
        assert loaded.stats["by_type"]["RESEARCH_PAPER"] == 1
        assert loaded.stats["by_type"]["NON_ACADEMIC"] == 1

    def test_load_corrupt_file_raises(self, store):
        path = store.index_path
        path.write_text("not json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            store.load()

    def test_save_creates_parent_dir(self, tmp_path):
        nested = tmp_path / "a" / "b"
        s = ClassificationStore(index_dir=nested)
        s.save(ClassificationIndex())
        assert s.index_path.exists()


class TestClassificationStoreQuery:
    def test_get_extractable_ids(self, store, sample_record):
        index = ClassificationIndex()
        index.papers["p1"] = sample_record
        non_ext = ClassificationRecord(
            title="X",
            item_type="presentation",
            document_type="NON_ACADEMIC",
            confidence=0.95,
            tier=1,
            reasons=[],
            extractable=False,
            word_count=50,
            page_count=1,
            section_markers=0,
            classified_at=datetime.now().isoformat(),
        )
        index.papers["p2"] = non_ext
        store.save(index)

        loaded = store.load()
        ids = store.get_extractable_ids(loaded)
        assert ids == {"p1"}

    def test_get_paper(self, store, sample_record):
        index = ClassificationIndex()
        index.papers["p1"] = sample_record
        assert store.get_paper(index, "p1") == sample_record
        assert store.get_paper(index, "missing") is None


class TestClassificationStoreClassify:
    @patch("src.analysis.classification_store.classify")
    def test_classify_paper_uses_existing_classifier(self, mock_classify, store):
        from src.zotero.models import Author, PaperMetadata

        from src.analysis.document_types import DocumentType
        mock_classify.return_value = (DocumentType.RESEARCH_PAPER, 0.85)

        paper = PaperMetadata(
            zotero_key="ZK001",
            zotero_item_id="ZI001",
            title="Test Paper",
            item_type="journalArticle",
            date_added="2026-01-01",
            date_modified="2026-01-01",
            authors=[Author(first_name="A", last_name="B")],
        )

        record = store.classify_paper(paper, text=None)
        assert record.document_type == "RESEARCH_PAPER"
        assert record.confidence == 0.85
        mock_classify.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_classification_store.py -v`
Expected: FAIL -- module `src.analysis.classification_store` does not exist.

- [ ] **Step 3: Implement ClassificationStore**

Create `src/analysis/classification_store.py`:

```python
"""Persistent classification index for document type pre-filtering.

Stores per-paper classification results (document type, confidence,
extractability) to disk as JSON. Used by build_index.py to skip
non-academic content before expensive LLM extraction.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.analysis.document_classifier import classify
from src.utils.logging_config import get_logger

if TYPE_CHECKING:
    from src.references.models import PaperMetadata

logger = get_logger(__name__)

INDEX_FILENAME = "classification_index.json"
SCHEMA_VERSION = "1.0.0"


@dataclass
class ClassificationRecord:
    """Classification result for a single paper."""

    title: str
    item_type: str
    document_type: str
    confidence: float
    tier: int
    reasons: list[str]
    extractable: bool
    word_count: int | None
    page_count: int | None
    section_markers: int | None
    classified_at: str
    classification_error: str | None = None
    extraction_method: str | None = None


@dataclass
class ClassificationIndex:
    """In-memory representation of the classification index."""

    schema_version: str = SCHEMA_VERSION
    classified_at: str = ""
    stats: dict = field(default_factory=dict)
    papers: dict[str, ClassificationRecord] = field(default_factory=dict)


def _is_extractable(
    document_type: str,
    word_count: int | None,
    page_count: int | None,
    min_publication_words: int = 500,
    min_publication_pages: int = 2,
) -> bool:
    """Determine if a paper should be extracted.

    A paper is extractable when ALL of:
    - document_type is not NON_ACADEMIC
    - word_count >= min_publication_words (if known)
    - page_count >= min_publication_pages (if known)
    """
    if document_type == "NON_ACADEMIC":
        return False
    if word_count is not None and word_count < min_publication_words:
        return False
    if page_count is not None and page_count < min_publication_pages:
        return False
    return True


def _compute_stats(papers: dict[str, ClassificationRecord]) -> dict:
    """Recompute summary statistics from paper records."""
    by_type: dict[str, int] = {}
    extractable_count = 0
    non_extractable_count = 0

    for record in papers.values():
        by_type[record.document_type] = by_type.get(record.document_type, 0) + 1
        if record.extractable:
            extractable_count += 1
        else:
            non_extractable_count += 1

    return {
        "total": len(papers),
        "by_type": by_type,
        "extractable_count": extractable_count,
        "non_extractable_count": non_extractable_count,
    }


class ClassificationStore:
    """Read/write/query the classification index on disk."""

    def __init__(
        self,
        index_dir: Path,
        min_publication_words: int = 500,
        min_publication_pages: int = 2,
    ):
        """Initialize store.

        Args:
            index_dir: Directory containing classification_index.json.
            min_publication_words: Minimum word count for extractability.
            min_publication_pages: Minimum page count for extractability.
        """
        self.index_dir = index_dir
        self.index_path = index_dir / INDEX_FILENAME
        self.min_publication_words = min_publication_words
        self.min_publication_pages = min_publication_pages

    def load(self) -> ClassificationIndex:
        """Load classification index from disk.

        Returns:
            ClassificationIndex (empty if file does not exist).

        Raises:
            json.JSONDecodeError: If the file is corrupt.
        """
        if not self.index_path.exists():
            return ClassificationIndex()

        data = json.loads(self.index_path.read_text(encoding="utf-8"))
        index = ClassificationIndex(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            classified_at=data.get("classified_at", ""),
            stats=data.get("stats", {}),
        )
        for paper_id, rec_data in data.get("papers", {}).items():
            index.papers[paper_id] = ClassificationRecord(**rec_data)
        return index

    def save(self, index: ClassificationIndex) -> None:
        """Save classification index to disk with recomputed stats.

        Args:
            index: Classification index to save.
        """
        index.classified_at = datetime.now().isoformat()
        index.stats = _compute_stats(index.papers)

        data = {
            "schema_version": index.schema_version,
            "classified_at": index.classified_at,
            "stats": index.stats,
            "papers": {
                pid: asdict(rec) for pid, rec in index.papers.items()
            },
        }

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def classify_paper(
        self,
        paper: PaperMetadata,
        text: str | None = None,
        word_count: int | None = None,
        page_count: int | None = None,
        section_markers: int | None = None,
    ) -> ClassificationRecord:
        """Classify a single paper using the existing classifier.

        Args:
            paper: Paper metadata.
            text: Optional full text (triggers Tier 2 if needed).
            word_count: Precomputed word count.
            page_count: Precomputed page count.
            section_markers: Precomputed section marker count.

        Returns:
            ClassificationRecord with classification results.
        """
        doc_type, confidence = classify(
            paper=paper,
            text=text,
            word_count=word_count,
            page_count=page_count,
            section_marker_count=section_markers,
        )

        tier = 1 if text is None else 2
        extractable = _is_extractable(
            doc_type.value if hasattr(doc_type, "value") else str(doc_type),
            word_count,
            page_count,
            self.min_publication_words,
            self.min_publication_pages,
        )

        doc_type_str = doc_type.value if hasattr(doc_type, "value") else str(doc_type)

        return ClassificationRecord(
            title=paper.title,
            item_type=paper.item_type,
            document_type=doc_type_str,
            confidence=confidence,
            tier=tier,
            reasons=[],  # Could be enriched later
            extractable=extractable,
            word_count=word_count,
            page_count=page_count,
            section_markers=section_markers,
            classified_at=datetime.now().isoformat(),
        )

    @staticmethod
    def get_extractable_ids(index: ClassificationIndex) -> set[str]:
        """Return IDs of all extractable papers.

        Args:
            index: Classification index.

        Returns:
            Set of paper IDs where extractable is True.
        """
        return {
            pid for pid, rec in index.papers.items() if rec.extractable
        }

    @staticmethod
    def get_paper(
        index: ClassificationIndex, paper_id: str
    ) -> ClassificationRecord | None:
        """Look up a single paper's classification.

        Args:
            index: Classification index.
            paper_id: Paper ID to look up.

        Returns:
            ClassificationRecord or None if not found.
        """
        return index.papers.get(paper_id)

    @staticmethod
    def get_stats(index: ClassificationIndex) -> dict:
        """Return summary statistics.

        Args:
            index: Classification index.

        Returns:
            Stats dictionary with totals and breakdowns.
        """
        return index.stats
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_classification_store.py -v`
Expected: PASS (all ~11 tests)

- [ ] **Step 5: Commit**

```bash
git add src/analysis/classification_store.py tests/test_classification_store.py
git commit -m "feat: add ClassificationStore for persistent document classification"
```

---

## Chunk 4: Build Index Integration -- CLI Flags, Gating, and Report

### Task 9: Add --classify-only, --index-all, and --reclassify flags

**Files:**
- Modify: `scripts/build_index.py:35-196` (argparse), `build_index.py:700-720` (main logic)

- [ ] **Step 1: Add CLI flags to argparse**

In `scripts/build_index.py`, add after the `--collection` argument (around line 195):

```python
    parser.add_argument(
        "--classify-only",
        action="store_true",
        help="Run classification pre-pass only (no extraction). "
        "Builds/updates classification_index.json.",
    )
    parser.add_argument(
        "--index-all",
        action="store_true",
        help="Extract all papers regardless of classification. "
        "Default behavior filters to academic content only.",
    )
    parser.add_argument(
        "--reclassify",
        action="store_true",
        help="Force re-classification of all papers (use with --classify-only).",
    )
```

- [ ] **Step 2: Commit**

```bash
git add scripts/build_index.py
git commit -m "feat: add --classify-only, --index-all, --reclassify CLI flags"
```

---

### Task 10: Implement --classify-only mode

**Files:**
- Modify: `scripts/build_index.py` (main function)
- Test: `tests/test_build_gating.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_build_gating.py`:

```python
"""Tests for build_index classification gating."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.analysis.classification_store import (
    ClassificationIndex,
    ClassificationRecord,
    ClassificationStore,
)


@pytest.fixture
def index_dir(tmp_path):
    return tmp_path / "index"


@pytest.fixture
def store(index_dir):
    return ClassificationStore(index_dir=index_dir)


def _make_record(doc_type="RESEARCH_PAPER", extractable=True, word_count=5000):
    from datetime import datetime
    return ClassificationRecord(
        title="Test",
        item_type="journalArticle",
        document_type=doc_type,
        confidence=0.9,
        tier=1,
        reasons=[],
        extractable=extractable,
        word_count=word_count,
        page_count=10,
        section_markers=5,
        classified_at=datetime.now().isoformat(),
    )


class TestGatingFilter:
    def test_academic_only_filters_non_extractable(self, store, index_dir):
        index = ClassificationIndex()
        index.papers["p1"] = _make_record(extractable=True)
        index.papers["p2"] = _make_record(
            doc_type="NON_ACADEMIC", extractable=False, word_count=100,
        )
        store.save(index)

        loaded = store.load()
        ids = store.get_extractable_ids(loaded)
        assert ids == {"p1"}

    def test_index_all_keeps_everything(self, store, index_dir):
        index = ClassificationIndex()
        index.papers["p1"] = _make_record(extractable=True)
        index.papers["p2"] = _make_record(
            doc_type="NON_ACADEMIC", extractable=False,
        )
        store.save(index)

        loaded = store.load()
        # --index-all means we use all paper IDs, not just extractable
        all_ids = set(loaded.papers.keys())
        assert all_ids == {"p1", "p2"}

    def test_classify_only_produces_index_file(self, store, index_dir):
        """Verifies classification store creates the index file."""
        index = ClassificationIndex()
        index.papers["p1"] = _make_record()
        store.save(index)
        assert (index_dir / "classification_index.json").exists()

    def test_reclassify_replaces_all_records(self, store, index_dir):
        """--reclassify replaces existing records."""
        index = ClassificationIndex()
        index.papers["p1"] = _make_record()
        store.save(index)

        # Simulate reclassification
        new_index = ClassificationIndex()
        new_record = _make_record(doc_type="REVIEW_PAPER")
        new_index.papers["p1"] = new_record
        store.save(new_index)

        loaded = store.load()
        assert loaded.papers["p1"].document_type == "REVIEW_PAPER"


class TestSkipReport:
    def test_skip_summary_counts(self):
        """Skip summary correctly counts reasons."""
        index = ClassificationIndex()
        index.papers["p1"] = _make_record(extractable=True)
        index.papers["p2"] = _make_record(
            doc_type="NON_ACADEMIC", extractable=False,
        )
        index.papers["p3"] = _make_record(
            doc_type="RESEARCH_PAPER", extractable=False, word_count=100,
        )

        skipped = {
            pid: rec for pid, rec in index.papers.items()
            if not rec.extractable
        }
        assert len(skipped) == 2

        non_academic = sum(
            1 for r in skipped.values() if r.document_type == "NON_ACADEMIC"
        )
        assert non_academic == 1


class TestInlineClassification:
    def test_inline_fallback_when_no_index(self, index_dir):
        """Store returns empty index when no file exists."""
        store = ClassificationStore(index_dir=index_dir)
        index = store.load()
        assert index.papers == {}
```

- [ ] **Step 2: Run tests to verify they pass (these test the store, not build_index)**

Run: `python -m pytest tests/test_build_gating.py -v`
Expected: PASS -- these tests exercise ClassificationStore directly.

- [ ] **Step 3: Implement --classify-only in build_index.py**

In `scripts/build_index.py`, add import at top:
```python
from src.analysis.classification_store import ClassificationStore
```

In the main function, after papers are loaded (after line 721), add the classify-only early return:

```python
    # Classification pre-pass
    index_dir = Path(config.get_index_dir())
    class_store = ClassificationStore(
        index_dir=index_dir,
        min_publication_words=config.processing.min_publication_words,
        min_publication_pages=config.processing.min_publication_pages,
    )

    if args.classify_only:
        class_index = class_store.load() if not args.reclassify else ClassificationIndex()
        pdf_extractor = PDFExtractor(
            enable_ocr=config.processing.ocr_enabled or config.processing.ocr_on_fail,
        )
        text_cleaner = TextCleaner()

        for paper in tqdm(all_papers, desc="Classifying"):
            if not args.reclassify and paper.paper_id in class_index.papers:
                continue

            # Tier 1: metadata only
            text = None
            word_count = None
            page_count = None
            section_markers = None

            # Tier 2: text heuristics if needed
            if paper.pdf_path and Path(paper.pdf_path).exists():
                try:
                    raw_text, _ = pdf_extractor.extract_text_with_method(
                        Path(paper.pdf_path)
                    )
                    text = text_cleaner.clean(raw_text)
                    stats = text_cleaner.get_stats(text)
                    word_count = stats.word_count
                    page_count = stats.page_count
                    section_markers = text_cleaner.count_section_markers(text)
                except Exception as e:
                    logger.warning(f"Text extraction failed for {paper.title}: {e}")

            record = class_store.classify_paper(
                paper, text=text, word_count=word_count,
                page_count=page_count, section_markers=section_markers,
            )
            class_index.papers[paper.paper_id] = record

        class_store.save(class_index)
        _print_classification_report(class_index)
        return
```

Add the report helper function before main:

```python
def _print_classification_report(index: ClassificationIndex) -> None:
    """Print classification summary to terminal."""
    stats = index.stats
    total = stats.get("total", 0)
    if total == 0:
        print("No papers classified.")
        return

    print("\nClassification summary:")
    for doc_type, count in sorted(
        stats.get("by_type", {}).items(), key=lambda x: -x[1]
    ):
        pct = count / total * 100
        print(f"  {doc_type:<22} {count:>5} ({pct:.0f}%)")

    ext = stats.get("extractable_count", 0)
    non_ext = stats.get("non_extractable_count", 0)
    print(f"\nExtractable: {ext}  |  Skippable: {non_ext}")
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_build_gating.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/build_index.py tests/test_build_gating.py
git commit -m "feat: implement --classify-only mode in build_index"
```

---

### Task 11: Implement build index gating (academic-only filtering)

**Files:**
- Modify: `scripts/build_index.py` (main function, after classification block)

- [ ] **Step 1: Add gating logic after papers are loaded**

In `scripts/build_index.py`, after the `--classify-only` early return block, add the gating logic:

```python
    # Load or build classification index for gating
    from src.analysis.classification_store import ClassificationIndex

    class_index = class_store.load()
    papers_needing_classification = [
        p for p in all_papers if p.paper_id not in class_index.papers
    ]

    if papers_needing_classification:
        if not class_index.papers:
            logger.info(
                "No classification index found. Running inline classification. "
                "Use --classify-only for a preview."
            )
        pdf_extractor_for_class = PDFExtractor(
            ocr_enabled=config.processing.ocr_enabled,
            ocr_on_fail=config.processing.ocr_on_fail,
        )
        text_cleaner_for_class = TextCleaner()

        for paper in papers_needing_classification:
            text = None
            word_count = None
            page_count = None
            section_markers = None
            if paper.pdf_path and Path(paper.pdf_path).exists():
                try:
                    raw_text, _ = pdf_extractor_for_class.extract_text_with_method(
                        Path(paper.pdf_path)
                    )
                    cleaned = text_cleaner_for_class.clean(raw_text)
                    stats = text_cleaner_for_class.get_stats(cleaned)
                    word_count = stats.word_count
                    page_count = stats.page_count
                    section_markers = text_cleaner_for_class.count_section_markers(
                        cleaned
                    )
                except Exception:
                    pass
            record = class_store.classify_paper(
                paper, text=text, word_count=word_count,
                page_count=page_count, section_markers=section_markers,
            )
            class_index.papers[paper.paper_id] = record

        class_store.save(class_index)

    # Apply gating filter
    if not args.index_all:
        extractable_ids = class_store.get_extractable_ids(class_index)
        skipped = [p for p in all_papers if p.paper_id not in extractable_ids]
        all_papers = [p for p in all_papers if p.paper_id in extractable_ids]

        if skipped:
            non_academic = sum(
                1 for p in skipped
                if class_index.papers.get(p.paper_id)
                and class_index.papers[p.paper_id].document_type == "NON_ACADEMIC"
            )
            other = len(skipped) - non_academic
            logger.info(
                f"Skipping {len(skipped)} papers "
                f"({non_academic} non-academic, {other} insufficient text). "
                f"Use --index-all to include."
            )

    _print_classification_report(class_index)
```

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add scripts/build_index.py
git commit -m "feat: add academic-only gating with inline classification fallback"
```

---

## Chunk 5: BaseReferenceDB Wiring and Final Integration

### Task 12: Replace hardcoded ZoteroDatabase with adapter factory

**Files:**
- Modify: `scripts/build_index.py:31` (import), `build_index.py:702-715` (instantiation)

- [ ] **Step 1: Add reference source config**

In `scripts/build_index.py`, add argparse flag (near `--collection`):

```python
    parser.add_argument(
        "--source",
        type=str,
        default="zotero",
        choices=["zotero", "bibtex", "pdffolder", "mendeley", "endnote", "paperpile"],
        help="Reference source to load papers from (default: zotero).",
    )
    parser.add_argument(
        "--source-path",
        type=str,
        default=None,
        help="Path for non-Zotero sources (BibTeX file, PDF folder, etc.).",
    )
```

- [ ] **Step 2: Replace ZoteroDatabase instantiation**

Replace the ZoteroDatabase import (line 31):
```python
from src.references.factory import create_reference_db
```

Remove: `from src.zotero.database import ZoteroDatabase`

Replace the instantiation block (lines 702-715) with:

```python
    # Load papers from configured reference source
    if args.source == "zotero":
        ref_db = create_reference_db(
            "zotero",
            db_path=config.get_zotero_db_path(),
            storage_path=config.get_storage_path(),
        )
    elif args.source_path:
        # Map provider names to their expected keyword arguments
        source_kwarg_map = {
            "bibtex": "bibtex_path",
            "pdffolder": "folder_path",
            "endnote": "xml_path",
            "mendeley": "db_path",
            "paperpile": "bibtex_path",
        }
        kwarg_name = source_kwarg_map.get(args.source, "path")
        ref_db = create_reference_db(args.source, **{kwarg_name: args.source_path})
    else:
        parser.error(f"--source-path required for source '{args.source}'")

    all_papers = list(ref_db.get_all_papers())

    # Collection filter (Zotero-specific; warn for other sources)
    if args.collection:
        if args.source != "zotero":
            logger.warning(
                f"--collection filter is Zotero-specific; "
                f"ignoring for source '{args.source}'"
            )
        else:
            all_papers = [
                p for p in all_papers
                if any(args.collection.lower() in c.lower() for c in (p.collections or []))
            ]
```

- [ ] **Step 3: Filter to papers with PDFs (preserve existing behavior)**

Keep the existing PDF filter (around line 717-721):
```python
    all_papers = [p for p in all_papers if p.pdf_path and Path(p.pdf_path).exists()]
```

- [ ] **Step 4: Run existing tests**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All existing tests pass. The factory wiring is backward compatible since `zotero` is the default source.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_index.py
git commit -m "feat: replace hardcoded ZoteroDatabase with adapter factory"
```

---

### Task 13: Save extraction_method to classification index after extraction

**Files:**
- Modify: `scripts/build_index.py` (extraction results processing)

- [ ] **Step 1: Add extraction_method update after extraction**

In `scripts/build_index.py`, find where extraction results are processed (around line 354-364 where `result` from `extract_batch` is handled). Add after the result is saved:

```python
            # Track extraction method in classification index
            if result.success and result.extraction_method:
                if result.paper_id in class_index.papers:
                    class_index.papers[result.paper_id].extraction_method = (
                        result.extraction_method
                    )
```

At the end of the main function, save the updated classification index:

```python
    # Save updated classification index with extraction methods
    class_store.save(class_index)
```

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add scripts/build_index.py
git commit -m "feat: track extraction_method in classification index"
```

---

### Task 14: Final integration test and cleanup

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short 2>&1 | tee data/logs/post_cascade_tests.txt`
Expected: All tests pass, zero failures.

- [ ] **Step 2: Run ruff linting**

Run: `ruff check src/ tests/ scripts/`
Expected: No errors.

- [ ] **Step 3: Verify cascade_enabled=false backward compatibility**

Run: `python -c "from src.analysis.section_extractor import SectionExtractor; print('cascade_enabled=False OK')"` with cascade_enabled=False to verify it constructs without error.

- [ ] **Step 4: Verify --classify-only dry run (if Zotero available)**

Run: `python scripts/build_index.py --classify-only --limit 3`
Expected: Classifies 3 papers, prints summary, creates classification_index.json.

- [ ] **Step 5: Commit any remaining changes**

```bash
git add -A
git commit -m "chore: post-integration cleanup"
```

- [ ] **Step 6: Push to remote**

```bash
git push
```
