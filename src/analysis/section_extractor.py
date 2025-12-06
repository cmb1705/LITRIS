"""Section extractor orchestrating PDF extraction and LLM analysis."""

from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from src.analysis.llm_client import ExtractionMode, LLMClient
from src.analysis.schemas import ExtractionResult
from src.extraction.pdf_extractor import PDFExtractor
from src.extraction.text_cleaner import TextCleaner
from src.utils.logging_config import LogContext, get_logger
from src.zotero.models import PaperMetadata

logger = get_logger(__name__)


@dataclass
class ExtractionStats:
    """Statistics for extraction run."""

    total: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_duration: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total == 0:
            return 0.0
        return self.successful / self.total


class SectionExtractor:
    """Orchestrates extraction of structured content from papers."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        mode: ExtractionMode = "api",
        model: str = "claude-opus-4-5-20251101",
        max_tokens: int = 8192,
        min_text_length: int = 100,
    ):
        """Initialize section extractor.

        Args:
            cache_dir: Directory for PDF text cache.
            mode: LLM extraction mode.
            model: Model to use.
            max_tokens: Maximum response tokens.
            min_text_length: Minimum text length to attempt extraction.
        """
        self.pdf_extractor = PDFExtractor(cache_dir=cache_dir)
        self.text_cleaner = TextCleaner()
        self.llm_client = LLMClient(mode=mode, model=model, max_tokens=max_tokens)
        self.min_text_length = min_text_length

    def extract_paper(self, paper: PaperMetadata) -> ExtractionResult:
        """Extract structured content from a single paper.

        Args:
            paper: Paper metadata including PDF path.

        Returns:
            ExtractionResult with extraction or error.
        """
        with LogContext(logger, f"Extracting paper: {paper.title[:50]}..."):
            # Check for PDF
            if not paper.pdf_path or not paper.pdf_path.exists():
                logger.warning(f"No PDF available for paper {paper.paper_id}")
                return ExtractionResult(
                    paper_id=paper.paper_id,
                    success=False,
                    error="No PDF available",
                )

            # Extract text from PDF
            try:
                text = self.pdf_extractor.extract_text(paper.pdf_path)
            except Exception as e:
                logger.error(f"PDF extraction failed: {e}")
                return ExtractionResult(
                    paper_id=paper.paper_id,
                    success=False,
                    error=f"PDF extraction failed: {e}",
                )

            # Clean text
            text = self.text_cleaner.clean(text)

            # Validate text
            if not self.text_cleaner.is_valid_extraction(
                text, min_words=self.min_text_length
            ):
                logger.warning(f"Insufficient text content for paper {paper.paper_id}")
                return ExtractionResult(
                    paper_id=paper.paper_id,
                    success=False,
                    error="Insufficient text content",
                )

            # Truncate for LLM if needed
            text = self.text_cleaner.truncate_for_llm(text)

            # Perform LLM extraction
            result = self.llm_client.extract(
                paper_id=paper.paper_id,
                title=paper.title,
                authors=paper.author_string,
                year=paper.publication_year,
                item_type=paper.item_type,
                text=text,
            )

            # Set indexed timestamp on success
            if result.success:
                paper.indexed_at = datetime.now()

            return result

    def extract_batch(
        self,
        papers: list[PaperMetadata],
        progress_callback: Callable | None = None,
    ) -> Generator[ExtractionResult, None, ExtractionStats]:
        """Extract content from multiple papers.

        Args:
            papers: List of papers to process.
            progress_callback: Optional callback(current, total, paper_title).

        Yields:
            ExtractionResult for each paper.

        Returns:
            ExtractionStats with overall statistics.
        """
        stats = ExtractionStats(total=len(papers))

        for i, paper in enumerate(papers):
            if progress_callback:
                progress_callback(i + 1, len(papers), paper.title)

            # Skip papers without PDFs
            if not paper.pdf_path:
                stats.skipped += 1
                continue

            result = self.extract_paper(paper)

            if result.success:
                stats.successful += 1
                stats.total_input_tokens += result.input_tokens
                stats.total_output_tokens += result.output_tokens
            else:
                stats.failed += 1

            stats.total_duration += result.duration_seconds

            yield result

        logger.info(
            f"Extraction complete: {stats.successful}/{stats.total} successful "
            f"({stats.success_rate:.1%}), {stats.skipped} skipped"
        )

        return stats

    def estimate_batch_cost(self, papers: list[PaperMetadata]) -> dict:
        """Estimate cost for batch extraction.

        Args:
            papers: List of papers to process.

        Returns:
            Dictionary with cost estimates.
        """
        papers_with_pdf = [p for p in papers if p.pdf_path and p.pdf_path.exists()]

        # Sample text lengths
        sample_size = min(10, len(papers_with_pdf))
        sample_lengths = []

        for paper in papers_with_pdf[:sample_size]:
            try:
                text = self.pdf_extractor.extract_text(paper.pdf_path)
                text = self.text_cleaner.clean(text)
                text = self.text_cleaner.truncate_for_llm(text)
                sample_lengths.append(len(text))
            except Exception:
                continue

        if not sample_lengths:
            return {
                "papers_with_pdf": len(papers_with_pdf),
                "estimated_cost": 0,
                "note": "Could not sample papers for estimate",
            }

        avg_length = sum(sample_lengths) / len(sample_lengths)
        per_paper_cost = self.llm_client.estimate_cost(int(avg_length))
        total_cost = per_paper_cost * len(papers_with_pdf)

        return {
            "papers_with_pdf": len(papers_with_pdf),
            "average_text_length": int(avg_length),
            "estimated_cost_per_paper": round(per_paper_cost, 4),
            "estimated_total_cost": round(total_cost, 2),
            "model": self.llm_client.model,
        }
