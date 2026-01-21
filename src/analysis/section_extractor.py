"""Section extractor orchestrating PDF extraction and LLM analysis."""

import hashlib
import json
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Callable

from src.analysis.base_llm import BaseLLMClient, ExtractionMode
from src.analysis.llm_factory import create_llm_client
from src.analysis.schemas import ExtractionResult, PaperExtraction
from src.extraction.pdf_extractor import PDFExtractor
from src.extraction.text_cleaner import TextCleaner
from src.utils.logging_config import LogContext, get_logger
from src.zotero.models import PaperMetadata

logger = get_logger(__name__)

PROMPT_VERSION = "1.0"


class ExtractionCache:
    """Cache for extraction results based on content hash."""

    def __init__(self, cache_dir: Path):
        """Initialize extraction cache.

        Args:
            cache_dir: Base cache directory.
        """
        self.cache_dir = cache_dir / "extractions"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def compute_content_hash(
        self, pdf_path: Path, model: str, prompt_version: str = PROMPT_VERSION
    ) -> str:
        """Compute content hash for cache key.

        Uses file modification time and size to detect changes without
        reading the entire file. Includes model and prompt version to
        invalidate cache when extraction logic changes.

        Args:
            pdf_path: Path to PDF file.
            model: Model identifier.
            prompt_version: Version string for extraction prompts.

        Returns:
            16-character hex hash string.
        """
        hasher = hashlib.sha256()
        stat = pdf_path.stat()
        hasher.update(f"{stat.st_mtime}:{stat.st_size}".encode())
        hasher.update(f":{model}:{prompt_version}".encode())
        return hasher.hexdigest()[:16]

    def _get_cache_path(self, paper_id: str, content_hash: str) -> Path:
        """Get cache file path for a paper."""
        return self.cache_dir / f"{paper_id}_{content_hash}.json"

    def get(self, paper_id: str, content_hash: str) -> ExtractionResult | None:
        """Retrieve cached extraction result.

        Args:
            paper_id: Paper identifier.
            content_hash: Content hash for cache key.

        Returns:
            Cached ExtractionResult or None if not found/invalid.
        """
        cache_path = self._get_cache_path(paper_id, content_hash)

        if not cache_path.exists():
            return None

        try:
            with self._lock:
                with open(cache_path, encoding="utf-8") as f:
                    data = json.load(f)

            # Reconstruct nested objects
            if "extraction" in data and data["extraction"]:
                from src.analysis.schemas import KeyClaim, KeyFinding, Methodology

                ext_data = data["extraction"]
                if "methodology" in ext_data and isinstance(ext_data["methodology"], dict):
                    ext_data["methodology"] = Methodology(**ext_data["methodology"])
                if "key_findings" in ext_data and isinstance(ext_data["key_findings"], list):
                    ext_data["key_findings"] = [
                        KeyFinding(**f) if isinstance(f, dict) else f
                        for f in ext_data["key_findings"]
                    ]
                if "key_claims" in ext_data and isinstance(ext_data["key_claims"], list):
                    ext_data["key_claims"] = [
                        KeyClaim(**c) if isinstance(c, dict) else c
                        for c in ext_data["key_claims"]
                    ]
                data["extraction"] = PaperExtraction(**ext_data)

            return ExtractionResult(**data)

        except Exception as e:
            logger.debug(f"Cache read failed for {paper_id}: {e}")
            return None

    def set(self, paper_id: str, content_hash: str, result: ExtractionResult) -> None:
        """Store extraction result in cache.

        Args:
            paper_id: Paper identifier.
            content_hash: Content hash for cache key.
            result: Extraction result to cache.
        """
        if not result.success:
            return  # Only cache successful extractions

        cache_path = self._get_cache_path(paper_id, content_hash)

        try:
            # Convert to serializable dict
            data = {
                "paper_id": result.paper_id,
                "success": result.success,
                "error": result.error,
                "duration_seconds": result.duration_seconds,
                "model_used": result.model_used,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            }

            if result.extraction:
                data["extraction"] = result.extraction.model_dump()

            with self._lock:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

            logger.debug(f"Cached extraction for {paper_id}")

        except Exception as e:
            logger.warning(f"Cache write failed for {paper_id}: {e}")

    def clear(self) -> int:
        """Clear all cached extractions.

        Returns:
            Number of cache entries removed.
        """
        count = 0
        with self._lock:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    cache_file.unlink()
                    count += 1
                except Exception:
                    pass
        logger.info(f"Cleared {count} cached extractions")
        return count


@dataclass
class ExtractionStats:
    """Statistics for extraction run."""

    total: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    cached: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_duration: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total == 0:
            return 0.0
        return self.successful / self.total

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        processed = self.successful + self.cached
        if processed == 0:
            return 0.0
        return self.cached / processed


class SectionExtractor:
    """Orchestrates extraction of structured content from papers."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        provider: str = "anthropic",
        mode: ExtractionMode = "api",
        model: str | None = None,
        max_tokens: int = 8192,
        min_text_length: int = 100,
        ocr_enabled: bool = False,
        ocr_config: dict | None = None,
        use_cache: bool = True,
        parallel_workers: int = 1,
        reasoning_effort: str | None = None,
    ):
        """Initialize section extractor.

        Args:
            cache_dir: Directory for PDF text cache.
            provider: LLM provider ('anthropic' or 'openai').
            mode: LLM extraction mode.
            model: Model to use (None uses provider default).
            max_tokens: Maximum response tokens.
            min_text_length: Minimum text length to attempt extraction.
            ocr_enabled: Enable OCR fallback for scanned PDFs.
            ocr_config: Optional OCR handler config (tesseract_cmd, poppler_path, dpi, lang).
            use_cache: Enable extraction caching to skip unchanged papers.
            parallel_workers: Number of parallel workers for CLI mode (1 = sequential).
            reasoning_effort: For OpenAI GPT-5.2: none/low/medium/high/xhigh.
        """
        self.pdf_extractor = PDFExtractor(
            cache_dir=cache_dir,
            enable_ocr=ocr_enabled,
            ocr_config=ocr_config,
        )
        self.text_cleaner = TextCleaner()

        # Create LLM client using factory
        self.llm_client: BaseLLMClient = create_llm_client(
            provider=provider,
            mode=mode,
            model=model,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        self.min_text_length = min_text_length
        self.model = self.llm_client.model  # Use resolved model from client
        self.mode = mode
        self.provider = provider
        self.parallel_workers = parallel_workers
        self.use_cache = use_cache

        # Initialize extraction cache
        if cache_dir and use_cache:
            self.extraction_cache = ExtractionCache(cache_dir)
        else:
            self.extraction_cache = None

    def extract_paper(
        self, paper: PaperMetadata, check_cache: bool = True
    ) -> tuple[ExtractionResult, bool]:
        """Extract structured content from a single paper.

        Args:
            paper: Paper metadata including PDF path.
            check_cache: Whether to check cache before extracting.

        Returns:
            Tuple of (ExtractionResult, from_cache).
        """
        with LogContext(logger, f"Extracting paper: {paper.title[:50]}..."):
            # Check for PDF
            if not paper.pdf_path or not paper.pdf_path.exists():
                logger.warning(f"No PDF available for paper {paper.paper_id}")
                return ExtractionResult(
                    paper_id=paper.paper_id,
                    success=False,
                    error="No PDF available",
                ), False

            # Check cache
            content_hash = None
            if check_cache and self.extraction_cache:
                content_hash = self.extraction_cache.compute_content_hash(
                    paper.pdf_path, self.model
                )
                cached_result = self.extraction_cache.get(paper.paper_id, content_hash)
                if cached_result:
                    logger.info(f"Using cached extraction for {paper.paper_id}")
                    return cached_result, True

            # Extract text from PDF
            try:
                text = self.pdf_extractor.extract_text(paper.pdf_path)
            except Exception as e:
                logger.error(f"PDF extraction failed: {e}")
                return ExtractionResult(
                    paper_id=paper.paper_id,
                    success=False,
                    error=f"PDF extraction failed: {e}",
                ), False

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
                ), False

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

                # Cache the result
                if self.extraction_cache and content_hash:
                    self.extraction_cache.set(paper.paper_id, content_hash, result)

            return result, False

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
        # Use parallel extraction for CLI mode with multiple workers
        if self.mode == "cli" and self.parallel_workers > 1:
            return self._extract_batch_parallel(papers, progress_callback)
        else:
            return self._extract_batch_sequential(papers, progress_callback)

    def _extract_batch_sequential(
        self,
        papers: list[PaperMetadata],
        progress_callback: Callable | None = None,
    ) -> Generator[ExtractionResult, None, ExtractionStats]:
        """Sequential batch extraction.

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

            result, from_cache = self.extract_paper(paper)

            if from_cache:
                stats.cached += 1
                stats.successful += 1
            elif result.success:
                stats.successful += 1
                stats.total_input_tokens += result.input_tokens
                stats.total_output_tokens += result.output_tokens
                stats.total_duration += result.duration_seconds
            else:
                stats.failed += 1
                stats.total_duration += result.duration_seconds

            yield result

        self._log_completion_stats(stats)
        return stats

    def _extract_batch_parallel(
        self,
        papers: list[PaperMetadata],
        progress_callback: Callable | None = None,
    ) -> Generator[ExtractionResult, None, ExtractionStats]:
        """Parallel batch extraction using thread pool.

        Args:
            papers: List of papers to process.
            progress_callback: Optional callback(current, total, paper_title).

        Yields:
            ExtractionResult for each paper.

        Returns:
            ExtractionStats with overall statistics.
        """
        stats = ExtractionStats(total=len(papers))
        completed = 0
        stats_lock = Lock()

        # Filter papers with PDFs
        papers_to_process = []
        for paper in papers:
            if not paper.pdf_path:
                stats.skipped += 1
            else:
                papers_to_process.append(paper)

        if not papers_to_process:
            return stats

        def extract_one(paper: PaperMetadata) -> tuple[PaperMetadata, ExtractionResult, bool]:
            """Extract a single paper (thread-safe)."""
            result, from_cache = self.extract_paper(paper)
            return paper, result, from_cache

        logger.info(
            f"Starting parallel extraction with {self.parallel_workers} workers "
            f"for {len(papers_to_process)} papers"
        )

        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            # Submit all tasks
            future_to_paper = {
                executor.submit(extract_one, paper): paper
                for paper in papers_to_process
            }

            # Process results as they complete
            for future in as_completed(future_to_paper):
                paper, result, from_cache = future.result()

                with stats_lock:
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, len(papers_to_process), paper.title)

                    if from_cache:
                        stats.cached += 1
                        stats.successful += 1
                    elif result.success:
                        stats.successful += 1
                        stats.total_input_tokens += result.input_tokens
                        stats.total_output_tokens += result.output_tokens
                        stats.total_duration += result.duration_seconds
                    else:
                        stats.failed += 1
                        stats.total_duration += result.duration_seconds

                yield result

        self._log_completion_stats(stats)
        return stats

    def _log_completion_stats(self, stats: ExtractionStats) -> None:
        """Log completion statistics."""
        cache_info = ""
        if stats.cached > 0:
            cache_info = f", {stats.cached} from cache ({stats.cache_hit_rate:.1%} hit rate)"

        logger.info(
            f"Extraction complete: {stats.successful}/{stats.total} successful "
            f"({stats.success_rate:.1%}), {stats.skipped} skipped{cache_info}"
        )

    def clear_cache(self) -> int:
        """Clear the extraction cache.

        Returns:
            Number of cache entries removed.
        """
        if self.extraction_cache:
            return self.extraction_cache.clear()
        return 0

    def estimate_batch_cost(self, papers: list[PaperMetadata]) -> dict:
        """Estimate cost for batch extraction.

        Args:
            papers: List of papers to process.

        Returns:
            Dictionary with cost estimates.
        """
        papers_with_pdf = [p for p in papers if p.pdf_path and p.pdf_path.exists()]

        # Check how many would be cached
        cached_count = 0
        if self.extraction_cache:
            for paper in papers_with_pdf:
                content_hash = self.extraction_cache.compute_content_hash(
                    paper.pdf_path, self.model
                )
                if self.extraction_cache.get(paper.paper_id, content_hash):
                    cached_count += 1

        papers_to_extract = len(papers_with_pdf) - cached_count

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
                "papers_cached": cached_count,
                "papers_to_extract": papers_to_extract,
                "estimated_cost": 0,
                "note": "Could not sample papers for estimate",
            }

        avg_length = sum(sample_lengths) / len(sample_lengths)
        per_paper_cost = self.llm_client.estimate_cost(int(avg_length))
        total_cost = per_paper_cost * papers_to_extract

        return {
            "papers_with_pdf": len(papers_with_pdf),
            "papers_cached": cached_count,
            "papers_to_extract": papers_to_extract,
            "average_text_length": int(avg_length),
            "estimated_cost_per_paper": round(per_paper_cost, 4),
            "estimated_total_cost": round(total_cost, 2),
            "model": self.llm_client.model,
        }
