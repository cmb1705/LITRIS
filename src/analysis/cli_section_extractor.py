"""CLI-based section extractor using Claude Code headless mode."""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from src.analysis.cli_executor import (
    ClaudeCliExecutor,
    CliExecutionError,
    ExtractionTimeoutError,
    ParseError,
    RateLimitError,
)
from src.analysis.progress_tracker import ProgressTracker
from src.analysis.prompts import build_extraction_prompt
from src.analysis.rate_limit_handler import RateLimitHandler
from src.analysis.schemas import (
    KeyClaim,
    KeyFinding,
    Methodology,
    PaperExtraction,
)
from src.utils.logging_config import get_logger
from src.zotero.models import PaperMetadata

logger = get_logger(__name__)


def _normalize_discipline_tags(tags: list[str] | None) -> list[str]:
    """Normalize discipline tags to lowercase and deduplicate.

    Args:
        tags: Raw discipline tags from extraction.

    Returns:
        Normalized, deduplicated list of discipline tags.
    """
    if not tags:
        return []

    normalized = []
    seen = set()
    for tag in tags:
        if not isinstance(tag, str):
            continue
        # Normalize: lowercase, strip whitespace
        clean = tag.lower().strip()
        if clean and clean not in seen:
            normalized.append(clean)
            seen.add(clean)
    return normalized


class CliSectionExtractor:
    """Orchestrate CLI-based extraction with rate limiting and progress tracking.

    Uses Claude Code CLI in headless mode for cost-free extraction
    with Max subscription.
    """

    def __init__(
        self,
        cache_dir: Path,
        executor: ClaudeCliExecutor | None = None,
        rate_handler: RateLimitHandler | None = None,
        progress_tracker: ProgressTracker | None = None,
        max_text_length: int = 100000,
    ):
        """Initialize CLI section extractor.

        Args:
            cache_dir: Directory for progress tracking.
            executor: CLI executor (created if not provided).
            rate_handler: Rate limit handler (created if not provided).
            progress_tracker: Progress tracker (created if not provided).
            max_text_length: Maximum text length before truncation.
        """
        self.cache_dir = cache_dir
        self.executor = executor or ClaudeCliExecutor()
        self.rate_handler = rate_handler or RateLimitHandler(
            pause_on_limit=True,
            auto_resume=False,
        )
        self.progress = progress_tracker or ProgressTracker(cache_dir)
        self.max_text_length = max_text_length

        # Ensure cache directory exists
        cache_dir.mkdir(parents=True, exist_ok=True)

    def verify_setup(self) -> bool:
        """Verify CLI is properly set up.

        Returns:
            True if ready for extraction.

        Raises:
            CliExecutionError: If setup is invalid.
        """
        return self.executor.verify_authentication()

    def extract_single(
        self,
        paper_id: str,
        paper_text: str,
        metadata: PaperMetadata,
    ) -> PaperExtraction:
        """Extract structured data from a single paper via CLI.

        Args:
            paper_id: Unique paper identifier.
            paper_text: Full text of the paper.
            metadata: Paper metadata from Zotero.

        Returns:
            PaperExtraction with extracted data.

        Raises:
            RateLimitError: If rate limit is hit.
            CliExecutionError: For other CLI errors.
        """
        # Truncate if needed
        if len(paper_text) > self.max_text_length:
            logger.warning(
                f"Truncating paper {paper_id} from {len(paper_text)} "
                f"to {self.max_text_length} chars"
            )
            paper_text = paper_text[: self.max_text_length]

        # Build prompt
        prompt = build_extraction_prompt(
            title=metadata.title,
            authors=", ".join(a.full_name for a in metadata.authors) if metadata.authors else "",
            year=metadata.publication_year,
            item_type=metadata.item_type,
            text=paper_text,
        )

        # Execute extraction
        try:
            response = self.executor.extract(prompt, paper_text)
            self.rate_handler.record_request()
        except RateLimitError:
            # Let caller handle rate limit
            raise

        # Parse response into PaperExtraction
        return self._parse_response(response, paper_id, metadata)

    def extract_all(
        self,
        papers: list[tuple[str, str, PaperMetadata]],
        resume: bool = True,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[PaperExtraction]:
        """Extract structured data from multiple papers.

        Args:
            papers: List of (paper_id, text, metadata) tuples.
            resume: Whether to resume from previous progress.
            progress_callback: Optional callback(current, total, paper_title).

        Returns:
            List of PaperExtraction objects.
        """
        # Verify setup
        self.verify_setup()

        # Get paper IDs for filtering
        all_paper_ids = [p[0] for p in papers]
        papers_by_id = {p[0]: p for p in papers}

        # Load or initialize progress
        if resume:
            state = self.progress.load()
            if state:
                pending_ids = self.progress.get_pending_papers(all_paper_ids)
                logger.info(
                    f"Resuming: {len(state.completed)} completed, "
                    f"{len(state.failed)} failed, {len(pending_ids)} pending"
                )
            else:
                self.progress.initialize(len(papers))
                pending_ids = all_paper_ids
        else:
            self.progress.reset()
            self.progress.initialize(len(papers))
            pending_ids = all_paper_ids

        # Start rate limit session
        self.rate_handler.start_session()
        self.progress.start_session()

        # Process papers
        extractions = []
        papers_to_process = [(pid, papers_by_id[pid]) for pid in pending_ids if pid in papers_by_id]

        for i, (paper_id, (_, text, metadata)) in enumerate(
            tqdm(papers_to_process, desc="Extracting papers (CLI)")
        ):
            try:
                # Check if approaching rate limit
                if self.rate_handler.is_approaching_limit():
                    logger.warning(
                        f"Approaching rate limit at {self.rate_handler.get_session_request_count()} requests"
                    )

                # Extract
                extraction = self.extract_single(paper_id, text, metadata)
                extractions.append(extraction)
                self.progress.mark_completed(paper_id)

                # Callback
                if progress_callback:
                    progress_callback(i + 1, len(papers_to_process), metadata.title)

            except RateLimitError:
                logger.warning(f"Rate limit hit at paper {paper_id}")

                # Handle rate limit
                def save_progress():
                    self.progress.save()

                can_continue = self.rate_handler.handle_limit_hit(save_progress)
                if not can_continue:
                    # Exit cleanly
                    break

            except ExtractionTimeoutError as e:
                logger.error(f"Timeout extracting {paper_id}: {e}")
                self.progress.mark_failed(paper_id, str(e))

            except ParseError as e:
                logger.error(f"Parse error for {paper_id}: {e}")
                self.progress.mark_failed(paper_id, str(e))

            except CliExecutionError as e:
                logger.error(f"CLI error for {paper_id}: {e}")
                self.progress.mark_failed(paper_id, str(e))

            except Exception as e:
                logger.error(f"Unexpected error for {paper_id}: {e}")
                self.progress.mark_failed(paper_id, str(e))

        # Log summary
        summary = self.progress.get_summary()
        logger.info(
            f"Extraction complete: {summary['completed']} completed, "
            f"{summary['failed']} failed, {summary['pending']} pending"
        )

        return extractions

    def _parse_response(
        self,
        response: dict,
        paper_id: str,
        metadata: PaperMetadata,
    ) -> PaperExtraction:
        """Parse CLI response into PaperExtraction.

        Args:
            response: Parsed JSON response.
            paper_id: Paper identifier.
            metadata: Paper metadata.

        Returns:
            PaperExtraction object.
        """
        _now = datetime.now()

        # Parse methodology
        methodology_data = response.get("methodology", {})
        methodology = None
        if methodology_data:
            methodology = Methodology(
                approach=methodology_data.get("approach"),
                design=methodology_data.get("design"),
                data_sources=methodology_data.get("data_sources", []),
                analysis_methods=methodology_data.get("analysis_methods", []),
                sample_size=methodology_data.get("sample_size"),
                time_period=methodology_data.get("time_period"),
            )

        # Parse findings
        findings = []
        for f in response.get("key_findings", []):
            if isinstance(f, dict):
                findings.append(KeyFinding(
                    finding=f.get("finding", ""),
                    evidence_type=f.get("evidence_type", "empirical"),
                    significance=f.get("significance", "medium"),
                    page_reference=f.get("page_reference"),
                ))
            elif isinstance(f, str):
                findings.append(KeyFinding(finding=f))

        # Parse claims
        claims = []
        for c in response.get("key_claims", []):
            if isinstance(c, dict):
                claims.append(KeyClaim(
                    claim=c.get("claim", ""),
                    support_type=c.get("support_type", "data"),
                    page_reference=c.get("page_reference"),
                ))
            elif isinstance(c, str):
                claims.append(KeyClaim(claim=c))

        # Build extraction (PaperExtraction doesn't include paper_id - that's in ExtractionResult)
        extraction = PaperExtraction(
            thesis_statement=response.get("thesis_statement"),
            research_questions=response.get("research_questions", []),
            theoretical_framework=response.get("theoretical_framework"),
            methodology=methodology if methodology else Methodology(),
            key_findings=findings,
            key_claims=claims,
            conclusions=response.get("conclusions"),
            limitations=response.get("limitations", []),
            future_directions=response.get("future_directions", []),
            contribution_summary=response.get("contribution_summary"),
            keywords=response.get("keywords", []),
            discipline_tags=_normalize_discipline_tags(response.get("discipline_tags", [])),
            extraction_confidence=response.get("extraction_confidence", 0.7),
            extraction_notes=response.get("extraction_notes"),
        )

        return extraction

    def get_progress_summary(self) -> dict:
        """Get current progress summary.

        Returns:
            Dict with progress statistics.
        """
        return self.progress.get_summary()

    def get_failed_papers(self) -> list:
        """Get list of failed papers.

        Returns:
            List of failed paper records.
        """
        return self.progress.get_failed_papers()
