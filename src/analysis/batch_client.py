"""Anthropic Batch API client for paper extraction."""

import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic

from src.analysis.prompts import EXTRACTION_SYSTEM_PROMPT, build_extraction_prompt
from src.analysis.schemas import ExtractionResult, PaperExtraction
from src.utils.logging_config import get_logger
from src.utils.secrets import get_anthropic_api_key
from src.zotero.models import PaperMetadata

logger = get_logger(__name__)


@dataclass
class BatchRequest:
    """A single request in a batch."""

    custom_id: str  # paper_id
    paper: PaperMetadata
    prompt: str


@dataclass
class BatchStatus:
    """Status of a batch job."""

    batch_id: str
    status: str  # validating, in_progress, ended, canceling, canceled, expired
    created_at: datetime
    total_requests: int
    completed_requests: int = 0
    failed_requests: int = 0
    expired_at: datetime | None = None
    results_url: str | None = None


class BatchExtractionClient:
    """Client for batch extraction using Anthropic Batch API.

    The Batch API is cost-effective (50% discount) but asynchronous.
    Batches are processed within 24 hours.

    Typical workflow:
    1. Create batch from papers
    2. Submit batch
    3. Poll for completion (or wait)
    4. Retrieve results
    """

    def __init__(
        self,
        model: str = "claude-opus-4-5-20251101",
        max_tokens: int = 8192,
        batch_dir: Path | None = None,
    ):
        """Initialize batch client.

        Args:
            model: Model to use for extraction.
            max_tokens: Maximum tokens for response.
            batch_dir: Directory to store batch state files.
        """
        api_key = get_anthropic_api_key()
        if not api_key:
            raise ValueError(
                "Anthropic API key required for batch API. "
                "Set ANTHROPIC_API_KEY or store it in the OS keyring "
                "(service: 'litris', key: 'ANTHROPIC_API_KEY')."
            )

        self.client = Anthropic(api_key=api_key)
        self.model = model
        # Claude batch API hard-limits output tokens to 64k; clamp to avoid rejections.
        if max_tokens > 64000:
            logger.info(
                "Clamping max_tokens from %s to 64000 to satisfy Claude batch API limits",
                max_tokens,
            )
        self.max_tokens = min(max_tokens, 64000)
        self.batch_dir = batch_dir or Path("data/batches")
        self.batch_dir.mkdir(parents=True, exist_ok=True)

    def create_batch_requests(
        self,
        papers: list[PaperMetadata],
        text_getter: Callable,
    ) -> list[BatchRequest]:
        """Create batch requests from papers.

        Args:
            papers: List of papers to process.
            text_getter: Function(paper) -> str that returns cleaned text.

        Returns:
            List of BatchRequest objects.
        """
        requests = []

        for paper in papers:
            try:
                text = text_getter(paper)
                prompt = build_extraction_prompt(
                    title=paper.title,
                    authors=paper.author_string,
                    year=paper.publication_year,
                    item_type=paper.item_type,
                    text=text,
                )
                requests.append(
                    BatchRequest(
                        custom_id=paper.paper_id,
                        paper=paper,
                        prompt=prompt,
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to create request for {paper.paper_id}: {e}")

        return requests

    def submit_batch(self, requests: list[BatchRequest]) -> str:
        """Submit a batch of requests.

        Args:
            requests: List of batch requests.

        Returns:
            Batch ID for tracking.
        """
        if not requests:
            raise ValueError("No requests to submit")

        # Build the batch request format
        batch_requests = []
        for req in requests:
            batch_requests.append({
                "custom_id": req.custom_id,
                "params": {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "system": EXTRACTION_SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": req.prompt}],
                },
            })

        logger.info(f"Submitting batch with {len(batch_requests)} requests...")

        # Create the batch
        response = self.client.messages.batches.create(requests=batch_requests)

        batch_id = response.id
        logger.info(f"Batch submitted: {batch_id}")

        # Save batch state
        self._save_batch_state(batch_id, requests)

        return batch_id

    def get_batch_status(self, batch_id: str) -> BatchStatus:
        """Get status of a batch.

        Args:
            batch_id: Batch ID from submit_batch.

        Returns:
            BatchStatus with current state.
        """
        response = self.client.messages.batches.retrieve(batch_id)

        return BatchStatus(
            batch_id=response.id,
            status=response.processing_status,
            created_at=response.created_at,
            total_requests=response.request_counts.processing
            + response.request_counts.succeeded
            + response.request_counts.errored
            + response.request_counts.canceled
            + response.request_counts.expired,
            completed_requests=response.request_counts.succeeded,
            failed_requests=response.request_counts.errored,
            results_url=response.results_url,
        )

    def wait_for_batch(
        self,
        batch_id: str,
        poll_interval: int = 60,
        max_wait: int = 86400,
        progress_callback: Callable | None = None,
    ) -> BatchStatus:
        """Wait for batch to complete.

        Args:
            batch_id: Batch ID from submit_batch.
            poll_interval: Seconds between status checks.
            max_wait: Maximum seconds to wait (default 24 hours).
            progress_callback: Optional callback(status) for progress updates.

        Returns:
            Final BatchStatus.

        Raises:
            TimeoutError: If batch doesn't complete within max_wait.
        """
        start_time = time.time()

        while True:
            status = self.get_batch_status(batch_id)

            if progress_callback:
                progress_callback(status)

            if status.status == "ended":
                return status

            if status.status in ("canceled", "expired"):
                raise RuntimeError(f"Batch {batch_id} {status.status}")

            elapsed = time.time() - start_time
            if elapsed > max_wait:
                raise TimeoutError(
                    f"Batch {batch_id} did not complete within {max_wait}s"
                )

            logger.info(
                f"Batch {batch_id}: {status.completed_requests}/{status.total_requests} "
                f"complete, waiting {poll_interval}s..."
            )
            time.sleep(poll_interval)

    def get_results(self, batch_id: str) -> Iterator[ExtractionResult]:
        """Retrieve results from a completed batch.

        Args:
            batch_id: Batch ID from submit_batch.

        Yields:
            ExtractionResult for each paper.
        """
        # Stream results from the batch
        for result in self.client.messages.batches.results(batch_id):
            paper_id = result.custom_id

            if result.result.type == "succeeded":
                try:
                    message = result.result.message
                    response_text = message.content[0].text
                    extraction = self._parse_response(response_text)

                    yield ExtractionResult(
                        paper_id=paper_id,
                        success=True,
                        extraction=extraction,
                        model_used=self.model,
                        input_tokens=message.usage.input_tokens,
                        output_tokens=message.usage.output_tokens,
                    )
                except Exception as e:
                    logger.error(f"Failed to parse result for {paper_id}: {e}")
                    yield ExtractionResult(
                        paper_id=paper_id,
                        success=False,
                        error=f"Parse error: {e}",
                        model_used=self.model,
                    )
            else:
                error_msg = getattr(result.result, "error", {})
                yield ExtractionResult(
                    paper_id=paper_id,
                    success=False,
                    error=f"Batch error: {error_msg}",
                    model_used=self.model,
                )

    def _parse_response(self, response_text: str) -> PaperExtraction:
        """Parse JSON response into PaperExtraction."""
        from src.analysis.schemas import KeyClaim, KeyFinding, Methodology

        # Clean response - remove any markdown formatting
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Parse JSON
        data = json.loads(text)

        # Handle nested objects
        if "methodology" in data and isinstance(data["methodology"], dict):
            data["methodology"] = Methodology(**data["methodology"])

        if "key_findings" in data and isinstance(data["key_findings"], list):
            data["key_findings"] = [
                KeyFinding(**f) if isinstance(f, dict) else f
                for f in data["key_findings"]
            ]

        if "key_claims" in data and isinstance(data["key_claims"], list):
            data["key_claims"] = [
                KeyClaim(**c) if isinstance(c, dict) else c
                for c in data["key_claims"]
            ]

        return PaperExtraction(**data)

    def _save_batch_state(self, batch_id: str, requests: list[BatchRequest]) -> None:
        """Save batch state to disk for recovery."""
        state = {
            "batch_id": batch_id,
            "created_at": datetime.now().isoformat(),
            "model": self.model,
            "paper_ids": [r.custom_id for r in requests],
        }

        state_file = self.batch_dir / f"{batch_id}.json"
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def list_pending_batches(self) -> list[str]:
        """List batch IDs that haven't completed yet."""
        pending = []

        for state_file in self.batch_dir.glob("*.json"):
            try:
                with open(state_file, encoding="utf-8") as f:
                    state = json.load(f)
                batch_id = state.get("batch_id")
                if batch_id:
                    status = self.get_batch_status(batch_id)
                    if status.status not in ("ended", "canceled", "expired"):
                        pending.append(batch_id)
            except Exception:
                continue

        return pending

    def estimate_cost(self, num_papers: int, avg_text_length: int) -> dict:
        """Estimate cost for batch extraction.

        Args:
            num_papers: Number of papers.
            avg_text_length: Average text length per paper.

        Returns:
            Cost estimate dictionary.
        """
        # Rough estimate: 4 chars per token
        input_tokens_per_paper = avg_text_length // 4 + 500
        output_tokens_per_paper = 2000

        total_input = input_tokens_per_paper * num_papers
        total_output = output_tokens_per_paper * num_papers

        # Batch API is 50% cheaper
        input_cost_per_million = 15.0 * 0.5
        output_cost_per_million = 75.0 * 0.5

        input_cost = (total_input / 1_000_000) * input_cost_per_million
        output_cost = (total_output / 1_000_000) * output_cost_per_million

        return {
            "num_papers": num_papers,
            "estimated_input_tokens": total_input,
            "estimated_output_tokens": total_output,
            "estimated_cost": round(input_cost + output_cost, 2),
            "discount": "50% (batch API)",
            "model": self.model,
        }
