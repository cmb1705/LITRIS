"""Anthropic Batch API client for 6-pass SemanticAnalysis extraction."""

import json
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic

from src.analysis.constants import ANTHROPIC_BATCH_PRICING, DEFAULT_MODELS
from src.analysis.coverage import score_coverage, score_dimension_values
from src.analysis.dimensions import DimensionProfile
from src.analysis.schemas import DimensionedExtraction, ExtractionResult, SemanticAnalysis
from src.analysis.semantic_prompts import (
    SEMANTIC_PROMPT_VERSION,
    SEMANTIC_SYSTEM_PROMPT,
    build_pass_user_prompt,
    get_pass_definitions,
)
from src.utils.logging_config import get_logger
from src.utils.secrets import get_anthropic_api_key
from src.zotero.models import PaperMetadata

logger = get_logger(__name__)

CUSTOM_ID_PASS_SEPARATOR = "__pass"


@dataclass
class BatchRequest:
    """A single request in a batch."""

    custom_id: str  # {paper_id}__pass{N}
    paper: PaperMetadata
    prompt: str
    pass_number: int = 1


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


@dataclass
class _PaperPassResults:
    """Accumulator for per-paper pass results during batch reassembly."""

    answers: dict[str, str | None] = field(default_factory=dict)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    errors: list[str] = field(default_factory=list)


class BatchExtractionClient:
    """Client for batch extraction using Anthropic Batch API.

    The Batch API is cost-effective (50% discount) but asynchronous.
    Batches are processed within 24 hours.

    Uses the 6-pass SemanticAnalysis pipeline: each paper generates 6
    requests (one per pass), which are reassembled into a single
    SemanticAnalysis after completion.

    Typical workflow:
    1. Create batch from papers (6 requests per paper)
    2. Submit batch
    3. Poll for completion (or wait)
    4. Retrieve and reassemble results
    """

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = 8192,
        batch_dir: Path | None = None,
    ):
        """Initialize batch client.

        Args:
            model: Model to use for extraction.
            max_tokens: Maximum tokens per response.
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
        self.model = model or DEFAULT_MODELS["anthropic"]
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
        pass_definitions: list[tuple[str, list[tuple[str, str]]]] | None = None,
    ) -> list[BatchRequest]:
        """Create batch requests from papers.

        Generates 6 requests per paper (one per pass in the SemanticAnalysis
        pipeline).

        Args:
            papers: List of papers to process.
            text_getter: Function(paper) -> str that returns cleaned text.

        Returns:
            List of BatchRequest objects (6 per paper).
        """
        requests = []
        resolved_pass_definitions = pass_definitions or get_pass_definitions()
        num_passes = len(resolved_pass_definitions)

        for paper in papers:
            try:
                text = text_getter(paper)

                for pass_num in range(1, num_passes + 1):
                    prompt = build_pass_user_prompt(
                        pass_number=pass_num,
                        title=paper.title,
                        authors=paper.author_string,
                        year=paper.publication_year,
                        document_type=paper.item_type,
                        text=text,
                        pass_definitions=resolved_pass_definitions,
                    )
                    custom_id = self._build_custom_id(paper.paper_id, pass_num)
                    requests.append(
                        BatchRequest(
                            custom_id=custom_id,
                            paper=paper,
                            prompt=prompt,
                            pass_number=pass_num,
                        )
                    )
            except Exception as e:
                logger.warning(f"Failed to create requests for {paper.paper_id}: {e}")

        return requests

    def submit_batch(self, requests: list[BatchRequest], *, persist_state: bool = True) -> str:
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
            batch_requests.append(
                {
                    "custom_id": req.custom_id,
                    "params": {
                        "model": self.model,
                        "max_tokens": self.max_tokens,
                        "system": SEMANTIC_SYSTEM_PROMPT,
                        "messages": [{"role": "user", "content": req.prompt}],
                    },
                }
            )

        logger.info(f"Submitting batch with {len(batch_requests)} requests...")

        # Create the batch
        response = self.client.messages.batches.create(requests=batch_requests)

        batch_id = response.id
        logger.info(f"Batch submitted: {batch_id}")

        # Save batch state
        if persist_state:
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
                raise TimeoutError(f"Batch {batch_id} did not complete within {max_wait}s")

            logger.info(
                f"Batch {batch_id}: {status.completed_requests}/{status.total_requests} "
                f"complete, waiting {poll_interval}s..."
            )
            time.sleep(poll_interval)

    def get_results(
        self,
        batch_id: str,
        *,
        pass_definitions: list[tuple[str, list[tuple[str, str]]]] | None = None,
        profile: DimensionProfile | None = None,
        prompt_version: str = SEMANTIC_PROMPT_VERSION,
    ) -> Iterator[ExtractionResult]:
        """Retrieve results from a completed batch.

        Reassembles 6 per-pass responses into one SemanticAnalysis per paper.

        Args:
            batch_id: Batch ID from submit_batch.

        Yields:
            ExtractionResult for each paper.
        """
        # Collect all pass results grouped by paper_id
        paper_results: dict[str, _PaperPassResults] = defaultdict(_PaperPassResults)

        resolved_pass_definitions = pass_definitions or get_pass_definitions()
        field_to_dimension = _field_to_dimension_id(profile)

        for result in self.client.messages.batches.results(batch_id):
            custom_id = result.custom_id

            # Parse paper_id and pass number from custom_id
            parsed_id = self._parse_custom_id(custom_id)
            if parsed_id is None:
                # Legacy single-pass format (no :pass suffix)
                paper_id = custom_id
                pass_num = 0
                logger.warning(
                    f"Legacy single-pass custom_id format: {custom_id}. "
                    "Cannot reassemble into SemanticAnalysis."
                )
                continue
            paper_id, pass_num = parsed_id

            acc = paper_results[paper_id]

            if result.result.type == "succeeded":
                try:
                    message = result.result.message
                    response_text = message.content[0].text
                    pass_data = self._parse_pass_response(response_text)

                    # Get expected fields for this pass
                    if 1 <= pass_num <= len(resolved_pass_definitions):
                        _, pass_questions = resolved_pass_definitions[pass_num - 1]
                        expected_fields = [q[0] for q in pass_questions]

                        for field_name in expected_fields:
                            acc.answers[field_name] = pass_data.get(field_name)

                    acc.total_input_tokens += message.usage.input_tokens
                    acc.total_output_tokens += message.usage.output_tokens

                except Exception as e:
                    logger.error(f"Failed to parse pass {pass_num} for {paper_id}: {e}")
                    acc.errors.append(f"pass {pass_num}: parse error: {e}")
            else:
                error_msg = getattr(result.result, "error", {})
                acc.errors.append(f"pass {pass_num}: {error_msg}")

        # Reassemble each paper's passes into a SemanticAnalysis
        for paper_id, acc in paper_results.items():
            num_passes = len(resolved_pass_definitions)
            if len(acc.errors) == num_passes:
                yield ExtractionResult(
                    paper_id=paper_id,
                    success=False,
                    error=f"All {num_passes} passes failed: " + "; ".join(acc.errors),
                    model_used=self.model,
                    input_tokens=acc.total_input_tokens,
                    output_tokens=acc.total_output_tokens,
                )
                continue

            try:
                extraction = _build_extraction(
                    paper_id=paper_id,
                    answers=acc.answers,
                    model=self.model,
                    field_to_dimension=field_to_dimension,
                    profile=profile,
                    prompt_version=prompt_version,
                )

                logger.info(
                    f"Reassembled {paper_id}: "
                    f"{num_passes - len(acc.errors)}/{num_passes} passes, "
                    f"coverage: {extraction.dimension_coverage:.0%}"
                )

                yield ExtractionResult(
                    paper_id=paper_id,
                    success=True,
                    extraction=extraction,
                    model_used=self.model,
                    input_tokens=acc.total_input_tokens,
                    output_tokens=acc.total_output_tokens,
                )
            except Exception as e:
                logger.error(f"Failed to build SemanticAnalysis for {paper_id}: {e}")
                yield ExtractionResult(
                    paper_id=paper_id,
                    success=False,
                    error=f"Reassembly error: {e}",
                    model_used=self.model,
                    input_tokens=acc.total_input_tokens,
                    output_tokens=acc.total_output_tokens,
                )

    def _parse_pass_response(self, response_text: str) -> dict[str, str | None]:
        """Parse a single pass JSON response into a dict of q-field answers.

        Args:
            response_text: Raw JSON response from LLM.

        Returns:
            Dict mapping field names to their string answers (or None).
        """
        # Clean response - remove any markdown formatting
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        if not text:
            raise ValueError("Cannot parse empty response from LLM")

        data = json.loads(text)

        # Validate that we got q-field keys
        has_q_fields = any(k.startswith("q") and k[1:3].isdigit() for k in data)
        if not has_q_fields:
            logger.warning(
                "Pass response does not contain q-field keys. Got keys: %s",
                list(data.keys()),
            )

        return data

    def _save_batch_state(self, batch_id: str, requests: list[BatchRequest]) -> None:
        """Save batch state to disk for recovery."""
        # Deduplicate paper_ids (each paper has 6 requests)
        paper_ids = sorted(
            {
                parsed[0] if parsed else r.custom_id
                for r in requests
                for parsed in [self._parse_custom_id(r.custom_id)]
            }
        )

        state = {
            "batch_id": batch_id,
            "created_at": datetime.now().isoformat(),
            "model": self.model,
            "pipeline": "semantic_6pass",
            "paper_ids": paper_ids,
            "total_requests": len(requests),
            "passes_per_paper": len(get_pass_definitions()),
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

    # Import batch pricing from centralized constants
    BATCH_PRICING = ANTHROPIC_BATCH_PRICING

    def estimate_cost(self, num_papers: int, avg_text_length: int) -> dict:
        """Estimate cost for batch extraction.

        Accounts for 6 passes per paper in the SemanticAnalysis pipeline.

        Args:
            num_papers: Number of papers.
            avg_text_length: Average text length per paper.

        Returns:
            Cost estimate dictionary.
        """
        # Rough estimate: 4 chars per token
        # Each pass sends the full text, so input tokens are multiplied by 6
        input_tokens_per_pass = avg_text_length // 4 + 500
        output_tokens_per_pass = 2000

        num_passes = len(get_pass_definitions())
        total_input = input_tokens_per_pass * num_papers * num_passes
        total_output = output_tokens_per_pass * num_papers * num_passes

        # Get batch pricing for model (default to Opus 4.6 pricing)
        input_cost_per_million, output_cost_per_million = self.BATCH_PRICING.get(
            self.model, (2.50, 12.50)
        )

        input_cost = (total_input / 1_000_000) * input_cost_per_million
        output_cost = (total_output / 1_000_000) * output_cost_per_million

        return {
            "num_papers": num_papers,
            "passes_per_paper": num_passes,
            "total_requests": num_papers * num_passes,
            "estimated_input_tokens": total_input,
            "estimated_output_tokens": total_output,
            "estimated_cost": round(input_cost + output_cost, 2),
            "discount": "50% (batch API)",
            "model": self.model,
            "pricing": f"${input_cost_per_million}/MTok in, ${output_cost_per_million}/MTok out",
        }

    @staticmethod
    def _build_custom_id(paper_id: str, pass_num: int) -> str:
        """Build an Anthropic-safe custom ID for a paper pass."""
        return f"{paper_id}{CUSTOM_ID_PASS_SEPARATOR}{pass_num}"

    @staticmethod
    def _parse_custom_id(custom_id: str) -> tuple[str, int] | None:
        """Parse a batch custom ID, accepting current and legacy separators."""
        for separator in (CUSTOM_ID_PASS_SEPARATOR, ":pass"):
            if separator not in custom_id:
                continue
            paper_id, pass_suffix = custom_id.rsplit(separator, 1)
            try:
                return paper_id, int(pass_suffix)
            except ValueError:
                logger.warning(f"Invalid pass number in custom_id: {custom_id}")
                return None
        return None


def _field_to_dimension_id(profile: DimensionProfile | None) -> dict[str, str]:
    """Return output-key -> canonical dimension id mapping for a profile."""

    if profile is None:
        return {}
    mapping: dict[str, str] = {}
    for dimension in profile.dimensions:
        for key in [dimension.id, dimension.legacy_field_name, dimension.legacy_short_name]:
            if key:
                mapping[key] = dimension.id
    return mapping


def _build_extraction(
    *,
    paper_id: str,
    answers: dict[str, str | None],
    model: str,
    field_to_dimension: dict[str, str],
    profile: DimensionProfile | None,
    prompt_version: str,
) -> SemanticAnalysis | DimensionedExtraction:
    """Build an extraction object from pass answers."""

    if profile is None:
        analysis = SemanticAnalysis(
            paper_id=paper_id,
            prompt_version=prompt_version,
            extraction_model=model,
            extracted_at=datetime.now().isoformat(),
            **answers,
        )
        coverage_result = score_coverage(analysis)
        analysis.dimension_coverage = coverage_result.coverage
        analysis.coverage_flags = coverage_result.flags
        return analysis

    dimensions = {dimension.id: None for dimension in profile.dimensions}
    for field_name, value in answers.items():
        canonical_id = field_to_dimension.get(field_name)
        if canonical_id:
            dimensions[canonical_id] = value

    extraction = DimensionedExtraction(
        paper_id=paper_id,
        profile_id=profile.profile_id,
        profile_version=profile.version,
        profile_fingerprint=profile.fingerprint,
        prompt_version=prompt_version,
        extraction_model=model,
        extracted_at=datetime.now().isoformat(),
        dimensions=dimensions,
    )
    coverage_result = score_dimension_values(
        paper_id=paper_id,
        profile=profile,
        dimensions=extraction.dimensions,
    )
    extraction.dimension_coverage = coverage_result.coverage
    extraction.coverage_flags = coverage_result.flags
    return extraction
