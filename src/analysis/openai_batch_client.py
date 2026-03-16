"""OpenAI Batch API client for 6-pass SemanticAnalysis extraction.

Submits extraction requests as JSONL files via the OpenAI Batch API,
which offers 50% cost savings with a 24-hour completion window.

Each paper generates 6 batch requests (one per pass), identified by
custom_id format: {paper_id}:pass{N}. Results are reassembled into
SemanticAnalysis objects on retrieval.
"""

import json
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.analysis.constants import DEFAULT_MODELS, OPENAI_PRICING
from src.analysis.coverage import score_coverage
from src.analysis.schemas import ExtractionResult, SemanticAnalysis
from src.analysis.semantic_prompts import (
    PASS_DEFINITIONS,
    SEMANTIC_PROMPT_VERSION,
    SEMANTIC_SYSTEM_PROMPT,
    build_pass_user_prompt,
)
from src.utils.logging_config import get_logger
from src.utils.secrets import get_openai_api_key
from src.zotero.models import PaperMetadata

logger = get_logger(__name__)

NUM_PASSES = 6


@dataclass
class BatchRequest:
    """A single request in a batch."""

    custom_id: str  # {paper_id}:pass{N}
    paper: PaperMetadata
    prompt: str
    pass_number: int = 1


@dataclass
class BatchStatus:
    """Status of a batch job."""

    batch_id: str
    status: str  # validating, in_progress, completed, failed, expired, cancelled
    created_at: str
    total_requests: int
    completed_requests: int = 0
    failed_requests: int = 0
    output_file_id: str | None = None
    error_file_id: str | None = None


@dataclass
class _PaperPassResults:
    """Accumulator for per-paper pass results during batch reassembly."""

    answers: dict[str, str | None] = field(default_factory=dict)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    errors: list[str] = field(default_factory=list)


class OpenAIBatchClient:
    """Client for batch extraction using OpenAI Batch API.

    The Batch API offers 50% cost savings and processes within 24 hours.

    Typical workflow:
    1. Create batch requests from papers (6 per paper)
    2. Submit batch (uploads JSONL, creates batch job)
    3. Poll for completion
    4. Retrieve and reassemble results
    """

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = 8192,
        batch_dir: Path | None = None,
        reasoning_effort: str | None = None,
    ):
        """Initialize OpenAI batch client.

        Args:
            model: Model to use. Defaults to provider default.
            max_tokens: Maximum tokens per response.
            batch_dir: Directory to store batch state files.
            reasoning_effort: Reasoning effort for GPT-5.x models.
        """
        api_key = get_openai_api_key()
        if not api_key:
            raise ValueError(
                "OpenAI API key required for batch API. "
                "Set OPENAI_API_KEY or store it in the OS keyring "
                "(service: 'litris', key: 'OPENAI_API_KEY')."
            )

        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model or DEFAULT_MODELS["openai"]
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort or "xhigh"
        self.batch_dir = batch_dir or Path("data/batches")
        self.batch_dir.mkdir(parents=True, exist_ok=True)

    def create_batch_requests(
        self,
        papers: list[PaperMetadata],
        text_getter: Callable,
    ) -> list[BatchRequest]:
        """Create batch requests from papers.

        Generates 6 requests per paper (one per extraction pass).

        Args:
            papers: List of papers to process.
            text_getter: Function(paper) -> str returning cleaned text.

        Returns:
            List of BatchRequest objects (6 per paper).
        """
        requests = []

        for paper in papers:
            try:
                text = text_getter(paper)

                for pass_num in range(1, NUM_PASSES + 1):
                    prompt = build_pass_user_prompt(
                        pass_number=pass_num,
                        title=paper.title,
                        authors=paper.author_string,
                        year=paper.publication_year,
                        document_type=paper.item_type,
                        text=text,
                    )
                    custom_id = f"{paper.paper_id}:pass{pass_num}"
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

    def submit_batch(self, requests: list[BatchRequest]) -> str:
        """Submit a batch of requests.

        Writes JSONL file, uploads to OpenAI, and creates batch job.

        Args:
            requests: List of batch requests.

        Returns:
            Batch ID for tracking.
        """
        if not requests:
            raise ValueError("No requests to submit")

        # Write JSONL to temp file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".jsonl",
            delete=False,
            encoding="utf-8",
        ) as f:
            for req in requests:
                body: dict = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SEMANTIC_SYSTEM_PROMPT},
                        {"role": "user", "content": req.prompt},
                    ],
                    "max_completion_tokens": self.max_tokens,
                    "user": "litris-batch-extract",
                }
                # Add reasoning effort for GPT-5.x
                if self.reasoning_effort and self.model.startswith("gpt-5"):
                    body["reasoning_effort"] = self.reasoning_effort

                line = json.dumps({
                    "custom_id": req.custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                })
                f.write(line + "\n")
            jsonl_path = f.name

        logger.info(
            f"Uploading batch file with {len(requests)} requests "
            f"({len(requests) // NUM_PASSES} papers)..."
        )

        # Upload file
        with open(jsonl_path, "rb") as f:
            file_obj = self.client.files.create(file=f, purpose="batch")

        # Create batch
        batch = self.client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        batch_id = batch.id
        logger.info(f"Batch submitted: {batch_id}")

        # Clean up temp file
        Path(jsonl_path).unlink(missing_ok=True)

        # Save state
        self._save_batch_state(batch_id, requests)

        return batch_id

    def get_batch_status(self, batch_id: str) -> BatchStatus:
        """Get status of a batch.

        Args:
            batch_id: Batch ID from submit_batch.

        Returns:
            BatchStatus with current state.
        """
        batch = self.client.batches.retrieve(batch_id)

        counts = batch.request_counts
        return BatchStatus(
            batch_id=batch.id,
            status=batch.status,
            created_at=str(batch.created_at),
            total_requests=(
                counts.total if counts else 0
            ),
            completed_requests=(
                counts.completed if counts else 0
            ),
            failed_requests=(
                counts.failed if counts else 0
            ),
            output_file_id=batch.output_file_id,
            error_file_id=batch.error_file_id,
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
            progress_callback: Optional callback(status).

        Returns:
            Final BatchStatus.
        """
        start_time = time.time()

        while True:
            status = self.get_batch_status(batch_id)

            if progress_callback:
                progress_callback(status)

            if status.status == "completed":
                return status

            if status.status in ("failed", "expired", "cancelled"):
                raise RuntimeError(f"Batch {batch_id} {status.status}")

            elapsed = time.time() - start_time
            if elapsed > max_wait:
                raise TimeoutError(
                    f"Batch {batch_id} did not complete within {max_wait}s"
                )

            logger.info(
                f"Batch {batch_id}: {status.completed_requests}/{status.total_requests} "
                f"complete ({status.status}), waiting {poll_interval}s..."
            )
            time.sleep(poll_interval)

    def get_results(self, batch_id: str) -> Iterator[ExtractionResult]:
        """Retrieve results from a completed batch.

        Reassembles 6 per-pass responses into one SemanticAnalysis per paper.

        Args:
            batch_id: Batch ID from submit_batch.

        Yields:
            ExtractionResult for each paper.
        """
        status = self.get_batch_status(batch_id)
        if not status.output_file_id:
            raise RuntimeError(f"Batch {batch_id} has no output file")

        # Download results
        content = self.client.files.content(status.output_file_id)
        lines = content.text.strip().split("\n")

        # Collect all pass results grouped by paper_id
        paper_results: dict[str, _PaperPassResults] = defaultdict(
            _PaperPassResults
        )

        for line in lines:
            result = json.loads(line)
            custom_id = result["custom_id"]

            # Parse paper_id and pass number
            if ":pass" in custom_id:
                paper_id, pass_suffix = custom_id.rsplit(":pass", 1)
                try:
                    pass_num = int(pass_suffix)
                except ValueError:
                    logger.warning(f"Invalid pass number in custom_id: {custom_id}")
                    continue
            else:
                logger.warning(f"Unexpected custom_id format: {custom_id}")
                continue

            acc = paper_results[paper_id]
            response = result.get("response", {})

            if response.get("status_code") == 200:
                try:
                    body = response["body"]
                    response_text = body["choices"][0]["message"]["content"]
                    pass_data = self._parse_pass_response(response_text)

                    # Get expected fields for this pass
                    if 1 <= pass_num <= NUM_PASSES:
                        _, pass_questions = PASS_DEFINITIONS[pass_num - 1]
                        expected_fields = [q[0] for q in pass_questions]

                        for field_name in expected_fields:
                            acc.answers[field_name] = pass_data.get(field_name)

                    usage = body.get("usage", {})
                    acc.total_input_tokens += usage.get("prompt_tokens", 0)
                    acc.total_output_tokens += usage.get("completion_tokens", 0)

                except Exception as e:
                    logger.error(
                        f"Failed to parse pass {pass_num} for {paper_id}: {e}"
                    )
                    acc.errors.append(f"pass {pass_num}: parse error: {e}")
            else:
                error = response.get("error", {})
                error_msg = error.get("message", str(error))
                acc.errors.append(f"pass {pass_num}: {error_msg}")

        # Reassemble each paper
        for paper_id, acc in paper_results.items():
            if len(acc.errors) == NUM_PASSES:
                yield ExtractionResult(
                    paper_id=paper_id,
                    success=False,
                    error=f"All {NUM_PASSES} passes failed: "
                    + "; ".join(acc.errors),
                    model_used=self.model,
                    input_tokens=acc.total_input_tokens,
                    output_tokens=acc.total_output_tokens,
                )
                continue

            try:
                analysis = SemanticAnalysis(
                    paper_id=paper_id,
                    prompt_version=SEMANTIC_PROMPT_VERSION,
                    extraction_model=self.model,
                    extracted_at=datetime.now().isoformat(),
                    **acc.answers,
                )

                coverage_result = score_coverage(analysis)
                analysis.dimension_coverage = coverage_result.coverage
                analysis.coverage_flags = coverage_result.flags

                # Reject partial extractions
                successful_passes = NUM_PASSES - len(acc.errors)
                if successful_passes < NUM_PASSES // 2:
                    yield ExtractionResult(
                        paper_id=paper_id,
                        success=False,
                        error=(
                            f"Too few passes ({successful_passes}/{NUM_PASSES}): "
                            + "; ".join(acc.errors)
                        ),
                        model_used=self.model,
                        input_tokens=acc.total_input_tokens,
                        output_tokens=acc.total_output_tokens,
                    )
                    continue

                logger.info(
                    f"Reassembled {paper_id}: "
                    f"{successful_passes}/{NUM_PASSES} passes, "
                    f"coverage: {analysis.dimension_coverage:.0%}"
                )

                yield ExtractionResult(
                    paper_id=paper_id,
                    success=True,
                    extraction=analysis,
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
        """Parse a single pass JSON response into a dict of q-field answers."""
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

        has_q_fields = any(k.startswith("q") and k[1:3].isdigit() for k in data)
        if not has_q_fields:
            logger.warning(
                "Pass response does not contain q-field keys. Got keys: %s",
                list(data.keys()),
            )

        return data

    def _save_batch_state(self, batch_id: str, requests: list[BatchRequest]) -> None:
        """Save batch state to disk for recovery."""
        paper_ids = sorted({
            r.custom_id.rsplit(":pass", 1)[0]
            if ":pass" in r.custom_id
            else r.custom_id
            for r in requests
        })

        state = {
            "batch_id": batch_id,
            "provider": "openai",
            "created_at": datetime.now().isoformat(),
            "model": self.model,
            "pipeline": "semantic_6pass",
            "paper_ids": paper_ids,
            "total_requests": len(requests),
            "passes_per_paper": NUM_PASSES,
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
                if state.get("provider") != "openai":
                    continue
                batch_id = state.get("batch_id")
                if batch_id:
                    status = self.get_batch_status(batch_id)
                    if status.status not in ("completed", "failed", "expired", "cancelled"):
                        pending.append(batch_id)
            except Exception:
                continue

        return pending

    def estimate_cost(self, num_papers: int, avg_text_length: int) -> dict:
        """Estimate cost for batch extraction (50% discount).

        Args:
            num_papers: Number of papers.
            avg_text_length: Average text length per paper in chars.

        Returns:
            Cost estimate dictionary.
        """
        input_tokens_per_pass = avg_text_length // 4 + 500
        output_tokens_per_pass = 2000

        total_input = input_tokens_per_pass * num_papers * NUM_PASSES
        total_output = output_tokens_per_pass * num_papers * NUM_PASSES

        # Standard pricing with 50% batch discount
        std_input, std_output = OPENAI_PRICING.get(
            self.model, (2.50, 15.0)
        )
        input_cost_per_million = std_input * 0.5
        output_cost_per_million = std_output * 0.5

        input_cost = (total_input / 1_000_000) * input_cost_per_million
        output_cost = (total_output / 1_000_000) * output_cost_per_million

        return {
            "num_papers": num_papers,
            "passes_per_paper": NUM_PASSES,
            "total_requests": num_papers * NUM_PASSES,
            "estimated_input_tokens": total_input,
            "estimated_output_tokens": total_output,
            "estimated_cost": round(input_cost + output_cost, 2),
            "discount": "50% (batch API)",
            "model": self.model,
            "pricing": f"${input_cost_per_million}/MTok in, ${output_cost_per_million}/MTok out",
        }
