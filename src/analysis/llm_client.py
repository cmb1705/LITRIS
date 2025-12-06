"""LLM client for paper extraction."""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Literal
import shutil

from anthropic import Anthropic

from src.analysis.prompts import EXTRACTION_SYSTEM_PROMPT, build_extraction_prompt
from src.analysis.schemas import ExtractionResult, PaperExtraction
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

ExtractionMode = Literal["api", "cli", "batch_api"]


class LLMClient:
    """Client for LLM-based paper extraction."""

    def __init__(
        self,
        mode: ExtractionMode = "api",
        model: str = "claude-opus-4-5-20251101",
        max_tokens: int = 8192,
        timeout: int = 120,
    ):
        """Initialize LLM client.

        Args:
            mode: Extraction mode ('api' for Anthropic API, 'cli' for Claude CLI).
            model: Model to use for extraction.
            max_tokens: Maximum tokens for response.
            timeout: Request timeout in seconds.
        """
        self.mode = mode
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout

        if mode == "batch_api":
            raise ValueError(
                "batch_api mode is not supported via LLMClient. "
                "Use scripts/batch_extract.py to submit and collect batch jobs."
            )
        if mode == "api":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable required for API mode"
                )
            self.client = Anthropic(api_key=api_key)
        else:
            self.client = None

    def extract(
        self,
        paper_id: str,
        title: str,
        authors: str,
        year: int | str | None,
        item_type: str,
        text: str,
    ) -> ExtractionResult:
        """Extract structured information from paper text.

        Args:
            paper_id: Unique paper identifier.
            title: Paper title.
            authors: Author string.
            year: Publication year.
            item_type: Type of paper.
            text: Full text content.

        Returns:
            ExtractionResult with extraction or error.
        """
        start_time = time.time()

        try:
            prompt = build_extraction_prompt(
                title=title,
                authors=authors,
                year=year,
                item_type=item_type,
                text=text,
            )

            if self.mode == "api":
                response_text, input_tokens, output_tokens = self._call_api(prompt)
            else:
                response_text, input_tokens, output_tokens = self._call_cli(prompt)

            # Parse JSON response
            extraction = self._parse_response(response_text)

            duration = time.time() - start_time
            logger.info(
                f"Extracted paper {paper_id} in {duration:.1f}s "
                f"(confidence: {extraction.extraction_confidence:.2f})"
            )

            return ExtractionResult(
                paper_id=paper_id,
                success=True,
                extraction=extraction,
                duration_seconds=duration,
                model_used=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Extraction failed for paper {paper_id}: {e}")
            return ExtractionResult(
                paper_id=paper_id,
                success=False,
                error=str(e),
                duration_seconds=duration,
                model_used=self.model,
            )

    def _call_api(self, prompt: str) -> tuple[str, int, int]:
        """Call Anthropic API.

        Args:
            prompt: User prompt.

        Returns:
            Tuple of (response_text, input_tokens, output_tokens).
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=EXTRACTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        return response_text, input_tokens, output_tokens

    def _call_cli(self, prompt: str) -> tuple[str, int, int]:
        """Call Claude CLI.

        Args:
            prompt: User prompt.

        Returns:
            Tuple of (response_text, input_tokens, output_tokens).
            Note: CLI mode does not provide token counts.
        """
        cli_path = shutil.which("claude") or "claude"
        cmd = [
            cli_path,
            "--print",
            "--model", self.model,
            "--system-prompt", EXTRACTION_SYSTEM_PROMPT,
            "--output-format", "text",
        ]

        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            encoding="utf-8",
        )

        if result.returncode != 0:
            raise RuntimeError(f"CLI error: {result.stderr or result.stdout}")

        return result.stdout, 0, 0

    def _parse_response(self, response_text: str) -> PaperExtraction:
        """Parse JSON response into PaperExtraction.

        Args:
            response_text: Raw response text.

        Returns:
            Parsed PaperExtraction.
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

        # Parse JSON
        data = json.loads(text)

        # Handle nested methodology
        if "methodology" in data and isinstance(data["methodology"], dict):
            from src.analysis.schemas import Methodology
            data["methodology"] = Methodology(**data["methodology"])

        # Handle nested key_findings
        if "key_findings" in data and isinstance(data["key_findings"], list):
            from src.analysis.schemas import KeyFinding
            data["key_findings"] = [
                KeyFinding(**f) if isinstance(f, dict) else f
                for f in data["key_findings"]
            ]

        # Handle nested key_claims
        if "key_claims" in data and isinstance(data["key_claims"], list):
            from src.analysis.schemas import KeyClaim
            data["key_claims"] = [
                KeyClaim(**c) if isinstance(c, dict) else c
                for c in data["key_claims"]
            ]

        return PaperExtraction(**data)

    def estimate_cost(self, text_length: int) -> float:
        """Estimate cost for extraction.

        Args:
            text_length: Length of input text.

        Returns:
            Estimated cost in USD.
        """
        # Rough estimate: 4 chars per token
        input_tokens = text_length // 4 + 500  # Add prompt overhead
        output_tokens = 2000  # Typical extraction output

        # Claude Opus pricing (as of 2024)
        input_cost_per_million = 15.0
        output_cost_per_million = 75.0

        input_cost = (input_tokens / 1_000_000) * input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * output_cost_per_million

        return input_cost + output_cost
