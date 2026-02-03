"""Anthropic (Claude) LLM client for paper extraction."""

import json
import time

from anthropic import Anthropic, APIConnectionError, APIError, RateLimitError
from pydantic import ValidationError

from src.analysis.base_llm import BaseLLMClient, ExtractionMode, LLMProvider
from src.analysis.cli_executor import ClaudeCliExecutor, CliExecutionError
from src.analysis.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    build_cli_extraction_prompt,
    build_extraction_prompt,
)
from src.analysis.schemas import ExtractionResult, PaperExtraction
from src.utils.logging_config import get_logger
from src.utils.secrets import get_anthropic_api_key

logger = get_logger(__name__)


class AnthropicLLMClient(BaseLLMClient):
    """Client for Anthropic Claude-based paper extraction."""

    # Available Claude models
    MODELS = {
        "claude-opus-4-5-20251101": "Claude Opus 4.5 (Latest, most capable)",
        "claude-sonnet-4-20250514": "Claude Sonnet 4 (Fast, capable)",
        "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet (Legacy)",
        "claude-3-opus-20240229": "Claude 3 Opus (Legacy)",
    }

    # Model pricing per million tokens (input, output) in USD
    # Source: https://docs.anthropic.com/en/docs/about-claude/models#model-comparison
    # Updated: 2026-02
    MODEL_PRICING = {
        "claude-opus-4-5-20251101": (5.0, 25.0),      # Opus 4.5: $5/$25 per MTok
        "claude-sonnet-4-20250514": (3.0, 15.0),      # Sonnet 4: $3/$15 per MTok
        "claude-sonnet-4-5-20250514": (3.0, 15.0),    # Sonnet 4.5: $3/$15 per MTok
        "claude-haiku-4-5-20250514": (1.0, 5.0),      # Haiku 4.5: $1/$5 per MTok
        "claude-3-5-sonnet-20241022": (3.0, 15.0),    # Legacy Sonnet 3.5
        "claude-3-5-haiku-20241022": (0.80, 4.0),     # Legacy Haiku 3.5
        "claude-3-opus-20240229": (15.0, 75.0),       # Legacy Opus 3 (deprecated)
    }

    # Batch API pricing (50% discount on all models)
    BATCH_PRICING = {
        "claude-opus-4-5-20251101": (2.50, 12.50),    # Opus 4.5 batch
        "claude-sonnet-4-20250514": (1.50, 7.50),     # Sonnet 4 batch
        "claude-sonnet-4-5-20250514": (1.50, 7.50),   # Sonnet 4.5 batch
        "claude-haiku-4-5-20250514": (0.50, 2.50),    # Haiku 4.5 batch
        "claude-3-5-sonnet-20241022": (1.50, 7.50),   # Legacy Sonnet 3.5 batch
        "claude-3-5-haiku-20241022": (0.40, 2.0),     # Legacy Haiku 3.5 batch
        "claude-3-opus-20240229": (7.50, 37.50),      # Legacy Opus 3 batch
    }

    def __init__(
        self,
        mode: ExtractionMode = "api",
        model: str | None = None,
        max_tokens: int = 8192,
        timeout: int = 120,
    ):
        """Initialize Anthropic LLM client.

        Args:
            mode: Extraction mode ('api' for Anthropic API, 'cli' for Claude CLI).
            model: Model to use for extraction.
            max_tokens: Maximum tokens for response.
            timeout: Request timeout in seconds.
        """
        super().__init__(mode=mode, model=model, max_tokens=max_tokens, timeout=timeout)
        self.cli_executor = None
        self.client = None

        self.validate_mode()

        if mode == "batch_api":
            raise ValueError(
                "batch_api mode is not supported via AnthropicLLMClient. "
                "Use scripts/batch_extract.py to submit and collect batch jobs."
            )
        if mode == "api":
            api_key = get_anthropic_api_key()
            if not api_key:
                raise ValueError(
                    "Anthropic API key required for API mode. "
                    "Set ANTHROPIC_API_KEY or store it in the OS keyring "
                    "(service: 'litris', key: 'ANTHROPIC_API_KEY')."
                )
            self.client = Anthropic(api_key=api_key)
        elif mode == "cli":
            self.cli_executor = ClaudeCliExecutor(timeout=timeout)

    @property
    def provider(self) -> LLMProvider:
        """Return the provider identifier."""
        return "anthropic"

    @property
    def default_model(self) -> str:
        """Return the default model for Anthropic."""
        return "claude-opus-4-5-20251101"

    @property
    def supported_modes(self) -> list[ExtractionMode]:
        """Return list of supported extraction modes."""
        return ["api", "cli"]

    @classmethod
    def list_models(cls) -> dict[str, str]:
        """Return available models with descriptions."""
        return cls.MODELS.copy()

    def extract(
        self,
        paper_id: str,
        title: str,
        authors: str,
        year: int | str | None,
        item_type: str,
        text: str,
        prompt_override: str | None = None,
    ) -> ExtractionResult:
        """Extract structured information from paper text.

        Args:
            paper_id: Unique paper identifier.
            title: Paper title.
            authors: Author string.
            year: Publication year.
            item_type: Type of paper.
            text: Full text content.
            prompt_override: Optional pre-built prompt to use instead of default.

        Returns:
            ExtractionResult with extraction or error.
        """
        start_time = time.time()

        try:
            if self.mode == "api":
                # API mode: build full prompt with text included
                prompt = prompt_override or build_extraction_prompt(
                    title=title,
                    authors=authors,
                    year=year,
                    item_type=item_type,
                    text=text,
                )
                response_text, input_tokens, output_tokens = self._call_api(prompt)
            else:
                # CLI mode: use separate prompt and text for proper handling
                # The cli_executor.extract() method writes text to temp file and
                # passes prompt via -p flag, which handles large documents better
                # Include system prompt in user prompt since CLI doesn't have separate system prompt
                user_prompt = build_cli_extraction_prompt(
                    title=title,
                    authors=authors,
                    year=year,
                    item_type=item_type,
                )
                prompt = f"{EXTRACTION_SYSTEM_PROMPT}\n\n---\n\n{user_prompt}"
                response_dict = self.cli_executor.extract(prompt, text)
                response_text = json.dumps(response_dict)
                input_tokens, output_tokens = 0, 0

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

        except RateLimitError as e:
            duration = time.time() - start_time
            logger.warning(f"Rate limit hit for paper {paper_id}: {e}")
            return ExtractionResult(
                paper_id=paper_id,
                success=False,
                error=f"Rate limit exceeded: {e}",
                duration_seconds=duration,
                model_used=self.model,
            )
        except APIConnectionError as e:
            duration = time.time() - start_time
            logger.error(f"API connection failed for paper {paper_id}: {e}")
            return ExtractionResult(
                paper_id=paper_id,
                success=False,
                error=f"API connection error: {e}",
                duration_seconds=duration,
                model_used=self.model,
            )
        except APIError as e:
            duration = time.time() - start_time
            logger.error(f"API error for paper {paper_id}: {e}")
            return ExtractionResult(
                paper_id=paper_id,
                success=False,
                error=f"API error: {e}",
                duration_seconds=duration,
                model_used=self.model,
            )
        except CliExecutionError as e:
            duration = time.time() - start_time
            logger.error(f"CLI execution failed for paper {paper_id}: {e}")
            return ExtractionResult(
                paper_id=paper_id,
                success=False,
                error=f"CLI error: {e}",
                duration_seconds=duration,
                model_used=self.model,
            )
        except (json.JSONDecodeError, ValidationError) as e:
            duration = time.time() - start_time
            logger.error(f"Response parsing failed for paper {paper_id}: {e}")
            return ExtractionResult(
                paper_id=paper_id,
                success=False,
                error=f"Parse error: {e}",
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
        """Call Claude CLI using the ClaudeCliExecutor.

        Args:
            prompt: User prompt.

        Returns:
            Tuple of (response_text, input_tokens, output_tokens).
            Note: CLI mode does not provide token counts.
        """
        response_text = self.cli_executor.call_with_prompt(prompt)
        return response_text, 0, 0

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

        # Guard against empty response
        if not text:
            raise ValueError("Cannot parse empty response from LLM")

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

        # Get pricing for model (default to Opus pricing)
        pricing = self.MODEL_PRICING.get(self.model, (15.0, 75.0))
        input_cost_per_million, output_cost_per_million = pricing

        input_cost = (input_tokens / 1_000_000) * input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * output_cost_per_million

        return input_cost + output_cost
