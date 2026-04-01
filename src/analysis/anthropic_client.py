"""Anthropic (Claude) LLM client for paper extraction."""

import json
import time

from anthropic import Anthropic, APIConnectionError, APIError, RateLimitError
from pydantic import ValidationError

from src.analysis.base_llm import BaseLLMClient, ExtractionMode, LLMProvider
from src.analysis.cli_executor import ClaudeCliExecutor, CliExecutionError
from src.analysis.dimensions import (
    EXTRACTION_METADATA_KEYS,
    get_default_dimension_registry,
    is_dimension_payload,
    normalize_dimension_payload,
)
from src.analysis.constants import (
    ANTHROPIC_BATCH_PRICING,
    ANTHROPIC_MODELS,
    ANTHROPIC_PRICING,
    DEFAULT_MODELS,
)
from src.analysis.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    build_cli_extraction_prompt,
    build_extraction_prompt,
)
from src.analysis.retry import with_retry
from src.analysis.schemas import ExtractionResult, PaperExtraction, SemanticAnalysis
from src.utils.logging_config import get_logger
from src.utils.secrets import get_anthropic_api_key

logger = get_logger(__name__)


class AnthropicLLMClient(BaseLLMClient):
    """Client for Anthropic Claude-based paper extraction."""

    # Import from centralized constants
    MODELS = ANTHROPIC_MODELS
    MODEL_PRICING = ANTHROPIC_PRICING
    BATCH_PRICING = ANTHROPIC_BATCH_PRICING

    def __init__(
        self,
        mode: ExtractionMode = "api",
        model: str | None = None,
        max_tokens: int = 8192,
        timeout: int = 120,
        effort: str | None = None,
    ):
        """Initialize Anthropic LLM client.

        Args:
            mode: Extraction mode ('api' for Anthropic API, 'cli' for Claude CLI).
            model: Model to use for extraction.
            max_tokens: Maximum tokens for response.
            timeout: Request timeout in seconds.
            effort: Claude CLI effort level for extended thinking (low/medium/high).
        """
        super().__init__(mode=mode, model=model, max_tokens=max_tokens, timeout=timeout)
        self.cli_executor = None
        self.client = None
        self.effort = effort

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
            self.cli_executor = ClaudeCliExecutor(
                timeout=timeout,
                model=self.model,
                effort=effort,
            )

    @property
    def provider(self) -> LLMProvider:
        """Return the provider identifier."""
        return "anthropic"

    @property
    def default_model(self) -> str:
        """Return the default model for Anthropic."""
        return DEFAULT_MODELS["anthropic"]

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
                # CLI mode: use prompt_override if provided (from 6-pass pipeline),
                # otherwise fall back to legacy single-pass prompt
                if prompt_override:
                    prompt = prompt_override
                else:
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
            confidence = getattr(extraction, "extraction_confidence", None)
            conf_str = f", confidence: {confidence:.2f}" if confidence else ""
            logger.info(
                f"Extracted paper {paper_id} in {duration:.1f}s{conf_str}"
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
            if e.stdout:
                logger.debug("Claude CLI stdout for %s: %s", paper_id, e.stdout[:500])
            if e.stderr:
                logger.debug("Claude CLI stderr for %s: %s", paper_id, e.stderr[:500])
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
            if e.stdout:
                logger.debug("Claude CLI stdout for %s: %s", paper_id, e.stdout[:500])
            if e.stderr:
                logger.debug("Claude CLI stderr for %s: %s", paper_id, e.stderr[:500])
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

    @with_retry(max_retries=3, retry_delay=2.0)
    def _call_api(self, prompt: str) -> tuple[str, int, int]:
        """Call Anthropic API with automatic retry for transient errors.

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

    def _parse_response(
        self, response_text: str
    ) -> SemanticAnalysis | PaperExtraction:
        """Parse JSON response into SemanticAnalysis or PaperExtraction.

        Detects q-field keys (from 6-pass pipeline) and returns SemanticAnalysis.
        Falls back to PaperExtraction for legacy single-pass responses.

        Args:
            response_text: Raw response text.

        Returns:
            Parsed SemanticAnalysis (6-pass) or PaperExtraction (legacy).
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

        if is_dimension_payload(data):
            # 6-pass pipeline: build SemanticAnalysis with placeholder metadata.
            # The caller (section_extractor._extract_6_pass) builds the final
            # SemanticAnalysis from merged answers; these placeholders are
            # only used by _extract_pass_answers via getattr.
            profile_id = data.get("profile_id") or get_default_dimension_registry().active_profile_id
            return SemanticAnalysis(
                paper_id=data.get("paper_id", "pending"),
                profile_id=profile_id,
                profile_version=data.get("profile_version", ""),
                profile_fingerprint=data.get("profile_fingerprint", ""),
                prompt_version=data.get("prompt_version", "2.0.0"),
                extraction_model=data.get("extraction_model", self.model),
                extracted_at=data.get("extracted_at", ""),
                dimensions=normalize_dimension_payload(data, profile_id=profile_id),
                **{
                    k: v
                    for k, v in data.items()
                    if k not in EXTRACTION_METADATA_KEYS
                },
            )

        # Legacy single-pass: handle nested models
        if "methodology" in data and isinstance(data["methodology"], dict):
            from src.analysis.schemas import Methodology
            data["methodology"] = Methodology(**data["methodology"])

        if "key_findings" in data and isinstance(data["key_findings"], list):
            from src.analysis.schemas import KeyFinding
            data["key_findings"] = [
                KeyFinding(**f) if isinstance(f, dict) else f
                for f in data["key_findings"]
            ]

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
