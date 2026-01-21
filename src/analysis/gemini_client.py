"""Google Gemini LLM client for paper extraction.

Supports Gemini 2.5 and 3.x models via the Google Gen AI Python SDK.

References:
- https://ai.google.dev/gemini-api/docs/models
- https://ai.google.dev/gemini-api/docs/pricing
- https://googleapis.github.io/python-genai/
"""

import json
import os
import time

from pydantic import ValidationError

from src.analysis.base_llm import BaseLLMClient, ExtractionMode, LLMProvider
from src.analysis.prompts import EXTRACTION_SYSTEM_PROMPT, build_extraction_prompt
from src.analysis.schemas import ExtractionResult, PaperExtraction
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class GeminiLLMClient(BaseLLMClient):
    """Client for Google Gemini-based paper extraction.

    Supports Gemini 2.5 and 3.x models via the Google Gen AI SDK.
    """

    # Available Gemini models with descriptions
    MODELS = {
        # Gemini 3 family (latest)
        "gemini-3-flash": "Gemini 3 Flash (Pro intelligence at Flash speed)",
        "gemini-3-pro": "Gemini 3 Pro Preview (Highest capability)",
        # Gemini 2.5 family (stable)
        "gemini-2.5-flash": "Gemini 2.5 Flash (Best price-performance)",
        "gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite (Fastest, cost-effective)",
        "gemini-2.5-pro": "Gemini 2.5 Pro (State-of-the-art reasoning)",
        # Legacy (deprecated March 2026)
        "gemini-2.0-flash": "Gemini 2.0 Flash (Legacy - deprecated March 2026)",
    }

    # Model pricing per million tokens (input, output) in USD
    MODEL_PRICING = {
        "gemini-3-flash": (0.50, 3.00),
        "gemini-3-pro": (2.00, 12.00),  # <=200K context
        "gemini-2.5-flash": (0.15, 0.60),
        "gemini-2.5-flash-lite": (0.10, 0.40),
        "gemini-2.5-pro": (1.25, 10.00),
        "gemini-2.0-flash": (0.10, 0.40),
    }

    def __init__(
        self,
        mode: ExtractionMode = "api",
        model: str | None = None,
        max_tokens: int = 8192,
        timeout: int = 120,
    ):
        """Initialize Gemini LLM client.

        Args:
            mode: Extraction mode ('api' only for Gemini).
            model: Model to use. Defaults to gemini-3-pro.
            max_tokens: Maximum tokens for response.
            timeout: Request timeout in seconds.
        """
        super().__init__(mode=mode, model=model, max_tokens=max_tokens, timeout=timeout)
        self.client = None

        self.validate_mode()

        if mode == "api":
            api_key = self._get_api_key()
            if not api_key:
                raise ValueError(
                    "Google API key required for Gemini. "
                    "Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
                )
            # Lazy import to avoid requiring google-genai if not used
            try:
                from google import genai
                self.client = genai.Client(api_key=api_key)
            except ImportError:
                raise ImportError(
                    "Google Gen AI package not installed. "
                    "Install with: pip install google-genai"
                )

    @property
    def provider(self) -> LLMProvider:
        """Return the provider identifier."""
        return "google"

    @property
    def default_model(self) -> str:
        """Return the default model for Gemini."""
        return "gemini-3-pro"

    @property
    def supported_modes(self) -> list[ExtractionMode]:
        """Return list of supported extraction modes."""
        return ["api"]

    @classmethod
    def list_models(cls) -> dict[str, str]:
        """Return available models with descriptions."""
        return cls.MODELS.copy()

    def _get_api_key(self) -> str | None:
        """Get Google API key from environment.

        Checks GOOGLE_API_KEY first, then GEMINI_API_KEY.
        """
        return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

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

            response_text, input_tokens, output_tokens = self._call_api(prompt)

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

        except ImportError as e:
            duration = time.time() - start_time
            logger.error(f"Import error for paper {paper_id}: {e}")
            return ExtractionResult(
                paper_id=paper_id,
                success=False,
                error=f"Missing dependency: {e}",
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
        except Exception as e:
            duration = time.time() - start_time
            error_type = type(e).__name__
            logger.error(f"{error_type} for paper {paper_id}: {e}")
            return ExtractionResult(
                paper_id=paper_id,
                success=False,
                error=f"{error_type}: {e}",
                duration_seconds=duration,
                model_used=self.model,
            )

    def _call_api(self, prompt: str) -> tuple[str, int, int]:
        """Call Gemini API.

        Args:
            prompt: User prompt.

        Returns:
            Tuple of (response_text, input_tokens, output_tokens).
        """
        # Combine system prompt and user prompt
        # Gemini uses system_instruction for system prompts
        full_prompt = f"{EXTRACTION_SYSTEM_PROMPT}\n\n---\n\n{prompt}"

        # Make API call using generate_content
        response = self.client.models.generate_content(
            model=self.model,
            contents=full_prompt,
            config={
                "max_output_tokens": self.max_tokens,
                "temperature": 0.1,  # Low temperature for consistent extraction
            },
        )

        response_text = response.text

        # Extract token counts from usage metadata
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata"):
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)

        return response_text, input_tokens, output_tokens

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

        # Get pricing for model (default to gemini-2.5-flash pricing)
        pricing = self.MODEL_PRICING.get(self.model, (0.15, 0.60))
        input_cost_per_million, output_cost_per_million = pricing

        input_cost = (input_tokens / 1_000_000) * input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * output_cost_per_million

        return input_cost + output_cost
