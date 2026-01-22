"""Ollama LLM client for local paper extraction.

Supports local LLM inference via Ollama server.

References:
- https://ollama.com/
- https://github.com/ollama/ollama-python
"""

import json
import os
import time

from pydantic import ValidationError

from src.analysis.base_llm import BaseLLMClient, ExtractionMode
from src.analysis.prompts import EXTRACTION_SYSTEM_PROMPT, build_extraction_prompt
from src.analysis.schemas import ExtractionResult, PaperExtraction
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Type alias for Ollama provider (to be added to base_llm.py)
OllamaProvider = str


class OllamaLLMClient(BaseLLMClient):
    """Client for Ollama-based local paper extraction.

    Connects to a local or remote Ollama server for inference.
    No API costs - runs entirely on local hardware.
    """

    # Common Ollama models with descriptions
    MODELS = {
        # Llama 3 family
        "llama3.3": "Llama 3.3 70B (Meta, high capability)",
        "llama3.2": "Llama 3.2 (Meta, 1B/3B lightweight)",
        "llama3.1": "Llama 3.1 (Meta, 8B/70B/405B)",
        "llama3": "Llama 3 8B (Meta, general purpose)",
        # Mistral family
        "mistral": "Mistral 7B (Mistral AI, efficient)",
        "mixtral": "Mixtral 8x7B (Mistral AI, MoE)",
        "mistral-nemo": "Mistral Nemo 12B (Mistral AI)",
        # Gemma family
        "gemma2": "Gemma 2 (Google, 2B/9B/27B)",
        "gemma": "Gemma 7B (Google)",
        # Code models
        "codellama": "Code Llama (Meta, code-focused)",
        "deepseek-coder": "DeepSeek Coder (code generation)",
        # Qwen family
        "qwen2.5": "Qwen 2.5 (Alibaba, multilingual)",
        "qwen2": "Qwen 2 (Alibaba)",
        # Phi family
        "phi3": "Phi-3 (Microsoft, small but capable)",
        # Other popular models
        "neural-chat": "Neural Chat (Intel, conversational)",
        "starling-lm": "Starling LM (Berkeley, RLHF-tuned)",
        "vicuna": "Vicuna (LMSYS, instruction-following)",
        "orca-mini": "Orca Mini (Microsoft, reasoning)",
    }

    def __init__(
        self,
        mode: ExtractionMode = "api",
        model: str | None = None,
        max_tokens: int = 8192,
        timeout: int = 300,
        host: str | None = None,
    ):
        """Initialize Ollama LLM client.

        Args:
            mode: Extraction mode ('api' only for Ollama).
            model: Model to use. Defaults to llama3.
            max_tokens: Maximum tokens for response.
            timeout: Request timeout in seconds (default 300 for local inference).
            host: Ollama server URL. Defaults to http://localhost:11434.
        """
        super().__init__(mode=mode, model=model, max_tokens=max_tokens, timeout=timeout)
        self.client = None
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")

        self.validate_mode()

        if mode == "api":
            # Lazy import to avoid requiring ollama if not used
            try:
                import ollama
                self.client = ollama.Client(host=self.host)
            except ImportError:
                raise ImportError(
                    "Ollama package not installed. "
                    "Install with: pip install ollama"
                )

    @property
    def provider(self) -> str:
        """Return the provider identifier."""
        return "ollama"

    @property
    def default_model(self) -> str:
        """Return the default model for Ollama."""
        return "llama3"

    @property
    def supported_modes(self) -> list[ExtractionMode]:
        """Return list of supported extraction modes."""
        return ["api"]

    @classmethod
    def list_models(cls) -> dict[str, str]:
        """Return available models with descriptions."""
        return cls.MODELS.copy()

    def list_local_models(self) -> list[str]:
        """List models currently available on the Ollama server.

        Returns:
            List of model names available locally.
        """
        if self.client is None:
            return []

        try:
            response = self.client.list()
            return [model["name"] for model in response.get("models", [])]
        except Exception as e:
            logger.warning(f"Failed to list Ollama models: {e}")
            return []

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
            prompt = prompt_override or build_extraction_prompt(
                title=title,
                authors=authors,
                year=year,
                item_type=item_type,
                text=text,
            )

            response_text = self._call_api(prompt)

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
                input_tokens=0,  # Ollama doesn't provide token counts in basic API
                output_tokens=0,
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

    def _call_api(self, prompt: str) -> str:
        """Call Ollama API.

        Args:
            prompt: User prompt.

        Returns:
            Response text from model.
        """
        # Build messages with system prompt
        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # Make API call
        response = self.client.chat(
            model=self.model,
            messages=messages,
            options={
                "num_predict": self.max_tokens,
                "temperature": 0.1,  # Low temperature for consistent extraction
            },
        )

        return response["message"]["content"]

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

        Local inference has no API cost.

        Args:
            text_length: Length of input text.

        Returns:
            0.0 (local inference is free).
        """
        return 0.0

    def check_connection(self) -> bool:
        """Check if Ollama server is accessible.

        Returns:
            True if server is reachable and responding.
        """
        if self.client is None:
            return False

        try:
            self.client.list()
            return True
        except Exception:
            return False

    def pull_model(self, model: str | None = None) -> bool:
        """Pull a model to the Ollama server.

        Args:
            model: Model to pull. Defaults to self.model.

        Returns:
            True if pull was successful.
        """
        if self.client is None:
            return False

        target_model = model or self.model
        try:
            logger.info(f"Pulling model {target_model}...")
            self.client.pull(target_model)
            logger.info(f"Model {target_model} pulled successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to pull model {target_model}: {e}")
            return False
