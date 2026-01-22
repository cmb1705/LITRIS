"""llama.cpp LLM client for local paper extraction.

Supports direct local LLM inference via llama-cpp-python bindings.

References:
- https://github.com/ggerganov/llama.cpp
- https://github.com/abetlen/llama-cpp-python
"""

import json
import time
from pathlib import Path

from pydantic import ValidationError

from src.analysis.base_llm import BaseLLMClient, ExtractionMode
from src.analysis.prompts import EXTRACTION_SYSTEM_PROMPT, build_extraction_prompt
from src.analysis.schemas import ExtractionResult, PaperExtraction
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class LlamaCppLLMClient(BaseLLMClient):
    """Client for llama.cpp-based local paper extraction.

    Loads GGUF models directly for inference. No external server required.
    No API costs - runs entirely on local hardware.
    """

    # Common GGUF model families (actual model file must be downloaded)
    MODELS = {
        # These are model family names - actual model path must be provided
        "llama-3": "Llama 3 family (Meta, general purpose)",
        "llama-3.1": "Llama 3.1 family (Meta, 8B/70B/405B)",
        "llama-3.2": "Llama 3.2 family (Meta, 1B/3B)",
        "llama-3.3": "Llama 3.3 family (Meta, 70B)",
        "mistral": "Mistral family (Mistral AI)",
        "mixtral": "Mixtral 8x7B MoE (Mistral AI)",
        "gemma": "Gemma family (Google)",
        "gemma-2": "Gemma 2 family (Google)",
        "phi-3": "Phi-3 family (Microsoft)",
        "qwen2": "Qwen 2 family (Alibaba)",
        "codellama": "Code Llama family (Meta)",
        "deepseek-coder": "DeepSeek Coder (code generation)",
    }

    def __init__(
        self,
        mode: ExtractionMode = "api",
        model: str | None = None,
        max_tokens: int = 8192,
        timeout: int = 600,
        model_path: str | Path | None = None,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        verbose: bool = False,
    ):
        """Initialize llama.cpp LLM client.

        Args:
            mode: Extraction mode ('api' only for llama.cpp).
            model: Model identifier (for logging only, actual model from model_path).
            max_tokens: Maximum tokens for response.
            timeout: Request timeout in seconds (default 600 for local inference).
            model_path: Path to GGUF model file. Required.
            n_ctx: Context window size. Default 8192.
            n_gpu_layers: Number of layers to offload to GPU. -1 for all.
            verbose: Enable verbose llama.cpp output.
        """
        super().__init__(mode=mode, model=model, max_tokens=max_tokens, timeout=timeout)
        self.llm = None
        self.model_path = Path(model_path) if model_path else None
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose

        self.validate_mode()

        if mode == "api":
            if not self.model_path:
                raise ValueError(
                    "model_path is required for llama.cpp provider. "
                    "Provide path to a GGUF model file."
                )

            if not self.model_path.exists():
                raise ValueError(f"Model file not found: {self.model_path}")

            # Lazy import to avoid requiring llama-cpp-python if not used
            try:
                from llama_cpp import Llama
                logger.info(f"Loading model from {self.model_path}...")
                self.llm = Llama(
                    model_path=str(self.model_path),
                    n_ctx=self.n_ctx,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=self.verbose,
                )
                logger.info("Model loaded successfully")
            except ImportError:
                raise ImportError(
                    "llama-cpp-python package not installed. "
                    "Install with: pip install llama-cpp-python"
                )

    @property
    def provider(self) -> str:
        """Return the provider identifier."""
        return "llamacpp"

    @property
    def default_model(self) -> str:
        """Return the default model for llama.cpp."""
        return "llama-3"

    @property
    def supported_modes(self) -> list[ExtractionMode]:
        """Return list of supported extraction modes."""
        return ["api"]

    @classmethod
    def list_models(cls) -> dict[str, str]:
        """Return available model families with descriptions."""
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
            prompt = prompt_override or build_extraction_prompt(
                title=title,
                authors=authors,
                year=year,
                item_type=item_type,
                text=text,
            )

            response_text = self._call_llm(prompt)

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
                model_used=self.model or str(self.model_path),
                input_tokens=0,  # Token counts not easily available
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
                model_used=self.model or str(self.model_path),
            )
        except (json.JSONDecodeError, ValidationError) as e:
            duration = time.time() - start_time
            logger.error(f"Response parsing failed for paper {paper_id}: {e}")
            return ExtractionResult(
                paper_id=paper_id,
                success=False,
                error=f"Parse error: {e}",
                duration_seconds=duration,
                model_used=self.model or str(self.model_path),
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
                model_used=self.model or str(self.model_path),
            )

    def _call_llm(self, prompt: str) -> str:
        """Call llama.cpp model.

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

        # Make chat completion call
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0.1,  # Low temperature for consistent extraction
        )

        return response["choices"][0]["message"]["content"]

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

    def unload_model(self) -> None:
        """Unload the model from memory."""
        if self.llm is not None:
            del self.llm
            self.llm = None
            logger.info("Model unloaded")
