"""Abstract base class for LLM clients."""

from abc import ABC, abstractmethod
from typing import Literal

from src.analysis.schemas import ExtractionResult

ExtractionMode = Literal["api", "cli", "batch_api"]
LLMProvider = Literal["anthropic", "openai", "google"]


class BaseLLMClient(ABC):
    """Abstract base class for LLM-based paper extraction.

    All LLM provider implementations must inherit from this class
    and implement the required methods.
    """

    def __init__(
        self,
        mode: ExtractionMode = "api",
        model: str | None = None,
        max_tokens: int = 8192,
        timeout: int = 120,
    ):
        """Initialize LLM client.

        Args:
            mode: Extraction mode ('api' for direct API, 'cli' for CLI tool).
            model: Model identifier. If None, uses provider's default.
            max_tokens: Maximum tokens for response.
            timeout: Request timeout in seconds.
        """
        self.mode = mode
        self.model = model or self.default_model
        self.max_tokens = max_tokens
        self.timeout = timeout

    @property
    @abstractmethod
    def provider(self) -> LLMProvider:
        """Return the provider identifier."""
        ...

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Return the default model for this provider."""
        ...

    @property
    @abstractmethod
    def supported_modes(self) -> list[ExtractionMode]:
        """Return list of supported extraction modes."""
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    def estimate_cost(self, text_length: int) -> float:
        """Estimate cost for extraction.

        Args:
            text_length: Length of input text in characters.

        Returns:
            Estimated cost in USD.
        """
        ...

    def validate_mode(self) -> None:
        """Validate that the current mode is supported.

        Raises:
            ValueError: If mode is not supported by this provider.
        """
        if self.mode not in self.supported_modes:
            raise ValueError(
                f"{self.provider} does not support mode '{self.mode}'. "
                f"Supported modes: {self.supported_modes}"
            )

    @staticmethod
    def get_available_providers() -> list[str]:
        """Return list of available LLM providers."""
        return ["anthropic", "openai", "google"]
