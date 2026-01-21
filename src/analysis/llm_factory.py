"""Factory for creating LLM clients based on provider configuration."""

from typing import Literal

from src.analysis.base_llm import BaseLLMClient, ExtractionMode, LLMProvider

# Provider type for configuration
Provider = Literal["anthropic", "openai"]


def create_llm_client(
    provider: Provider = "anthropic",
    mode: ExtractionMode = "api",
    model: str | None = None,
    max_tokens: int = 8192,
    timeout: int = 120,
    **kwargs,
) -> BaseLLMClient:
    """Create an LLM client for the specified provider.

    This is the main entry point for creating LLM clients. It handles
    provider selection and passes configuration to the appropriate client.

    Args:
        provider: LLM provider ('anthropic' or 'openai').
        mode: Extraction mode ('api' for direct API, 'cli' for CLI tool).
        model: Model identifier. If None, uses provider's default.
        max_tokens: Maximum tokens for response.
        timeout: Request timeout in seconds.
        **kwargs: Additional provider-specific arguments.
            For OpenAI: reasoning_effort ('none', 'low', 'medium', 'high', 'xhigh')

    Returns:
        Configured LLM client instance.

    Raises:
        ValueError: If provider is not supported.

    Example:
        # Create Anthropic client (default)
        client = create_llm_client(provider="anthropic", mode="cli")

        # Create OpenAI client with GPT-5.2
        client = create_llm_client(
            provider="openai",
            model="gpt-5.2",
            reasoning_effort="high"
        )

        # Create OpenAI client with Codex CLI
        client = create_llm_client(
            provider="openai",
            mode="cli",
            model="gpt-5.2-codex"
        )
    """
    if provider == "anthropic":
        from src.analysis.anthropic_client import AnthropicLLMClient
        return AnthropicLLMClient(
            mode=mode,
            model=model,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    elif provider == "openai":
        from src.analysis.openai_client import OpenAILLMClient
        return OpenAILLMClient(
            mode=mode,
            model=model,
            max_tokens=max_tokens,
            timeout=timeout,
            reasoning_effort=kwargs.get("reasoning_effort"),
        )
    else:
        raise ValueError(
            f"Unsupported LLM provider: {provider}. "
            f"Supported providers: {get_available_providers()}"
        )


def get_available_providers() -> list[str]:
    """Return list of available LLM providers."""
    return ["anthropic", "openai"]


def get_provider_models(provider: Provider) -> dict[str, str]:
    """Get available models for a provider.

    Args:
        provider: LLM provider name.

    Returns:
        Dictionary of model_id -> description.
    """
    if provider == "anthropic":
        from src.analysis.anthropic_client import AnthropicLLMClient
        return AnthropicLLMClient.list_models()
    elif provider == "openai":
        from src.analysis.openai_client import OpenAILLMClient
        return OpenAILLMClient.list_models()
    else:
        return {}


def get_default_model(provider: Provider) -> str:
    """Get the default model for a provider.

    Args:
        provider: LLM provider name.

    Returns:
        Default model identifier.
    """
    defaults = {
        "anthropic": "claude-opus-4-5-20251101",
        "openai": "gpt-5.2",
    }
    return defaults.get(provider, "")


# Backward compatibility alias
def get_llm_client(
    mode: ExtractionMode = "api",
    model: str = "claude-opus-4-5-20251101",
    max_tokens: int = 8192,
    timeout: int = 120,
) -> BaseLLMClient:
    """Create an Anthropic LLM client (backward compatibility).

    Deprecated: Use create_llm_client() instead.

    Args:
        mode: Extraction mode.
        model: Model to use.
        max_tokens: Maximum tokens.
        timeout: Timeout in seconds.

    Returns:
        AnthropicLLMClient instance.
    """
    return create_llm_client(
        provider="anthropic",
        mode=mode,
        model=model,
        max_tokens=max_tokens,
        timeout=timeout,
    )
