"""Factory for creating LLM clients based on provider configuration."""

from typing import Literal

from src.analysis.base_llm import BaseLLMClient, ExtractionMode

# Provider type for configuration
Provider = Literal["anthropic", "openai", "google", "ollama", "llamacpp"]


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
        provider: LLM provider ('anthropic', 'openai', 'google', 'ollama', or 'llamacpp').
        mode: Extraction mode ('api' for direct API, 'cli' for CLI tool).
        model: Model identifier. If None, uses provider's default.
        max_tokens: Maximum tokens for response.
        timeout: Request timeout in seconds.
        **kwargs: Additional provider-specific arguments.
            For OpenAI: reasoning_effort ('none', 'low', 'medium', 'high', 'xhigh')
            For Ollama: ollama_host (server URL, default http://localhost:11434)
            For llama.cpp: model_path (path to GGUF file), n_ctx (context size),
                n_gpu_layers (GPU offload layers), verbose (enable logging)

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

        # Create Google Gemini client
        client = create_llm_client(
            provider="google",
            model="gemini-2.5-flash"
        )

        # Create Ollama client for local inference
        client = create_llm_client(
            provider="ollama",
            model="llama3",
            ollama_host="http://localhost:11434"
        )

        # Create llama.cpp client for direct model loading
        client = create_llm_client(
            provider="llamacpp",
            model_path="/path/to/model.gguf",
            n_gpu_layers=-1  # Offload all layers to GPU
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
    elif provider == "google":
        from src.analysis.gemini_client import GeminiLLMClient
        return GeminiLLMClient(
            mode=mode,
            model=model,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    elif provider == "ollama":
        from src.analysis.ollama_client import OllamaLLMClient
        return OllamaLLMClient(
            mode=mode,
            model=model,
            max_tokens=max_tokens,
            timeout=timeout,
            host=kwargs.get("ollama_host"),
        )
    elif provider == "llamacpp":
        from src.analysis.llamacpp_client import LlamaCppLLMClient
        return LlamaCppLLMClient(
            mode=mode,
            model=model,
            max_tokens=max_tokens,
            timeout=timeout,
            model_path=kwargs.get("model_path"),
            n_ctx=kwargs.get("n_ctx", 8192),
            n_gpu_layers=kwargs.get("n_gpu_layers", -1),
            verbose=kwargs.get("verbose", False),
        )
    else:
        raise ValueError(
            f"Unsupported LLM provider: {provider}. "
            f"Supported providers: {get_available_providers()}"
        )


def get_available_providers() -> list[str]:
    """Return list of available LLM providers."""
    return ["anthropic", "openai", "google", "ollama", "llamacpp"]


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
    elif provider == "google":
        from src.analysis.gemini_client import GeminiLLMClient
        return GeminiLLMClient.list_models()
    elif provider == "ollama":
        from src.analysis.ollama_client import OllamaLLMClient
        return OllamaLLMClient.list_models()
    elif provider == "llamacpp":
        from src.analysis.llamacpp_client import LlamaCppLLMClient
        return LlamaCppLLMClient.list_models()
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
        "google": "gemini-2.5-flash",
        "ollama": "llama3",
        "llamacpp": "llama-3",
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
