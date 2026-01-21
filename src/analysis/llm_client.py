"""LLM client for paper extraction.

This module provides backward compatibility with the original LLMClient.
New code should use the llm_factory module instead.

Example (new way):
    from src.analysis.llm_factory import create_llm_client
    client = create_llm_client(provider="openai", model="gpt-5.2")

Example (legacy, still supported):
    from src.analysis.llm_client import LLMClient
    client = LLMClient(mode="api", model="claude-opus-4-5-20251101")
"""

# Re-export for backward compatibility
from src.analysis.anthropic_client import AnthropicLLMClient
from src.analysis.base_llm import BaseLLMClient, ExtractionMode
from src.analysis.llm_factory import (
    create_llm_client,
    get_available_providers,
    get_default_model,
    get_provider_models,
)

# Backward compatibility: LLMClient is now AnthropicLLMClient
LLMClient = AnthropicLLMClient

__all__ = [
    "LLMClient",
    "AnthropicLLMClient",
    "BaseLLMClient",
    "ExtractionMode",
    "create_llm_client",
    "get_available_providers",
    "get_default_model",
    "get_provider_models",
]
