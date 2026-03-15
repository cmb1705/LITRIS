"""Centralized model defaults and pricing for LLM providers.

This module provides a single source of truth for:
- Default models per provider
- Model pricing (API and batch)
- Model descriptions

All LLM clients should import from here rather than defining their own constants.
"""

from typing import Literal

# Provider type
LLMProvider = Literal["anthropic", "openai", "google", "ollama", "llamacpp"]

# Default models per provider
DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-opus-4-6",
    "openai": "gpt-5.4",
    "google": "gemini-2.5-flash",
    "ollama": "llama3",
    "llamacpp": "llama-3",
}

# Anthropic models and pricing
# Source: https://docs.anthropic.com/en/docs/about-claude/models
# Updated: 2026-03
ANTHROPIC_MODELS: dict[str, str] = {
    "claude-opus-4-6": "Claude Opus 4.6 (Latest, most capable)",
    "claude-sonnet-4-6": "Claude Sonnet 4.6 (Fast, capable)",
    "claude-opus-4-5-20251101": "Claude Opus 4.5",
    "claude-sonnet-4-5-20250514": "Claude Sonnet 4.5 (Balanced)",
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5 (Fast, cost-effective)",
    "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet (Legacy)",
    "claude-3-5-haiku-20241022": "Claude 3.5 Haiku (Legacy)",
    "claude-3-opus-20240229": "Claude 3 Opus (Legacy)",
}

# Anthropic API pricing per million tokens (input, output) in USD
ANTHROPIC_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-6": (5.0, 25.0),               # Opus 4.6: $5/$25 per MTok
    "claude-sonnet-4-6": (3.0, 15.0),             # Sonnet 4.6: $3/$15 per MTok
    "claude-opus-4-5-20251101": (5.0, 25.0),      # Opus 4.5: $5/$25 per MTok
    "claude-sonnet-4-5-20250514": (3.0, 15.0),    # Sonnet 4.5: $3/$15 per MTok
    "claude-sonnet-4-20250514": (3.0, 15.0),      # Sonnet 4: $3/$15 per MTok
    "claude-haiku-4-5-20251001": (1.0, 5.0),      # Haiku 4.5: $1/$5 per MTok
    "claude-3-5-sonnet-20241022": (3.0, 15.0),    # Legacy Sonnet 3.5
    "claude-3-5-haiku-20241022": (0.80, 4.0),     # Legacy Haiku 3.5
    "claude-3-opus-20240229": (15.0, 75.0),       # Legacy Opus 3 (deprecated)
}

# Anthropic Batch API pricing (50% discount on all models)
ANTHROPIC_BATCH_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-6": (2.50, 12.50),             # Opus 4.6 batch
    "claude-sonnet-4-6": (1.50, 7.50),            # Sonnet 4.6 batch
    "claude-opus-4-5-20251101": (2.50, 12.50),    # Opus 4.5 batch
    "claude-sonnet-4-5-20250514": (1.50, 7.50),   # Sonnet 4.5 batch
    "claude-sonnet-4-20250514": (1.50, 7.50),     # Sonnet 4 batch
    "claude-haiku-4-5-20251001": (0.50, 2.50),    # Haiku 4.5 batch
    "claude-3-5-sonnet-20241022": (1.50, 7.50),   # Legacy Sonnet 3.5 batch
    "claude-3-5-haiku-20241022": (0.40, 2.0),     # Legacy Haiku 3.5 batch
    "claude-3-opus-20240229": (7.50, 37.50),      # Legacy Opus 3 batch
}

# OpenAI models and pricing
# Source: https://developers.openai.com/codex/models/
# Updated: 2026-03
OPENAI_MODELS: dict[str, str] = {
    # GPT-5.4 family (latest)
    "gpt-5.4": "GPT-5.4 (Latest, most capable)",
    "gpt-5.4-pro": "GPT-5.4 Pro (Highest quality, complex tasks)",
    # GPT-5 utility models
    "gpt-5-mini": "GPT-5 Mini (Fast, cost-efficient)",
    "gpt-5-nano": "GPT-5 Nano (Fastest, most economical)",
    # Legacy GPT-5.x
    "gpt-5.3-codex": "GPT-5.3-Codex (Legacy agentic coding)",
    "gpt-5.2": "GPT-5.2 (Legacy flagship)",
    "gpt-5.1": "GPT-5.1 (Legacy)",
    "gpt-5": "GPT-5 (Legacy base)",
    # o3 family
    "o3": "o3 (Reasoning)",
    "o3-pro": "o3 Pro (Deep reasoning)",
    "o3-mini": "o3-mini (Fast reasoning)",
}

OPENAI_PRICING: dict[str, tuple[float, float]] = {
    # GPT-5.4 family
    "gpt-5.4": (2.50, 15.0),
    "gpt-5.4-pro": (24.0, 192.0),
    # GPT-5 utility models
    "gpt-5-mini": (0.50, 4.0),
    "gpt-5-nano": (0.10, 0.80),
    # Legacy GPT-5.x
    "gpt-5.3-codex": (1.75, 14.0),
    "gpt-5.2": (1.75, 14.0),
    "gpt-5.1": (1.25, 10.0),
    "gpt-5": (1.25, 10.0),
    # o3 family
    "o3": (2.0, 8.0),
    "o3-pro": (20.0, 80.0),
    "o3-mini": (1.1, 4.4),
}

# Google Gemini models and pricing
# Source: https://ai.google.dev/gemini-api/docs/models
# Updated: 2026-03-14
GEMINI_MODELS: dict[str, str] = {
    # Gemini 3.1 family (latest, March 2026)
    "gemini-3.1-flash-lite": "Gemini 3.1 Flash Lite (Fastest, 2.5x faster TTFAT)",
    # Gemini 3 family
    "gemini-3-flash": "Gemini 3 Flash (Pro intelligence at Flash speed)",
    # Gemini 2.5 family (stable)
    "gemini-2.5-flash": "Gemini 2.5 Flash (Best price-performance)",
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite (Cost-effective)",
    "gemini-2.5-pro": "Gemini 2.5 Pro (State-of-the-art reasoning)",
    # Deprecated
    "gemini-3-pro": "Gemini 3 Pro Preview (Deprecated - shut down March 9, 2026)",
    "gemini-2.0-flash": "Gemini 2.0 Flash (Legacy - deprecated March 2026)",
}

GEMINI_PRICING: dict[str, tuple[float, float]] = {
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3-flash": (0.50, 3.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-3-pro": (2.00, 12.00),  # Deprecated March 9, 2026
    "gemini-2.0-flash": (0.10, 0.40),
}


def get_default_model(provider: str) -> str:
    """Get the default model for a provider.

    Args:
        provider: LLM provider name.

    Returns:
        Default model identifier.
    """
    return DEFAULT_MODELS.get(provider, DEFAULT_MODELS["anthropic"])


def get_model_pricing(provider: str, model: str) -> tuple[float, float]:
    """Get pricing for a specific model.

    Args:
        provider: LLM provider name.
        model: Model identifier.

    Returns:
        Tuple of (input_cost_per_million, output_cost_per_million) in USD.
    """
    pricing_maps = {
        "anthropic": ANTHROPIC_PRICING,
        "openai": OPENAI_PRICING,
        "google": GEMINI_PRICING,
    }

    pricing_map = pricing_maps.get(provider, {})

    # Return model-specific pricing or provider default
    if model in pricing_map:
        return pricing_map[model]

    # Default pricing by provider
    defaults = {
        "anthropic": (5.0, 25.0),   # Opus 4.6
        "openai": (2.0, 16.0),      # GPT-5.4
        "google": (0.15, 0.60),     # Gemini 2.5 Flash
    }
    return defaults.get(provider, (5.0, 25.0))


def get_batch_pricing(model: str) -> tuple[float, float]:
    """Get batch API pricing for a model.

    Currently only Anthropic supports batch API.

    Args:
        model: Model identifier.

    Returns:
        Tuple of (input_cost_per_million, output_cost_per_million) in USD.
    """
    return ANTHROPIC_BATCH_PRICING.get(model, (2.50, 12.50))
