"""Tests for multi-provider LLM support."""

import pytest
from unittest.mock import MagicMock, patch

from src.analysis.base_llm import BaseLLMClient, ExtractionMode
from src.analysis.llm_factory import (
    create_llm_client,
    get_available_providers,
    get_default_model,
    get_provider_models,
)


class TestBaseLLMClient:
    """Tests for BaseLLMClient abstract class."""

    def test_base_client_is_abstract(self):
        """BaseLLMClient cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLLMClient()

    def test_get_available_providers(self):
        """Should return list of providers."""
        providers = BaseLLMClient.get_available_providers()
        assert "anthropic" in providers
        assert "openai" in providers
        assert "google" in providers
        assert "ollama" in providers
        assert "llamacpp" in providers


class TestLLMFactory:
    """Tests for LLM client factory."""

    def test_get_available_providers(self):
        """Should return list of available providers."""
        providers = get_available_providers()
        assert isinstance(providers, list)
        assert "anthropic" in providers
        assert "openai" in providers
        assert "google" in providers
        assert "ollama" in providers
        assert "llamacpp" in providers

    def test_get_default_model_anthropic(self):
        """Should return Claude default model."""
        model = get_default_model("anthropic")
        assert "claude" in model.lower()

    def test_get_default_model_openai(self):
        """Should return GPT default model."""
        model = get_default_model("openai")
        assert "gpt" in model.lower()

    def test_get_default_model_google(self):
        """Should return Gemini default model."""
        model = get_default_model("google")
        assert "gemini" in model.lower()

    def test_get_default_model_ollama(self):
        """Should return llama3 default model."""
        model = get_default_model("ollama")
        assert "llama" in model.lower()

    def test_get_default_model_llamacpp(self):
        """Should return llama-3 default model."""
        model = get_default_model("llamacpp")
        assert "llama" in model.lower()

    def test_get_provider_models_anthropic(self):
        """Should return Anthropic models."""
        models = get_provider_models("anthropic")
        assert isinstance(models, dict)
        assert len(models) > 0
        # Should have Claude models
        assert any("claude" in k.lower() for k in models.keys())

    def test_get_provider_models_openai(self):
        """Should return OpenAI models."""
        models = get_provider_models("openai")
        assert isinstance(models, dict)
        assert len(models) > 0
        # Should have GPT models
        assert any("gpt" in k.lower() for k in models.keys())

    def test_get_provider_models_google(self):
        """Should return Google models."""
        models = get_provider_models("google")
        assert isinstance(models, dict)
        assert len(models) > 0
        # Should have Gemini models
        assert any("gemini" in k.lower() for k in models.keys())

    def test_get_provider_models_ollama(self):
        """Should return Ollama models."""
        models = get_provider_models("ollama")
        assert isinstance(models, dict)
        assert len(models) > 0
        # Should have llama models
        assert any("llama" in k.lower() for k in models.keys())

    def test_get_provider_models_llamacpp(self):
        """Should return llama.cpp model families."""
        models = get_provider_models("llamacpp")
        assert isinstance(models, dict)
        assert len(models) > 0
        # Should have llama model family
        assert any("llama" in k.lower() for k in models.keys())

    def test_create_invalid_provider(self):
        """Should raise error for invalid provider."""
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            create_llm_client(provider="invalid_provider")


class TestAnthropicClient:
    """Tests for AnthropicLLMClient."""

    def test_provider_property(self):
        """Should return 'anthropic' as provider."""
        from src.analysis.anthropic_client import AnthropicLLMClient

        # Create without actually initializing API (will fail without key)
        client = AnthropicLLMClient.__new__(AnthropicLLMClient)
        assert client.provider == "anthropic"

    def test_default_model(self):
        """Should have Claude default model."""
        from src.analysis.anthropic_client import AnthropicLLMClient

        client = AnthropicLLMClient.__new__(AnthropicLLMClient)
        assert "claude" in client.default_model.lower()

    def test_supported_modes(self):
        """Should support api and cli modes."""
        from src.analysis.anthropic_client import AnthropicLLMClient

        client = AnthropicLLMClient.__new__(AnthropicLLMClient)
        modes = client.supported_modes
        assert "api" in modes
        assert "cli" in modes

    def test_list_models(self):
        """Should list available Claude models."""
        from src.analysis.anthropic_client import AnthropicLLMClient

        models = AnthropicLLMClient.list_models()
        assert isinstance(models, dict)
        assert "claude-opus-4-5-20251101" in models

    def test_model_pricing(self):
        """Should have pricing for all listed models."""
        from src.analysis.anthropic_client import AnthropicLLMClient

        for model in AnthropicLLMClient.MODELS.keys():
            assert model in AnthropicLLMClient.MODEL_PRICING


class TestAnthropicClientEstimateCost:
    """Tests for Anthropic cost estimation."""

    def test_estimate_cost_opus(self):
        """Should estimate cost for Claude Opus."""
        from src.analysis.anthropic_client import AnthropicLLMClient

        client = AnthropicLLMClient.__new__(AnthropicLLMClient)
        client.model = "claude-opus-4-5-20251101"

        cost = client.estimate_cost(10000)  # 10k chars
        assert cost > 0
        assert isinstance(cost, float)

    def test_estimate_cost_sonnet_cheaper_than_opus(self):
        """Sonnet should be cheaper than Opus."""
        from src.analysis.anthropic_client import AnthropicLLMClient

        client_opus = AnthropicLLMClient.__new__(AnthropicLLMClient)
        client_opus.model = "claude-opus-4-5-20251101"

        client_sonnet = AnthropicLLMClient.__new__(AnthropicLLMClient)
        client_sonnet.model = "claude-sonnet-4-20250514"

        cost_opus = client_opus.estimate_cost(10000)
        cost_sonnet = client_sonnet.estimate_cost(10000)

        assert cost_sonnet < cost_opus


class TestOpenAIClient:
    """Tests for OpenAILLMClient."""

    def test_provider_property(self):
        """Should return 'openai' as provider."""
        from src.analysis.openai_client import OpenAILLMClient

        client = OpenAILLMClient.__new__(OpenAILLMClient)
        assert client.provider == "openai"

    def test_default_model(self):
        """Should have GPT-5.2 default model."""
        from src.analysis.openai_client import OpenAILLMClient

        client = OpenAILLMClient.__new__(OpenAILLMClient)
        assert "gpt-5.2" in client.default_model.lower()

    def test_supported_modes(self):
        """Should support api and cli modes."""
        from src.analysis.openai_client import OpenAILLMClient

        client = OpenAILLMClient.__new__(OpenAILLMClient)
        modes = client.supported_modes
        assert "api" in modes
        assert "cli" in modes

    def test_list_models(self):
        """Should list available GPT models."""
        from src.analysis.openai_client import OpenAILLMClient

        models = OpenAILLMClient.list_models()
        assert isinstance(models, dict)
        assert "gpt-5.2" in models
        assert "gpt-5.2-instant" in models
        assert "gpt-5.2-codex" in models

    def test_model_pricing(self):
        """Should have pricing for all listed models."""
        from src.analysis.openai_client import OpenAILLMClient

        for model in OpenAILLMClient.MODELS.keys():
            assert model in OpenAILLMClient.MODEL_PRICING


class TestOpenAIClientEstimateCost:
    """Tests for OpenAI cost estimation."""

    def test_estimate_cost_gpt_5_2(self):
        """Should estimate cost for GPT-5.2."""
        from src.analysis.openai_client import OpenAILLMClient

        # Create client without API
        client = OpenAILLMClient.__new__(OpenAILLMClient)
        client.model = "gpt-5.2"

        cost = client.estimate_cost(10000)  # 10k chars
        assert cost > 0
        assert isinstance(cost, float)

    def test_estimate_cost_gpt_4o_mini_cheaper(self):
        """GPT-4o-mini should be cheaper than GPT-5.2."""
        from src.analysis.openai_client import OpenAILLMClient

        client1 = OpenAILLMClient.__new__(OpenAILLMClient)
        client1.model = "gpt-5.2"

        client2 = OpenAILLMClient.__new__(OpenAILLMClient)
        client2.model = "gpt-4o-mini"

        cost_5_2 = client1.estimate_cost(10000)
        cost_mini = client2.estimate_cost(10000)

        assert cost_mini < cost_5_2


class TestGeminiClient:
    """Tests for GeminiLLMClient."""

    def test_provider_property(self):
        """Should return 'google' as provider."""
        from src.analysis.gemini_client import GeminiLLMClient

        client = GeminiLLMClient.__new__(GeminiLLMClient)
        assert client.provider == "google"

    def test_default_model(self):
        """Should have Gemini default model."""
        from src.analysis.gemini_client import GeminiLLMClient

        client = GeminiLLMClient.__new__(GeminiLLMClient)
        assert "gemini" in client.default_model.lower()

    def test_supported_modes(self):
        """Should support api mode only."""
        from src.analysis.gemini_client import GeminiLLMClient

        client = GeminiLLMClient.__new__(GeminiLLMClient)
        modes = client.supported_modes
        assert "api" in modes
        # Gemini does not have a CLI mode
        assert "cli" not in modes

    def test_list_models(self):
        """Should list available Gemini models."""
        from src.analysis.gemini_client import GeminiLLMClient

        models = GeminiLLMClient.list_models()
        assert isinstance(models, dict)
        assert "gemini-2.5-flash" in models
        assert "gemini-2.5-pro" in models

    def test_model_pricing(self):
        """Should have pricing for all listed models."""
        from src.analysis.gemini_client import GeminiLLMClient

        for model in GeminiLLMClient.MODELS.keys():
            assert model in GeminiLLMClient.MODEL_PRICING


class TestGeminiClientEstimateCost:
    """Tests for Google Gemini cost estimation."""

    def test_estimate_cost_gemini_flash(self):
        """Should estimate cost for Gemini 2.5 Flash."""
        from src.analysis.gemini_client import GeminiLLMClient

        client = GeminiLLMClient.__new__(GeminiLLMClient)
        client.model = "gemini-2.5-flash"

        cost = client.estimate_cost(10000)  # 10k chars
        assert cost > 0
        assert isinstance(cost, float)

    def test_estimate_cost_gemini_flash_cheaper_than_pro(self):
        """Gemini Flash should be cheaper than Gemini Pro."""
        from src.analysis.gemini_client import GeminiLLMClient

        client_flash = GeminiLLMClient.__new__(GeminiLLMClient)
        client_flash.model = "gemini-2.5-flash"

        client_pro = GeminiLLMClient.__new__(GeminiLLMClient)
        client_pro.model = "gemini-2.5-pro"

        cost_flash = client_flash.estimate_cost(10000)
        cost_pro = client_pro.estimate_cost(10000)

        assert cost_flash < cost_pro


class TestOllamaClient:
    """Tests for OllamaLLMClient."""

    def test_provider_property(self):
        """Should return 'ollama' as provider."""
        from src.analysis.ollama_client import OllamaLLMClient

        client = OllamaLLMClient.__new__(OllamaLLMClient)
        assert client.provider == "ollama"

    def test_default_model(self):
        """Should have llama3 default model."""
        from src.analysis.ollama_client import OllamaLLMClient

        client = OllamaLLMClient.__new__(OllamaLLMClient)
        assert "llama" in client.default_model.lower()

    def test_supported_modes(self):
        """Should support api mode only."""
        from src.analysis.ollama_client import OllamaLLMClient

        client = OllamaLLMClient.__new__(OllamaLLMClient)
        modes = client.supported_modes
        assert "api" in modes
        # Ollama does not have CLI mode
        assert "cli" not in modes

    def test_list_models(self):
        """Should list available Ollama models."""
        from src.analysis.ollama_client import OllamaLLMClient

        models = OllamaLLMClient.list_models()
        assert isinstance(models, dict)
        assert "llama3" in models
        assert "mistral" in models
        assert "gemma2" in models

    def test_estimate_cost_always_zero(self):
        """Should return zero cost for local inference."""
        from src.analysis.ollama_client import OllamaLLMClient

        client = OllamaLLMClient.__new__(OllamaLLMClient)

        cost = client.estimate_cost(10000)  # 10k chars
        assert cost == 0.0


class TestLlamaCppClient:
    """Tests for LlamaCppLLMClient."""

    def test_provider_property(self):
        """Should return 'llamacpp' as provider."""
        from src.analysis.llamacpp_client import LlamaCppLLMClient

        client = LlamaCppLLMClient.__new__(LlamaCppLLMClient)
        assert client.provider == "llamacpp"

    def test_default_model(self):
        """Should have llama-3 default model."""
        from src.analysis.llamacpp_client import LlamaCppLLMClient

        client = LlamaCppLLMClient.__new__(LlamaCppLLMClient)
        assert "llama" in client.default_model.lower()

    def test_supported_modes(self):
        """Should support api mode only."""
        from src.analysis.llamacpp_client import LlamaCppLLMClient

        client = LlamaCppLLMClient.__new__(LlamaCppLLMClient)
        modes = client.supported_modes
        assert "api" in modes
        # llama.cpp does not have CLI mode
        assert "cli" not in modes

    def test_list_models(self):
        """Should list available llama.cpp model families."""
        from src.analysis.llamacpp_client import LlamaCppLLMClient

        models = LlamaCppLLMClient.list_models()
        assert isinstance(models, dict)
        assert "llama-3" in models
        assert "mistral" in models
        assert "gemma" in models

    def test_estimate_cost_always_zero(self):
        """Should return zero cost for local inference."""
        from src.analysis.llamacpp_client import LlamaCppLLMClient

        client = LlamaCppLLMClient.__new__(LlamaCppLLMClient)

        cost = client.estimate_cost(10000)  # 10k chars
        assert cost == 0.0


class TestLocalLLMCostComparison:
    """Tests comparing local vs cloud LLM costs."""

    def test_local_cheaper_than_anthropic(self):
        """Local inference should be cheaper (free) than Anthropic."""
        from src.analysis.anthropic_client import AnthropicLLMClient
        from src.analysis.ollama_client import OllamaLLMClient

        anthropic_client = AnthropicLLMClient.__new__(AnthropicLLMClient)
        anthropic_client.model = "claude-opus-4-5-20251101"

        ollama_client = OllamaLLMClient.__new__(OllamaLLMClient)

        anthropic_cost = anthropic_client.estimate_cost(10000)
        ollama_cost = ollama_client.estimate_cost(10000)

        assert ollama_cost == 0.0
        assert ollama_cost < anthropic_cost

    def test_local_cheaper_than_openai(self):
        """Local inference should be cheaper (free) than OpenAI."""
        from src.analysis.openai_client import OpenAILLMClient
        from src.analysis.llamacpp_client import LlamaCppLLMClient

        openai_client = OpenAILLMClient.__new__(OpenAILLMClient)
        openai_client.model = "gpt-5.2"

        llamacpp_client = LlamaCppLLMClient.__new__(LlamaCppLLMClient)

        openai_cost = openai_client.estimate_cost(10000)
        llamacpp_cost = llamacpp_client.estimate_cost(10000)

        assert llamacpp_cost == 0.0
        assert llamacpp_cost < openai_cost


class TestConfigExtraction:
    """Tests for extraction config with provider support."""

    def test_config_default_provider(self, sample_config_dict):
        """Default provider should be anthropic."""
        from src.config import ExtractionConfig

        config = ExtractionConfig()
        assert config.provider == "anthropic"

    def test_config_openai_provider(self, sample_config_dict):
        """Should accept openai provider."""
        from src.config import ExtractionConfig

        config = ExtractionConfig(provider="openai")
        assert config.provider == "openai"

    def test_config_google_provider(self, sample_config_dict):
        """Should accept google provider."""
        from src.config import ExtractionConfig

        config = ExtractionConfig(provider="google")
        assert config.provider == "google"

    def test_config_ollama_provider(self, sample_config_dict):
        """Should accept ollama provider."""
        from src.config import ExtractionConfig

        config = ExtractionConfig(provider="ollama")
        assert config.provider == "ollama"

    def test_config_llamacpp_provider(self, sample_config_dict):
        """Should accept llamacpp provider."""
        from src.config import ExtractionConfig

        config = ExtractionConfig(provider="llamacpp")
        assert config.provider == "llamacpp"

    def test_config_invalid_provider(self, sample_config_dict):
        """Should reject invalid provider."""
        from src.config import ExtractionConfig

        with pytest.raises(ValueError, match="provider must be one of"):
            ExtractionConfig(provider="invalid")

    def test_config_reasoning_effort(self, sample_config_dict):
        """Should accept valid reasoning effort."""
        from src.config import ExtractionConfig

        config = ExtractionConfig(provider="openai", reasoning_effort="high")
        assert config.reasoning_effort == "high"

    def test_config_invalid_reasoning_effort(self, sample_config_dict):
        """Should reject invalid reasoning effort."""
        from src.config import ExtractionConfig

        with pytest.raises(ValueError, match="reasoning_effort must be one of"):
            ExtractionConfig(reasoning_effort="super_high")

    def test_config_get_model_or_default_anthropic(self, sample_config_dict):
        """Should return Claude default when no model specified."""
        from src.config import ExtractionConfig

        config = ExtractionConfig(provider="anthropic", model=None)
        model = config.get_model_or_default()
        assert "claude" in model.lower()

    def test_config_get_model_or_default_openai(self, sample_config_dict):
        """Should return GPT default when no model specified."""
        from src.config import ExtractionConfig

        config = ExtractionConfig(provider="openai", model=None)
        model = config.get_model_or_default()
        assert "gpt" in model.lower()

    def test_config_get_model_or_default_google(self, sample_config_dict):
        """Should return Gemini default when no model specified."""
        from src.config import ExtractionConfig

        config = ExtractionConfig(provider="google", model=None)
        model = config.get_model_or_default()
        assert "gemini" in model.lower()

    def test_config_get_model_or_default_ollama(self, sample_config_dict):
        """Should return llama3 default when no model specified."""
        from src.config import ExtractionConfig

        config = ExtractionConfig(provider="ollama", model=None)
        model = config.get_model_or_default()
        assert "llama" in model.lower()

    def test_config_get_model_or_default_llamacpp(self, sample_config_dict):
        """Should return llama-3 default when no model specified."""
        from src.config import ExtractionConfig

        config = ExtractionConfig(provider="llamacpp", model=None)
        model = config.get_model_or_default()
        assert "llama" in model.lower()

    def test_config_get_model_explicit(self, sample_config_dict):
        """Should return explicit model when specified."""
        from src.config import ExtractionConfig

        config = ExtractionConfig(provider="openai", model="gpt-4o-mini")
        model = config.get_model_or_default()
        assert model == "gpt-4o-mini"


class TestModelOverrides:
    """Tests for per-item-type model selection."""

    def test_model_overrides_none_by_default(self):
        """Model overrides should be None by default."""
        from src.config import ExtractionConfig

        config = ExtractionConfig()
        assert config.model_overrides is None

    def test_model_overrides_journal_article(self):
        """Should use override for journal articles."""
        from src.config import ExtractionConfig, ModelOverrides

        overrides = ModelOverrides(journal_article="claude-sonnet-4-20250514")
        config = ExtractionConfig(model_overrides=overrides)

        model = config.get_model_or_default(item_type="journalArticle")
        assert model == "claude-sonnet-4-20250514"

    def test_model_overrides_book(self):
        """Should use override for books."""
        from src.config import ExtractionConfig, ModelOverrides

        overrides = ModelOverrides(book="claude-opus-4-5-20251101")
        config = ExtractionConfig(model_overrides=overrides)

        model = config.get_model_or_default(item_type="book")
        assert model == "claude-opus-4-5-20251101"

    def test_model_overrides_conference_paper(self):
        """Should use override for conference papers."""
        from src.config import ExtractionConfig, ModelOverrides

        overrides = ModelOverrides(conference_paper="gemini-2.5-flash")
        config = ExtractionConfig(model_overrides=overrides)

        model = config.get_model_or_default(item_type="conferencePaper")
        assert model == "gemini-2.5-flash"

    def test_model_overrides_fallback_to_explicit(self):
        """Should fall back to explicit model when no override."""
        from src.config import ExtractionConfig, ModelOverrides

        overrides = ModelOverrides(journal_article="claude-sonnet-4-20250514")
        config = ExtractionConfig(model="gpt-5.2", model_overrides=overrides)

        # No override for thesis, should use explicit model
        model = config.get_model_or_default(item_type="thesis")
        assert model == "gpt-5.2"

    def test_model_overrides_fallback_to_default(self):
        """Should fall back to provider default when no override or explicit."""
        from src.config import ExtractionConfig, ModelOverrides

        overrides = ModelOverrides(journal_article="claude-sonnet-4-20250514")
        config = ExtractionConfig(provider="google", model_overrides=overrides)

        # No override for report, should use Google default
        model = config.get_model_or_default(item_type="report")
        assert "gemini" in model.lower()

    def test_model_overrides_without_item_type(self):
        """Should return default model when item_type not provided."""
        from src.config import ExtractionConfig, ModelOverrides

        overrides = ModelOverrides(journal_article="claude-sonnet-4-20250514")
        config = ExtractionConfig(model="gpt-5.2", model_overrides=overrides)

        # No item_type, should use explicit model
        model = config.get_model_or_default()
        assert model == "gpt-5.2"

    def test_model_overrides_all_types(self):
        """Should handle all supported item types."""
        from src.config import ExtractionConfig, ModelOverrides

        overrides = ModelOverrides(
            journal_article="model-a",
            book="model-b",
            book_section="model-c",
            thesis="model-d",
            conference_paper="model-e",
            report="model-f",
            preprint="model-g",
        )
        config = ExtractionConfig(model_overrides=overrides)

        assert config.get_model_or_default(item_type="journalArticle") == "model-a"
        assert config.get_model_or_default(item_type="book") == "model-b"
        assert config.get_model_or_default(item_type="bookSection") == "model-c"
        assert config.get_model_or_default(item_type="thesis") == "model-d"
        assert config.get_model_or_default(item_type="conferencePaper") == "model-e"
        assert config.get_model_or_default(item_type="report") == "model-f"
        assert config.get_model_or_default(item_type="preprint") == "model-g"


class TestBackwardCompatibility:
    """Tests for backward compatibility with old API."""

    def test_llm_client_import(self):
        """LLMClient should still be importable from llm_client module."""
        from src.analysis.llm_client import LLMClient
        from src.analysis.anthropic_client import AnthropicLLMClient

        # LLMClient should be an alias for AnthropicLLMClient
        assert LLMClient is AnthropicLLMClient

    def test_extraction_mode_import(self):
        """ExtractionMode should be importable from llm_client."""
        from src.analysis.llm_client import ExtractionMode

        assert "api" in ExtractionMode.__args__
        assert "cli" in ExtractionMode.__args__

    def test_create_llm_client_import(self):
        """create_llm_client should be importable from llm_client."""
        from src.analysis.llm_client import create_llm_client

        assert callable(create_llm_client)
