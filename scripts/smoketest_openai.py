#!/usr/bin/env python
"""Smoketest for OpenAI integration.

This script tests the OpenAI client integration for LITRIS.
For API mode, set OPENAI_API_KEY environment variable.
For CLI mode (Codex), authenticate with: codex login
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_openai_cli_mode():
    """Test OpenAI CLI mode (requires Codex CLI)."""
    print("=" * 60)
    print("Testing OpenAI CLI Mode (Codex)")
    print("=" * 60)

    try:
        from src.analysis.openai_client import OpenAILLMClient

        # Note: Codex CLI with ChatGPT auth only supports gpt-5.2
        client = OpenAILLMClient(
            mode="cli",
            model="gpt-5.2",  # Default for ChatGPT Plus/Pro users
        )
        print(f"Provider: {client.provider}")
        print(f"Model: {client.model}")
        print(f"Codex path: {client._codex_path}")

        # Simple extraction test
        print("\nRunning CLI extraction test...")
        result = client.extract(
            paper_id="test_cli_001",
            title="Simple Test Paper",
            authors="Test Author",
            year=2024,
            item_type="journalArticle",
            text="""
            This paper examines the impact of AI on research workflows.
            Our main finding is that AI tools improve productivity by 30%.
            We used a mixed-methods approach combining surveys and interviews.
            """,
        )

        print(f"\nResult:")
        print(f"  Success: {result.success}")
        print(f"  Duration: {result.duration_seconds:.2f}s")

        if result.success and result.extraction:
            thesis = result.extraction.thesis_statement or "N/A"
            print(f"  Thesis: {thesis[:80]}...")
            print(f"  Confidence: {result.extraction.extraction_confidence}")
        elif result.error:
            error_preview = result.error[:150] if len(result.error) > 150 else result.error
            print(f"  Error: {error_preview}...")

        return result.success

    except ValueError as e:
        if "Codex CLI not found" in str(e):
            print("Codex CLI not installed - skipping CLI mode test")
            print("To install: npm i -g @openai/codex")
            print("Then authenticate: codex login")
            return None
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        return False


def test_openai_api_mode():
    """Test OpenAI API mode (requires OPENAI_API_KEY)."""
    print("\n" + "=" * 60)
    print("Testing OpenAI API Mode")
    print("=" * 60)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set - skipping API mode test")
        print("To test: set OPENAI_API_KEY=sk-...")
        return None

    try:
        from src.analysis.openai_client import OpenAILLMClient

        client = OpenAILLMClient(
            mode="api",
            model="gpt-4o-mini",  # Use mini for cost-effective testing
        )
        print(f"Provider: {client.provider}")
        print(f"Model: {client.model}")

        # Simple extraction
        print("\nRunning extraction test...")
        result = client.extract(
            paper_id="test_api_001",
            title="Machine Learning Overview",
            authors="Jane Smith",
            year=2024,
            item_type="journalArticle",
            text="""
            This paper provides an overview of machine learning techniques.
            We discuss supervised learning methods including classification
            and regression, as well as unsupervised learning approaches
            like clustering and dimensionality reduction.
            Our key finding is that ensemble methods often outperform
            single models in practice.
            """,
        )

        print(f"\nResult:")
        print(f"  Success: {result.success}")
        print(f"  Duration: {result.duration_seconds:.2f}s")

        if result.success and result.extraction:
            print(f"  Thesis: {result.extraction.thesis_statement[:80]}...")
            print(f"  Confidence: {result.extraction.extraction_confidence}")
        elif result.error:
            print(f"  Error: {result.error[:100]}...")

        return result.success

    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        return False


def test_openai_models():
    """Test that OpenAI model listing works."""
    print("\n" + "=" * 60)
    print("Available OpenAI Models")
    print("=" * 60)

    from src.analysis.openai_client import OpenAILLMClient

    models = OpenAILLMClient.list_models()
    for model_id, description in models.items():
        print(f"  {model_id}: {description}")

    return True


def test_cost_estimation():
    """Test cost estimation for OpenAI models."""
    print("\n" + "=" * 60)
    print("Cost Estimation Test")
    print("=" * 60)

    from src.analysis.openai_client import OpenAILLMClient

    # Test cost estimation for different models
    test_text_length = 10000  # ~10k chars typical paper

    for model in ["gpt-5.2", "gpt-4o-mini", "gpt-4o"]:
        client = OpenAILLMClient.__new__(OpenAILLMClient)
        client.model = model
        cost = client.estimate_cost(test_text_length)
        print(f"  {model}: ${cost:.4f} per paper (~10k chars)")

    return True


def test_factory():
    """Test the LLM factory with OpenAI provider."""
    print("\n" + "=" * 60)
    print("LLM Factory Test")
    print("=" * 60)

    from src.analysis.llm_factory import create_llm_client, get_available_providers

    providers = get_available_providers()
    print(f"Available providers: {providers}")

    # Test creating OpenAI client (will fail without API key, but shows it works)
    try:
        client = create_llm_client(
            provider="openai",
            mode="api",
            model="gpt-5.2",
        )
        print(f"Created OpenAI client: {client.provider}, {client.model}")
        return True
    except ValueError as e:
        if "API key required" in str(e):
            print("Factory works - API key needed for actual calls")
            return True
        raise


def main():
    """Run smoketests."""
    print("LITRIS OpenAI Integration Smoketest")
    print("=" * 60)
    print()

    results = {}

    # Test model listing
    results["models"] = test_openai_models()

    # Test cost estimation
    results["costs"] = test_cost_estimation()

    # Test factory
    results["factory"] = test_factory()

    # Test CLI mode (optional - requires Codex CLI)
    results["cli"] = test_openai_cli_mode()

    # Test API mode (optional - requires key)
    results["api"] = test_openai_api_mode()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for test, result in results.items():
        if result is None:
            status = "SKIP"
        elif result:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"  {test}: {status}")

    # Check if any required tests failed
    required_tests = ["models", "costs", "factory"]
    all_passed = all(results.get(t, False) for t in required_tests)

    if all_passed:
        print("\nOpenAI integration is ready!")
        if results.get("cli") is None:
            print("Install Codex CLI for subscription-based usage: npm i -g @openai/codex")
        if results.get("api") is None:
            print("Set OPENAI_API_KEY to enable API mode extraction.")
    else:
        print("\nSome tests failed.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
