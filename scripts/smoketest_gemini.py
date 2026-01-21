#!/usr/bin/env python
"""Smoketest for Google Gemini integration.

This script tests the Gemini client integration for LITRIS.
Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_gemini_models():
    """Test that Gemini model listing works."""
    print("=" * 60)
    print("Available Gemini Models")
    print("=" * 60)

    from src.analysis.gemini_client import GeminiLLMClient

    models = GeminiLLMClient.list_models()
    for model_id, description in models.items():
        print(f"  {model_id}: {description}")

    return True


def test_cost_estimation():
    """Test cost estimation for Gemini models."""
    print("\n" + "=" * 60)
    print("Cost Estimation Test")
    print("=" * 60)

    from src.analysis.gemini_client import GeminiLLMClient

    # Test cost estimation for different models
    test_text_length = 10000  # ~10k chars typical paper

    for model in ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"]:
        client = GeminiLLMClient.__new__(GeminiLLMClient)
        client.model = model
        cost = client.estimate_cost(test_text_length)
        print(f"  {model}: ${cost:.4f} per paper (~10k chars)")

    return True


def test_factory():
    """Test the LLM factory with Google provider."""
    print("\n" + "=" * 60)
    print("LLM Factory Test")
    print("=" * 60)

    from src.analysis.llm_factory import create_llm_client, get_available_providers

    providers = get_available_providers()
    print(f"Available providers: {providers}")

    if "google" not in providers:
        print("ERROR: 'google' not in available providers!")
        return False

    # Test creating Google client (will fail without API key)
    try:
        client = create_llm_client(
            provider="google",
            model="gemini-2.5-flash",
        )
        print(f"Created Gemini client: {client.provider}, {client.model}")
        return True
    except ValueError as e:
        if "API key required" in str(e):
            print("Factory works - API key needed for actual calls")
            return True
        raise


def test_gemini_api():
    """Test Gemini API mode (requires GOOGLE_API_KEY)."""
    print("\n" + "=" * 60)
    print("Testing Gemini API Mode")
    print("=" * 60)

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY/GEMINI_API_KEY not set - skipping API test")
        print("To test: set GOOGLE_API_KEY=...")
        return None

    try:
        from src.analysis.gemini_client import GeminiLLMClient

        client = GeminiLLMClient(
            mode="api",
            model="gemini-2.5-flash-lite",  # Use cheapest model for testing
        )
        print(f"Provider: {client.provider}")
        print(f"Model: {client.model}")

        # Simple extraction
        print("\nRunning extraction test...")
        result = client.extract(
            paper_id="test_gemini_001",
            title="AI Research Overview",
            authors="Jane Smith",
            year=2024,
            item_type="journalArticle",
            text="""
            This paper provides an overview of AI research trends.
            We examine deep learning, reinforcement learning, and
            natural language processing advances from 2020-2024.
            Our key finding is that transformer architectures have
            become dominant across all three areas.
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

    except ImportError as e:
        print(f"Import error: {e}")
        print("Install with: pip install google-genai")
        return False
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        return False


def main():
    """Run smoketests."""
    print("LITRIS Google Gemini Integration Smoketest")
    print("=" * 60)
    print()

    results = {}

    # Test model listing
    results["models"] = test_gemini_models()

    # Test cost estimation
    results["costs"] = test_cost_estimation()

    # Test factory
    results["factory"] = test_factory()

    # Test API mode (optional - requires key)
    results["api"] = test_gemini_api()

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
        print("\nGemini integration is ready!")
        if results.get("api") is None:
            print("Set GOOGLE_API_KEY to enable API mode extraction.")
    else:
        print("\nSome tests failed.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
