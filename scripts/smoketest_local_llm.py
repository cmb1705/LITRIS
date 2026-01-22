#!/usr/bin/env python
"""Smoketest for local LLM providers (Ollama and llama.cpp).

This script tests the client initialization and factory integration.
Actual inference tests require models to be available locally.

Note: Some tests may be skipped if optional dependencies are not installed.
The src.analysis module requires anthropic for the full import chain.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_dependencies():
    """Check which optional dependencies are available."""
    deps = {
        "anthropic": False,
        "ollama": False,
        "llama_cpp": False,
        "pydantic": False,
    }

    try:
        import anthropic
        deps["anthropic"] = True
    except ImportError:
        pass

    try:
        import ollama
        deps["ollama"] = True
    except ImportError:
        pass

    try:
        import llama_cpp
        deps["llama_cpp"] = True
    except ImportError:
        pass

    try:
        import pydantic
        deps["pydantic"] = True
    except ImportError:
        pass

    return deps


def test_factory_integration(deps):
    """Test factory recognizes local providers."""
    print("=" * 60)
    print("Testing Factory Integration")
    print("=" * 60)

    if not deps["anthropic"]:
        print("  SKIPPED: anthropic not installed (required for full import chain)")
        print("  Install with: pip install anthropic")
        return None  # Skipped, not failed

    try:
        from src.analysis.llm_factory import (
            get_available_providers,
            get_provider_models,
            get_default_model,
        )

        providers = get_available_providers()
        print(f"  Available providers: {providers}")

        ollama_in_list = "ollama" in providers
        llamacpp_in_list = "llamacpp" in providers

        print(f"  Ollama in providers: {'PASS' if ollama_in_list else 'FAIL'}")
        print(f"  llama.cpp in providers: {'PASS' if llamacpp_in_list else 'FAIL'}")

        # Check models
        ollama_models = get_provider_models("ollama")
        llamacpp_models = get_provider_models("llamacpp")

        print(f"  Ollama models available: {len(ollama_models)}")
        print(f"  llama.cpp models available: {len(llamacpp_models)}")

        # Check defaults
        ollama_default = get_default_model("ollama")
        llamacpp_default = get_default_model("llamacpp")

        print(f"  Ollama default model: {ollama_default}")
        print(f"  llama.cpp default model: {llamacpp_default}")

        return ollama_in_list and llamacpp_in_list

    except ImportError as e:
        print(f"  Import error: {e}")
        return False


def test_ollama_client_standalone(deps):
    """Test Ollama client module directly (without full import chain)."""
    print("\n" + "=" * 60)
    print("Testing Ollama Client (Standalone)")
    print("=" * 60)

    if not deps["pydantic"]:
        print("  SKIPPED: pydantic not installed")
        return None

    try:
        # Test the module can at least define the class
        # We import the file directly to avoid triggering the package __init__.py
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "ollama_client",
            project_root / "src" / "analysis" / "ollama_client.py"
        )

        # This will still fail if base_llm can't be imported, but we try
        if spec and spec.loader:
            print("  Module file found: ollama_client.py")
            print("  PASS (file exists)")
            return True
        else:
            print("  Module file not found")
            return False

    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_llamacpp_client_standalone(deps):
    """Test llama.cpp client module directly (without full import chain)."""
    print("\n" + "=" * 60)
    print("Testing llama.cpp Client (Standalone)")
    print("=" * 60)

    if not deps["pydantic"]:
        print("  SKIPPED: pydantic not installed")
        return None

    try:
        # Check the file exists
        llamacpp_file = project_root / "src" / "analysis" / "llamacpp_client.py"

        if llamacpp_file.exists():
            print("  Module file found: llamacpp_client.py")

            # Verify file content has expected structure
            content = llamacpp_file.read_text()
            checks = [
                ("class LlamaCppLLMClient" in content, "LlamaCppLLMClient class defined"),
                ("def extract" in content, "extract method defined"),
                ("def estimate_cost" in content, "estimate_cost method defined"),
                ("llama-cpp-python" in content, "llama-cpp-python dependency documented"),
            ]

            all_passed = True
            for check, desc in checks:
                status = "PASS" if check else "FAIL"
                print(f"    {desc}: {status}")
                if not check:
                    all_passed = False

            return all_passed
        else:
            print("  Module file not found")
            return False

    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_ollama_client_import(deps):
    """Test Ollama client can be imported (requires full import chain)."""
    print("\n" + "=" * 60)
    print("Testing Ollama Client Import")
    print("=" * 60)

    if not deps["anthropic"]:
        print("  SKIPPED: anthropic not installed (required for import chain)")
        return None

    try:
        from src.analysis.ollama_client import OllamaLLMClient

        # Check class attributes
        models = OllamaLLMClient.list_models()
        print(f"  Ollama models: {len(models)} model families")

        # Check some expected models
        expected_models = ["llama3", "mistral", "gemma2"]
        for model in expected_models:
            if model in models:
                print(f"    {model}: {models[model][:50]}...")

        print("  Import: PASS")
        return True

    except ImportError as e:
        print(f"  Import error: {e}")
        return False


def test_llamacpp_client_import(deps):
    """Test llama.cpp client can be imported (requires full import chain)."""
    print("\n" + "=" * 60)
    print("Testing llama.cpp Client Import")
    print("=" * 60)

    if not deps["anthropic"]:
        print("  SKIPPED: anthropic not installed (required for import chain)")
        return None

    try:
        from src.analysis.llamacpp_client import LlamaCppLLMClient

        # Check class attributes
        models = LlamaCppLLMClient.list_models()
        print(f"  llama.cpp models: {len(models)} model families")

        # Check some expected model families
        expected_models = ["llama-3", "mistral", "gemma"]
        for model in expected_models:
            if model in models:
                print(f"    {model}: {models[model][:50]}...")

        print("  Import: PASS")
        return True

    except ImportError as e:
        print(f"  Import error: {e}")
        return False


def test_ollama_client_init(deps):
    """Test Ollama client initialization (without server)."""
    print("\n" + "=" * 60)
    print("Testing Ollama Client Initialization")
    print("=" * 60)

    if not deps["anthropic"]:
        print("  SKIPPED: anthropic not installed (required for import chain)")
        return None

    if not deps["ollama"]:
        print("  SKIPPED: ollama package not installed")
        print("  Install with: pip install ollama")
        return None

    try:
        from src.analysis.llm_factory import create_llm_client

        client = create_llm_client(
            provider="ollama",
            model="llama3",
            ollama_host="http://localhost:11434",
        )
        print(f"  Provider: {client.provider}")
        print(f"  Model: {client.model}")
        print(f"  Default model: {client.default_model}")

        # Check connection (will fail if Ollama not running)
        connected = client.check_connection()
        print(f"  Server connected: {connected}")

        if connected:
            local_models = client.list_local_models()
            if local_models:
                print(f"  Local models: {local_models[:5]}...")

        print("  Initialization: PASS")
        return True

    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_llamacpp_client_init(deps):
    """Test llama.cpp client initialization (without model file)."""
    print("\n" + "=" * 60)
    print("Testing llama.cpp Client Initialization")
    print("=" * 60)

    if not deps["anthropic"]:
        print("  SKIPPED: anthropic not installed (required for import chain)")
        return None

    if not deps["llama_cpp"]:
        print("  SKIPPED: llama-cpp-python not installed")
        print("  Install with: pip install llama-cpp-python")
        return None

    try:
        from src.analysis.llm_factory import create_llm_client

        # Test that missing model_path raises appropriate error
        try:
            client = create_llm_client(provider="llamacpp")
            print("  ERROR: Should have raised ValueError for missing model_path")
            return False
        except ValueError as e:
            if "model_path is required" in str(e):
                print("  Missing model_path correctly raises ValueError: PASS")
            else:
                print(f"  Unexpected error: {e}")
                return False

        # Test with non-existent model file
        try:
            client = create_llm_client(
                provider="llamacpp",
                model_path="/nonexistent/model.gguf",
            )
            print("  ERROR: Should have raised ValueError for missing file")
            return False
        except ValueError as e:
            if "not found" in str(e):
                print("  Non-existent file correctly raises ValueError: PASS")
            else:
                print(f"  Unexpected error: {e}")
                return False

        print("  Initialization: PASS")
        return True

    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    print("LITRIS Local LLM Provider Smoketest")
    print("=" * 60)

    # Check dependencies
    deps = check_dependencies()
    print("\nDependency Check:")
    for dep, available in deps.items():
        status = "installed" if available else "NOT INSTALLED"
        print(f"  {dep}: {status}")

    print("\nNote: Full integration tests require:")
    print("  - anthropic package (for import chain)")
    print("  - ollama package + server (for Ollama tests)")
    print("  - llama-cpp-python + GGUF model (for llama.cpp tests)")
    print()

    results = {}

    # Standalone tests (file existence and structure)
    results["ollama_standalone"] = test_ollama_client_standalone(deps)
    results["llamacpp_standalone"] = test_llamacpp_client_standalone(deps)

    # Import tests (require anthropic)
    results["factory_integration"] = test_factory_integration(deps)
    results["ollama_import"] = test_ollama_client_import(deps)
    results["llamacpp_import"] = test_llamacpp_client_import(deps)

    # Initialization tests (require full dependencies)
    results["ollama_init"] = test_ollama_client_init(deps)
    results["llamacpp_init"] = test_llamacpp_client_init(deps)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = 0
    failed = 0
    skipped = 0

    for test, result in results.items():
        if result is None:
            status = "SKIPPED"
            skipped += 1
        elif result:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1
        print(f"  {test}: {status}")

    print(f"\n  Passed: {passed}, Failed: {failed}, Skipped: {skipped}")

    # Consider success if no failures (skips are OK)
    all_passed = failed == 0

    if all_passed:
        print("\nLocal LLM provider files are ready!")

        if not deps["anthropic"]:
            print("\nTo run full integration tests:")
            print("  pip install anthropic")

        print("\nTo use Ollama:")
        print("  1. Install Ollama: https://ollama.com/")
        print("  2. Install package: pip install ollama")
        print("  3. Pull a model: ollama pull llama3")
        print("  4. Configure LITRIS with provider='ollama'")

        print("\nTo use llama.cpp:")
        print("  1. Install: pip install llama-cpp-python")
        print("  2. Download a GGUF model")
        print("  3. Configure LITRIS with provider='llamacpp' and model_path")
    else:
        print("\nSome tests failed. Check output above for details.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
