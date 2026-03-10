"""Helpers for securely loading secrets (e.g., API keys)."""

import os

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

try:
    import keyring  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    keyring = None


def get_anthropic_api_key(
    service_name: str = "litris",
    key_name: str = "ANTHROPIC_API_KEY",
) -> str | None:
    """Return the Anthropic API key from env or OS keyring.

    Order:
    1) Environment variable ANTHROPIC_API_KEY
    2) keyring entry (service: service_name, username: key_name)
    """
    env_key = os.getenv("ANTHROPIC_API_KEY")
    if env_key:
        return env_key

    if keyring:
        try:
            return keyring.get_password(service_name, key_name)
        except Exception as exc:  # pragma: no cover - keyring backend issues
            logger.warning("Failed to read Anthropic key from keyring: %s", exc)

    return None
