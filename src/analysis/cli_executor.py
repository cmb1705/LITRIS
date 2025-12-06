"""CLI executor for Claude Code headless mode extraction."""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class CliExecutionError(Exception):
    """Error during CLI execution."""

    pass


class ExtractionTimeoutError(CliExecutionError):
    """Extraction timed out."""

    pass


class RateLimitError(CliExecutionError):
    """Rate limit hit during extraction."""

    def __init__(self, message: str, reset_time: str | None = None):
        super().__init__(message)
        self.reset_time = reset_time


class ParseError(CliExecutionError):
    """Failed to parse CLI response."""

    def __init__(self, message: str, raw_output: str):
        super().__init__(message)
        self.raw_output = raw_output


class ClaudeCliExecutor:
    """Execute Claude Code CLI commands for LLM extraction.

    Uses Claude Code CLI in headless mode with Max subscription
    for cost-free extraction (no API billing).
    """

    def __init__(
        self,
        timeout: int = 120,
        output_format: str = "json",
    ):
        """Initialize CLI executor.

        Args:
            timeout: Seconds before command timeout.
            output_format: CLI output format (json recommended).
        """
        self.timeout = timeout
        self.output_format = output_format
        self._cli_path: str | None = None

    def verify_authentication(self) -> bool:
        """Verify CLI is using subscription, not API billing.

        Returns:
            True if ready for subscription-based extraction.

        Raises:
            CliExecutionError: If API key is set or CLI not installed.
        """
        # Check if ANTHROPIC_API_KEY is set (would trigger API billing)
        if os.environ.get("ANTHROPIC_API_KEY"):
            raise CliExecutionError(
                "ANTHROPIC_API_KEY environment variable is set. "
                "This will use API billing instead of your Max subscription. "
                "Unset it with: $env:ANTHROPIC_API_KEY = $null (PowerShell) "
                "or: unset ANTHROPIC_API_KEY (bash)"
            )

        # Find claude CLI
        self._cli_path = shutil.which("claude")
        if not self._cli_path:
            raise CliExecutionError(
                "Claude Code CLI not found in PATH. "
                "Install with: npm install -g @anthropic-ai/claude-code"
            )

        # Verify CLI works
        try:
            result = subprocess.run(
                [self._cli_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise CliExecutionError(
                    f"Claude CLI returned error: {result.stderr}"
                )
            logger.debug(f"Claude CLI version: {result.stdout.strip()}")
        except subprocess.TimeoutExpired:
            raise CliExecutionError("Claude CLI version check timed out")
        except FileNotFoundError:
            raise CliExecutionError("Claude CLI executable not found")

        return True

    def extract(self, prompt: str, input_text: str) -> dict:
        """Execute extraction via CLI.

        Args:
            prompt: The extraction prompt/instructions.
            input_text: The paper text to extract from.

        Returns:
            Parsed JSON response from Claude.

        Raises:
            ExtractionTimeoutError: If extraction times out.
            RateLimitError: If rate limit is hit.
            ParseError: If response cannot be parsed.
            CliExecutionError: For other CLI errors.
        """
        if not self._cli_path:
            self.verify_authentication()

        # Write input to temp file to handle large texts
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(input_text)
            input_file = Path(f.name)

        try:
            # Build command
            # Use --print to get output only (no interactive mode)
            # Use --output-format json for structured output
            cmd = [
                self._cli_path,
                "--print",
                "--output-format", self.output_format,
                "-p", prompt,
            ]

            # Execute with input from file
            with open(input_file, "r", encoding="utf-8") as f:
                result = subprocess.run(
                    cmd,
                    stdin=f,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )

            # Check for rate limit
            if self._is_rate_limited(result.stdout, result.stderr):
                reset_time = self._extract_reset_time(result.stdout + result.stderr)
                raise RateLimitError(
                    "Rate limit hit during extraction",
                    reset_time=reset_time,
                )

            # Check for errors
            if result.returncode != 0:
                error_msg = result.stderr or result.stdout
                raise CliExecutionError(f"CLI returned error: {error_msg}")

            # Parse response
            return self._parse_response(result.stdout)

        except subprocess.TimeoutExpired:
            raise ExtractionTimeoutError(
                f"Extraction timed out after {self.timeout} seconds"
            )
        finally:
            # Clean up temp file
            input_file.unlink(missing_ok=True)

    def _is_rate_limited(self, stdout: str, stderr: str) -> bool:
        """Check if response indicates rate limiting.

        Args:
            stdout: Standard output from CLI.
            stderr: Standard error from CLI.

        Returns:
            True if rate limit was hit.
        """
        combined = (stdout + stderr).lower()
        rate_limit_indicators = [
            "rate limit",
            "usage limit",
            "try again later",
            "too many requests",
            "quota exceeded",
        ]
        return any(indicator in combined for indicator in rate_limit_indicators)

    def _extract_reset_time(self, text: str) -> str | None:
        """Try to extract rate limit reset time from response.

        Args:
            text: Response text to search.

        Returns:
            Reset time string if found, else None.
        """
        # Look for common patterns like "try again in X minutes"
        import re

        patterns = [
            r"try again in (\d+\s*(?:minute|hour|second)s?)",
            r"reset in (\d+\s*(?:minute|hour|second)s?)",
            r"wait (\d+\s*(?:minute|hour|second)s?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1)
        return None

    def _parse_response(self, output: str) -> dict:
        """Parse JSON response from CLI output.

        Args:
            output: Raw CLI output.

        Returns:
            Parsed JSON as dict.

        Raises:
            ParseError: If JSON parsing fails.
        """
        output = output.strip()

        # Try direct JSON parse first
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block in response
        # Look for content between ```json and ``` or { and }
        import re

        # Try markdown code block
        json_match = re.search(r"```json\s*(.*?)\s*```", output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find raw JSON object
        json_match = re.search(r"\{.*\}", output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        raise ParseError(
            "Could not parse JSON from CLI response",
            raw_output=output,
        )
