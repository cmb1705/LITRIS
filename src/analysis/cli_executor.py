"""CLI executor for Claude Code headless mode extraction."""

import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class CliExecutionError(Exception):
    """Error during CLI execution."""

    pass


class AuthenticationError(CliExecutionError):
    """Authentication required for CLI mode."""

    def __init__(self, message: str, setup_instructions: str | None = None):
        super().__init__(message)
        self.setup_instructions = setup_instructions


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


class EmptyResponseError(CliExecutionError):
    """CLI returned empty response - transient failure."""

    pass


class TransientError(CliExecutionError):
    """Transient error that may succeed on retry."""

    pass


class ClaudeCliAuthenticator:
    """Manage Claude Code CLI authentication for headless environments."""

    def __init__(self):
        """Initialize authenticator with platform-appropriate paths."""
        self.creds_path = self._get_credentials_path()

    def _get_credentials_path(self) -> Path:
        """Get the credentials file path based on platform."""
        home = Path.home()
        # Primary location for all platforms
        primary = home / ".claude" / ".credentials.json"
        if primary.exists():
            return primary
        # Alternative location on some Linux systems
        alt = home / ".config" / "claude-code" / ".credentials.json"
        if alt.exists():
            return alt
        return primary  # Default to primary even if doesn't exist

    def get_auth_method(self) -> str:
        """Return which authentication method the CLI will use.

        The Claude CLI checks in this order:
        1. CLAUDE_CODE_OAUTH_TOKEN env var (subscription)
        2. Credentials file with valid OAuth (subscription)
        3. ANTHROPIC_API_KEY env var (API billing)

        Returns:
            One of: 'oauth_token', 'credentials_file', 'api_key', 'none'
        """
        # OAuth token env var takes highest priority
        if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
            return "oauth_token"

        # Credentials file with valid OAuth is next (CLI prefers this over API key)
        if self.creds_path.exists():
            try:
                with open(self.creds_path, "r", encoding="utf-8") as f:
                    creds = json.load(f)
                if "claudeAiOauth" in creds:
                    oauth = creds["claudeAiOauth"]
                    expires_at = oauth.get("expiresAt", 0)
                    if expires_at >= int(time.time() * 1000):
                        return "credentials_file"
            except Exception:
                pass

        # API key is fallback (only used if no OAuth available)
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "api_key"

        return "none"

    def is_authenticated(self) -> tuple[bool, str]:
        """Check if Claude Code is authenticated.

        Returns:
            Tuple of (is_authenticated, status_message)
        """
        # Check for OAuth token env var (highest priority for headless)
        if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
            return True, "Using CLAUDE_CODE_OAUTH_TOKEN (subscription)"

        # Check for credentials file - CLI prefers this over API key
        if self.creds_path.exists():
            try:
                with open(self.creds_path, "r", encoding="utf-8") as f:
                    creds = json.load(f)

                if "claudeAiOauth" in creds:
                    oauth = creds["claudeAiOauth"]
                    expires_at = oauth.get("expiresAt", 0)
                    # Check if token is expired (expires_at is in milliseconds)
                    if expires_at < int(time.time() * 1000):
                        # Token expired - fall through to check API key
                        pass
                    else:
                        return True, "Using credentials file OAuth (subscription)"
            except json.JSONDecodeError:
                pass
            except Exception:
                pass

        # Check for API key - only used as fallback if no valid OAuth
        if os.environ.get("ANTHROPIC_API_KEY"):
            return True, "Using ANTHROPIC_API_KEY (API billing)"

        return False, "No credentials found"

    def get_setup_instructions(self) -> str:
        """Get instructions for setting up authentication.

        Returns:
            Multi-line string with setup instructions.
        """
        return """
Claude Code CLI Authentication Setup
=====================================

For CLI mode extraction (free with Max subscription), you need to authenticate:

Option 1: Interactive Login (Recommended for first-time setup)
--------------------------------------------------------------
1. Run in a terminal with browser access:
   > claude

2. The CLI will open your browser for OAuth authentication
3. Sign in with your Claude account
4. Once authenticated, generate a long-lived token:
   > claude setup-token

5. Set the token as an environment variable:
   Windows PowerShell:
     $env:CLAUDE_CODE_OAUTH_TOKEN = "your-token-here"

   Linux/Mac:
     export CLAUDE_CODE_OAUTH_TOKEN="your-token-here"

Option 2: Use API Key (Paid - uses API billing)
-----------------------------------------------
Set your Anthropic API key and change config.yaml to use 'api' mode:
   Windows PowerShell:
     $env:ANTHROPIC_API_KEY = "sk-ant-..."

   Linux/Mac:
     export ANTHROPIC_API_KEY="sk-ant-..."

Then in config.yaml:
   extraction:
     mode: "api"

For more information, see: https://docs.anthropic.com/claude-code
"""

    def trigger_interactive_login(self) -> bool:
        """Attempt to trigger interactive login via CLI.

        This will open a browser window for OAuth authentication.

        Returns:
            True if login was triggered (user should complete in browser).

        Raises:
            CliExecutionError: If CLI is not installed.
        """
        cli_path = shutil.which("claude")
        if not cli_path:
            raise CliExecutionError(
                "Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
            )

        print("\n" + "=" * 60)
        print("Claude Code Authentication Required")
        print("=" * 60)
        print("\nOpening browser for authentication...")
        print("Please sign in with your Claude account.\n")

        try:
            # Start interactive Claude session which will trigger OAuth
            # On Windows, we need to use 'start' to open in a new window
            if platform.system() == "Windows":
                subprocess.Popen(
                    ["start", "cmd", "/c", cli_path],
                    shell=True,
                )
            else:
                subprocess.Popen(
                    [cli_path],
                    start_new_session=True,
                )

            print("A new terminal window should open with Claude Code.")
            print("After authenticating, run: claude setup-token")
            print("Then set the CLAUDE_CODE_OAUTH_TOKEN environment variable.")
            print("\n" + "=" * 60 + "\n")
            return True

        except Exception as e:
            raise CliExecutionError(f"Failed to launch Claude CLI: {e}")


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
        self.authenticator = ClaudeCliAuthenticator()

    def verify_authentication(self) -> bool:
        """Verify CLI is authenticated and ready for extraction.

        Returns:
            True if ready for extraction.

        Raises:
            AuthenticationError: If not authenticated with setup instructions.
            CliExecutionError: If CLI not installed.
        """
        # Find claude CLI
        self._cli_path = shutil.which("claude")
        if not self._cli_path:
            raise CliExecutionError(
                "Claude Code CLI not found in PATH. "
                "Install with: npm install -g @anthropic-ai/claude-code"
            )

        # Check authentication status
        is_auth, status = self.authenticator.is_authenticated()
        auth_method = self.authenticator.get_auth_method()

        if not is_auth:
            raise AuthenticationError(
                f"Claude Code authentication required: {status}",
                setup_instructions=self.authenticator.get_setup_instructions(),
            )

        # Log which auth method is being used
        logger.info(f"Claude CLI authenticated via: {auth_method}")

        # Warn if using API key (will incur charges)
        if auth_method == "api_key":
            logger.warning(
                "Using ANTHROPIC_API_KEY - this will use API billing, not your subscription. "
                "For free extraction with Max subscription, use CLAUDE_CODE_OAUTH_TOKEN instead."
            )

        # Verify CLI works with a simple version check
        try:
            result = subprocess.run(
                [self._cli_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise CliExecutionError(f"Claude CLI returned error: {result.stderr}")
            logger.debug(f"Claude CLI version: {result.stdout.strip()}")
        except subprocess.TimeoutExpired:
            raise CliExecutionError("Claude CLI version check timed out")
        except FileNotFoundError:
            raise CliExecutionError("Claude CLI executable not found")

        return True

    def setup_authentication_interactive(self) -> bool:
        """Interactively set up authentication.

        Opens browser for OAuth flow.

        Returns:
            True if setup was initiated.
        """
        return self.authenticator.trigger_interactive_login()

    def call_with_prompt(
        self,
        prompt: str,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> str:
        """Execute CLI with a prompt and return raw response text.

        This is a simpler interface for when you have a complete prompt
        and just need the raw text response (e.g., for custom parsing).

        Args:
            prompt: The complete prompt to send.
            max_retries: Maximum retry attempts for transient failures.
            retry_delay: Base delay between retries (uses exponential backoff).

        Returns:
            Raw response text from Claude.

        Raises:
            AuthenticationError: If not authenticated.
            ExtractionTimeoutError: If extraction times out.
            RateLimitError: If rate limit is hit.
            CliExecutionError: For CLI errors after retries exhausted.
        """
        if not self._cli_path:
            self.verify_authentication()

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return self._execute_prompt(prompt)
            except (EmptyResponseError, TransientError) as e:
                last_error = e
                if attempt < max_retries:
                    delay = retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Transient error (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"All {max_retries + 1} attempts failed: {e}")

        raise CliExecutionError(f"CLI failed after {max_retries + 1} attempts: {last_error}")

    def _execute_prompt(self, prompt: str) -> str:
        """Execute a single prompt and return raw response.

        Args:
            prompt: The complete prompt to send.

        Returns:
            Raw response text.

        Raises:
            EmptyResponseError: If CLI returns empty output.
            TransientError: For retriable errors.
            AuthenticationError: If not authenticated.
            ExtractionTimeoutError: If extraction times out.
            RateLimitError: If rate limit is hit.
            CliExecutionError: For other CLI errors.
        """
        # Use stdin for prompt to avoid Windows command line length limits
        # The -p flag with long text exceeds the ~8KB limit on Windows
        cmd = [
            self._cli_path,
            "--print",
            "--output-format",
            self.output_format,
        ]

        env = os.environ.copy()

        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
                encoding="utf-8",
                errors="replace",  # Replace unencodable chars instead of failing
            )

            # Check for authentication errors
            combined_output = (result.stdout + result.stderr).lower()
            if any(
                x in combined_output
                for x in ["invalid api key", "authentication", "unauthorized", "not authenticated"]
            ):
                raise AuthenticationError(
                    "Authentication failed during extraction. Please re-authenticate.",
                    setup_instructions=self.authenticator.get_setup_instructions(),
                )

            # Check for rate limit
            if self._is_rate_limited(result.stdout, result.stderr):
                reset_time = self._extract_reset_time(result.stdout + result.stderr)
                raise RateLimitError(
                    "Rate limit hit during extraction",
                    reset_time=reset_time,
                )

            # Check for empty response
            stdout_stripped = result.stdout.strip()
            if not stdout_stripped:
                stderr_info = result.stderr.strip()[:200] if result.stderr else "none"
                raise EmptyResponseError(
                    f"CLI returned empty response (returncode={result.returncode}, "
                    f"stderr={stderr_info})"
                )

            # Check for errors
            if result.returncode != 0:
                error_msg = result.stderr or result.stdout
                if "error" in error_msg.lower():
                    raise TransientError(f"CLI error (may retry): {error_msg[:200]}")
                raise CliExecutionError(f"CLI returned error: {error_msg}")

            return result.stdout

        except subprocess.TimeoutExpired:
            raise ExtractionTimeoutError(
                f"Extraction timed out after {self.timeout} seconds"
            )

    def extract(
        self,
        prompt: str,
        input_text: str,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> dict:
        """Execute extraction via CLI with automatic retry for transient failures.

        Args:
            prompt: The extraction prompt/instructions.
            input_text: The paper text to extract from.
            max_retries: Maximum retry attempts for transient failures.
            retry_delay: Base delay between retries (uses exponential backoff).

        Returns:
            Parsed JSON response from Claude.

        Raises:
            AuthenticationError: If not authenticated.
            ExtractionTimeoutError: If extraction times out.
            RateLimitError: If rate limit is hit.
            ParseError: If response cannot be parsed after retries.
            CliExecutionError: For other CLI errors.
        """
        if not self._cli_path:
            self.verify_authentication()

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return self._execute_single_extraction(prompt, input_text)
            except (EmptyResponseError, TransientError) as e:
                last_error = e
                if attempt < max_retries:
                    delay = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Transient error (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"All {max_retries + 1} attempts failed: {e}")

        # Convert to ParseError for consistent handling
        raise ParseError(
            f"Extraction failed after {max_retries + 1} attempts: {last_error}",
            raw_output="",
        )

    def _execute_single_extraction(self, prompt: str, input_text: str) -> dict:
        """Execute a single extraction attempt.

        Args:
            prompt: The extraction prompt/instructions.
            input_text: The paper text to extract from.

        Returns:
            Parsed JSON response from Claude.

        Raises:
            EmptyResponseError: If CLI returns empty output.
            TransientError: For retriable errors.
            AuthenticationError: If not authenticated.
            ExtractionTimeoutError: If extraction times out.
            RateLimitError: If rate limit is hit.
            ParseError: If response cannot be parsed.
            CliExecutionError: For other CLI errors.
        """
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
                "--output-format",
                self.output_format,
                "-p",
                prompt,
            ]

            # Set up environment - ensure OAuth token is passed if set
            env = os.environ.copy()

            # Execute with input from file
            with open(input_file, "r", encoding="utf-8") as f:
                result = subprocess.run(
                    cmd,
                    stdin=f,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env=env,
                )

            # Check for authentication errors in response
            combined_output = (result.stdout + result.stderr).lower()
            if any(
                x in combined_output
                for x in ["invalid api key", "authentication", "unauthorized", "not authenticated"]
            ):
                raise AuthenticationError(
                    "Authentication failed during extraction. Please re-authenticate.",
                    setup_instructions=self.authenticator.get_setup_instructions(),
                )

            # Check for rate limit
            if self._is_rate_limited(result.stdout, result.stderr):
                reset_time = self._extract_reset_time(result.stdout + result.stderr)
                raise RateLimitError(
                    "Rate limit hit during extraction",
                    reset_time=reset_time,
                )

            # Check for empty response (transient failure)
            stdout_stripped = result.stdout.strip()
            if not stdout_stripped:
                stderr_info = result.stderr.strip()[:200] if result.stderr else "none"
                raise EmptyResponseError(
                    f"CLI returned empty response (returncode={result.returncode}, "
                    f"stderr={stderr_info})"
                )

            # Check for errors with non-empty output
            if result.returncode != 0:
                error_msg = result.stderr or result.stdout
                # Some non-zero returns with output might be transient
                if "error" in error_msg.lower() and "json" not in stdout_stripped.lower():
                    raise TransientError(f"CLI error (may retry): {error_msg[:200]}")
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
            EmptyResponseError: If output is empty.
            ParseError: If JSON parsing fails.
        """
        output = output.strip()

        # Guard against empty output
        if not output:
            raise EmptyResponseError("Cannot parse empty response")

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
