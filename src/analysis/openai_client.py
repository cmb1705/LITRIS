"""OpenAI LLM client for paper extraction.

Supports GPT-5.2 models including GPT-5.2-Codex for agentic coding tasks.
Also supports Codex CLI for subscription-based usage.

References:
- https://platform.openai.com/docs/models/gpt-5.2
- https://developers.openai.com/codex/cli/
"""

import json
import re
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from pydantic import ValidationError

from src.analysis.base_llm import BaseLLMClient, ExtractionMode, LLMProvider
from src.analysis.prompts import EXTRACTION_SYSTEM_PROMPT, build_extraction_prompt
from src.analysis.schemas import ExtractionResult, PaperExtraction
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class OpenAILLMClient(BaseLLMClient):
    """Client for OpenAI GPT-based paper extraction.

    Supports both API mode (direct OpenAI API) and CLI mode (Codex CLI).
    """

    # Available models with descriptions
    MODELS = {
        # o3 family (recommended for Codex CLI with ChatGPT)
        "o3": "o3 (High intelligence, complex tasks)",
        "o3-mini": "o3-mini (Fast, cost-effective reasoning)",
        # GPT-5.2 family (latest API models)
        "gpt-5.2": "GPT-5.2 (Latest flagship model)",
        "gpt-5.2-instant": "GPT-5.2 Instant (Fast, everyday tasks)",
        "gpt-5.2-pro": "GPT-5.2 Pro (Highest quality, complex tasks)",
        "gpt-5.2-codex": "GPT-5.2-Codex (Optimized for agentic coding)",
        # GPT-5 family
        "gpt-5": "GPT-5 (Previous generation flagship)",
        # GPT-4 family (API mode only)
        "gpt-4o": "GPT-4o (Multimodal, fast) - API mode only",
        "gpt-4o-mini": "GPT-4o Mini (Cost-effective) - API mode only",
        "gpt-4-turbo": "GPT-4 Turbo (Legacy) - API mode only",
    }

    # Model pricing per million tokens (input, output) in USD
    MODEL_PRICING = {
        # o3 family
        "o3": (10.0, 40.0),
        "o3-mini": (1.1, 4.4),
        # GPT-5.2 family
        "gpt-5.2": (10.0, 30.0),
        "gpt-5.2-instant": (2.5, 10.0),
        "gpt-5.2-pro": (20.0, 60.0),
        "gpt-5.2-codex": (10.0, 30.0),
        "gpt-5": (10.0, 30.0),
        # GPT-4 family
        "gpt-4o": (2.5, 10.0),
        "gpt-4o-mini": (0.15, 0.6),
        "gpt-4-turbo": (10.0, 30.0),
    }

    # Models supported by Codex CLI with ChatGPT authentication
    # Default is gpt-5.2 for ChatGPT Plus/Pro subscribers
    CLI_SUPPORTED_MODELS = {"gpt-5.2"}

    def __init__(
        self,
        mode: ExtractionMode = "api",
        model: str | None = None,
        max_tokens: int = 8192,
        timeout: int = 120,
        reasoning_effort: str | None = None,
    ):
        """Initialize OpenAI LLM client.

        Args:
            mode: Extraction mode ('api' for OpenAI API, 'cli' for Codex CLI).
            model: Model to use for extraction. Defaults to gpt-5.2.
            max_tokens: Maximum tokens for response.
            timeout: Request timeout in seconds.
            reasoning_effort: Reasoning effort level for GPT-5.2 models.
                Options: 'none', 'low', 'medium', 'high', 'xhigh'.
                Only applies to GPT-5.2 family models.
        """
        super().__init__(mode=mode, model=model, max_tokens=max_tokens, timeout=timeout)
        self.reasoning_effort = reasoning_effort
        self.client = None

        self.validate_mode()

        if mode == "api":
            api_key = self._get_api_key()
            if not api_key:
                raise ValueError(
                    "OpenAI API key required for API mode. "
                    "Set OPENAI_API_KEY environment variable."
                )
            # Lazy import to avoid requiring openai package if not used
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
            except ImportError as e:
                raise ImportError(
                    "OpenAI package not installed. Install with: pip install openai"
                ) from e
        elif mode == "cli":
            self._verify_codex_cli()
            # Check if model is supported by CLI mode with ChatGPT auth
            if self.model not in self.CLI_SUPPORTED_MODELS:
                logger.warning(
                    f"Model '{self.model}' may not be supported by Codex CLI with ChatGPT auth. "
                    f"Supported models: {', '.join(self.CLI_SUPPORTED_MODELS)}. "
                    f"Using default 'gpt-5.2'."
                )
                self.model = "gpt-5.2"

    @property
    def provider(self) -> LLMProvider:
        """Return the provider identifier."""
        return "openai"

    @property
    def default_model(self) -> str:
        """Return the default model for OpenAI."""
        return "gpt-5.2"

    @property
    def supported_modes(self) -> list[ExtractionMode]:
        """Return list of supported extraction modes."""
        return ["api", "cli"]

    @classmethod
    def list_models(cls) -> dict[str, str]:
        """Return available models with descriptions."""
        return cls.MODELS.copy()

    def _get_api_key(self) -> str | None:
        """Get OpenAI API key from environment."""
        return os.environ.get("OPENAI_API_KEY")

    def _find_codex_path(self) -> str | None:
        """Find the Codex CLI executable path.

        Searches standard PATH plus common npm global bin locations.

        Returns:
            Path to codex executable or None.
        """
        # Try standard PATH first
        codex_path = shutil.which("codex")
        if codex_path:
            return codex_path

        # Check common npm global bin locations
        import platform
        npm_paths = []

        if platform.system() == "Windows":
            # Windows npm global locations
            appdata = os.environ.get("APPDATA", "")
            if appdata:
                npm_paths.append(os.path.join(appdata, "npm", "codex.cmd"))
                npm_paths.append(os.path.join(appdata, "npm", "codex"))
            # Also check user profile
            userprofile = os.environ.get("USERPROFILE", "")
            if userprofile:
                npm_paths.append(os.path.join(userprofile, "AppData", "Roaming", "npm", "codex.cmd"))
        else:
            # Unix-like systems
            home = os.environ.get("HOME", "")
            if home:
                npm_paths.extend([
                    os.path.join(home, ".npm-global", "bin", "codex"),
                    os.path.join(home, ".nvm", "versions", "node", "*", "bin", "codex"),
                    "/usr/local/bin/codex",
                    "/opt/homebrew/bin/codex",
                ])

        for path in npm_paths:
            if "*" in path:
                # Handle glob patterns (for nvm)
                import glob
                matches = glob.glob(path)
                for match in matches:
                    if os.path.isfile(match) and os.access(match, os.X_OK):
                        return match
            elif os.path.isfile(path):
                return path

        return None

    def _verify_codex_cli(self) -> None:
        """Verify Codex CLI is installed and authenticated.

        Raises:
            ValueError: If Codex CLI is not available or not authenticated.
        """
        self._codex_path = self._find_codex_path()
        if not self._codex_path:
            raise ValueError(
                "Codex CLI not found. Install with:\n"
                "  npm i -g @openai/codex\n"
                "  -- or --\n"
                "  brew install --cask codex\n"
                "Then authenticate with: codex login"
            )

        logger.debug(f"Found Codex CLI at: {self._codex_path}")

        # Check authentication status using 'codex login status'
        try:
            result = subprocess.run(
                [self._codex_path, "login", "status"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                logger.warning(
                    "Codex CLI may not be authenticated. "
                    "Run 'codex login' to authenticate."
                )
            else:
                logger.debug(f"Codex auth: {result.stdout.strip()}")
        except subprocess.TimeoutExpired:
            logger.warning("Timeout checking Codex CLI auth status")
        except FileNotFoundError as e:
            raise ValueError(f"Codex CLI not found at {self._codex_path}") from e

    def extract(
        self,
        paper_id: str,
        title: str,
        authors: str,
        year: int | str | None,
        item_type: str,
        text: str,
        prompt_override: str | None = None,
    ) -> ExtractionResult:
        """Extract structured information from paper text.

        Args:
            paper_id: Unique paper identifier.
            title: Paper title.
            authors: Author string.
            year: Publication year.
            item_type: Type of paper.
            text: Full text content.
            prompt_override: Optional pre-built prompt to use instead of default.

        Returns:
            ExtractionResult with extraction or error.
        """
        start_time = time.time()

        try:
            prompt = prompt_override or build_extraction_prompt(
                title=title,
                authors=authors,
                year=year,
                item_type=item_type,
                text=text,
            )

            if self.mode == "api":
                response_text, input_tokens, output_tokens = self._call_api(prompt)
            else:
                response_text, input_tokens, output_tokens = self._call_cli(prompt)

            # Parse JSON response
            extraction = self._parse_response(response_text)

            duration = time.time() - start_time
            logger.info(
                f"Extracted paper {paper_id} in {duration:.1f}s "
                f"(confidence: {extraction.extraction_confidence:.2f})"
            )

            return ExtractionResult(
                paper_id=paper_id,
                success=True,
                extraction=extraction,
                duration_seconds=duration,
                model_used=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except ImportError as e:
            duration = time.time() - start_time
            logger.error(f"Import error for paper {paper_id}: {e}")
            return ExtractionResult(
                paper_id=paper_id,
                success=False,
                error=f"Missing dependency: {e}",
                duration_seconds=duration,
                model_used=self.model,
            )
        except (json.JSONDecodeError, ValidationError) as e:
            duration = time.time() - start_time
            logger.error(f"Response parsing failed for paper {paper_id}: {e}")
            return ExtractionResult(
                paper_id=paper_id,
                success=False,
                error=f"Parse error: {e}",
                duration_seconds=duration,
                model_used=self.model,
            )
        except Exception as e:
            duration = time.time() - start_time
            error_type = type(e).__name__
            logger.error(f"{error_type} for paper {paper_id}: {e}")
            return ExtractionResult(
                paper_id=paper_id,
                success=False,
                error=f"{error_type}: {e}",
                duration_seconds=duration,
                model_used=self.model,
            )

    def _call_api(self, prompt: str) -> tuple[str, int, int]:
        """Call OpenAI API.

        Args:
            prompt: User prompt.

        Returns:
            Tuple of (response_text, input_tokens, output_tokens).
        """
        # Build messages
        messages = [
            {"role": "developer", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # Build request kwargs
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": self.max_tokens,
        }

        # Add reasoning effort for GPT-5.2 models
        if self.reasoning_effort and self.model.startswith("gpt-5"):
            kwargs["reasoning"] = {"effort": self.reasoning_effort}

        # Make API call
        response = self.client.chat.completions.create(**kwargs)

        response_text = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        return response_text, input_tokens, output_tokens

    def _call_cli(self, prompt: str) -> tuple[str, int, int]:
        """Call Codex CLI.

        Uses Codex CLI to process the prompt. The CLI uses ChatGPT
        subscription for billing (free with Plus/Pro plans).

        Args:
            prompt: User prompt.

        Returns:
            Tuple of (response_text, input_tokens, output_tokens).
            Note: CLI mode does not provide token counts.
        """
        # Create temp file for output
        output_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
            encoding="utf-8",
        )
        output_file.close()
        output_path = output_file.name

        try:
            # Combine system prompt and user prompt
            full_prompt = f"{EXTRACTION_SYSTEM_PROMPT}\n\n---\n\n{prompt}"

            # Run Codex CLI exec command with prompt via stdin
            codex_cmd = getattr(self, "_codex_path", None) or "codex"
            cmd = [
                codex_cmd,
                "exec",  # Non-interactive mode
                "-m", self.model,  # Model selection
                "-o", output_path,  # Output file for response
                "--skip-git-repo-check",  # Allow running outside git repo
                "-",  # Read prompt from stdin
            ]

            result = subprocess.run(
                cmd,
                input=full_prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                encoding="utf-8",
                errors="replace",  # Replace unencodable chars instead of failing
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"Codex CLI failed (exit {result.returncode}): {result.stderr}"
                )

            # Read response from output file
            response_text = Path(output_path).read_text(encoding="utf-8").strip()
            return response_text, 0, 0

        finally:
            # Clean up temp file
            Path(output_path).unlink(missing_ok=True)

    def _parse_response(self, response_text: str) -> PaperExtraction:
        """Parse JSON response into PaperExtraction.

        Args:
            response_text: Raw response text.

        Returns:
            Parsed PaperExtraction.
        """
        # Clean response - remove any markdown formatting
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Guard against empty response
        if not text:
            raise ValueError("Cannot parse empty response from LLM")

        # Parse JSON
        data = json.loads(text)

        def _coerce_str_list(value: object) -> list[str]:
            """Coerce a value into a list of strings."""
            if value is None:
                return []
            if isinstance(value, list):
                items = []
                for item in value:
                    if item is None:
                        continue
                    text_item = str(item).strip()
                    if text_item:
                        items.append(text_item)
                return items
            if isinstance(value, str):
                text = value.strip()
                if not text:
                    return []
                parts = [p.strip() for p in re.split(r"[;,\n]+", text) if p.strip()]
                return parts if parts else [text]
            return [str(value).strip()]

        def _normalize_enum(
            value: object,
            allowed: set[str],
            synonyms: dict[str, str] | None = None,
            default: str | None = None,
        ) -> object:
            """Normalize enum-like values to allowed strings."""
            if value is None:
                return None
            if isinstance(value, (int, float)):
                return default if default is not None else value
            if not isinstance(value, str):
                return value

            raw = value.strip().strip('"').strip("'").lower()
            if not raw:
                return None
            raw = re.sub(r"\s*\([^)]*\)", "", raw).strip()
            if not raw:
                return None
            raw = raw.replace("&", "and").replace("\\", "/")

            def _match(token: str) -> str | None:
                token = token.strip()
                if not token:
                    return None
                token = re.sub(r"[^\w\s/-]", " ", token)
                token = token.replace("-", " ")
                token = " ".join(token.split())
                token_key = token.replace(" ", "_")
                if token_key in allowed:
                    return token_key
                if token in allowed:
                    return token
                if synonyms:
                    if token_key in synonyms:
                        return synonyms[token_key]
                    if token in synonyms:
                        return synonyms[token]
                return None

            direct = _match(raw)
            if direct:
                return direct

            for part in re.split(r"[/,;|]+", raw):
                matched = _match(part)
                if matched:
                    return matched

            raw_key = raw.replace(" ", "_")
            for allowed_value in allowed:
                if allowed_value in raw_key:
                    return allowed_value

            return default if default is not None else value

        def _normalize_significance(value: object) -> object:
            """Normalize significance values into high/medium/low."""
            if isinstance(value, (int, float)):
                score = float(value)
                if score >= 0.67:
                    return "high"
                if score >= 0.34:
                    return "medium"
                return "low"
            if isinstance(value, str):
                numeric = value.strip()
                if re.fullmatch(r"\d+(\.\d+)?", numeric):
                    return _normalize_significance(float(numeric))
            return _normalize_enum(
                value,
                {"high", "medium", "low"},
                synonyms={
                    "major": "high",
                    "significant": "high",
                    "substantial": "high",
                    "strong": "high",
                    "important": "high",
                    "novel": "high",
                    "groundbreaking": "high",
                    "moderate": "medium",
                    "mixed": "medium",
                    "partial": "medium",
                    "average": "medium",
                    "minor": "low",
                    "limited": "low",
                    "small": "low",
                    "weak": "low",
                    "modest": "low",
                    "incremental": "low",
                    "preliminary": "low",
                },
                default="medium",
            )

        def _normalize_evidence_type(value: object) -> object:
            """Normalize evidence_type values to allowed enum entries."""
            return _normalize_enum(
                value,
                {
                    "empirical",
                    "theoretical",
                    "methodological",
                    "case_study",
                    "survey",
                    "experimental",
                    "qualitative",
                    "quantitative",
                    "mixed",
                },
                synonyms={
                    "empiric": "empirical",
                    "data": "empirical",
                    "evidence": "empirical",
                    "theory": "theoretical",
                    "methods": "methodological",
                    "methodology": "methodological",
                    "case": "case_study",
                    "case_studies": "case_study",
                    "case_study": "case_study",
                    "case study": "case_study",
                    "survey_based": "survey",
                    "experiment": "experimental",
                    "experiments": "experimental",
                    "qual": "qualitative",
                    "quant": "quantitative",
                    "mixed_methods": "mixed",
                    "mixed methods": "mixed",
                },
                default="empirical",
            )

        def _normalize_support_type(value: object) -> object:
            """Normalize support_type values to allowed enum entries."""
            return _normalize_enum(
                value,
                {"data", "citation", "logic", "example", "authority"},
                synonyms={
                    "empirical": "data",
                    "empiric": "data",
                    "experimental": "data",
                    "quantitative": "data",
                    "qualitative": "data",
                    "survey": "data",
                    "methodological": "data",
                    "methodology": "data",
                    "methods": "data",
                    "evidence": "data",
                    "data": "data",
                    "citation": "citation",
                    "cite": "citation",
                    "reference": "citation",
                    "references": "citation",
                    "literature": "citation",
                    "logic": "logic",
                    "reasoning": "logic",
                    "rationale": "logic",
                    "argument": "logic",
                    "theoretical": "logic",
                    "theory": "logic",
                    "example": "example",
                    "case": "example",
                    "case_study": "example",
                    "case study": "example",
                    "illustration": "example",
                    "instance": "example",
                    "authority": "authority",
                    "expert": "authority",
                    "expert_opinion": "authority",
                    "consensus": "authority",
                },
                default="logic",
            )

        # Normalize list-like fields to avoid schema failures
        for list_field in [
            "research_questions",
            "limitations",
            "future_directions",
            "keywords",
            "discipline_tags",
        ]:
            if list_field in data:
                data[list_field] = _coerce_str_list(data.get(list_field))

        # Normalize nested methodology fields
        if "methodology" in data:
            if not isinstance(data["methodology"], dict):
                data["methodology"] = {}
            else:
                methodology = data["methodology"]
                methodology["data_sources"] = _coerce_str_list(
                    methodology.get("data_sources")
                )
                methodology["analysis_methods"] = _coerce_str_list(
                    methodology.get("analysis_methods")
                )
                for field in ["approach", "design", "sample_size", "time_period"]:
                    if field in methodology and methodology[field] is not None:
                        if not isinstance(methodology[field], str):
                            methodology[field] = str(methodology[field])

        # Normalize known enum fields to avoid validation failures
        if isinstance(data.get("key_findings"), list):
            normalized_findings = []
            for finding in data["key_findings"]:
                if isinstance(finding, str):
                    finding = {"finding": finding}
                if isinstance(finding, dict):
                    if "finding" not in finding:
                        for alt_key in ("result", "finding_text", "summary"):
                            if alt_key in finding:
                                finding["finding"] = finding[alt_key]
                                break
                    if "finding" not in finding:
                        continue
                    finding["significance"] = _normalize_significance(
                        finding.get("significance")
                    )
                    finding["evidence_type"] = _normalize_evidence_type(
                        finding.get("evidence_type")
                    )
                    normalized_findings.append(finding)
            data["key_findings"] = normalized_findings

        if isinstance(data.get("key_claims"), list):
            normalized_claims = []
            for claim in data["key_claims"]:
                if isinstance(claim, str):
                    claim = {"claim": claim}
                if isinstance(claim, dict):
                    if "claim" not in claim:
                        for alt_key in ("statement", "claim_text", "text"):
                            if alt_key in claim:
                                claim["claim"] = claim[alt_key]
                                break
                    if "claim" not in claim:
                        continue
                    claim["support_type"] = _normalize_support_type(
                        claim.get("support_type")
                    )
                    claim["strength"] = _normalize_significance(
                        claim.get("strength")
                    )
                    normalized_claims.append(claim)
            data["key_claims"] = normalized_claims

        # Handle nested methodology
        if "methodology" in data and isinstance(data["methodology"], dict):
            from src.analysis.schemas import Methodology
            data["methodology"] = Methodology(**data["methodology"])

        # Handle nested key_findings
        if "key_findings" in data and isinstance(data["key_findings"], list):
            from src.analysis.schemas import KeyFinding
            data["key_findings"] = [
                KeyFinding(**f) if isinstance(f, dict) else f
                for f in data["key_findings"]
            ]

        # Handle nested key_claims
        if "key_claims" in data and isinstance(data["key_claims"], list):
            from src.analysis.schemas import KeyClaim
            data["key_claims"] = [
                KeyClaim(**c) if isinstance(c, dict) else c
                for c in data["key_claims"]
            ]

        return PaperExtraction(**data)

    def estimate_cost(self, text_length: int) -> float:
        """Estimate cost for extraction.

        Args:
            text_length: Length of input text.

        Returns:
            Estimated cost in USD.
        """
        # Rough estimate: 4 chars per token
        input_tokens = text_length // 4 + 500  # Add prompt overhead
        output_tokens = 2000  # Typical extraction output

        # Get pricing for model (default to gpt-5.2 pricing)
        pricing = self.MODEL_PRICING.get(self.model, (10.0, 30.0))
        input_cost_per_million, output_cost_per_million = pricing

        input_cost = (input_tokens / 1_000_000) * input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * output_cost_per_million

        return input_cost + output_cost


class CodexCliExecutor:
    """Executor for OpenAI Codex CLI commands.

    Provides a higher-level interface for Codex CLI operations,
    similar to ClaudeCliExecutor for Anthropic.
    """

    def __init__(self, timeout: int = 120):
        """Initialize Codex CLI executor.

        Args:
            timeout: Default timeout for CLI commands in seconds.
        """
        self.timeout = timeout
        self._codex_path = self._find_codex_path()
        if not self._codex_path:
            raise RuntimeError(
                "Codex CLI not installed. Install with:\n"
                "  npm i -g @openai/codex\n"
                "  brew install --cask codex"
            )

    def _find_codex_path(self) -> str | None:
        """Find the Codex CLI executable path."""
        # Try standard PATH first
        codex_path = shutil.which("codex")
        if codex_path:
            return codex_path

        # Check common npm global bin locations
        import platform
        npm_paths = []

        if platform.system() == "Windows":
            appdata = os.environ.get("APPDATA", "")
            if appdata:
                npm_paths.append(os.path.join(appdata, "npm", "codex.cmd"))
                npm_paths.append(os.path.join(appdata, "npm", "codex"))
            userprofile = os.environ.get("USERPROFILE", "")
            if userprofile:
                npm_paths.append(os.path.join(userprofile, "AppData", "Roaming", "npm", "codex.cmd"))
        else:
            home = os.environ.get("HOME", "")
            if home:
                npm_paths.extend([
                    os.path.join(home, ".npm-global", "bin", "codex"),
                    "/usr/local/bin/codex",
                    "/opt/homebrew/bin/codex",
                ])

        for path in npm_paths:
            if os.path.isfile(path):
                return path

        return None

    def is_authenticated(self) -> tuple[bool, str]:
        """Check if Codex CLI is authenticated.

        Returns:
            Tuple of (is_authenticated, status_message).
        """
        try:
            result = subprocess.run(
                [self._codex_path, "login", "status"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return True, result.stdout.strip() or "Authenticated via ChatGPT"
            return False, result.stderr.strip() or "Not authenticated"
        except subprocess.TimeoutExpired:
            return False, "Timeout checking auth status"
        except FileNotFoundError:
            return False, "Codex CLI not found"

    def login(self, method: str = "oauth") -> bool:
        """Trigger Codex login.

        Args:
            method: Login method ('oauth', 'device', or 'api-key').

        Returns:
            True if login was initiated.
        """
        cmd = [self._codex_path, "login"]
        if method == "device":
            cmd.append("--device-auth")

        try:
            subprocess.Popen(cmd)
            return True
        except FileNotFoundError:
            return False

    def get_setup_instructions(self) -> str:
        """Get instructions for setting up Codex CLI."""
        return """
Codex CLI Setup Instructions
============================

1. Install Codex CLI:
   npm i -g @openai/codex
   -- or --
   brew install --cask codex

2. Authenticate:
   codex login

   This opens a browser for ChatGPT OAuth authentication.
   Your ChatGPT Plus/Pro subscription covers Codex usage.

3. For headless environments, use device auth:
   codex login --device-auth

4. Verify authentication:
   codex auth status

For more information: https://developers.openai.com/codex/cli/
"""
