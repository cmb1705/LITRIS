"""OpenDataLoader PDF extraction helpers."""

from __future__ import annotations

import atexit
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from threading import local
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

try:
    import opendataloader_pdf

    OPENDATALOADER_AVAILABLE = True
except ImportError:
    OPENDATALOADER_AVAILABLE = False
    opendataloader_pdf = None

_MIN_JAVA_VERSION = 11
DEFAULT_HYBRID_BACKEND = "docling-fast"
DEFAULT_HYBRID_CLIENT_MODE = "auto"
DEFAULT_HYBRID_HOST = "127.0.0.1"
DEFAULT_HYBRID_PORT = 5002
DEFAULT_HYBRID_TIMEOUT_MS = 0
DEFAULT_HYBRID_STARTUP_TIMEOUT_SECONDS = 30.0
DEFAULT_HYBRID_DEVICE = "cuda"

_HYBRID_SERVER_PROCESS: subprocess.Popen[str] | None = None
_HYBRID_SERVER_COMMAND: tuple[str, ...] | None = None
_ATTEMPT_STATE = local()


@dataclass(frozen=True)
class OpenDataLoaderHybridConfig:
    """Configuration for OpenDataLoader hybrid extraction."""

    enabled: bool = False
    backend: str = DEFAULT_HYBRID_BACKEND
    client_mode: str = DEFAULT_HYBRID_CLIENT_MODE
    server_url: str | None = None
    timeout_ms: int = DEFAULT_HYBRID_TIMEOUT_MS
    fallback_to_fast: bool = False
    autostart: bool = False
    host: str = DEFAULT_HYBRID_HOST
    port: int = DEFAULT_HYBRID_PORT
    startup_timeout_seconds: float = DEFAULT_HYBRID_STARTUP_TIMEOUT_SECONDS
    force_ocr: bool = False
    ocr_lang: str | None = None
    enrich_formula: bool = False
    enrich_picture_description: bool = False
    picture_description_prompt: str | None = None
    device: str = DEFAULT_HYBRID_DEVICE
    python_executable: str | None = None
    managed_servers: tuple[ManagedHybridServerSpec, ...] = ()

    def matched_managed_server(self) -> ManagedHybridServerSpec | None:
        """Return the fixed managed endpoint matching this hybrid profile."""
        for server in self.managed_servers:
            if server.matches(self):
                return server
        return None

    def uses_managed_server_pool(self) -> bool:
        """Return whether this request resolves to a managed fixed endpoint."""
        return self.matched_managed_server() is not None

    def resolved_server_url(self) -> str:
        """Return the fully qualified hybrid server URL."""
        matched_server = self.matched_managed_server()
        if matched_server is not None:
            return matched_server.url
        if self.server_url:
            return _normalize_server_url(self.server_url)
        return f"http://{self.host}:{self.port}"

    def effective_client_mode(self) -> str:
        """Return the required hybrid client mode for the configured features."""
        if self.enrich_formula or self.enrich_picture_description:
            return "full"
        return self.client_mode


@dataclass(frozen=True)
class OpenDataLoaderAttemptResult:
    """Structured result for the last OpenDataLoader extraction attempt."""

    mode: str
    text: str | None
    error: str | None = None
    word_count: int = 0


@dataclass(frozen=True)
class ManagedHybridServerSpec:
    """A fixed managed OpenDataLoader hybrid server profile."""

    name: str
    url: str
    force_ocr: bool = False
    enrich_formula: bool = False
    enrich_picture_description: bool = False

    def matches(self, config: OpenDataLoaderHybridConfig) -> bool:
        """Return whether this server matches the requested hybrid features."""
        return (
            self.force_ocr == config.force_ocr
            and self.enrich_formula == config.enrich_formula
            and (self.enrich_picture_description == config.enrich_picture_description)
        )


def _hybrid_endpoint_label(config: OpenDataLoaderHybridConfig) -> str:
    """Return a concise label describing how the hybrid endpoint was resolved."""
    matched_server = config.matched_managed_server()
    if matched_server is not None:
        return f"managed:{matched_server.name}"
    if config.server_url:
        return "legacy:url"
    return "legacy:host_port"


def build_managed_server_specs(
    server_configs: dict[str, Any] | None,
) -> tuple[ManagedHybridServerSpec, ...]:
    """Convert config models or dicts into fixed hybrid server specs."""
    if not server_configs:
        return ()

    specs: list[ManagedHybridServerSpec] = []
    for name, raw_server in server_configs.items():
        if isinstance(raw_server, dict):
            server_url = str(raw_server.get("url") or "").strip()
            force_ocr = bool(raw_server.get("force_ocr", False))
            enrich_formula = bool(raw_server.get("enrich_formula", False))
            enrich_picture_description = bool(raw_server.get("enrich_picture_description", False))
        else:
            server_url = str(getattr(raw_server, "url", "") or "").strip()
            force_ocr = bool(getattr(raw_server, "force_ocr", False))
            enrich_formula = bool(getattr(raw_server, "enrich_formula", False))
            enrich_picture_description = bool(
                getattr(raw_server, "enrich_picture_description", False)
            )

        if not server_url:
            continue

        specs.append(
            ManagedHybridServerSpec(
                name=name,
                url=_normalize_server_url(server_url),
                force_ocr=force_ocr,
                enrich_formula=enrich_formula,
                enrich_picture_description=enrich_picture_description,
            )
        )

    return tuple(specs)


def build_hybrid_config(
    *,
    enabled: bool = False,
    backend: str = DEFAULT_HYBRID_BACKEND,
    client_mode: str = DEFAULT_HYBRID_CLIENT_MODE,
    server_url: str | None = None,
    timeout_ms: int = DEFAULT_HYBRID_TIMEOUT_MS,
    fallback_to_fast: bool = False,
    autostart: bool = False,
    host: str = DEFAULT_HYBRID_HOST,
    port: int = DEFAULT_HYBRID_PORT,
    startup_timeout_seconds: float = DEFAULT_HYBRID_STARTUP_TIMEOUT_SECONDS,
    force_ocr: bool = False,
    ocr_lang: str | None = None,
    enrich_formula: bool = False,
    enrich_picture_description: bool = False,
    picture_description_prompt: str | None = None,
    device: str = DEFAULT_HYBRID_DEVICE,
    python_executable: str | None = None,
    managed_servers: Iterable[ManagedHybridServerSpec] | None = None,
) -> OpenDataLoaderHybridConfig | None:
    """Build a hybrid config when hybrid behavior is requested."""
    wants_hybrid = (
        enabled
        or autostart
        or bool(server_url)
        or force_ocr
        or enrich_formula
        or enrich_picture_description
    )
    if not wants_hybrid:
        return None
    return OpenDataLoaderHybridConfig(
        enabled=wants_hybrid,
        backend=backend,
        client_mode=client_mode,
        server_url=server_url,
        timeout_ms=timeout_ms,
        fallback_to_fast=fallback_to_fast,
        autostart=autostart,
        host=host,
        port=port,
        startup_timeout_seconds=startup_timeout_seconds,
        force_ocr=force_ocr,
        ocr_lang=ocr_lang,
        enrich_formula=enrich_formula,
        enrich_picture_description=enrich_picture_description,
        picture_description_prompt=picture_description_prompt,
        device=device,
        python_executable=python_executable,
        managed_servers=tuple(managed_servers or ()),
    )


def build_hybrid_config_from_processing(
    processing: Any,
    *,
    enabled: bool | None = None,
    fallback_to_fast: bool | None = None,
) -> OpenDataLoaderHybridConfig | None:
    """Build a hybrid config directly from ProcessingConfig-like data."""
    return build_hybrid_config(
        enabled=(
            bool(enabled)
            if enabled is not None
            else bool(getattr(processing, "opendataloader_hybrid_enabled", False))
        ),
        backend=getattr(
            processing,
            "opendataloader_hybrid_backend",
            DEFAULT_HYBRID_BACKEND,
        ),
        client_mode=getattr(
            processing,
            "opendataloader_hybrid_client_mode",
            DEFAULT_HYBRID_CLIENT_MODE,
        ),
        server_url=getattr(processing, "opendataloader_hybrid_url", None),
        timeout_ms=getattr(
            processing,
            "opendataloader_hybrid_timeout_ms",
            DEFAULT_HYBRID_TIMEOUT_MS,
        ),
        fallback_to_fast=(
            bool(fallback_to_fast)
            if fallback_to_fast is not None
            else bool(getattr(processing, "opendataloader_hybrid_fallback", False))
        ),
        autostart=bool(getattr(processing, "opendataloader_hybrid_autostart", False)),
        host=getattr(processing, "opendataloader_hybrid_host", DEFAULT_HYBRID_HOST),
        port=int(getattr(processing, "opendataloader_hybrid_port", DEFAULT_HYBRID_PORT)),
        startup_timeout_seconds=float(
            getattr(
                processing,
                "opendataloader_hybrid_startup_timeout_seconds",
                DEFAULT_HYBRID_STARTUP_TIMEOUT_SECONDS,
            )
        ),
        force_ocr=bool(getattr(processing, "opendataloader_hybrid_force_ocr", False)),
        ocr_lang=getattr(processing, "opendataloader_hybrid_ocr_lang", None),
        enrich_formula=bool(getattr(processing, "opendataloader_hybrid_enrich_formula", False)),
        enrich_picture_description=bool(
            getattr(
                processing,
                "opendataloader_hybrid_enrich_picture_description",
                False,
            )
        ),
        picture_description_prompt=getattr(
            processing,
            "opendataloader_hybrid_picture_description_prompt",
            None,
        ),
        device=getattr(processing, "opendataloader_hybrid_device", DEFAULT_HYBRID_DEVICE),
        python_executable=getattr(
            processing,
            "opendataloader_hybrid_python_executable",
            None,
        ),
        managed_servers=build_managed_server_specs(
            getattr(processing, "opendataloader_hybrid_servers", None)
        ),
    )


def _normalize_server_url(server_url: str) -> str:
    """Normalize a hybrid server URL to include a scheme."""
    url = server_url.strip()
    if "://" not in url:
        url = f"http://{url}"
    return url.rstrip("/")


def _healthcheck_url(server_url: str) -> str:
    """Return the hybrid health endpoint for a server URL."""
    return f"{_normalize_server_url(server_url)}/health"


def hybrid_server_executable_for_python(
    python_executable: str | Path | None,
    *,
    allow_path_fallback: bool = True,
) -> str | None:
    """Return the hybrid server executable associated with a Python install."""
    if python_executable is None:
        return None

    python_path = Path(python_executable).resolve()
    script_dirs = [
        python_path.parent,
        python_path.parent / "Scripts",
        python_path.parent / "bin",
        python_path.parent.parent / "Scripts",
        python_path.parent.parent / "bin",
    ]
    executable_names = (
        "opendataloader-pdf-hybrid.exe",
        "opendataloader-pdf-hybrid",
    )
    seen: set[Path] = set()

    for script_dir in script_dirs:
        resolved_dir = script_dir.resolve()
        if resolved_dir in seen:
            continue
        seen.add(resolved_dir)
        for executable_name in executable_names:
            candidate = resolved_dir / executable_name
            if candidate.is_file():
                return str(candidate)

    if not allow_path_fallback:
        return None

    return shutil.which("opendataloader-pdf-hybrid") or shutil.which(
        "opendataloader-pdf-hybrid.exe"
    )


def _hybrid_server_executable() -> str | None:
    """Return the preferred hybrid server executable path, if installed.

    Prefer the launcher that lives alongside the current interpreter so
    autostart and preflight stay pinned to the active virtual environment
    instead of a different global Python installation that happens to be on
    ``PATH``.
    """
    return hybrid_server_executable_for_python(
        sys.executable,
        allow_path_fallback=True,
    )


def hybrid_extra_installed() -> bool:
    """Return True when optional local hybrid backend dependencies exist."""
    return all(find_spec(name) is not None for name in ("docling", "fastapi", "uvicorn"))


def _set_last_attempt_result(result: OpenDataLoaderAttemptResult) -> None:
    """Store the most recent OpenDataLoader attempt result for this thread."""
    _ATTEMPT_STATE.last_result = result


def get_last_attempt_result() -> OpenDataLoaderAttemptResult | None:
    """Return the most recent OpenDataLoader attempt result for this thread."""
    return getattr(_ATTEMPT_STATE, "last_result", None)


def _find_java() -> str | None:
    """Find a Java 11+ executable."""
    candidates: list[str] = []
    java_on_path = shutil.which("java")
    if java_on_path:
        candidates.append(java_on_path)

    import os

    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    ms_jdk_dir = Path(program_files) / "Microsoft"
    if ms_jdk_dir.is_dir():
        for jdk_dir in sorted(ms_jdk_dir.glob("jdk-*"), reverse=True):
            java_exe = jdk_dir / "bin" / "java.exe"
            if java_exe.is_file():
                candidates.append(str(java_exe))

    for java_path in candidates:
        version = _get_java_version(java_path)
        if version is not None and version >= _MIN_JAVA_VERSION:
            return java_path
    return None


def _get_java_version(java_path: str) -> int | None:
    """Parse the major Java version from `java -version` output."""
    try:
        result = subprocess.run(
            [java_path, "-version"],
            capture_output=True,
            text=True,
            timeout=10,
            encoding="utf-8",
            errors="replace",
        )
        output = result.stderr or result.stdout
        for line in output.splitlines():
            line = line.strip().strip('"')
            if "version" not in line.lower():
                continue
            parts = line.split('"')
            ver_str = parts[1] if len(parts) >= 2 else line.split()[-1]
            ver_parts = ver_str.split(".")
            major = int(ver_parts[0])
            if major == 1 and len(ver_parts) > 1:
                return int(ver_parts[1])
            return major
    except Exception:
        return None
    return None


_JAVA_PATH: str | None = _find_java() if OPENDATALOADER_AVAILABLE else None


def is_available() -> bool:
    """Return True when base OpenDataLoader extraction is usable."""
    return OPENDATALOADER_AVAILABLE and _JAVA_PATH is not None


def get_java_path() -> str | None:
    """Return the detected Java path."""
    return _JAVA_PATH


def is_hybrid_server_reachable(
    server_url: str | None = None,
    timeout_seconds: float = 1.0,
) -> bool:
    """Return True when the hybrid backend health check succeeds."""
    target = _healthcheck_url(server_url or f"{DEFAULT_HYBRID_HOST}:{DEFAULT_HYBRID_PORT}")
    try:
        with urlopen(target, timeout=timeout_seconds) as response:
            return 200 <= response.status < 300
    except (OSError, URLError, ValueError):
        return False


def _stop_hybrid_server() -> None:
    """Terminate any locally managed hybrid server."""
    global _HYBRID_SERVER_PROCESS, _HYBRID_SERVER_COMMAND
    if _HYBRID_SERVER_PROCESS is None:
        return
    if _HYBRID_SERVER_PROCESS.poll() is None:
        _HYBRID_SERVER_PROCESS.terminate()
        try:
            _HYBRID_SERVER_PROCESS.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _HYBRID_SERVER_PROCESS.kill()
    _HYBRID_SERVER_PROCESS = None
    _HYBRID_SERVER_COMMAND = None


atexit.register(_stop_hybrid_server)


def ensure_hybrid_server(
    config: OpenDataLoaderHybridConfig | None,
) -> OpenDataLoaderHybridConfig | None:
    """Ensure a hybrid backend is reachable, autostarting a local one if allowed."""
    global _HYBRID_SERVER_PROCESS, _HYBRID_SERVER_COMMAND

    if config is None or not config.enabled:
        return None

    matched_managed_server = config.matched_managed_server()
    resolved_url = config.resolved_server_url()
    if is_hybrid_server_reachable(resolved_url):
        return config
    if matched_managed_server is not None:
        logger.info(
            "Managed OpenDataLoader hybrid backend '%s' unavailable at %s",
            matched_managed_server.name,
            resolved_url,
        )
        return None
    if not config.autostart:
        return None

    executable = _hybrid_server_executable()
    if executable is None:
        logger.info("OpenDataLoader hybrid executable not installed; hybrid disabled")
        return None
    if not hybrid_extra_installed():
        logger.info("OpenDataLoader hybrid extra not installed; hybrid disabled")
        return None

    parsed = urlparse(resolved_url)
    if parsed.hostname not in {None, "127.0.0.1", "localhost"}:
        logger.info("Hybrid autostart only supports local backends")
        return None

    command = [
        executable,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--device",
        config.device,
    ]
    if config.force_ocr:
        command.append("--force-ocr")
    if config.ocr_lang:
        command.extend(["--ocr-lang", config.ocr_lang])
    if config.enrich_formula:
        command.append("--enrich-formula")
    if config.enrich_picture_description:
        command.append("--enrich-picture-description")
    if config.picture_description_prompt:
        command.extend(["--picture-description-prompt", config.picture_description_prompt])

    if (
        _HYBRID_SERVER_PROCESS is not None
        and _HYBRID_SERVER_PROCESS.poll() is None
        and _HYBRID_SERVER_COMMAND == tuple(command)
    ):
        deadline = time.perf_counter() + config.startup_timeout_seconds
        while time.perf_counter() < deadline:
            if is_hybrid_server_reachable(resolved_url):
                return config
            time.sleep(0.25)
        return None

    if _HYBRID_SERVER_PROCESS is not None and _HYBRID_SERVER_PROCESS.poll() is None:
        _stop_hybrid_server()

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    logger.info("Starting OpenDataLoader hybrid backend at %s", resolved_url)
    _HYBRID_SERVER_PROCESS = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        creationflags=creationflags,
    )
    _HYBRID_SERVER_COMMAND = tuple(command)

    deadline = time.perf_counter() + config.startup_timeout_seconds
    while time.perf_counter() < deadline:
        if _HYBRID_SERVER_PROCESS.poll() is not None:
            _stop_hybrid_server()
            return None
        if is_hybrid_server_reachable(resolved_url):
            return config
        time.sleep(0.25)

    logger.warning("OpenDataLoader hybrid backend failed to become healthy")
    _stop_hybrid_server()
    return None


def extract_with_opendataloader(
    pdf_path: Path,
    mode: str = "fast",
    use_struct_tree: bool = True,
    hybrid_config: OpenDataLoaderHybridConfig | None = None,
) -> str | None:
    """Extract text from PDF using OpenDataLoader."""

    def _fail(reason: str) -> None:
        _set_last_attempt_result(
            OpenDataLoaderAttemptResult(mode=mode, text=None, error=reason, word_count=0)
        )

    if not OPENDATALOADER_AVAILABLE:
        logger.debug("opendataloader-pdf not installed, skipping")
        _fail("opendataloader-pdf not installed")
        return None
    if _JAVA_PATH is None:
        logger.debug("Java 11+ not found, skipping OpenDataLoader")
        _fail("java 11+ not found")
        return None
    if not pdf_path.exists():
        logger.warning("PDF not found: %s", pdf_path)
        _fail("pdf not found")
        return None
    if mode == "hybrid" and hybrid_config is None:
        logger.debug("Hybrid mode requested without a hybrid backend config")
        _fail("hybrid requested without backend config")
        return None

    if mode == "hybrid" and hybrid_config is not None:
        logger.info(
            "OpenDataLoader extraction (hybrid): %s | endpoint=%s url=%s "
            "backend=%s client_mode=%s force_ocr=%s enrich_formula=%s "
            "enrich_picture_description=%s fallback_to_fast=%s timeout_ms=%s",
            pdf_path.name,
            _hybrid_endpoint_label(hybrid_config),
            hybrid_config.resolved_server_url(),
            hybrid_config.backend,
            hybrid_config.effective_client_mode(),
            hybrid_config.force_ocr,
            hybrid_config.enrich_formula,
            hybrid_config.enrich_picture_description,
            hybrid_config.fallback_to_fast,
            hybrid_config.timeout_ms,
        )
    else:
        logger.info("OpenDataLoader extraction (%s): %s", mode, pdf_path.name)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            java_dir = str(Path(_JAVA_PATH).parent)
            kwargs: dict[str, object] = {
                "input_path": [str(pdf_path)],
                "output_dir": tmpdir,
                "format": "markdown",
                "quiet": True,
                "image_output": "off",
                "use_struct_tree": use_struct_tree,
            }
            if mode == "hybrid" and hybrid_config is not None:
                kwargs["hybrid"] = hybrid_config.backend
                kwargs["hybrid_mode"] = hybrid_config.effective_client_mode()
                kwargs["hybrid_url"] = hybrid_config.resolved_server_url()
                if hybrid_config.timeout_ms > 0:
                    kwargs["hybrid_timeout"] = str(hybrid_config.timeout_ms)
                if hybrid_config.fallback_to_fast:
                    kwargs["hybrid_fallback"] = True

            old_path = os.environ.get("PATH", "")
            os.environ["PATH"] = java_dir + os.pathsep + old_path
            try:
                opendataloader_pdf.convert(**kwargs)
            finally:
                os.environ["PATH"] = old_path

            md_files = list(Path(tmpdir).rglob("*.md"))
            if not md_files:
                logger.debug("OpenDataLoader produced no markdown for %s", pdf_path.name)
                _fail("no markdown output produced")
                return None

            text = md_files[0].read_text(encoding="utf-8")
            if text and len(text.split()) > 100:
                word_count = len(text.split())
                _set_last_attempt_result(
                    OpenDataLoaderAttemptResult(
                        mode=mode,
                        text=text,
                        word_count=word_count,
                    )
                )
                logger.info(
                    "OpenDataLoader (%s): extracted %s words from %s",
                    mode,
                    word_count,
                    pdf_path.name,
                )
                return text

            logger.debug("OpenDataLoader: insufficient text from %s", pdf_path.name)
            _fail(f"insufficient text ({len(text.split()) if text else 0} words)")
            return None

    except FileNotFoundError as exc:
        if "java" in str(exc).lower():
            logger.warning("Java not found during OpenDataLoader extraction: %s", exc)
            _fail(f"java not found during extraction: {exc}")
        else:
            logger.warning("OpenDataLoader extraction failed for %s: %s", pdf_path.name, exc)
            _fail(f"file error: {exc}")
        return None
    except subprocess.CalledProcessError as exc:
        logger.warning(
            "OpenDataLoader CLI failed for %s: exit code %s",
            pdf_path.name,
            exc.returncode,
        )
        _fail(f"cli exit code {exc.returncode}")
        return None
    except Exception as exc:
        logger.warning("OpenDataLoader extraction failed for %s: %s", pdf_path.name, exc)
        _fail(str(exc))
        return None
