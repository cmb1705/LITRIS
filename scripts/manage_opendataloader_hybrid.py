#!/usr/bin/env python
"""Manage the fixed OpenDataLoader hybrid GPU server pool."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config  # noqa: E402
from src.extraction.opendataloader_extractor import (  # noqa: E402
    DEFAULT_MANAGED_HYBRID_STARTUP_TIMEOUT_SECONDS,
    build_managed_server_specs,
    hybrid_server_executable_for_python,
    is_hybrid_server_reachable,
)

STATE_FILENAME = "opendataloader_hybrid_pool_state.json"


@dataclass(frozen=True)
class ManagedLaunchTarget:
    """Launch metadata for one managed hybrid backend."""

    name: str
    url: str
    command: tuple[str, ...]
    log_path: Path


def runtime_log_dir(project_root: Path = PROJECT_ROOT) -> Path:
    """Return the runtime log directory for managed hybrid services."""
    return project_root / "data" / "logs"


def state_path(project_root: Path = PROJECT_ROOT) -> Path:
    """Return the persisted state path for managed hybrid services."""
    return runtime_log_dir(project_root) / STATE_FILENAME


def load_state(project_root: Path = PROJECT_ROOT) -> dict[str, dict]:
    """Load the persisted process state."""
    path = state_path(project_root)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    servers = data.get("servers")
    return servers if isinstance(servers, dict) else {}


def save_state(state: dict[str, dict], project_root: Path = PROJECT_ROOT) -> None:
    """Persist the managed process state."""
    log_dir = runtime_log_dir(project_root)
    log_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "servers": state,
    }
    state_path(project_root).write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_config(config_path: Path | None = None) -> Config:
    """Load the active project config."""
    return Config.load(config_path or (PROJECT_ROOT / "config.yaml"))


def _parse_host_port(url: str) -> tuple[str, int]:
    """Parse a managed hybrid server URL into host and port."""
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return host, port


def build_server_commands(
    processing,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, ManagedLaunchTarget]:
    """Build launch commands for every configured managed hybrid endpoint."""
    specs = build_managed_server_specs(getattr(processing, "opendataloader_hybrid_servers", None))
    if not specs:
        return {}
    python_executable = getattr(
        processing,
        "opendataloader_hybrid_python_executable",
        None,
    )
    executable = hybrid_server_executable_for_python(
        python_executable,
        allow_path_fallback=False,
    )
    if executable is None:
        raise FileNotFoundError(
            f"opendataloader-pdf-hybrid executable not found next to {python_executable!r}"
        )

    targets: dict[str, ManagedLaunchTarget] = {}
    for spec in specs:
        host, port = _parse_host_port(spec.url)
        command = [
            executable,
            "--host",
            host,
            "--port",
            str(port),
            "--device",
            str(getattr(processing, "opendataloader_hybrid_device", "cuda")),
        ]
        if spec.force_ocr:
            command.append("--force-ocr")
            ocr_lang = getattr(processing, "opendataloader_hybrid_ocr_lang", None)
            if ocr_lang:
                command.extend(["--ocr-lang", str(ocr_lang)])
        if spec.enrich_formula:
            command.append("--enrich-formula")
        if spec.enrich_picture_description:
            command.append("--enrich-picture-description")
            picture_prompt = getattr(
                processing,
                "opendataloader_hybrid_picture_description_prompt",
                None,
            )
            if picture_prompt:
                command.extend(["--picture-description-prompt", str(picture_prompt)])

        targets[spec.name] = ManagedLaunchTarget(
            name=spec.name,
            url=spec.url,
            command=tuple(command),
            log_path=runtime_log_dir(project_root) / f"opendataloader_hybrid_{spec.name}.log",
        )

    return targets


def _pid_running(pid: int | None) -> bool:
    """Return whether a PID is currently running."""
    if pid is None or pid <= 0:
        return False
    result = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return str(pid) in result.stdout


def _terminate_pid(pid: int) -> None:
    """Terminate a managed server PID."""
    result = subprocess.run(
        ["taskkill", "/PID", str(pid), "/T"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )


def _print_status_line(
    target: ManagedLaunchTarget,
    *,
    pid: int | None,
    managed: bool,
    healthy: bool,
) -> None:
    """Print one human-readable status line."""
    state = "up" if healthy else "down"
    owner = "managed" if managed else "untracked"
    pid_label = str(pid) if pid is not None else "-"
    print(f"{target.name:20} {state:4} pid={pid_label:>6} owner={owner:8} url={target.url}")


def command_status(config: Config, *, project_root: Path = PROJECT_ROOT) -> int:
    """Show health and process state for the configured server pool."""
    targets = build_server_commands(config.processing, project_root=project_root)
    state = load_state(project_root)
    all_healthy = True

    for name, target in targets.items():
        entry = state.get(name, {})
        pid = int(entry["pid"]) if "pid" in entry and entry["pid"] is not None else None
        managed = _pid_running(pid)
        healthy = is_hybrid_server_reachable(target.url, timeout_seconds=1.5)
        _print_status_line(target, pid=pid, managed=managed, healthy=healthy)
        all_healthy = all_healthy and healthy

    return 0 if all_healthy else 1


def command_stop(config: Config, *, project_root: Path = PROJECT_ROOT) -> int:
    """Stop any managed hybrid servers recorded in state."""
    _ = config
    state = load_state(project_root)
    if not state:
        print("No managed OpenDataLoader hybrid servers recorded.")
        return 0

    for name, entry in sorted(state.items()):
        pid = int(entry["pid"]) if "pid" in entry and entry["pid"] is not None else None
        if _pid_running(pid):
            print(f"Stopping {name} (pid={pid})")
            _terminate_pid(pid)
        else:
            print(f"{name} is not running (pid={pid})")

    save_state({}, project_root)
    return 0


def resolve_startup_timeout(processing, timeout_seconds: float | None = None) -> float:
    """Resolve managed hybrid startup timeout."""
    if timeout_seconds is not None:
        return timeout_seconds
    return max(
        float(processing.opendataloader_hybrid_startup_timeout_seconds),
        DEFAULT_MANAGED_HYBRID_STARTUP_TIMEOUT_SECONDS,
    )


def command_start(
    config: Config,
    *,
    project_root: Path = PROJECT_ROOT,
    timeout_seconds: float | None = None,
) -> int:
    """Start the managed hybrid server pool."""
    targets = build_server_commands(config.processing, project_root=project_root)
    state = load_state(project_root)
    failure = False
    timeout = resolve_startup_timeout(config.processing, timeout_seconds)
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    for name, target in targets.items():
        entry = state.get(name, {})
        pid = int(entry["pid"]) if "pid" in entry and entry["pid"] is not None else None
        if _pid_running(pid) and is_hybrid_server_reachable(target.url, timeout_seconds=1.5):
            print(f"{name} already running on {target.url} (pid={pid})")
            continue
        if is_hybrid_server_reachable(target.url, timeout_seconds=1.5):
            print(f"{name} already healthy on {target.url}; leaving external process in place")
            state[name] = {
                "url": target.url,
                "pid": None,
                "managed": False,
                "command": list(target.command),
                "log_path": str(target.log_path),
            }
            continue

        target.log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Starting {name} on {target.url}")
        with target.log_path.open("a", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                list(target.command),
                stdout=log_handle,
                stderr=log_handle,
                cwd=project_root,
                creationflags=creationflags,
            )

        deadline = time.perf_counter() + timeout
        started = False
        while time.perf_counter() < deadline:
            if process.poll() is not None:
                break
            if is_hybrid_server_reachable(target.url, timeout_seconds=1.5):
                started = True
                break
            time.sleep(0.25)

        if not started:
            failure = True
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            print(f"Failed to start {name} on {target.url}")
            continue

        state[name] = {
            "url": target.url,
            "pid": process.pid,
            "managed": True,
            "command": list(target.command),
            "log_path": str(target.log_path),
        }
        print(f"{name} healthy on {target.url} (pid={process.pid})")

    save_state(state, project_root)
    return 1 if failure else 0


def command_restart(
    config: Config,
    *,
    project_root: Path = PROJECT_ROOT,
    timeout_seconds: float | None = None,
) -> int:
    """Restart the managed hybrid server pool."""
    stop_rc = command_stop(config, project_root=project_root)
    start_rc = command_start(
        config,
        project_root=project_root,
        timeout_seconds=timeout_seconds,
    )
    return max(stop_rc, start_rc)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Manage the fixed OpenDataLoader hybrid GPU server pool",
    )
    parser.add_argument(
        "command",
        choices=("start", "stop", "status", "restart"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Path to config.yaml (default: ./config.yaml)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Override startup timeout seconds for start/restart",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    args = build_parser().parse_args(argv)
    config = load_config(args.config)

    if args.command == "start":
        return command_start(
            config,
            project_root=PROJECT_ROOT,
            timeout_seconds=args.timeout,
        )
    if args.command == "stop":
        return command_stop(config, project_root=PROJECT_ROOT)
    if args.command == "restart":
        return command_restart(
            config,
            project_root=PROJECT_ROOT,
            timeout_seconds=args.timeout,
        )
    return command_status(config, project_root=PROJECT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
