"""Cooperative pause control for long-running extraction runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Event, Lock
from typing import Any

from src.utils.file_utils import safe_write_json
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

CONTROL_FILENAME = "run_control.json"
PAUSE_ACTION = "pause"


class PauseRequested(BaseException):
    """Graceful pause request for the active run."""


@dataclass(frozen=True)
class RunControlRequest:
    """Persisted run-control request payload."""

    action: str
    requested_at: str
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the request payload."""
        payload = {
            "action": self.action,
            "requested_at": self.requested_at,
        }
        if self.reason:
            payload["reason"] = self.reason
        if self.details:
            payload["details"] = self.details
        return payload


def default_control_path(index_dir: Path) -> Path:
    """Return the cooperative run-control path for an index directory."""
    return Path(index_dir) / CONTROL_FILENAME


def read_control_request(control_path: Path | None) -> RunControlRequest | None:
    """Read a run-control request from disk."""
    if control_path is None:
        return None

    path = Path(control_path)
    if not path.exists():
        return None

    try:
        raw_text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None

    if not raw_text:
        return None

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        action = raw_text.lower()
        if action:
            return RunControlRequest(
                action=action,
                requested_at=datetime.now().isoformat(),
            )
        return None

    if not isinstance(payload, dict):
        return None

    action = str(payload.get("action", "")).strip().lower()
    if not action:
        return None

    requested_at = str(payload.get("requested_at") or datetime.now().isoformat())
    reason = payload.get("reason")
    details = payload.get("details")
    return RunControlRequest(
        action=action,
        requested_at=requested_at,
        reason=str(reason) if isinstance(reason, str) and reason.strip() else None,
        details=details if isinstance(details, dict) else {},
    )


def write_pause_request(
    control_path: Path,
    *,
    reason: str | None = None,
    details: dict[str, Any] | None = None,
) -> RunControlRequest:
    """Write a pause request for the active run."""
    request = RunControlRequest(
        action=PAUSE_ACTION,
        requested_at=datetime.now().isoformat(),
        reason=reason.strip() if isinstance(reason, str) and reason.strip() else None,
        details=details or {},
    )
    safe_write_json(Path(control_path), request.to_dict())
    logger.info("Wrote pause request to %s", control_path)
    return request


def clear_control_request(control_path: Path | None) -> bool:
    """Remove an existing run-control request."""
    if control_path is None:
        return False

    path = Path(control_path)
    if not path.exists():
        return False

    try:
        path.unlink()
    except OSError:
        return False
    logger.info("Cleared run-control request at %s", control_path)
    return True


class RunControlPoller:
    """Thread-safe poller for cooperative run-control requests."""

    def __init__(self, control_path: Path | None = None):
        self.control_path = Path(control_path) if control_path else None
        self._pause_requested = Event()
        self._lock = Lock()
        self._request: RunControlRequest | None = None

    @property
    def request(self) -> RunControlRequest | None:
        """Return the cached request when one has been observed."""
        return self._request

    def check_pause_requested(self) -> bool:
        """Return whether a pause request has been observed."""
        if self.control_path is None:
            return False
        if self._pause_requested.is_set():
            return True

        with self._lock:
            if self._pause_requested.is_set():
                return True
            request = read_control_request(self.control_path)
            if request and request.action == PAUSE_ACTION:
                self._request = request
                self._pause_requested.set()
                logger.warning(
                    "Pause requested via %s%s",
                    self.control_path,
                    f" ({request.reason})" if request.reason else "",
                )
                return True
        return False

    def raise_if_pause_requested(self, context: str | None = None) -> None:
        """Raise PauseRequested when a pause request has been observed."""
        if not self.check_pause_requested():
            return

        message = "Pause requested"
        if self._request and self._request.reason:
            message += f": {self._request.reason}"
        if context:
            message += f" [{context}]"
        raise PauseRequested(message)

    def clear(self) -> bool:
        """Clear the current request and cached pause state."""
        cleared = clear_control_request(self.control_path)
        with self._lock:
            self._request = None
            self._pause_requested.clear()
        return cleared
