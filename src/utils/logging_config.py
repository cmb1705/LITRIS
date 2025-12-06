"""Logging configuration for the Literature Review system."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def setup_logging(
    level: LogLevel = "INFO",
    log_dir: Path | None = None,
    log_file: str | None = None,
    console: bool = True,
    file_logging: bool = True,
) -> logging.Logger:
    """Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Directory for log files. Defaults to data/logs/.
        log_file: Log filename. Defaults to lit_review_YYYYMMDD.log.
        console: Whether to log to console.
        file_logging: Whether to log to file.

    Returns:
        Configured root logger.
    """
    # Determine log directory
    if log_dir is None:
        log_dir = _find_project_root() / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Determine log filename
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = f"lit_review_{timestamp}.log"

    # Get root logger for our application
    logger = logging.getLogger("lit_review")
    logger.setLevel(getattr(logging, level))

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatters
    console_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level))
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler
    if file_logging:
        file_path = log_dir / log_file
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module.

    Args:
        name: Module name (typically __name__).

    Returns:
        Logger instance.
    """
    # Create child logger under lit_review namespace
    if name.startswith("src."):
        name = name[4:]  # Remove 'src.' prefix
    return logging.getLogger(f"lit_review.{name}")


def _find_project_root() -> Path:
    """Find the project root directory.

    Returns:
        Path to project root (directory containing config.yaml).
    """
    current = Path.cwd()

    # Check current directory and up to 3 parent levels
    for _ in range(4):
        if (current / "config.yaml").exists():
            return current
        if current.parent == current:
            break
        current = current.parent

    # Default to current directory
    return Path.cwd()


class LogContext:
    """Context manager for logging operations with timing."""

    def __init__(self, logger: logging.Logger, operation: str, level: LogLevel = "INFO"):
        """Initialize log context.

        Args:
            logger: Logger instance to use.
            operation: Description of the operation being performed.
            level: Log level for start/end messages.
        """
        self.logger = logger
        self.operation = operation
        self.level = getattr(logging, level)
        self.start_time: datetime | None = None

    def __enter__(self) -> "LogContext":
        """Log operation start."""
        self.start_time = datetime.now()
        self.logger.log(self.level, f"Starting: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Log operation completion with duration."""
        duration = datetime.now() - self.start_time
        duration_str = f"{duration.total_seconds():.2f}s"

        if exc_type is not None:
            self.logger.error(
                f"Failed: {self.operation} after {duration_str} - {exc_type.__name__}: {exc_val}"
            )
        else:
            self.logger.log(self.level, f"Completed: {self.operation} in {duration_str}")

        return False  # Don't suppress exceptions
