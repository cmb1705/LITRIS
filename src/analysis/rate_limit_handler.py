"""Rate limit handler for Claude CLI Max subscription limits."""

import time
from datetime import datetime, timedelta

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class RateLimitExceededError(Exception):
    """Rate limit exceeded and cannot continue."""

    pass


class RateLimitHandler:
    """Monitor and handle Max subscription rate limits.

    Max subscription rate limits:
    - Max 5x: 50-200 prompts per 5 hours
    - Max 20x: 200-800 prompts per 5 hours

    Rate limit windows reset every 5 hours from first request.
    """

    # Rate limit window in hours
    WINDOW_HOURS = 5

    def __init__(
        self,
        pause_on_limit: bool = True,
        auto_resume: bool = False,
        check_interval: int = 60,
        conservative_limit: int = 150,
    ):
        """Initialize rate limit handler.

        Args:
            pause_on_limit: Pause processing when limit hit (vs raise error).
            auto_resume: Automatically wait and resume after reset.
            check_interval: Seconds between status checks during wait.
            conservative_limit: Conservative request count before warning.
        """
        self.pause_on_limit = pause_on_limit
        self.auto_resume = auto_resume
        self.check_interval = check_interval
        self.conservative_limit = conservative_limit

        # Track session
        self._session_start: datetime | None = None
        self._request_count = 0
        self._limit_hit = False
        self._last_limit_time: datetime | None = None

    def start_session(self) -> None:
        """Start a new rate limit tracking session."""
        self._session_start = datetime.now()
        self._request_count = 0
        self._limit_hit = False
        logger.info(f"Started rate limit session at {self._session_start}")

    def record_request(self) -> None:
        """Record a successful request."""
        if not self._session_start:
            self.start_session()

        self._request_count += 1

        # Warn when approaching conservative limit
        if self._request_count == self.conservative_limit:
            logger.warning(
                f"Approaching rate limit: {self._request_count} requests in session. "
                "Consider pausing to avoid hitting limits."
            )

    def get_session_request_count(self) -> int:
        """Get number of requests in current session.

        Returns:
            Request count.
        """
        return self._request_count

    def check_response_for_limit(self, response: str) -> bool:
        """Detect rate limit indicators in CLI output.

        Args:
            response: CLI response text.

        Returns:
            True if rate limit was hit.
        """
        response_lower = response.lower()
        indicators = [
            "rate limit",
            "usage limit",
            "try again",
            "too many requests",
            "quota exceeded",
            "limit reached",
        ]
        return any(ind in response_lower for ind in indicators)

    def handle_limit_hit(self, save_progress_callback=None) -> bool:
        """Handle when rate limit is hit.

        Args:
            save_progress_callback: Optional callback to save current progress.

        Returns:
            True if processing can continue, False if should stop.

        Raises:
            RateLimitExceededError: If not pausing on limit.
        """
        self._limit_hit = True
        self._last_limit_time = datetime.now()

        logger.warning(
            f"Rate limit hit after {self._request_count} requests. "
            f"Session started at {self._session_start}"
        )

        # Save progress if callback provided
        if save_progress_callback:
            try:
                save_progress_callback()
                logger.info("Progress saved successfully")
            except Exception as e:
                logger.error(f"Failed to save progress: {e}")

        if self.auto_resume:
            # Wait for reset and continue
            reset_time = self.get_estimated_reset_time()
            wait_seconds = (reset_time - datetime.now()).total_seconds()

            if wait_seconds > 0:
                logger.info(
                    f"Auto-resume enabled. Waiting until {reset_time} "
                    f"({wait_seconds/60:.1f} minutes)"
                )
                self._wait_for_reset(wait_seconds)

            # Reset session and continue
            self.start_session()
            return True

        elif self.pause_on_limit:
            # Clean exit with resume instructions
            reset_time = self.get_estimated_reset_time()
            print("\n" + "=" * 60)
            print("RATE LIMIT REACHED")
            print("=" * 60)
            print(f"Requests this session: {self._request_count}")
            print(f"Estimated reset time: {reset_time}")
            print("\nTo resume, run the build command with --resume flag:")
            print("  python scripts/build_index.py --mode cli --resume")
            print("=" * 60 + "\n")

            return False

        else:
            raise RateLimitExceededError(
                f"Rate limit exceeded after {self._request_count} requests. "
                f"Reset expected at {self.get_estimated_reset_time()}"
            )

    def get_estimated_reset_time(self) -> datetime:
        """Estimate when rate limit will reset.

        Returns:
            Estimated reset datetime.
        """
        if self._session_start:
            return self._session_start + timedelta(hours=self.WINDOW_HOURS)
        return datetime.now() + timedelta(hours=self.WINDOW_HOURS)

    def get_time_until_reset(self) -> timedelta:
        """Get time remaining until rate limit reset.

        Returns:
            Time until reset.
        """
        reset_time = self.get_estimated_reset_time()
        remaining = reset_time - datetime.now()
        return max(remaining, timedelta(0))

    def is_approaching_limit(self, threshold: int | None = None) -> bool:
        """Check if approaching rate limit.

        Args:
            threshold: Custom threshold (defaults to conservative_limit).

        Returns:
            True if approaching limit.
        """
        threshold = threshold or self.conservative_limit
        return self._request_count >= threshold

    def _wait_for_reset(self, seconds: float) -> None:
        """Wait for rate limit reset with periodic status updates.

        Args:
            seconds: Total seconds to wait.
        """
        end_time = time.time() + seconds
        while time.time() < end_time:
            remaining = end_time - time.time()
            if remaining <= 0:
                break

            # Status update
            minutes_left = remaining / 60
            if minutes_left > 1:
                logger.info(f"Waiting for rate limit reset: {minutes_left:.1f} minutes remaining")
            else:
                logger.info(f"Waiting for rate limit reset: {remaining:.0f} seconds remaining")

            # Sleep for check interval or remaining time, whichever is smaller
            sleep_time = min(self.check_interval, remaining)
            time.sleep(sleep_time)

        logger.info("Rate limit reset complete")

    def get_session_stats(self) -> dict:
        """Get current session statistics.

        Returns:
            Dict with session statistics.
        """
        return {
            "session_start": self._session_start.isoformat() if self._session_start else None,
            "request_count": self._request_count,
            "limit_hit": self._limit_hit,
            "last_limit_time": self._last_limit_time.isoformat() if self._last_limit_time else None,
            "estimated_reset": self.get_estimated_reset_time().isoformat(),
            "approaching_limit": self.is_approaching_limit(),
        }
