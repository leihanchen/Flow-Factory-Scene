"""Lightweight wall-clock timer for per-stage training profiling.

All durations are reported in **milliseconds** (float).
"""
import time
from typing import Optional


class StepTimer:
    """Accumulates named timing intervals using ``time.perf_counter``.

    Usage::

        timer = StepTimer()
        with timer("rollout"):
            samples = self.sample()
        with timer("reward"):
            self.prepare_feedback(samples)
        timer.log_elapsed("optimize", optimize_ms)
        metrics = timer.collect()  # {"rollout_ms": 123.4, "reward_ms": 56.7, ...}
        timer.reset()
    """

    def __init__(self) -> None:
        self._records: dict[str, float] = {}
        self._start: Optional[float] = None
        self._current_key: Optional[str] = None

    def __call__(self, key: str) -> "StepTimer":
        """Start a new timed interval (closes any previous one)."""
        self._close()
        self._current_key = key
        self._start = time.perf_counter()
        return self

    def __enter__(self) -> "StepTimer":
        return self

    def __exit__(self, *exc) -> None:
        self._close()

    def _close(self) -> None:
        if self._start is not None and self._current_key is not None:
            elapsed_ms = (time.perf_counter() - self._start) * 1000.0
            self._records[self._current_key] = elapsed_ms
            self._start = None
            self._current_key = None

    def log_elapsed(self, key: str, ms: float) -> None:
        """Manually record an elapsed time for *key*."""
        self._records[key] = ms

    def collect(self) -> dict[str, float]:
        """Return all recorded timings with ``_ms`` suffix and reset."""
        self._close()
        result = {f"{k}_ms": v for k, v in self._records.items()}
        self._records.clear()
        return result

    def reset(self) -> None:
        """Discard all recorded timings."""
        self._records.clear()
        self._start = None
        self._current_key = None
