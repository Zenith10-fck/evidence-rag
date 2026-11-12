"""Per-caller usage limits for the routes that cost money (/ask, /retrieve).

Two limits, both enforced in code before any paid API call:
- per minute: a sliding 60-second window (stops a runaway loop or a leaked key burning money fast),
- per day: a quota that resets at 00:00 UTC (caps what one caller can spend in a day).

A request over a limit is refused with HTTP 429 and a Retry-After header; it is never queued.

State is in memory and per process. One uvicorn worker on one host enforces exactly; with several
workers or hosts each keeps its own count, so put the limit in an API gateway or Redis instead.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

WINDOW_S = 60.0
DAY_S = 86_400
MAX_TRACKED = 10_000  # demo mode tracks client IPs; old entries are dropped past this many


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str = ""  # "per_minute" | "per_day" when refused
    retry_after_s: int = 0
    remaining_minute: int | None = None  # None = no limit
    remaining_day: int | None = None


@dataclass
class _Usage:
    recent: deque
    day: int
    day_count: int


class UsageLimiter:
    def __init__(self, per_minute: int, per_day: int, clock=time.time):
        self.per_minute = per_minute
        self.per_day = per_day
        self._clock = clock
        self._usage: dict[str, _Usage] = {}
        self._lock = threading.Lock()

    def check(self, caller: str) -> Decision:
        """Count one request for `caller` if it fits; refuse it (and do not count it) if not."""
        now = self._clock()
        today = int(now // DAY_S)
        with self._lock:
            u = self._usage.get(caller)
            if u is None:
                if len(self._usage) >= MAX_TRACKED:
                    self._prune(now, today)
                u = self._usage[caller] = _Usage(deque(), today, 0)
            if u.day != today:
                u.day, u.day_count = today, 0
            while u.recent and now - u.recent[0] >= WINDOW_S:
                u.recent.popleft()

            if self.per_day and u.day_count >= self.per_day:
                wait = (today + 1) * DAY_S - now
                return Decision(False, "per_day", _ceil(wait), *self._remaining(u))
            if self.per_minute and len(u.recent) >= self.per_minute:
                wait = WINDOW_S - (now - u.recent[0])
                return Decision(False, "per_minute", _ceil(wait), *self._remaining(u))

            u.recent.append(now)
            u.day_count += 1
            return Decision(True, "", 0, *self._remaining(u))

    def _remaining(self, u: _Usage) -> tuple[int | None, int | None]:
        minute = max(self.per_minute - len(u.recent), 0) if self.per_minute else None
        day = max(self.per_day - u.day_count, 0) if self.per_day else None
        return minute, day

    def _prune(self, now: float, today: int) -> None:
        idle = [
            k
            for k, u in self._usage.items()
            if (not u.recent or now - u.recent[-1] >= WINDOW_S) and u.day != today
        ]
        for k in idle or list(self._usage)[: MAX_TRACKED // 10]:
            del self._usage[k]


def _ceil(seconds: float) -> int:
    return max(1, int(seconds) + (seconds % 1 > 0))
