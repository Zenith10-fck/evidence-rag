"""A small client-side rate limiter (requests and tokens per minute, sliding 60 s window).

Used for the Voyage API: accounts without a payment method get 3 requests and 10K tokens per
minute. Setting VOYAGE_RPM / VOYAGE_TPM keeps every call under those limits instead of failing.
Token counts are estimated from characters (about 4 per token); pass a TPM a little under the
real limit (e.g. 9000 for 10K) to keep a margin.
"""

from __future__ import annotations

import threading
import time
from collections import deque


def estimate_tokens(texts: list[str]) -> int:
    return sum(len(t) // 4 + 2 for t in texts)  # ~4 characters per token, plus a small margin


class RateLimiter:
    def __init__(
        self,
        rpm: int = 0,
        tpm: int = 0,
        window_s: float = 60.0,
        clock=time.monotonic,
        sleep=time.sleep,
    ):
        self.rpm = rpm
        self.tpm = tpm
        self.window = window_s
        self._clock = clock
        self._sleep = sleep
        self._events: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.rpm or self.tpm)

    def acquire(self, tokens: int) -> float:
        """Block until a request of `tokens` fits. Returns the seconds waited."""
        if not self.enabled:
            return 0.0
        waited = 0.0
        with self._lock:
            while True:
                now = self._clock()
                while self._events and now - self._events[0][0] >= self.window:
                    self._events.popleft()
                used_req = len(self._events)
                used_tok = sum(t for _, t in self._events)
                fits_req = not self.rpm or used_req < self.rpm
                fits_tok = not self.tpm or used_tok + tokens <= self.tpm or not self._events
                if fits_req and fits_tok:
                    self._events.append((now, tokens))
                    return waited
                pause = self.window - (now - self._events[0][0]) + 0.5
                self._sleep(pause)
                waited += pause


def token_batches(texts: list[str], max_tokens: int, max_items: int) -> list[list[int]]:
    """Split indices of `texts` into batches under a token budget (each batch >= 1 item)."""
    batches: list[list[int]] = []
    current: list[int] = []
    used = 0
    for i, t in enumerate(texts):
        cost = estimate_tokens([t])
        if current and (used + cost > max_tokens or len(current) >= max_items):
            batches.append(current)
            current, used = [], 0
        current.append(i)
        used += cost
    if current:
        batches.append(current)
    return batches


def call_with_rate_limit_retry(
    fn, is_rate_limit, attempts: int = 6, wait_s: float = 65.0, sleep=time.sleep
):
    """Call `fn()`. On a rate-limit error, wait a full window and try again (up to `attempts`).
    Any other error is raised at once."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if not is_rate_limit(exc) or attempt == attempts - 1:
                raise
            sleep(wait_s)
    raise AssertionError("unreachable")
