import pytest

from app.ratelimit import (
    RateLimiter,
    call_with_rate_limit_retry,
    estimate_tokens,
    token_batches,
)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_requests_per_minute_are_respected():
    clock = FakeClock()
    lim = RateLimiter(rpm=3, clock=clock, sleep=clock.sleep)
    waits = [lim.acquire(1) for _ in range(4)]
    assert waits[:3] == [0.0, 0.0, 0.0] and waits[3] >= 60.0


def test_tokens_per_minute_are_respected():
    clock = FakeClock()
    lim = RateLimiter(tpm=1000, clock=clock, sleep=clock.sleep)
    assert lim.acquire(800) == 0.0
    assert lim.acquire(300) >= 60.0


def test_a_single_request_larger_than_the_limit_still_goes_through_alone():
    clock = FakeClock()
    lim = RateLimiter(tpm=100, clock=clock, sleep=clock.sleep)
    assert lim.acquire(500) == 0.0


def test_disabled_limiter_never_waits():
    assert RateLimiter().acquire(10**9) == 0.0


def test_token_batches_split_by_budget():
    texts = ["x" * 400] * 5  # ~102 tokens each
    batches = token_batches(texts, max_tokens=250, max_items=128)
    assert [len(b) for b in batches] == [2, 2, 1]
    assert sum(estimate_tokens([texts[i]]) for i in batches[0]) <= 250


class Limited(Exception):
    pass


def test_retry_waits_after_rate_limit_then_succeeds():
    calls, slept = [], []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise Limited()
        return "ok"

    assert (
        call_with_rate_limit_retry(fn, lambda e: isinstance(e, Limited), sleep=slept.append) == "ok"
    )
    assert len(calls) == 3 and slept == [65.0, 65.0]


def test_other_errors_are_not_retried():
    def fn():
        raise ValueError("bad")

    with pytest.raises(ValueError):
        call_with_rate_limit_retry(fn, lambda e: isinstance(e, Limited), sleep=lambda s: None)
