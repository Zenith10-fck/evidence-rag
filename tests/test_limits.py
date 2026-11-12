"""UsageLimiter: per-minute window, daily quota, independence between callers. A fake clock is
used, so no test sleeps."""

from __future__ import annotations

from app import limits
from app.limits import DAY_S, UsageLimiter


class Clock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_per_minute_limit_refuses_and_does_not_count_the_refused_request():
    clock = Clock(1000.0)
    lim = UsageLimiter(per_minute=3, per_day=0, clock=clock)
    assert [lim.check("a").allowed for _ in range(3)] == [True, True, True]
    refused = lim.check("a")
    assert not refused.allowed and refused.reason == "per_minute"
    assert 1 <= refused.retry_after_s <= 60
    # Hammering while refused must not extend the penalty: only counted requests fill the window.
    for _ in range(50):
        lim.check("a")
    clock.t += 60
    assert lim.check("a").allowed


def test_window_slides_instead_of_resetting_all_at_once():
    clock = Clock(1000.0)
    lim = UsageLimiter(per_minute=3, per_day=0, clock=clock)
    for _ in range(3):  # t = 1000, 1010, 1020
        assert lim.check("a").allowed
        clock.t += 10
    # t = 1030: window is full, the oldest request (1000) leaves at 1060
    refused = lim.check("a")
    assert not refused.allowed and refused.retry_after_s == 30
    clock.t = 1060
    assert lim.check("a").allowed  # one slot freed, not three
    assert not lim.check("a").allowed


def test_daily_quota_resets_at_midnight_utc():
    midnight = 10 * DAY_S
    clock = Clock(midnight - 100)
    lim = UsageLimiter(per_minute=0, per_day=2, clock=clock)
    assert lim.check("a").allowed and lim.check("a").allowed
    refused = lim.check("a")
    assert not refused.allowed and refused.reason == "per_day"
    assert refused.retry_after_s == 100
    clock.t = midnight + 1
    assert lim.check("a").allowed


def test_callers_do_not_share_limits():
    lim = UsageLimiter(per_minute=1, per_day=0, clock=Clock(5.0))
    assert lim.check("acme").allowed
    assert not lim.check("acme").allowed
    assert lim.check("beta").allowed


def test_zero_means_no_limit_and_remaining_counts_down():
    unlimited = UsageLimiter(per_minute=0, per_day=0, clock=Clock())
    d = unlimited.check("a")
    assert d.allowed and d.remaining_minute is None and d.remaining_day is None
    assert all(unlimited.check("a").allowed for _ in range(500))

    lim = UsageLimiter(per_minute=5, per_day=10, clock=Clock())
    first, second = lim.check("a"), lim.check("a")
    assert (first.remaining_minute, first.remaining_day) == (4, 9)
    assert (second.remaining_minute, second.remaining_day) == (3, 8)


def test_memory_is_bounded_when_many_callers_appear(monkeypatch):
    monkeypatch.setattr(limits, "MAX_TRACKED", 20)
    lim = UsageLimiter(per_minute=5, per_day=0, clock=Clock(1.0))
    for i in range(200):  # e.g. many client IPs in demo mode
        lim.check(f"ip:{i}")
    assert len(lim._usage) <= 20
