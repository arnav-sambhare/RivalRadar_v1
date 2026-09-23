"""Groq rate-limit handling (no network)."""

import pytest

from rivalradar import llm


def test_parse_reset_durations():
    assert llm.parse_reset("1.17s") == 1.17
    assert round(llm.parse_reset("1m26.4s"), 2) == 86.4
    assert llm.parse_reset("250ms") == 0.25
    assert llm.parse_reset(None) == 0.0


def _budget(monkeypatch, now=100.0):
    sleeps = []
    monkeypatch.setattr(llm.time, "monotonic", lambda: now)
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    return llm.RateBudget(), sleeps


def test_waits_for_token_budget_to_reset(monkeypatch):
    budget, sleeps = _budget(monkeypatch)
    budget.update({"x-ratelimit-remaining-tokens": "500", "x-ratelimit-reset-tokens": "12s",
                   "x-ratelimit-remaining-requests": "900", "x-ratelimit-reset-requests": "1m"})
    budget.wait(needed_tokens=3000, interval=2)
    assert sleeps and sleeps[0] >= 12  # not enough tokens left: waited for the reset


def test_only_spaces_calls_when_budget_is_fine(monkeypatch):
    budget, sleeps = _budget(monkeypatch)
    budget.update({"x-ratelimit-remaining-tokens": "7800", "x-ratelimit-reset-tokens": "1s"})
    budget.wait(needed_tokens=3000, interval=2)
    assert sleeps == [2]


def test_stops_when_daily_requests_are_used_up(monkeypatch):
    budget, _ = _budget(monkeypatch)
    budget.update({"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "5h"})
    with pytest.raises(llm.LLMUnavailable, match="quota"):
        budget.wait(needed_tokens=10, interval=2)


def test_estimate_tokens_is_conservative():
    assert llm.estimate_tokens("a" * 400) >= 100


def test_waits_for_output_token_budget(monkeypatch):
    clock = {"now": 100.0}
    sleeps = []
    monkeypatch.setattr(llm.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    budget = llm.RateBudget()
    budget.record_output(600)            # a reply at t=100
    clock["now"] = 110.0
    budget.wait(needed_tokens=10, interval=0, max_output=600, output_per_minute=1000)
    assert sleeps and 50 <= sleeps[0] <= 51  # 600 + 600 > 1000: wait until t=160


def test_output_budget_with_room_does_not_wait(monkeypatch):
    clock = {"now": 100.0}
    sleeps = []
    monkeypatch.setattr(llm.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    budget = llm.RateBudget()
    budget.record_output(200)
    budget.wait(needed_tokens=10, interval=0, max_output=600, output_per_minute=1000)
    assert sleeps == []

