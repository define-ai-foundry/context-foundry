# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

"""Tests for context_foundry.fusion.sources.paced.

RealtimeReplaySource sleeps real wall-clock time between events, so every test
replaces `paced.time.monotonic`/`paced.time.sleep` with a fake, controllable
clock (module-level monkeypatch, exactly where the source looks them up) whose
sleep() advances its own clock instead of blocking. Nothing in this file ever
sleeps for real.
"""

from datetime import datetime, timedelta, timezone

import pytest

from context_foundry.fusion.sources import paced
from context_foundry.fusion.sources.paced import RealtimeReplaySource

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _FakeClock:
    """Controllable stand-in for the `time` module: sleep() advances `now`
    instead of blocking, and never allows a negative sleep (real time.sleep
    would raise ValueError, so a buggy caller is caught the same way here)."""

    def __init__(self, start=0.0):
        self.now = start
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        if seconds < 0:
            raise ValueError("sleep length must be non-negative")
        self.sleeps.append(seconds)
        self.now += seconds


class _FakeInnerSource:
    """Fake wrapped source: yields the given events on every iter_events()
    call and records reset() invocations."""

    def __init__(self, events):
        self._events = events
        self.reset_calls = 0

    def iter_events(self):
        yield from self._events

    def reset(self):
        self.reset_calls += 1


def _install_clock(monkeypatch, start=0.0):
    clock = _FakeClock(start=start)
    monkeypatch.setattr(paced.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(paced.time, "sleep", clock.sleep)
    return clock


def test_first_event_is_the_anchor_and_never_sleeps(monkeypatch):
    clock = _install_clock(monkeypatch, start=42.0)  # nonzero start: anchoring is relative
    inner = _FakeInnerSource([(T0, ["d0"])])
    source = RealtimeReplaySource(inner, factor=1.0)

    events = list(source.iter_events())

    assert clock.sleeps == []
    assert events == [(T0, ["d0"])]


def test_anchors_to_scenario_start_not_previous_event(monkeypatch):
    """The most important test in this file.

    Sleeps must be computed from the FIRST event's scenario time and wall
    time, never re-derived from the event immediately before. Event0 anchors
    at wall=0. Fusion then "takes" 2.5s to process it (simulated by advancing
    the fake clock directly, not via sleep) before event1 (t0+2s) is due: that
    is 0.5s behind schedule, under the warning threshold, so no sleep happens
    and the wall clock is untouched by this source. Event2 (t0+4s) is then
    requested immediately.

    Anchored on the scenario start (correct): offset is 4s from t0, wall is
    still 2.5, so the required sleep is 4 - 2.5 = 1.5s.

    Anchored on the previous event instead (bug): offset would be 2s from t1,
    wall would be re-based at 2.5 (when event1 was handled), so the required
    sleep would be 2 - 0 = 2.0s -- a different, larger number. This test
    would fail with 2.0 if the implementation regressed to that scheme.
    """
    clock = _install_clock(monkeypatch)
    t0 = T0
    t1 = T0 + timedelta(seconds=2)
    t2 = T0 + timedelta(seconds=4)
    inner = _FakeInnerSource([(t0, ["d0"]), (t1, ["d1"]), (t2, ["d2"])])
    source = RealtimeReplaySource(inner, factor=1.0)

    gen = source.iter_events()
    next(gen)  # event0: anchors, no sleep
    assert clock.sleeps == []

    clock.now += 2.5  # simulated processing time between event0 and event1
    next(gen)  # event1: 0.5s behind schedule, under LAG_WARN_SECONDS -> no sleep
    assert clock.sleeps == []

    next(gen)  # event2: requested immediately, no further simulated work
    assert clock.sleeps == [1.5]


def test_factor_scales_sleep_duration_inversely(monkeypatch):
    t0 = T0
    t1 = T0 + timedelta(seconds=10)

    clock_1x = _install_clock(monkeypatch)
    list(
        RealtimeReplaySource(
            _FakeInnerSource([(t0, ["d0"]), (t1, ["d1"])]), factor=1.0
        ).iter_events()
    )
    assert clock_1x.sleeps == [10.0]

    clock_5x = _install_clock(monkeypatch)
    list(
        RealtimeReplaySource(
            _FakeInnerSource([(t0, ["d0"]), (t1, ["d1"])]), factor=5.0
        ).iter_events()
    )
    assert clock_5x.sleeps == [2.0]

    assert clock_5x.sleeps[0] == pytest.approx(clock_1x.sleeps[0] / 5)


def test_backwards_timestamp_never_sleeps_negative(monkeypatch):
    clock = _install_clock(monkeypatch)
    t0 = T0
    t1 = T0 + timedelta(seconds=2)
    t2 = T0 + timedelta(seconds=1)  # earlier than t1: scenario clock moves backwards
    inner = _FakeInnerSource([(t0, ["d0"]), (t1, ["d1"]), (t2, ["d2"])])
    source = RealtimeReplaySource(inner, factor=1.0)

    list(source.iter_events())

    # Only the forward gap (event1) produced a sleep; the backwards step
    # (event2) is already behind schedule by construction, so it is handled
    # by the lag-reporting branch rather than by sleeping a negative amount.
    assert clock.sleeps == [2.0]


def test_persistent_lag_logs_exactly_one_warning_and_never_sleeps(monkeypatch, caplog):
    clock = _install_clock(monkeypatch)
    events = [(T0 + timedelta(seconds=i), [f"d{i}"]) for i in range(4)]
    inner = _FakeInnerSource(events)
    source = RealtimeReplaySource(inner, factor=1.0)

    gen = source.iter_events()
    next(gen)  # event0: anchors at wall=0, no sleep

    clock.now += 100  # fusion falls miles behind the scenario clock

    with caplog.at_level("WARNING"):
        list(gen)  # drains events 1..3, all still far behind schedule

    assert clock.sleeps == []  # never sleeps while behind schedule
    warnings = [r for r in caplog.records if "behind the scenario clock" in r.message]
    assert len(warnings) == 1  # latched after the first report, not one per event


def test_reset_delegates_to_inner_and_second_pass_reanchors(monkeypatch):
    clock = _install_clock(monkeypatch)
    t0 = T0
    t1 = T0 + timedelta(seconds=3)
    inner = _FakeInnerSource([(t0, ["d0"]), (t1, ["d1"])])
    source = RealtimeReplaySource(inner, factor=1.0)

    list(source.iter_events())  # pass 1
    assert clock.sleeps == [3.0]

    clock.now += 500  # e.g. a --loop-delay gap between replay passes
    source.reset()
    assert inner.reset_calls == 1

    list(source.iter_events())  # pass 2: must re-anchor to ITS OWN first event
    assert clock.sleeps == [3.0, 3.0]


def test_factor_zero_or_negative_raises_value_error():
    inner = _FakeInnerSource([])
    with pytest.raises(ValueError):
        RealtimeReplaySource(inner, factor=0)
    with pytest.raises(ValueError):
        RealtimeReplaySource(inner, factor=-1.0)
