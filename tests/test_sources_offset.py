# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

"""Tests for context_foundry.fusion.sources.offset.

OffsetReplaySource reads the wall clock to pick its offset, so every test
replaces `offset.datetime` with a controllable stand-in. That keeps the expected
timestamps exact rather than approximate.
"""

from datetime import datetime, timedelta, timezone

from context_foundry.fusion.sources import offset as offset_mod
from context_foundry.fusion.sources.offset import OffsetReplaySource

# Scenario clock: deliberately far from any plausible "now", which is the whole
# problem this source exists to solve.
T0 = datetime(2026, 11, 15, 2, 45, tzinfo=timezone.utc)
NOW = datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc)


class _FakeDetection:
    """Stands in for a Stone Soup Detection: only .timestamp matters here."""

    def __init__(self, timestamp):
        self.timestamp = timestamp


class _FakeInnerSource:
    """Yields freshly-built events on every iter_events() call, so each pass
    gets its own Detection objects exactly as a re-read file would."""

    def __init__(self, offsets_seconds):
        self._offsets = offsets_seconds
        self.reset_calls = 0

    def iter_events(self):
        for seconds in self._offsets:
            stamp = T0 + timedelta(seconds=seconds)
            yield stamp, [_FakeDetection(stamp)]

    def reset(self):
        self.reset_calls += 1


class _FakeDatetime:
    """Controllable `datetime` replacement whose now() returns queued values."""

    def __init__(self, *moments):
        self._moments = list(moments)
        self.calls = 0

    def now(self, tz=None):
        self.calls += 1
        moment = self._moments[min(self.calls - 1, len(self._moments) - 1)]
        return moment if tz is None else moment.astimezone(tz)


def _install_now(monkeypatch, *moments):
    fake = _FakeDatetime(*moments)
    monkeypatch.setattr(offset_mod, "datetime", fake)
    return fake


def test_first_event_lands_exactly_at_now(monkeypatch):
    _install_now(monkeypatch, NOW)
    source = OffsetReplaySource(_FakeInnerSource([0]))

    ((stamp, _),) = list(source.iter_events())

    assert stamp == NOW


def test_later_events_keep_their_original_spacing(monkeypatch):
    """The offset is one constant, so intervals -- and the dt the tracker's
    motion model depends on -- survive untouched."""
    _install_now(monkeypatch, NOW)
    source = OffsetReplaySource(_FakeInnerSource([0, 18, 41.5]))

    stamps = [stamp for stamp, _ in source.iter_events()]

    assert stamps == [NOW, NOW + timedelta(seconds=18), NOW + timedelta(seconds=41.5)]


def test_detection_timestamps_move_with_the_event(monkeypatch):
    """The regression that matters most.

    The tracker keys a track's state off detection.timestamp and the CLI only
    broadcasts when that equals the event timestamp. Shift one without the other
    and the pod goes silent with no error, so assert they stay equal.
    """
    _install_now(monkeypatch, NOW)
    source = OffsetReplaySource(_FakeInnerSource([0, 18]))

    for stamp, detections in source.iter_events():
        assert [d.timestamp for d in detections] == [stamp]


def test_offset_is_taken_once_per_iteration(monkeypatch):
    """One clock read per pass: a mid-pass read would smear the offset across
    events and distort their spacing."""
    fake = _install_now(monkeypatch, NOW)
    source = OffsetReplaySource(_FakeInnerSource([0, 18, 41.5]))

    list(source.iter_events())

    assert fake.calls == 1


def test_each_iteration_re_anchors_to_the_new_now(monkeypatch):
    """A looping replay must restart at the current time, not repeat the first
    pass's timestamps."""
    later = NOW + timedelta(minutes=38)
    _install_now(monkeypatch, NOW, later)
    inner = _FakeInnerSource([0, 18])
    source = OffsetReplaySource(inner)

    first = [stamp for stamp, _ in source.iter_events()]
    source.reset()
    second = [stamp for stamp, _ in source.iter_events()]

    assert first == [NOW, NOW + timedelta(seconds=18)]
    assert second == [later, later + timedelta(seconds=18)]
    assert inner.reset_calls == 1


def test_timestamps_only_move_forward_across_passes(monkeypatch):
    """Why the offset removes the backwards jump that forces a fresh tracker:
    pass two starts after pass one ended."""
    later = NOW + timedelta(minutes=38)
    _install_now(monkeypatch, NOW, later)
    source = OffsetReplaySource(_FakeInnerSource([0, 18]))

    first = [stamp for stamp, _ in source.iter_events()]
    source.reset()
    second = [stamp for stamp, _ in source.iter_events()]

    assert second[0] > first[-1]


def test_naive_scenario_timestamps_are_read_as_utc(monkeypatch):
    """A file without tzinfo must not silently shift by the host's offset."""
    _install_now(monkeypatch, NOW)

    class _NaiveInner:
        def iter_events(self):
            stamp = T0.replace(tzinfo=None)
            yield stamp, [_FakeDetection(stamp)]

        def reset(self):
            pass

    ((stamp, detections),) = list(OffsetReplaySource(_NaiveInner()).iter_events())

    assert stamp == NOW
    assert detections[0].timestamp == NOW


def test_reset_delegates_to_the_wrapped_source():
    inner = _FakeInnerSource([0])
    source = OffsetReplaySource(inner)

    source.reset()

    assert inner.reset_calls == 1


def test_empty_source_yields_nothing_and_never_reads_the_clock(monkeypatch):
    fake = _install_now(monkeypatch, NOW)
    source = OffsetReplaySource(_FakeInnerSource([]))

    assert list(source.iter_events()) == []
    assert fake.calls == 0


def test_logs_the_offset_it_applied(monkeypatch, caplog):
    _install_now(monkeypatch, NOW)
    source = OffsetReplaySource(_FakeInnerSource([0]))

    with caplog.at_level("INFO"):
        list(source.iter_events())

    assert "Offsetting replay timestamps" in caplog.text
    assert "--use-scenario-timestamps" in caplog.text
