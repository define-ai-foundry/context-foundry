"""Tests for context_foundry.fusion.sources.multiplex.

The point of the multiplexer is that a source which blocks does not stop the
others from being read, so the fakes here deliberately include one that never
returns; assertions have to hold while it is still blocked.
"""

import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from context_foundry.fusion.sources import multiplex
from context_foundry.fusion.sources.multiplex import MultiplexedSource

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _ListSource:
    """Yields a fixed list of events, then ends."""

    def __init__(self, events):
        self.events = events
        self.resets = 0

    def iter_events(self):
        yield from self.events

    def reset(self):
        self.resets += 1


class _BlockingSource:
    """Stands in for a live socket: yields one event, then blocks forever."""

    def __init__(self, event):
        self.event = event
        self.released = threading.Event()

    def iter_events(self):
        yield self.event
        self.released.wait()

    def reset(self):
        pass


class _ExplodingSource:
    def iter_events(self):
        raise RuntimeError("socket exploded")
        yield  # pragma: no cover  -- makes this a generator

    def reset(self):
        pass


def _event(seconds):
    return (T0 + timedelta(seconds=seconds), [f"det-{seconds}"])


def test_requires_at_least_one_source():
    with pytest.raises(ValueError, match="at least one source"):
        MultiplexedSource([])


def test_yields_events_from_every_source():
    a = _ListSource([_event(0), _event(1)])
    b = _ListSource([_event(2)])

    events = list(MultiplexedSource([a, b]).iter_events())

    assert sorted(timestamp for timestamp, _ in events) == [_event(i)[0] for i in (0, 1, 2)]


def test_a_blocked_source_does_not_starve_the_others():
    """The regression this class exists for: read sequentially, and everything
    after the first live source waits on a packet that may never come.
    """
    blocked = _BlockingSource(_event(0))
    other = _ListSource([_event(1), _event(2)])

    try:
        stream = MultiplexedSource([blocked, other]).iter_events()
        seen = [next(stream) for _ in range(3)]
    finally:
        blocked.released.set()

    assert sorted(timestamp for timestamp, _ in seen) == [_event(i)[0] for i in (0, 1, 2)]


def test_ends_only_once_every_source_has_ended():
    a = _ListSource([_event(0)])
    b = _ListSource([_event(1)])

    stream = MultiplexedSource([a, b]).iter_events()
    assert len(list(stream)) == 2


def test_a_source_that_raises_is_logged_and_does_not_stop_the_rest(caplog):
    good = _ListSource([_event(0)])

    with caplog.at_level("ERROR"):
        events = list(MultiplexedSource([_ExplodingSource(), good]).iter_events())

    assert len(events) == 1
    assert any("stopped with an error" in r.message for r in caplog.records)


def test_a_full_queue_back_pressures_instead_of_losing_events(caplog):
    """A slow consumer must not silently truncate a source."""
    source = _ListSource([_event(i) for i in range(50)])

    with caplog.at_level("WARNING"):
        events = list(MultiplexedSource([source], queue_size=1).iter_events())

    assert len(events) == 50
    assert sorted(timestamp for timestamp, _ in events) == [_event(i)[0] for i in range(50)]


def test_one_sustained_stall_is_one_log_line(caplog, monkeypatch):
    """A saturated queue makes every event wait, so reporting per event -- or on
    every momentary recovery, which under sustained overload is also every event
    -- is a log line per packet. Measured at 195 lines for one two-minute stall.
    """
    monkeypatch.setattr(multiplex, "PUT_TIMEOUT_SECONDS", 0.02)
    monkeypatch.setattr(multiplex, "STALL_REPORT_SECONDS", 30.0)
    source = _ListSource([_event(i) for i in range(12)])
    stream = MultiplexedSource([source], queue_size=2).iter_events()

    seen = []
    with caplog.at_level("WARNING"):
        for event in stream:
            # Slower than the reader, so the queue stays saturated throughout and
            # every event alternates blocked -> accepted.
            time.sleep(0.05)
            seen.append(event)

    assert len(seen) == 12
    stalls = [r for r in caplog.records if "Event queue full" in r.message]
    assert len(stalls) == 1
    # The line carries how many events the stall has held up, so its scale shows.
    assert "event(s) held up so far" in stalls[0].message


def test_a_continuing_stall_is_re_reported_on_its_own_schedule(caplog, monkeypatch):
    """One line for a stall that never ends would leave an operator watching a
    permanently degraded feed with nothing after the first minute."""
    monkeypatch.setattr(multiplex, "PUT_TIMEOUT_SECONDS", 0.02)
    monkeypatch.setattr(multiplex, "STALL_REPORT_SECONDS", 0.0)  # report every wait
    source = _ListSource([_event(i) for i in range(6)])
    stream = MultiplexedSource([source], queue_size=1).iter_events()

    with caplog.at_level("WARNING"):
        for _ in stream:
            time.sleep(0.05)

    stalls = [r for r in caplog.records if "Event queue full" in r.message]
    assert len(stalls) > 1


def test_a_stalled_reader_stops_with_the_consumer(caplog):
    source = _ListSource([_event(i) for i in range(3)])
    stream = MultiplexedSource([source], queue_size=1).iter_events()

    next(stream)  # leave the rest queued/blocked behind a consumer that stops
    time.sleep(multiplex.PUT_TIMEOUT_SECONDS * 3)
    stream.close()
    time.sleep(multiplex.PUT_TIMEOUT_SECONDS * 3)

    assert not [t for t in threading.enumerate() if t.name.startswith("source-")]


def test_a_reader_stops_between_events_once_the_consumer_is_gone(caplog):
    """A live socket wakes on its next packet, which must not be fused into a
    stream nobody is reading."""
    handed_on = []

    class _CountingSource:
        def iter_events(self):
            for i in range(10):
                handed_on.append(i)
                yield _event(i)

        def reset(self):
            pass

    stream = MultiplexedSource([_CountingSource()], queue_size=1).iter_events()
    next(stream)
    stream.close()
    time.sleep(multiplex.PUT_TIMEOUT_SECONDS * 3)

    # Stopped at the queue rather than running the source to exhaustion.
    assert len(handed_on) < 10


def test_reset_is_forwarded_to_every_source():
    a, b = _ListSource([]), _ListSource([])
    MultiplexedSource([a, b]).reset()
    assert (a.resets, b.resets) == (1, 1)
