"""Tests for context_foundry.fusion.sources.multiplex.

The point of the multiplexer is that a source which blocks does not stop the
others from being read, so the fakes here deliberately include one that never
returns; assertions have to hold while it is still blocked.
"""

import threading
from datetime import datetime, timedelta, timezone

import pytest

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


def test_events_beyond_the_queue_size_are_dropped_with_a_warning(caplog):
    source = _ListSource([_event(i) for i in range(5)])

    with caplog.at_level("WARNING"):
        events = list(MultiplexedSource([source], queue_size=1).iter_events())

    # A full queue drops rather than blocking every other source behind a slow one.
    assert len(events) < 5
    assert any("Event queue full" in r.message for r in caplog.records)


def test_reset_is_forwarded_to_every_source():
    a, b = _ListSource([]), _ListSource([])
    MultiplexedSource([a, b]).reset()
    assert (a.resets, b.resets) == (1, 1)
