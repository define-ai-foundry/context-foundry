"""Tests for context_foundry.fusion.sources.frames (FrameAssembler).

The assembler is exercised without a socket: it only decides what the next read's
timeout should be and when an event is complete. Every test that depends on the
window drives a fake monotonic clock, so nothing here sleeps or races.
"""

from datetime import datetime, timedelta, timezone

import pytest

from context_foundry.fusion.sources import frames
from context_foundry.fusion.sources.frames import (
    DEFAULT_FRAME_WINDOW_SECONDS,
    MAX_OPEN_FRAMES,
    MIN_FRAME_TIMEOUT_SECONDS,
    FrameAssembler,
)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
T1 = T0 + timedelta(seconds=18)

# How often a sensor reports the same object in the reference scenario.
SWEEP_INTERVAL_SECONDS = 18.0


class _FakeClock:
    """A monotonic clock the test advances, so nothing here sleeps."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = _FakeClock()
    monkeypatch.setattr(frames.time, "monotonic", fake)
    return fake


def test_idle_assembler_blocks_indefinitely():
    """No frame open means nothing to hand on when a read expires; don't poll."""
    assert FrameAssembler().next_timeout() is None


def test_open_frame_bounds_the_next_read_by_its_remaining_window(clock):
    """The read must expire when the window does, not a whole window later."""
    assembler = FrameAssembler(window_seconds=0.05)
    assert assembler.add(("k", T0), T0, "det-1") == []

    clock.advance(0.03)

    assert assembler.next_timeout() == pytest.approx(0.02)


def test_next_timeout_is_the_shortest_remaining_window(clock):
    """Several frames are open; the read must not outlast the first to expire."""
    assembler = FrameAssembler(window_seconds=0.05)
    assembler.add(("a", T0), T0, "det-a")
    clock.advance(0.02)
    assembler.add(("b", T0), T0, "det-b")

    assert assembler.next_timeout() == pytest.approx(0.03)


def test_elapsed_window_never_asks_for_a_zero_timeout(clock):
    """settimeout(0) is non-blocking mode: an empty buffer raises BlockingIOError
    rather than timing out, so the source would log an error per expired frame.
    Asserted as a positive number, not just as the constant, or the constant
    itself is free to become 0."""
    assembler = FrameAssembler(window_seconds=0.05)
    assembler.add(("k", T0), T0, "det-1")
    clock.advance(0.06)
    assert assembler.next_timeout() > 0
    assert assembler.next_timeout() == MIN_FRAME_TIMEOUT_SECONDS


def test_same_key_detections_accumulate_into_one_event(clock):
    assembler = FrameAssembler(window_seconds=0.05)
    assert assembler.add(("k", T0), T0, "det-1") == []
    clock.advance(0.01)
    assert assembler.add(("k", T0), T0, "det-2") == []

    assert assembler.take_expired() == []  # still inside the window
    clock.advance(0.05)
    assert assembler.take_expired() == [(T0, ["det-1", "det-2"])]


def test_a_new_key_does_not_close_the_open_frame(clock):
    """Closing on a key change would batch nothing once two sensors interleave."""
    assembler = FrameAssembler(window_seconds=0.05)
    assert assembler.add(("k", T0), T0, "det-1") == []
    assert assembler.add(("k", T1), T1, "det-2") == []

    clock.advance(0.06)
    assert assembler.take_expired() == [(T0, ["det-1"]), (T1, ["det-2"])]


def test_interleaved_keys_are_assembled_into_one_event_each(clock):
    """Two sensors' sweeps overlap on the wire; each still arrives as one event."""
    assembler = FrameAssembler(window_seconds=0.05)
    for i in range(3):
        assembler.add(("a", T0), T0, f"a-{i}")
        assembler.add(("b", T0), T0, f"b-{i}")
        clock.advance(0.001)

    clock.advance(0.05)
    assert assembler.take_expired() == [
        (T0, ["a-0", "a-1", "a-2"]),
        (T0, ["b-0", "b-1", "b-2"]),
    ]


def test_take_expired_is_oldest_opened_first(clock):
    """Arrival order, which is not necessarily ascending event timestamps -- two
    sensors whose clocks disagree hand on in the order they reported, and the
    fusion loop's gate deals with the consequence."""
    assembler = FrameAssembler(window_seconds=0.05)
    # Keys deliberately out of sort order, so only opening order can produce this.
    assembler.add(("z", T1), T1, "det-z")
    clock.advance(0.01)
    assembler.add(("a", T0), T0, "det-a")

    clock.advance(0.06)
    assert assembler.take_expired() == [(T1, ["det-z"]), (T0, ["det-a"])]


def test_take_expired_is_empty_while_the_window_holds(clock):
    assembler = FrameAssembler(window_seconds=0.05)
    assembler.add(("k", T0), T0, "det-1")
    clock.advance(0.04)
    assert assembler.take_expired() == []
    assert assembler.next_timeout() == pytest.approx(0.01)


def test_take_expired_on_an_idle_assembler_is_empty():
    assert FrameAssembler().take_expired() == []


def test_open_frames_are_capped_by_closing_the_oldest(clock):
    """Keys come off the wire, so a flood of distinct instants must not grow the table."""
    assembler = FrameAssembler(window_seconds=60.0)  # nothing expires during the test
    for i in range(MAX_OPEN_FRAMES):
        assert assembler.add(("node", i), T0, f"det-{i}") == []
        clock.advance(0.001)

    # One key too many: the oldest frame is handed on rather than buffered on top.
    assert assembler.add(("node", "extra"), T1, "det-extra") == [(T0, ["det-0"])]
    assert assembler.add(("node", "extra-2"), T1, "det-extra-2") == [(T0, ["det-1"])]


def test_a_distinct_key_flood_hands_on_everything_it_cannot_hold(clock):
    """Nothing is lost to the cap, and the table stays at its bound."""
    assembler = FrameAssembler(window_seconds=60.0)
    held_on = []
    for i in range(MAX_OPEN_FRAMES * 5):
        held_on += assembler.add(("node", i), T0, f"det-{i}")
        clock.advance(0.001)

    assert len(held_on) == MAX_OPEN_FRAMES * 4  # the remainder is still open
    clock.advance(60.0)
    assert len(held_on + assembler.take_expired()) == MAX_OPEN_FRAMES * 5


def test_default_window_is_short_enough_for_a_live_feed():
    """The window is latency on every live detection, so it stays a small
    fraction of the interval between a track's updates."""
    assert 0 < DEFAULT_FRAME_WINDOW_SECONDS < SWEEP_INTERVAL_SECONDS / 10


def test_default_window_assembles_a_sweep_spread_across_a_network(clock):
    """Datagrams arriving back to back is a loopback property, not a requirement
    a sensor on a routed network meets: milliseconds apart is normal, and a
    window too tight for that splits every sweep into same-timestamp events."""
    sweep = 20  # objects a single ASM reports per instant
    assembler = FrameAssembler()  # the shipped default, not a test-chosen window
    for i in range(sweep):
        assert assembler.add(("node-a", T0), T0, f"det-{i}") == []
        clock.advance(0.010)  # per-datagram spacing seen on a real socket

    clock.advance(DEFAULT_FRAME_WINDOW_SECONDS)
    assert assembler.take_expired() == [(T0, [f"det-{i}" for i in range(sweep)])]


def test_default_window_outlasts_a_fusion_pass(clock):
    """A fusion pass blocks the read loop, so every open frame ages by its
    duration; a window shorter than a pass splits the sweep it was assembling,
    and the extra tracks that produces make the next pass slower still."""
    assembler = FrameAssembler()
    assembler.add(("node-a", T0), T0, "det-0")

    clock.advance(0.140)  # one pass at a few tens of tracks

    assert assembler.add(("node-a", T0), T0, "det-1") == []


def test_an_elapsed_window_closes_the_frame_on_the_next_detection(clock):
    """A detection must not join a frame whose window ran out while the read blocked."""
    assembler = FrameAssembler(window_seconds=0.05)

    assert assembler.add("k", T0, "det-1") == []
    clock.advance(0.02)
    assert assembler.add("k", T0, "det-2") == []  # still inside the window

    clock.advance(0.06)  # past it
    assert assembler.add("k", T0, "det-3") == [(T0, ["det-1", "det-2"])]

    # The detection that closed it opened the next frame rather than being lost.
    clock.advance(0.06)
    assert assembler.take_expired() == [(T0, ["det-3"])]


def test_a_same_key_flood_keeps_producing_events(clock):
    """A frame must be bounded by the window however fast the datagrams arrive."""
    assembler = FrameAssembler(window_seconds=0.05)

    events = []
    for _ in range(500):
        clock.advance(0.001)  # 1kHz, all one key, socket never goes quiet
        events += assembler.add("k", T0, "det")

    assert len(events) >= 9  # ~one per window elapsed, not zero
    assert all(len(dets) <= 51 for _, dets in events)


def test_a_full_table_reports_the_frames_it_hands_on_early(clock, caplog):
    """Silent eviction leaves an operator with sweeps splitting for no visible
    reason; the cap only fires when something is stamping per detection."""
    assembler = FrameAssembler(window_seconds=10.0)  # nothing expires on its own
    with caplog.at_level("WARNING"):
        for i in range(frames.MAX_OPEN_FRAMES + 3):
            clock.advance(0.001)
            assembler.add((f"k{i}", T0), T0, f"det-{i}")

    reports = [r for r in caplog.records if "instants open at once" in r.message]
    # Rate-limited: one line, not one per evicted frame. It carries the running
    # total, so the first says 1 and a later one says how far it has got.
    assert len(reports) == 1
    assert "1 frame(s)" in reports[0].message


def test_the_eviction_report_is_rate_limited_not_one_shot(clock, caplog, monkeypatch):
    monkeypatch.setattr(frames, "EVICTION_REPORT_SECONDS", 0.0)
    assembler = FrameAssembler(window_seconds=10.0)
    with caplog.at_level("WARNING"):
        for i in range(frames.MAX_OPEN_FRAMES + 3):
            clock.advance(0.001)
            assembler.add((f"k{i}", T0), T0, f"det-{i}")

    reports = [r for r in caplog.records if "instants open at once" in r.message]
    assert len(reports) == 3
    # The count accumulates across reports, so the scale of the loss is visible.
    assert "3 frame(s)" in reports[-1].message
