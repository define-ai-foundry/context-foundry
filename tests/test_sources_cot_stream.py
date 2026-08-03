"""Tests for context_foundry.fusion.sources.cot_stream.

NetworkSapientStream/CotNetworkStream open real UDP sockets and loop forever on
recvfrom(). We mock socket.socket entirely (never bind a real socket) and use a
sentinel BaseException (not Exception, so the module's `except Exception` can't
swallow it) to deterministically break out of the infinite `while True` loop. The
frame-assembly window runs on a fake monotonic clock that the scripted reads
advance, so nothing here depends on wall-clock timing.
"""

import itertools
from unittest.mock import MagicMock, call

import pytest

from context_foundry.fusion.sources import frames
from context_foundry.fusion.sources.cot_stream import CotNetworkStream

TIME_A = "2026-01-01T00:00:00.000Z"
TIME_B = "2026-01-01T00:00:18.000Z"
SENDER_1 = ("10.0.0.1", 4242)
SENDER_2 = ("10.0.0.2", 4242)


def _cot_xml(uid="COT-SENSOR-1", time=TIME_A, hae='hae="100.0" '):
    return (
        f'<event version="2.0" uid="{uid}" type="a-f-A-M-F" '
        f'time="{time}" start="{time}" stale="2026-01-01T00:00:15.000Z" how="m-g">'
        f'<point lat="62.900000" lon="29.800000" {hae}ce="10.0" le="10.0"/>'
        f"</event>"
    ).encode()


VALID_COT_XML = _cot_xml()

# Missing the required hae attribute.
SCHEMA_INVALID_COT_XML = _cot_xml(hae="")


class _Stop(BaseException):
    """Sentinel used to break the infinite iter_events() loop from within a test."""


class _FakeClock:
    """A monotonic clock the scripted reads advance, so nothing here sleeps."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _packet(sender=SENDER_1, **kwargs):
    return _cot_xml(**kwargs), sender


def _make_stream(monkeypatch, recvfrom_side_effect, frame_window_seconds=0.05):
    scripted = iter(recvfrom_side_effect)
    clock = _FakeClock()
    monkeypatch.setattr(frames.time, "monotonic", clock)

    def recvfrom(bufsize):
        item = next(scripted)
        if isinstance(item, TimeoutError):
            # A read only times out once the window it was armed with has run out.
            clock.advance(frame_window_seconds * 2)
            raise item
        if isinstance(item, BaseException):
            raise item
        # A datagram takes a fraction of a window, so a sweep's packets share one.
        clock.advance(frame_window_seconds / 50)
        return item

    mock_sock = MagicMock()
    mock_sock.recvfrom.side_effect = recvfrom
    monkeypatch.setattr(
        "context_foundry.fusion.sources.cot_stream.socket.socket", lambda *a, **kw: mock_sock
    )
    return CotNetworkStream(port=6969, frame_window_seconds=frame_window_seconds)


def test_iter_events_yields_detection_for_valid_packet(monkeypatch):
    stream = _make_stream(monkeypatch, [(VALID_COT_XML, SENDER_1), TimeoutError(), _Stop()])
    gen = stream.iter_events()

    timestamp, detections = next(gen)
    assert len(detections) == 1
    det = detections[0]
    assert det.metadata == {
        "nodeId": "COT-SENSOR-1",
        "type": "CoT",
        "cot_type": "a-f-A-M-F",
    }
    assert timestamp.year == 2026

    with pytest.raises(_Stop):
        next(gen)


def test_iter_events_drops_schema_invalid_packet(monkeypatch):
    stream = _make_stream(monkeypatch, [(SCHEMA_INVALID_COT_XML, SENDER_1), _Stop()])
    gen = stream.iter_events()
    # Invalid packet is silently dropped; the next recvfrom is the sentinel.
    with pytest.raises(_Stop):
        next(gen)


def test_iter_events_generic_exception_is_caught_and_logged(monkeypatch, caplog):
    stream = _make_stream(monkeypatch, [(b"\xff\xfe-not-utf8", SENDER_1), _Stop()])
    gen = stream.iter_events()

    with caplog.at_level("ERROR"), pytest.raises(_Stop):
        next(gen)

    assert any("Failed to process CoT packet" in r.message for r in caplog.records)


def test_iter_events_batches_one_senders_instant_into_a_single_event(monkeypatch):
    """One sender's objects at one instant are one event: a uid names the object
    it describes, not the sender, so keying on it would batch nothing."""
    stream = _make_stream(
        monkeypatch,
        [_packet(uid="OBJ-1"), _packet(uid="OBJ-2"), TimeoutError(), _Stop()],
    )
    gen = stream.iter_events()

    timestamp, detections = next(gen)
    assert [d.metadata["nodeId"] for d in detections] == ["OBJ-1", "OBJ-2"]
    assert all(d.timestamp == timestamp for d in detections)

    with pytest.raises(_Stop):
        next(gen)


def test_iter_events_two_senders_at_one_instant_are_two_events(monkeypatch):
    """Merging two senders' reports of the same object into one event leaves JPDA
    one hit it can associate and one it cannot, which initiates a ghost track. A
    whole-second CoT time from two emitters normalises to the same instant, so
    the frame key carries the peer address the uid does not identify."""
    stream = _make_stream(
        monkeypatch,
        [
            _packet(sender=SENDER_1, uid="OBJ-1"),
            _packet(sender=SENDER_2, uid="OBJ-1"),
            TimeoutError(),
            _Stop(),
        ],
    )
    gen = stream.iter_events()

    ts_first, first = next(gen)
    ts_second, second = next(gen)

    assert ts_first == ts_second
    assert len(first) == 1
    assert len(second) == 1

    with pytest.raises(_Stop):
        next(gen)


def test_iter_events_a_new_instant_does_not_close_the_previous_frame(monkeypatch):
    """A frame is closed by its window, not by the next instant's first datagram:
    the next datagram may belong to another sender whose report overlaps this one."""
    stream = _make_stream(
        monkeypatch,
        [
            _packet(uid="OBJ-1"),
            _packet(uid="OBJ-2", time=TIME_B),
            _packet(uid="OBJ-3"),
            TimeoutError(),
            _Stop(),
        ],
    )
    gen = stream.iter_events()

    first_ts, first = next(gen)
    # OBJ-3 joined the frame OBJ-1 opened rather than finding it already handed on.
    assert [d.metadata["nodeId"] for d in first] == ["OBJ-1", "OBJ-3"]

    second_ts, second = next(gen)
    assert [d.metadata["nodeId"] for d in second] == ["OBJ-2"]
    assert second_ts > first_ts

    with pytest.raises(_Stop):
        next(gen)


def test_iter_events_invalid_packet_flood_still_releases_the_open_frame(monkeypatch):
    """A dropped packet never reaches the assembler, and a non-empty receive buffer
    never times out, so the window has to be enforced per read instead. Anyone can
    flood an open UDP port with XML that does not validate."""
    flood = itertools.chain(
        [_packet(uid="OBJ-1")],
        ((SCHEMA_INVALID_COT_XML, SENDER_2) for _ in range(500)),
        [_Stop()],
    )
    stream = _make_stream(monkeypatch, flood)

    _timestamp, detections = next(stream.iter_events())

    assert [d.metadata["nodeId"] for d in detections] == ["OBJ-1"]


def test_iter_events_window_expiry_emits_frame_without_logging(monkeypatch, caplog):
    """A timed-out recvfrom is the window closing, not a failed packet."""
    stream = _make_stream(monkeypatch, [_packet(), TimeoutError(), _Stop()], frame_window_seconds=5)
    gen = stream.iter_events()

    with caplog.at_level("ERROR"):
        _timestamp, detections = next(gen)

    assert len(detections) == 1
    assert caplog.records == []

    # Idle read blocks; the read after a frame opened is bounded by its window,
    # so this event came out without waiting for a second datagram.
    (idle, bounded) = stream.sock.settimeout.call_args_list
    assert idle == call(None)
    assert 0 < bounded.args[0] <= 5


def test_iter_events_timeout_with_no_open_frame_is_ignored(monkeypatch, caplog):
    stream = _make_stream(monkeypatch, [TimeoutError(), _Stop()])
    gen = stream.iter_events()

    with caplog.at_level("ERROR"), pytest.raises(_Stop):
        next(gen)

    assert caplog.records == []


def test_iter_events_detection_for_an_emitted_instant_becomes_its_own_event(monkeypatch):
    """No record of emitted instants is kept -- it would grow without bound -- so a
    late detection for one opens a frame of its own; the fusion loop gates it."""
    stream = _make_stream(
        monkeypatch,
        [
            _packet(uid="OBJ-1"),
            TimeoutError(),
            _packet(uid="OBJ-LATE"),
            TimeoutError(),
            _Stop(),
        ],
    )
    gen = stream.iter_events()

    ts_first, first = next(gen)
    ts_late, late = next(gen)

    assert [d.metadata["nodeId"] for d in first] == ["OBJ-1"]
    assert [d.metadata["nodeId"] for d in late] == ["OBJ-LATE"]
    assert ts_first == ts_late

    with pytest.raises(_Stop):
        next(gen)
