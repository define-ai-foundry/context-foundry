"""Tests for context_foundry.fusion.sources.stream (NetworkSapientStream).

We never bind a real socket: socket.socket is replaced with a MagicMock whose
recvfrom() is scripted via side_effect, terminated by a sentinel BaseException
that `except Exception`/`except DecodeError` cannot swallow. The frame-assembly
window runs on a fake monotonic clock that the scripted reads advance, so nothing
here depends on wall-clock timing.
"""

import itertools
from unittest.mock import MagicMock, call

import pytest
from google.protobuf.json_format import ParseDict

from context_foundry.fusion import config
from context_foundry.fusion.sources import frames
from context_foundry.fusion.sources.stream import NetworkSapientStream
from sapient_msg.bsi_flex_335_v2_0.sapient_message_pb2 import SapientMessage

TS_A = "2026-01-01T00:00:00Z"
TS_B = "2026-01-01T00:00:18Z"
NODE_A = "FI-MIL-RAD-KOLI-01"
NODE_B = "acoustic_array_01"

GARBAGE = (b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff", ("10.0.0.1", 1))


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


def _serialize(payload_dict):
    msg = SapientMessage()
    ParseDict(payload_dict, msg)
    return msg.SerializeToString()


def _valid_detection_report_payload(swarm=3, timestamp=TS_A, node_id=NODE_A, object_id="obj-1"):
    return {
        "timestamp": timestamp,
        "nodeId": node_id,
        "detectionReport": {
            "objectId": object_id,
            "location": {
                "x": 29.8,
                "y": 62.9,
                "z": 100.0,
                "coordinateSystem": "LOCATION_COORDINATE_SYSTEM_LAT_LNG_DEG_M",
                "datum": "LOCATION_DATUM_WGS84_E",
            },
            "objectInfo": [{"type": "estimatedSwarmCount", "value": str(swarm)}],
            "classification": [{"type": "UAS", "confidence": 0.9}],
        },
    }


def _packet(**kwargs):
    """One scripted datagram carrying a valid detection report."""
    return _serialize(_valid_detection_report_payload(**kwargs)), ("10.0.0.1", 1)


def _missing_location_oneof_payload():
    return {
        "timestamp": TS_A,
        "nodeId": NODE_A,
        "detectionReport": {"objectId": "obj-1"},
    }


def _rejected_packet():
    """One scripted datagram the validator drops, like a heartbeat or status update."""
    return _serialize(_missing_location_oneof_payload()), ("10.0.0.1", 1)


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
        payload, addr = item
        # A real socket truncates to the buffer the caller asked for, and the
        # mock has to as well or an undersized read looks fine in a test.
        return payload[:bufsize], addr

    mock_sock = MagicMock()
    mock_sock.recvfrom.side_effect = recvfrom
    monkeypatch.setattr(
        "context_foundry.fusion.sources.stream.socket.socket", lambda *a, **kw: mock_sock
    )
    return NetworkSapientStream(port=5000, frame_window_seconds=frame_window_seconds)


def test_iter_events_decode_error_is_caught_and_logged(monkeypatch, caplog):
    stream = _make_stream(monkeypatch, [GARBAGE, _Stop()])
    gen = stream.iter_events()
    with caplog.at_level("WARNING"), pytest.raises(_Stop):
        next(gen)
    assert any("malformed binary Protobuf" in r.message for r in caplog.records)


def test_iter_events_validator_rejects_message(monkeypatch):
    stream = _make_stream(monkeypatch, [_rejected_packet(), _Stop()])
    gen = stream.iter_events()
    with pytest.raises(_Stop):
        next(gen)


def test_iter_events_unexpected_error_is_caught_and_logged(monkeypatch, caplog):
    stream = _make_stream(monkeypatch, [_packet(), _Stop()])
    monkeypatch.setattr(
        "context_foundry.fusion.sources.stream.config.wgs84_to_enu",
        lambda *a: (_ for _ in ()).throw(ValueError("no projection")),
    )
    gen = stream.iter_events()

    with caplog.at_level("ERROR"), pytest.raises(_Stop):
        next(gen)

    assert any("Unexpected error in live stream ingestion" in r.message for r in caplog.records)


def test_iter_events_yields_detection_for_valid_packet(monkeypatch):
    config.load_sensor_network(
        sensor_network_list=[{"id": NODE_A, "lat": 62.9, "lon": 29.8, "alt": 100.0}]
    )
    stream = _make_stream(monkeypatch, [_packet(swarm=5), TimeoutError(), _Stop()])

    gen = stream.iter_events()
    timestamp, detections = next(gen)

    assert timestamp.year == 2026
    assert len(detections) == 1
    det = detections[0]
    # A 3x1 column vector: an inhomogeneous state vector raised on every packet,
    # so no live SAPIENT detection ever reached the tracker.
    assert det.state_vector.shape == (3, 1)
    assert det.metadata["nodeId"] == NODE_A
    assert det.metadata["objectId"] == "obj-1"
    assert det.metadata["classification"] == "UAS"
    assert det.metadata["swarm_count"] == 5
    assert det.metadata["sensor_geodetic"] == {
        "latitude": 62.9,
        "longitude": 29.8,
        "altitude": 100.0,
    }

    with pytest.raises(_Stop):
        next(gen)


def test_iter_events_reads_a_full_size_datagram(monkeypatch):
    """A report larger than 4096 bytes must not be truncated into a DecodeError."""
    payload_dict = _valid_detection_report_payload()
    # Pad with valid repeated objectInfo entries until the wire form is oversized.
    payload_dict["detectionReport"]["objectInfo"] += [
        {"type": f"filler-{i}", "value": "x" * 100} for i in range(50)
    ]
    payload = _serialize(payload_dict)
    assert len(payload) > 4096

    stream = _make_stream(monkeypatch, [(payload, ("10.0.0.1", 1)), TimeoutError(), _Stop()])
    _timestamp, detections = next(stream.iter_events())
    assert len(detections) == 1


def test_iter_events_sensor_geodetic_none_when_sensor_unregistered(monkeypatch):
    config.load_sensor_network(
        sensor_network_list=[{"id": "some-other-node", "lat": 62.9, "lon": 29.8, "alt": 100.0}]
    )
    stream = _make_stream(monkeypatch, [_packet(), TimeoutError(), _Stop()])

    gen = stream.iter_events()
    _, detections = next(gen)
    assert detections[0].metadata["sensor_geodetic"] is None

    with pytest.raises(_Stop):
        next(gen)


def test_iter_events_batches_one_instant_into_a_single_event(monkeypatch):
    """A sweep's datagrams share (timestamp, nodeId) and must arrive as one event."""
    stream = _make_stream(
        monkeypatch,
        [
            _packet(object_id="obj-1"),
            _packet(object_id="obj-2"),
            _packet(object_id="obj-3"),
            TimeoutError(),
            _Stop(),
        ],
    )
    gen = stream.iter_events()

    timestamp, detections = next(gen)
    assert timestamp.second == 0
    assert [d.metadata["objectId"] for d in detections] == ["obj-1", "obj-2", "obj-3"]
    assert all(d.timestamp == timestamp for d in detections)

    with pytest.raises(_Stop):
        next(gen)


def test_iter_events_a_new_instant_does_not_close_the_previous_frame(monkeypatch):
    """A frame is closed by its window, not by the next instant's first datagram:
    the next datagram may belong to another sensor whose sweep overlaps this one."""
    stream = _make_stream(
        monkeypatch,
        [
            _packet(object_id="obj-1"),
            _packet(timestamp=TS_B, object_id="obj-2"),
            _packet(object_id="obj-3"),
            TimeoutError(),
            _Stop(),
        ],
    )
    gen = stream.iter_events()

    first_ts, first = next(gen)
    assert first_ts.second == 0
    # obj-3 joined the frame obj-1 opened rather than finding it already handed on.
    assert [d.metadata["objectId"] for d in first] == ["obj-1", "obj-3"]

    second_ts, second = next(gen)
    assert second_ts.second == 18
    assert [d.metadata["objectId"] for d in second] == ["obj-2"]

    with pytest.raises(_Stop):
        next(gen)


def test_iter_events_two_nodes_at_one_instant_are_two_events(monkeypatch):
    """The frame key includes the node: one instant, two sensors, two events."""
    stream = _make_stream(
        monkeypatch,
        [
            _packet(node_id=NODE_A, object_id="obj-a"),
            _packet(node_id=NODE_B, object_id="obj-b"),
            TimeoutError(),
            _Stop(),
        ],
    )
    gen = stream.iter_events()

    ts_a, first = next(gen)
    ts_b, second = next(gen)
    assert ts_a == ts_b
    assert [d.metadata["nodeId"] for d in first] == [NODE_A]
    assert [d.metadata["nodeId"] for d in second] == [NODE_B]

    with pytest.raises(_Stop):
        next(gen)


def test_iter_events_interleaved_sweeps_are_one_event_per_node(monkeypatch):
    """Two nodes sharing the port interleave whenever their sweeps overlap. Each
    node's sweep must still be one event: a stream of single-detection events
    associates one hit per track and, because the timestamps then alternate, has
    the fusion loop's watermark gate drop whichever node is behind."""
    script = []
    for i in range(3):
        script.append(_packet(node_id=NODE_A, timestamp=TS_A, object_id=f"a-{i}"))
        script.append(_packet(node_id=NODE_B, timestamp=TS_B, object_id=f"b-{i}"))
    stream = _make_stream(monkeypatch, [*script, TimeoutError(), _Stop()])
    gen = stream.iter_events()

    ts_a, sweep_a = next(gen)
    ts_b, sweep_b = next(gen)

    assert ts_a.second == 0
    assert [d.metadata["objectId"] for d in sweep_a] == ["a-0", "a-1", "a-2"]
    assert ts_b.second == 18
    assert [d.metadata["objectId"] for d in sweep_b] == ["b-0", "b-1", "b-2"]

    with pytest.raises(_Stop):
        next(gen)


def test_iter_events_validator_rejection_flood_still_releases_the_open_frame(monkeypatch):
    """A rejected datagram never reaches the assembler, and a non-empty receive
    buffer never times out, so the window has to be enforced per read instead."""
    flood = itertools.chain(
        [_packet(object_id="obj-1")],
        (_rejected_packet() for _ in range(500)),  # heartbeats, status, registrations
        [_Stop()],
    )
    stream = _make_stream(monkeypatch, flood)

    _timestamp, detections = next(stream.iter_events())

    assert [d.metadata["objectId"] for d in detections] == ["obj-1"]


def test_iter_events_decode_error_flood_still_releases_the_open_frame(monkeypatch, caplog):
    """Same for undecodable datagrams, which anyone can send to an open UDP port."""
    flood = itertools.chain(
        [_packet(object_id="obj-1")],
        (GARBAGE for _ in range(500)),
        [_Stop()],
    )
    stream = _make_stream(monkeypatch, flood)

    with caplog.at_level("WARNING"):
        _timestamp, detections = next(stream.iter_events())

    assert [d.metadata["objectId"] for d in detections] == ["obj-1"]


def test_iter_events_window_expiry_emits_frame_without_logging(monkeypatch, caplog):
    """A timed-out recvfrom is the window closing, not an ingestion failure."""
    stream = _make_stream(monkeypatch, [_packet(), TimeoutError(), _Stop()])
    gen = stream.iter_events()

    with caplog.at_level("WARNING"):
        _timestamp, detections = next(gen)

    assert len(detections) == 1
    assert caplog.records == []

    with pytest.raises(_Stop):
        next(gen)


def test_iter_events_single_detection_frame_does_not_wait_for_the_next_packet(monkeypatch):
    """A lone detection is handed on when its window expires, not a sweep later."""
    stream = _make_stream(monkeypatch, [_packet(), TimeoutError(), _Stop()], frame_window_seconds=5)
    gen = stream.iter_events()

    _timestamp, detections = next(gen)
    assert len(detections) == 1

    # Idle read blocks; the read that follows an opened frame is bounded by its
    # remaining window -- so the event above came out without a second datagram.
    (idle, bounded) = stream.sock.settimeout.call_args_list
    assert idle == call(None)
    assert 0 < bounded.args[0] <= 5


def test_iter_events_timeout_with_no_open_frame_is_ignored(monkeypatch, caplog):
    stream = _make_stream(monkeypatch, [TimeoutError(), _Stop()])
    gen = stream.iter_events()

    with caplog.at_level("WARNING"), pytest.raises(_Stop):
        next(gen)

    assert caplog.records == []


def test_iter_events_detection_for_an_emitted_instant_becomes_its_own_event(monkeypatch):
    """No record of emitted instants is kept -- it would grow without bound -- so a
    late detection for one opens a frame of its own; the fusion loop gates it."""
    stream = _make_stream(
        monkeypatch,
        [
            _packet(object_id="obj-1"),
            TimeoutError(),
            _packet(object_id="obj-late"),
            TimeoutError(),
            _Stop(),
        ],
    )
    gen = stream.iter_events()

    ts_first, first = next(gen)
    ts_late, late = next(gen)

    assert [d.metadata["objectId"] for d in first] == ["obj-1"]
    assert [d.metadata["objectId"] for d in late] == ["obj-late"]
    assert ts_first == ts_late

    with pytest.raises(_Stop):
        next(gen)
