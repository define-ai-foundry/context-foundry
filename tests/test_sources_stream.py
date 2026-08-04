"""Tests for context_foundry.fusion.sources.stream (NetworkSapientStream).

We never bind a real socket: socket.socket is replaced with a MagicMock whose
recvfrom() is scripted via side_effect, terminated by a sentinel BaseException
that `except Exception`/`except DecodeError` cannot swallow.
"""

from unittest.mock import MagicMock

import pytest
from google.protobuf.json_format import ParseDict

from context_foundry.fusion import config
from context_foundry.fusion.sources.stream import NetworkSapientStream
from sapient_msg.bsi_flex_335_v2_0.sapient_message_pb2 import SapientMessage


class _Stop(BaseException):
    """Sentinel used to break the infinite iter_events() loop from within a test."""


def _serialize(payload_dict):
    msg = SapientMessage()
    ParseDict(payload_dict, msg)
    return msg.SerializeToString()


def _valid_detection_report_payload(swarm=3):
    return {
        "timestamp": "2026-01-01T00:00:00Z",
        "nodeId": "FI-MIL-RAD-KOLI-01",
        "detectionReport": {
            "objectId": "obj-1",
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


def _missing_location_oneof_payload():
    return {
        "timestamp": "2026-01-01T00:00:00Z",
        "nodeId": "FI-MIL-RAD-KOLI-01",
        "detectionReport": {"objectId": "obj-1"},
    }


def _make_stream(monkeypatch, recvfrom_side_effect):
    scripted = iter(recvfrom_side_effect)

    def recvfrom(bufsize):
        item = next(scripted)
        if isinstance(item, BaseException):
            raise item
        payload, addr = item
        # A real socket truncates to the buffer the caller asked for, and the
        # mock has to as well or an undersized read looks fine in a test.
        return payload[:bufsize], addr

    mock_sock = MagicMock()
    mock_sock.recvfrom.side_effect = recvfrom
    monkeypatch.setattr(
        "context_foundry.fusion.sources.stream.socket.socket", lambda *a, **kw: mock_sock
    )
    return NetworkSapientStream(port=5000)


def test_iter_events_decode_error_is_caught_and_logged(monkeypatch, caplog):
    stream = _make_stream(
        monkeypatch, [(b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff", ("10.0.0.1", 1)), _Stop()]
    )
    gen = stream.iter_events()
    with caplog.at_level("WARNING"), pytest.raises(_Stop):
        next(gen)
    assert any("malformed binary Protobuf" in r.message for r in caplog.records)


def test_iter_events_validator_rejects_message(monkeypatch):
    payload = _serialize(_missing_location_oneof_payload())
    stream = _make_stream(monkeypatch, [(payload, ("10.0.0.1", 1)), _Stop()])
    gen = stream.iter_events()
    with pytest.raises(_Stop):
        next(gen)


def test_iter_events_yields_detection_for_valid_packet(monkeypatch):
    config.load_sensor_network(
        sensor_network_list=[{"id": "FI-MIL-RAD-KOLI-01", "lat": 62.9, "lon": 29.8, "alt": 100.0}]
    )
    payload = _serialize(_valid_detection_report_payload(swarm=5))
    stream = _make_stream(monkeypatch, [(payload, ("10.0.0.1", 1)), _Stop()])

    gen = stream.iter_events()
    timestamp, detections = next(gen)

    assert timestamp.year == 2026
    assert len(detections) == 1
    det = detections[0]
    # A 3x1 column vector: an inhomogeneous state vector raised on every packet,
    # so no live SAPIENT detection ever reached the tracker.
    assert det.state_vector.shape == (3, 1)
    assert det.metadata["nodeId"] == "FI-MIL-RAD-KOLI-01"
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

    stream = _make_stream(monkeypatch, [(payload, ("10.0.0.1", 1)), _Stop()])
    _timestamp, detections = next(stream.iter_events())
    assert len(detections) == 1


def test_iter_events_sensor_geodetic_none_when_sensor_unregistered(monkeypatch):
    config.load_sensor_network(
        sensor_network_list=[{"id": "some-other-node", "lat": 62.9, "lon": 29.8, "alt": 100.0}]
    )
    payload = _serialize(_valid_detection_report_payload())
    stream = _make_stream(monkeypatch, [(payload, ("10.0.0.1", 1)), _Stop()])

    gen = stream.iter_events()
    _, detections = next(gen)
    assert detections[0].metadata["sensor_geodetic"] is None

    with pytest.raises(_Stop):
        next(gen)
