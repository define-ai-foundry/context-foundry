"""Tests for context_foundry.fusion.sources.stream (NetworkSapientStream).

KNOWN SOURCE BUG (see final report): the module builds detections with
`np.array([[e], [n], u])` -- note `u` is not wrapped in its own list, which makes
the list inhomogeneous and np.array() raises ValueError on every single call,
regardless of input. This means the "successful yield" path (building a
Detection, attaching metadata, yielding it) is unreachable dead code in the
current source; every valid SAPIENT packet actually falls through to the
generic `except Exception` handler and is silently dropped. We test the code
AS WRITTEN (asserting the bug fires) and additionally patch numpy.array for one
test to exercise the otherwise-dead metadata/yield lines for coverage.

We never bind a real socket: socket.socket is replaced with a MagicMock whose
recvfrom() is scripted via side_effect, terminated by a sentinel BaseException
that `except Exception`/`except DecodeError` cannot swallow.
"""

from unittest.mock import MagicMock

import numpy as np
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
    mock_sock = MagicMock()
    mock_sock.recvfrom.side_effect = recvfrom_side_effect
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


def test_iter_events_valid_message_hits_known_array_bug(monkeypatch, caplog):
    """Documents the bug: a perfectly valid, validator-accepted message still
    never produces a Detection because of the np.array([[e],[n],u]) typo.
    """
    payload = _serialize(_valid_detection_report_payload())
    stream = _make_stream(monkeypatch, [(payload, ("10.0.0.1", 1)), _Stop()])
    gen = stream.iter_events()

    with caplog.at_level("ERROR"), pytest.raises(_Stop):
        next(gen)

    assert any("Unexpected error in live stream ingestion" in r.message for r in caplog.records)


def test_iter_events_yields_detection_once_array_bug_is_worked_around(monkeypatch):
    """Same valid message as above, but with numpy.array patched to tolerate the
    malformed [[e],[n],u] nesting -- exercises the metadata/yield lines (87-102)
    that are otherwise permanently dead code given the real bug.
    """
    config.load_sensor_network(
        sensor_network_list=[{"id": "FI-MIL-RAD-KOLI-01", "lat": 62.9, "lon": 29.8, "alt": 100.0}]
    )
    payload = _serialize(_valid_detection_report_payload(swarm=5))
    stream = _make_stream(monkeypatch, [(payload, ("10.0.0.1", 1)), _Stop()])

    real_array = np.array

    def tolerant_array(seq, *a, **kw):
        try:
            return real_array(seq, *a, **kw)
        except ValueError:
            fixed = [x if isinstance(x, list) else [x] for x in seq]
            return real_array(fixed, *a, **kw)

    monkeypatch.setattr("context_foundry.fusion.sources.stream.np.array", tolerant_array)

    gen = stream.iter_events()
    _timestamp, detections = next(gen)

    assert len(detections) == 1
    det = detections[0]
    assert det.metadata["nodeId"] == "FI-MIL-RAD-KOLI-01"
    # objectId/swarm_count read from raw_metadata["original_report"], but
    # SapientValidator.normalize() actually stores it under "original_envelope"
    # (separate key-mismatch bug, see final report) -- so both fall back to
    # their defaults (None / 1) regardless of the input payload.
    assert det.metadata["objectId"] is None
    assert det.metadata["classification"] == "UAS"
    assert det.metadata["swarm_count"] == 1
    assert det.metadata["sensor_geodetic"] == {
        "latitude": 62.9,
        "longitude": 29.8,
        "altitude": 100.0,
    }

    with pytest.raises(_Stop):
        next(gen)


def test_iter_events_sensor_geodetic_none_when_sensor_unregistered(monkeypatch):
    payload = _serialize(_valid_detection_report_payload())
    stream = _make_stream(monkeypatch, [(payload, ("10.0.0.1", 1)), _Stop()])

    real_array = np.array

    def tolerant_array(seq, *a, **kw):
        try:
            return real_array(seq, *a, **kw)
        except ValueError:
            fixed = [x if isinstance(x, list) else [x] for x in seq]
            return real_array(fixed, *a, **kw)

    monkeypatch.setattr("context_foundry.fusion.sources.stream.np.array", tolerant_array)

    gen = stream.iter_events()
    _, detections = next(gen)
    assert detections[0].metadata["sensor_geodetic"] is None

    with pytest.raises(_Stop):
        next(gen)
