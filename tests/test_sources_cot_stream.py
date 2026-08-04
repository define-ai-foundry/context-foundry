"""Tests for context_foundry.fusion.sources.cot_stream.

NetworkSapientStream/CotNetworkStream open real UDP sockets and loop forever on
recvfrom(). We mock socket.socket entirely (never bind a real socket) and use a
sentinel BaseException (not Exception, so the module's `except Exception` can't
swallow it) to deterministically break out of the infinite `while True` loop.
"""

from unittest.mock import MagicMock

import pytest

from context_foundry.fusion.sources.cot_stream import CotNetworkStream

VALID_COT_XML = (
    b'<event version="2.0" uid="COT-SENSOR-1" type="a-f-A-M-F" '
    b'time="2026-01-01T00:00:00.000Z" start="2026-01-01T00:00:00.000Z" '
    b'stale="2026-01-01T00:00:15.000Z" how="m-g">'
    b'<point lat="62.900000" lon="29.800000" hae="100.0" ce="10.0" le="10.0"/>'
    b"</event>"
)

SCHEMA_INVALID_COT_XML = (
    b'<event version="2.0" uid="COT-SENSOR-1" type="a-f-A-M-F" '
    b'time="2026-01-01T00:00:00.000Z" start="2026-01-01T00:00:00.000Z" '
    b'stale="2026-01-01T00:00:15.000Z" how="m-g">'
    b'<point lat="62.900000" lon="29.800000" ce="10.0" le="10.0"/>'  # missing required hae
    b"</event>"
)


class _Stop(BaseException):
    """Sentinel used to break the infinite iter_events() loop from within a test."""


def _make_stream(monkeypatch, recvfrom_side_effect):
    mock_sock = MagicMock()
    mock_sock.recvfrom.side_effect = recvfrom_side_effect
    monkeypatch.setattr(
        "context_foundry.fusion.sources.cot_stream.socket.socket", lambda *a, **kw: mock_sock
    )
    return CotNetworkStream(port=6969)


def test_iter_events_yields_detection_for_valid_packet(monkeypatch):
    stream = _make_stream(monkeypatch, [(VALID_COT_XML, ("127.0.0.1", 12345)), _Stop()])
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
    stream = _make_stream(monkeypatch, [(SCHEMA_INVALID_COT_XML, ("127.0.0.1", 12345)), _Stop()])
    gen = stream.iter_events()
    # Invalid packet is silently dropped (continue); next recvfrom is the sentinel.
    with pytest.raises(_Stop):
        next(gen)


def test_iter_events_generic_exception_is_caught_and_logged(monkeypatch, caplog):
    stream = _make_stream(monkeypatch, [(b"\xff\xfe-not-utf8", ("127.0.0.1", 1)), _Stop()])
    gen = stream.iter_events()

    with caplog.at_level("ERROR"), pytest.raises(_Stop):
        next(gen)

    assert any("Failed to process CoT packet" in r.message for r in caplog.records)
