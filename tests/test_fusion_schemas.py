"""Tests for context_foundry.fusion.schemas."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from context_foundry.fusion.schemas import InternalDetection, TacticalTrack


def test_internal_detection_minimal_required_fields():
    det = InternalDetection(
        sensor_id="node-1",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        latitude=62.9,
        longitude=29.8,
    )
    assert det.altitude is None
    assert det.speed_mps is None
    assert det.heading_deg is None
    assert det.classification is None
    assert det.confidence is None
    assert det.raw_metadata == {}


def test_internal_detection_extra_fields_are_dropped():
    det = InternalDetection(
        sensor_id="node-1",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        latitude=62.9,
        longitude=29.8,
        totally_unexpected_field="ignored",
    )
    assert not hasattr(det, "totally_unexpected_field")


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_internal_detection_confidence_out_of_range_rejected(confidence):
    with pytest.raises(ValidationError):
        InternalDetection(
            sensor_id="node-1",
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            latitude=62.9,
            longitude=29.8,
            confidence=confidence,
        )


@pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
def test_internal_detection_confidence_boundaries_accepted(confidence):
    det = InternalDetection(
        sensor_id="node-1",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        latitude=62.9,
        longitude=29.8,
        confidence=confidence,
    )
    assert det.confidence == confidence


def test_internal_detection_missing_required_field_raises():
    with pytest.raises(ValidationError):
        InternalDetection(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc), latitude=62.9, longitude=29.8
        )


def test_tactical_track_defaults():
    track = TacticalTrack(track_id="abc123", timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert track.latitude is None
    assert track.swarm_count == 1
    assert track.threat_level == "unknown"
    assert track.metadata == {}
    assert track.velocity is None


def test_tactical_track_extra_fields_are_dropped():
    track = TacticalTrack(
        track_id="abc123",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        something_extra=123,
    )
    assert not hasattr(track, "something_extra")


def test_tactical_track_full_population():
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    track = TacticalTrack(
        track_id="abc123",
        timestamp=ts,
        latitude=62.9,
        longitude=29.8,
        altitude=100.0,
        speed_mps=12.5,
        heading_deg=270.0,
        classification="UAS",
        swarm_count=4,
        threat_level="hostile",
        velocity=[1.0, 2.0, 3.0],
        metadata={"foo": "bar"},
    )
    assert track.swarm_count == 4
    assert track.threat_level == "hostile"
    assert track.velocity == [1.0, 2.0, 3.0]
    assert track.metadata == {"foo": "bar"}


def test_naive_timestamps_are_treated_as_utc():
    """A CoT `time` attribute may legally omit its offset, and one naive
    timestamp mixed with the aware ones took the whole engine down comparing
    them."""
    det = InternalDetection(
        sensor_id="node-A",
        timestamp="2026-01-01T00:00:00.000",  # schema-valid CoT, no offset
        latitude=62.9,
        longitude=29.8,
    )

    assert det.timestamp.tzinfo is not None
    assert det.timestamp == datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_an_offset_timestamp_is_converted_to_utc():
    det = InternalDetection(
        sensor_id="node-A",
        timestamp="2026-01-01T02:00:00+02:00",
        latitude=62.9,
        longitude=29.8,
    )

    assert det.timestamp == datetime(2026, 1, 1, tzinfo=timezone.utc)
