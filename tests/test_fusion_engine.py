"""Tests for context_foundry.fusion.engine (FusionEngine), the step a pipeline stage calls."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from stonesoup.types.detection import Detection

from context_foundry.fusion import config
from context_foundry.fusion.engine import FusionEngine, sensor_ids
from context_foundry.fusion.measurement import DEFAULT_MEASUREMENT_MODEL
from context_foundry.fusion.publish import PublishPolicy
from context_foundry.fusion.tracker import SapientAsynchronousTracker

T0 = datetime(2026, 11, 15, 3, 0, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _origin():
    # Detections are made in the tracking frame directly; real ones set its origin.
    config.set_reference_origin(62.9, 29.8, 100.0)


def _det(seconds, east=0.0, node="RAD-1"):
    detection = Detection(
        state_vector=np.array([[east], [0.0], [100.0]]),
        measurement_model=DEFAULT_MEASUREMENT_MODEL,
        timestamp=T0 + timedelta(seconds=seconds),
    )
    detection.metadata = {"nodeId": node, "objectId": "obj", "classification": "UAV"}
    return detection


def _engine(confirm_hits=1):
    return FusionEngine(
        SapientAsynchronousTracker(confirm_hits=confirm_hits), PublishPolicy("window")
    )


def test_an_event_is_fused_and_its_track_published():
    engine = _engine()
    result = engine.process(T0, [_det(0)])
    assert not result.dropped
    assert result.mark == T0
    assert len(result.active) == 1
    ((track, tactical),) = result.published
    assert tactical.track_id == track.id


def test_the_publish_window_holds_back_a_track_sent_moments_ago():
    engine = _engine()
    engine.process(T0, [_det(0)])
    again = engine.process(T0 + timedelta(seconds=1), [_det(1, east=10.0)])
    assert len(again.active) == 1
    assert again.published == []


def test_an_event_far_behind_the_newest_is_dropped():
    engine = _engine()
    engine.process(T0 + timedelta(seconds=60), [_det(60)])
    late = engine.process(T0, [_det(0, node="SLOW-1")])
    assert late.dropped
    assert "behind the newest event" in late.skew
    assert engine.watermark_sensor == "RAD-1"


def test_a_live_event_from_the_future_is_dropped_and_a_near_one_clamped():
    engine = _engine()
    far = engine.process(T0 + timedelta(seconds=60), [_det(60)], now=T0)
    assert far.dropped and "ahead of the present" in far.skew
    detection = _det(5)
    near = engine.process(T0 + timedelta(seconds=5), [detection], now=T0)
    assert near.mark == T0
    assert detection.timestamp == T0


def test_reset_starts_a_new_scenario():
    engine = _engine()
    engine.process(T0 + timedelta(seconds=60), [_det(60)])
    engine.reset(SapientAsynchronousTracker(confirm_hits=1))
    assert not engine.process(T0, [_det(0)]).dropped


def test_sensor_ids_names_the_sensors_of_an_event():
    assert sensor_ids([_det(0, node="B"), _det(0, node="A")]) == "A, B"
    assert sensor_ids([]) == "an unnamed sensor"
