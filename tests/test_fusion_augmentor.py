"""Tests for context_foundry.fusion.augmentor."""

import math
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from context_foundry.fusion import config
from context_foundry.fusion.augmentor import TacticalContextAugmentor
from context_foundry.fusion.schemas import TacticalTrack


@pytest.fixture
def augmentor():
    return TacticalContextAugmentor()


def _make_track(vec, timestamp, metadata=None, track_id="track-1"):
    state = SimpleNamespace(state_vector=vec, timestamp=timestamp)
    if metadata is not None:
        state.metadata = metadata
    return SimpleNamespace(id=track_id, state=state)


# --- ecef_to_wgs84 -------------------------------------------------------------


def test_ecef_to_wgs84_round_trip(augmentor):
    from context_foundry.fusion.models import wgs84_to_ecef

    x, y, z = wgs84_to_ecef(62.9, 29.8, 250.0)
    lat, lon, alt = augmentor.ecef_to_wgs84(x, y, z)
    assert lat == pytest.approx(62.9, abs=1e-8)
    assert lon == pytest.approx(29.8, abs=1e-8)
    assert alt == pytest.approx(250.0, abs=1e-3)


def test_ecef_to_wgs84_pole_case(augmentor):
    lat, lon, _alt = augmentor.ecef_to_wgs84(0.0, 0.0, 6356752.0)
    assert lat == 90.0
    assert lon == 0.0

    lat, lon, _alt = augmentor.ecef_to_wgs84(0.0, 0.0, -6356752.0)
    assert lat == -90.0


# --- extract_tactical_track ----------------------------------------------------


def test_extract_tactical_track_slow_single_target_is_suspect(augmentor):
    config.set_reference_origin(62.9, 29.8, 0.0)
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    vec = np.zeros((9, 1))
    vec[0, 0] = 10.0  # east
    vec[1, 0] = 1.0  # v-east
    vec[3, 0] = 0.0  # north
    vec[4, 0] = 1.0  # v-north
    vec[6, 0] = 0.0  # up
    vec[7, 0] = 0.0  # v-up
    track = _make_track(vec, ts, metadata={"classification": "UAS", "swarm_count": 1})

    result = augmentor.extract_tactical_track(track)

    assert isinstance(result, TacticalTrack)
    assert result.track_id == "track-1"
    assert result.classification == "UAS"
    assert result.swarm_count == 1
    assert result.threat_level == "suspect"
    assert result.speed_mps == pytest.approx(math.sqrt(2), abs=1e-6)


def test_extract_tactical_track_fast_target_is_hostile(augmentor):
    config.set_reference_origin(62.9, 29.8, 0.0)
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    vec = np.zeros((9, 1))
    vec[1, 0] = 30.0  # v-east, fast
    track = _make_track(vec, ts, metadata={"classification": "UAS", "swarm_count": 1})

    result = augmentor.extract_tactical_track(track)
    assert result.threat_level == "hostile"


def test_extract_tactical_track_swarm_is_hostile(augmentor):
    config.set_reference_origin(62.9, 29.8, 0.0)
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    vec = np.zeros((9, 1))
    track = _make_track(vec, ts, metadata={"classification": "UAS", "swarm_count": 5})

    result = augmentor.extract_tactical_track(track)
    assert result.threat_level == "hostile"
    assert result.swarm_count == 5


def test_extract_tactical_track_missing_metadata_defaults(augmentor):
    config.set_reference_origin(62.9, 29.8, 0.0)
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    vec = np.zeros((9, 1))
    track = _make_track(vec, ts, metadata=None)

    result = augmentor.extract_tactical_track(track)
    assert result.classification == "Unknown"
    assert result.swarm_count == 1
    assert result.threat_level == "suspect"


def test_extract_tactical_track_heading_wraps_to_0_360(augmentor):
    config.set_reference_origin(62.9, 29.8, 0.0)
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    vec = np.zeros((9, 1))
    vec[1, 0] = -1.0  # v-east negative
    vec[4, 0] = -1.0  # v-north negative -> heading should be in (0, 360)
    track = _make_track(vec, ts, metadata={"classification": "UAS", "swarm_count": 1})

    result = augmentor.extract_tactical_track(track)
    assert 0.0 <= result.heading_deg < 360.0
