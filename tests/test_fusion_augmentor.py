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
    # Stone Soup accumulates detection metadata on the Track, not on its states.
    state = SimpleNamespace(state_vector=vec, timestamp=timestamp)
    return SimpleNamespace(id=track_id, state=state, metadata=metadata)


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


def test_extract_tactical_track_reads_metadata_off_a_real_stonesoup_track(augmentor):
    """Guards the contract the augmentor depends on: Stone Soup exposes detection
    metadata on Track, and its states carry none. Reading the state instead left
    every track classified 'Unknown' with swarm_count 1, whatever the sensor said.
    """
    from stonesoup.types.state import GaussianState
    from stonesoup.types.track import Track

    config.set_reference_origin(62.9, 29.8, 0.0)
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    prior = GaussianState(state_vector=np.zeros((9, 1)), covar=np.eye(9), timestamp=ts)
    track = Track([prior], init_metadata={"classification": "UAV_DECOY", "swarm_count": 12})

    assert not hasattr(track.state, "metadata")

    result = augmentor.extract_tactical_track(track)
    assert result.classification == "UAV_DECOY"
    assert result.swarm_count == 12
    assert result.threat_level == "hostile"


def test_extract_tactical_track_heading_wraps_to_0_360(augmentor):
    config.set_reference_origin(62.9, 29.8, 0.0)
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    vec = np.zeros((9, 1))
    vec[1, 0] = -1.0  # v-east negative
    vec[4, 0] = -1.0  # v-north negative -> heading should be in (0, 360)
    track = _make_track(vec, ts, metadata={"classification": "UAS", "swarm_count": 1})

    result = augmentor.extract_tactical_track(track)
    assert 0.0 <= result.heading_deg < 360.0


def test_cot_type_labels_a_track_a_sapient_sensor_left_unclassified(augmentor):
    """SAPIENT sources write the literal "Unknown" when the sensor gave nothing,
    so it has to be treated as absent or the CoT fallback never fires."""
    config.set_reference_origin(62.9, 29.8, 0.0)
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    track = _make_track(
        np.zeros((9, 1)), ts, metadata={"classification": "Unknown", "cot_type": "a-h-A-M-F-Q"}
    )

    assert augmentor.extract_tactical_track(track).classification == "a-h-A-M-F-Q"


def test_a_real_classification_outranks_a_cot_type(augmentor):
    """Stone Soup merges every hit's metadata into the track, so a CoT hit on a
    SAPIENT-tracked target must not downgrade its label to a 2525 code."""
    config.set_reference_origin(62.9, 29.8, 0.0)
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    track = _make_track(
        np.zeros((9, 1)), ts, metadata={"classification": "UAV_DECOY", "cot_type": "a-h-A-M-F-Q"}
    )

    assert augmentor.extract_tactical_track(track).classification == "UAV_DECOY"
