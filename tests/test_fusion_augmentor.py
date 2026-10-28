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


def test_extract_tactical_track_carries_the_confidences(augmentor):
    config.set_reference_origin(62.9, 29.8, 0.0)
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    metadata = {"classification": "Vehicle > Drone", "class_confidences": [0.62, None]}
    track = _make_track(np.zeros((9, 1)), ts, metadata={**metadata, "detection_confidence": 0.8})

    result = augmentor.extract_tactical_track(track)
    assert result.detection_confidence == pytest.approx(0.8)
    assert result.class_confidences == [pytest.approx(0.62), None]


def _updated_track(first, latest):
    """A real Stone Soup track started with metadata `first`, then updated by a detection
    carrying `latest`."""
    from stonesoup.types.detection import Detection
    from stonesoup.types.hypothesis import SingleHypothesis
    from stonesoup.types.prediction import GaussianStatePrediction
    from stonesoup.types.state import GaussianState
    from stonesoup.types.track import Track
    from stonesoup.types.update import GaussianStateUpdate

    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    zeros, eye = np.zeros((9, 1)), np.eye(9)
    track = Track([GaussianState(zeros, eye, timestamp=ts)], init_metadata=first)
    prediction = GaussianStatePrediction(zeros, eye, timestamp=ts)
    detection = Detection(np.zeros((3, 1)), timestamp=ts, metadata=latest)
    hypothesis = SingleHypothesis(prediction, detection)
    track.append(GaussianStateUpdate(zeros, eye, hypothesis=hypothesis, timestamp=ts))
    return track


def test_extract_tactical_track_takes_the_latest_detections_confidences(augmentor):
    config.set_reference_origin(62.9, 29.8, 0.0)
    track = _updated_track(
        {"detection_confidence": 0.4, "class_confidences": [0.4]},
        {"detection_confidence": 0.7, "class_confidences": [0.7]},
    )

    result = augmentor.extract_tactical_track(track)
    assert result.detection_confidence == pytest.approx(0.7)
    assert result.class_confidences == pytest.approx([0.7])


def test_extract_tactical_track_latest_detection_without_confidences_leaves_none(augmentor):
    config.set_reference_origin(62.9, 29.8, 0.0)
    track = _updated_track(
        {"detection_confidence": 0.4, "class_confidences": [0.4]},
        {"detection_confidence": None, "class_confidences": []},
    )

    result = augmentor.extract_tactical_track(track)
    assert result.detection_confidence is None
    assert result.class_confidences == []


def test_extract_tactical_track_cot_type_fallback_carries_no_class_confidences(augmentor):
    config.set_reference_origin(62.9, 29.8, 0.0)
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    metadata = {"classification": "Unknown", "class_confidences": [0.3], "cot_type": "a-h-A"}

    result = augmentor.extract_tactical_track(_make_track(np.zeros((9, 1)), ts, metadata=metadata))
    assert result.classification == "a-h-A"
    assert result.class_confidences == []


def test_extract_tactical_track_a_cot_hit_keeps_the_sapient_confidences(augmentor):
    config.set_reference_origin(62.9, 29.8, 0.0)
    track = _updated_track(
        {"classification": "Vehicle", "detection_confidence": 0.8, "class_confidences": [0.6]},
        {"nodeId": "cot", "type": "CoT", "cot_type": "a-h-A"},
    )

    result = augmentor.extract_tactical_track(track)
    assert result.classification == "Vehicle"
    assert result.detection_confidence == pytest.approx(0.8)
    assert result.class_confidences == pytest.approx([0.6])


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


def test_a_hostile_track_stays_hostile_within_the_hold(augmentor):
    from datetime import timedelta

    from context_foundry.fusion.augmentor import HOSTILE_HOLD_SECONDS

    config.set_reference_origin(62.9, 29.8, 0.0)
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    fast, slow = np.zeros((9, 1)), np.zeros((9, 1))
    fast[1, 0], slow[1, 0] = 25.0, 15.0
    meta = {"classification": "UAS", "swarm_count": 1}
    assert augmentor.extract_tactical_track(_make_track(fast, ts, meta)).threat_level == "hostile"
    # Slowing below the limit for a moment does not flip it back to suspect...
    soon = ts + timedelta(seconds=10)
    assert augmentor.extract_tactical_track(_make_track(slow, soon, meta)).threat_level == "hostile"
    # ...nor does it last forever.
    late = ts + timedelta(seconds=HOSTILE_HOLD_SECONDS + 1)
    assert augmentor.extract_tactical_track(_make_track(slow, late, meta)).threat_level == "suspect"
    # Another track's judgement is its own.
    other = _make_track(slow, soon, meta, track_id="track-2")
    assert augmentor.extract_tactical_track(other).threat_level == "suspect"


# --- declared affiliation ------------------------------------------------------


def _moving(speed_east):
    vec = np.zeros((9, 1))
    vec[1, 0] = speed_east
    return vec


def test_a_friendly_cot_track_stays_friend_when_fast(augmentor):
    config.set_reference_origin(62.9, 29.8, 0.0)
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    meta = {"cot_type": "a-f-A-M-F-Q", "swarm_count": 5}
    result = augmentor.extract_tactical_track(_make_track(_moving(40.0), ts, meta))
    assert result.threat_level == "friend"


def test_a_hostile_declared_cot_track_stays_hostile_when_slow(augmentor):
    config.set_reference_origin(62.9, 29.8, 0.0)
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    meta = {"cot_type": "a-h-G"}
    result = augmentor.extract_tactical_track(_make_track(_moving(1.0), ts, meta))
    assert result.threat_level == "hostile"


@pytest.mark.parametrize(
    ("cot_type", "threat"),
    [
        ("a-f-G", "friend"),
        ("a-a-G", "friend"),
        ("a-h-G", "hostile"),
        ("a-j-G", "hostile"),
        ("a-k-G", "hostile"),
        ("a-s-G", "suspect"),
        ("a-n-G", "neutral"),
    ],
)
def test_each_declared_affiliation_maps_to_its_threat(augmentor, cot_type, threat):
    config.set_reference_origin(62.9, 29.8, 0.0)
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    track = _make_track(_moving(40.0), ts, {"cot_type": cot_type})
    assert augmentor.extract_tactical_track(track).threat_level == threat


def test_declared_threat_reads_only_a_cot_atom_types_affiliation():
    from context_foundry.fusion.augmentor import declared_threat

    assert declared_threat("a-f-G-U-C") == "friend"
    assert declared_threat("a-F-G") is None
    assert declared_threat("b-f-G") is None
    assert declared_threat(None) is None
    assert declared_threat(7) is None


@pytest.mark.parametrize("cot_type", ["a-u-A", "a-p-A", "a-o-A", "a-x-A", "a-f", "b-m-p-s-m"])
def test_an_undeclared_affiliation_keeps_the_rule_and_its_hold(augmentor, cot_type):
    from datetime import timedelta

    config.set_reference_origin(62.9, 29.8, 0.0)
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    meta = {"cot_type": cot_type}
    fast = augmentor.extract_tactical_track(_make_track(_moving(25.0), ts, meta))
    assert fast.threat_level == "hostile"
    soon = ts + timedelta(seconds=10)
    slow = augmentor.extract_tactical_track(_make_track(_moving(1.0), soon, meta))
    assert slow.threat_level == "hostile"


def test_a_sapient_only_track_keeps_the_rule_and_its_hold(augmentor):
    from datetime import timedelta

    config.set_reference_origin(62.9, 29.8, 0.0)
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    meta = {"classification": "UAS", "swarm_count": 1}
    assert augmentor.extract_tactical_track(_make_track(_moving(1.0), ts, meta)).threat_level == (
        "suspect"
    )
    swarm = {"classification": "UAS", "swarm_count": 4}
    assert augmentor.extract_tactical_track(_make_track(_moving(1.0), ts, swarm)).threat_level == (
        "hostile"
    )
    soon = ts + timedelta(seconds=10)
    assert augmentor.extract_tactical_track(_make_track(_moving(1.0), soon, meta)).threat_level == (
        "hostile"
    )


def test_a_declared_affiliation_clears_the_hostile_hold(augmentor):
    from datetime import timedelta

    config.set_reference_origin(62.9, 29.8, 0.0)
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    undeclared = {"cot_type": "a-u-A"}
    fast = augmentor.extract_tactical_track(_make_track(_moving(25.0), ts, undeclared))
    assert fast.threat_level == "hostile"
    # The sender declares it friendly: the hold no longer applies...
    soon = ts + timedelta(seconds=5)
    friendly = {"cot_type": "a-f-A"}
    declared = augmentor.extract_tactical_track(_make_track(_moving(1.0), soon, friendly))
    assert declared.threat_level == "friend"
    # ...nor comes back when the declaration is gone, within what would have been the hold.
    later = ts + timedelta(seconds=10)
    after = augmentor.extract_tactical_track(_make_track(_moving(1.0), later, undeclared))
    assert after.threat_level == "suspect"
