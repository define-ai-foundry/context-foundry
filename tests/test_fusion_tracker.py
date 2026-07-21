"""Tests for context_foundry.fusion.tracker.SapientAsynchronousTracker.

IMPORTANT (see task brief / final report): JPDA.associate() enumerates joint
hypotheses over ALL current tracks and blows up combinatorially past ~15
concurrent tracks -- every scenario here stays to at most 1-2 tracks/detections.

KNOWN SOURCE BUG (see final report): `joint_hypothesis = hypotheses[0]` in
process_async_event always retrieves the *missed-detection* hypothesis (it is
always inserted first by stonesoup's JPDA.associate(), never sorted by
probability -- see stonesoup.dataassociator.probability.JPDA.associate). That
makes `if not joint_hypothesis` always True, so the real "association update"
branch (the `else:` block that actually updates a track from a detection) is
unreachable dead code under real JPDA -- confirmed empirically: a detection
1m from an existing track still spawns a new track instead of updating it.
We exercise the real bootstrap/coasting/unassociated-hit paths with genuine
tiny JPDA runs, and use a monkeypatched fake data_associator (as explicitly
sanctioned by the task brief) to reach the otherwise-dead association-update
branch and its two sensor_geodetic sub-branches.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.array import StateVector
from stonesoup.types.detection import Detection
from stonesoup.types.state import GaussianState

from context_foundry.fusion.tracker import SapientAsynchronousTracker

MEAS_MODEL = LinearGaussian(
    ndim_state=9, mapping=(0, 3, 6), noise_covar=np.diag([25.0, 25.0, 100.0])
)
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _FakeBearingOnlyDetection:
    """Duck-typed stand-in for a 1D bearing-only Detection (hashable, unlike
    SimpleNamespace which defines __eq__ and so loses the default __hash__)."""

    def __init__(self, state_vector, metadata, timestamp):
        self.state_vector = state_vector
        self.metadata = metadata
        self.timestamp = timestamp


def _det(e, n, u, timestamp, metadata=None):
    d = Detection(
        state_vector=np.array([[e], [n], [u]]), measurement_model=MEAS_MODEL, timestamp=timestamp
    )
    d.metadata = metadata if metadata is not None else {"classification": "UAS", "swarm_count": 1}
    return d


@pytest.fixture
def tracker():
    return SapientAsynchronousTracker()


# --- bootstrap ------------------------------------------------------------------


def test_bootstrap_creates_one_track_per_detection(tracker):
    d1 = _det(0.0, 0.0, 0.0, T0)
    d2 = _det(500.0, 500.0, 0.0, T0)

    result = tracker.process_async_event(T0, {d1, d2})

    assert result is tracker.tracks
    assert len(tracker.tracks) == 2
    for track in tracker.tracks:
        assert len(track.states) == 1
        assert track.state.timestamp == T0


# --- coasting (real, tiny JPDA: 1 track, 0 detections) ---------------------------


def test_process_async_event_coasts_track_when_no_detections(tracker):
    d1 = _det(0.0, 0.0, 0.0, T0)
    tracker.process_async_event(T0, {d1})
    (track,) = tuple(tracker.tracks)

    t1 = T0 + timedelta(seconds=2)
    tracker.process_async_event(t1, set())

    assert len(tracker.tracks) == 1
    assert track.state.timestamp == t1
    assert len(track.states) == 2


# --- unassociated hit becomes a new track (real, tiny JPDA: 1 track, 1 detection) -


def test_process_async_event_unassociated_detection_spawns_new_track(tracker):
    d1 = _det(0.0, 0.0, 0.0, T0)
    tracker.process_async_event(T0, {d1})

    t1 = T0 + timedelta(seconds=2)
    d2 = _det(5000.0, 5000.0, 0.0, t1)
    tracker.process_async_event(t1, {d2})

    # The pre-existing track always coasts (see module docstring / bug), and the
    # new detection is unassociated -> a second track is spawned.
    assert len(tracker.tracks) == 2
    lengths = sorted(len(t.states) for t in tracker.tracks)
    assert lengths == [1, 2]


# --- association-update branch (fake data_associator, per task brief) -----------


def _fake_truthy_associator(track, measurement):
    class FakeHypothesis:
        def __bool__(self):
            return True

    hyp = FakeHypothesis()
    hyp.measurement = measurement

    class FakeAssociator:
        def associate(self, tracks, detections, timestamp, **kwargs):
            return {track: [hyp]}

    return FakeAssociator()


def test_process_async_event_association_update_with_sensor_geodetic(tracker, monkeypatch):
    d0 = _det(0.0, 0.0, 0.0, T0)
    tracker.process_async_event(T0, {d0})
    (track,) = tuple(tracker.tracks)

    t1 = T0 + timedelta(seconds=2)
    sensor_geodetic = {"latitude": 62.9, "longitude": 29.8, "altitude": 0.0}
    d1 = _det(1.0, 1.0, 0.0, t1, metadata={"sensor_geodetic": sensor_geodetic})

    fake_updated_state = GaussianState(
        state_vector=StateVector(np.zeros(9)), covar=np.eye(9) * 5.0, timestamp=t1
    )
    update_mock = MagicMock(return_value=fake_updated_state)
    monkeypatch.setattr(tracker, "data_associator", _fake_truthy_associator(track, d1))
    monkeypatch.setattr(tracker.updater, "update", update_mock)

    result = tracker.process_async_event(t1, {d1})

    assert result == {track}  # no new track: d1 was associated, not unassociated
    assert track.state is fake_updated_state
    update_mock.assert_called_once()
    _, kwargs = update_mock.call_args
    assert kwargs == {"sensor_geodetic": sensor_geodetic}
    assert tracker.updater.measurement_model is MEAS_MODEL


def test_process_async_event_association_update_without_sensor_geodetic(tracker, monkeypatch):
    d0 = _det(0.0, 0.0, 0.0, T0)
    tracker.process_async_event(T0, {d0})
    (track,) = tuple(tracker.tracks)

    t1 = T0 + timedelta(seconds=2)
    d1 = _det(1.0, 1.0, 0.0, t1, metadata={"classification": "UAS", "swarm_count": 1})

    fake_updated_state = GaussianState(
        state_vector=StateVector(np.zeros(9)), covar=np.eye(9) * 5.0, timestamp=t1
    )
    update_mock = MagicMock(return_value=fake_updated_state)
    monkeypatch.setattr(tracker, "data_associator", _fake_truthy_associator(track, d1))
    monkeypatch.setattr(tracker.updater, "update", update_mock)

    tracker.process_async_event(t1, {d1})

    update_mock.assert_called_once()
    _, kwargs = update_mock.call_args
    assert kwargs == {}


# --- ndim == 1 bearing-only skip guard (fake data_associator: all-missed) -------


def test_process_async_event_skips_bearing_only_ndim1_detection(tracker, monkeypatch):
    d0 = _det(0.0, 0.0, 0.0, T0)
    tracker.process_async_event(T0, {d0})

    t1 = T0 + timedelta(seconds=2)

    class AllMissedHypothesis:
        def __bool__(self):
            return False

    class AllMissedAssociator:
        def associate(self, tracks, detections, timestamp, **kwargs):
            return {t: [AllMissedHypothesis()] for t in tracks}

    monkeypatch.setattr(tracker, "data_associator", AllMissedAssociator())

    bearing_only = _FakeBearingOnlyDetection(np.array([1.23]), {}, t1)
    real_hit = _det(5000.0, 5000.0, 0.0, t1)

    result = tracker.process_async_event(t1, {bearing_only, real_hit})

    # Original track coasted (1), real_hit spawned a new track (2); bearing_only skipped.
    assert len(result) == 2
    lengths = sorted(len(t.states) for t in result)
    assert lengths == [1, 2]


# --- _prune_stale_tracks (direct, white-box) -------------------------------------


class _FakeTimestampedTrack:
    """Minimal hashable stand-in exposing only `.state.timestamp`."""

    def __init__(self, timestamp):
        self.state = SimpleNamespace(timestamp=timestamp)


def test_prune_stale_tracks_drops_tracks_beyond_timeout(tracker):
    fresh = _FakeTimestampedTrack(T0)
    stale = _FakeTimestampedTrack(T0 - timedelta(seconds=100))
    tracker.tracks = {fresh, stale}

    tracker._prune_stale_tracks(T0, max_coastal_seconds=45.0)

    assert tracker.tracks == {fresh}


def test_prune_stale_tracks_keeps_tracks_within_timeout(tracker):
    fresh = _FakeTimestampedTrack(T0)
    almost_stale = _FakeTimestampedTrack(T0 - timedelta(seconds=30))
    tracker.tracks = {fresh, almost_stale}

    tracker._prune_stale_tracks(T0, max_coastal_seconds=45.0)

    assert tracker.tracks == {fresh, almost_stale}


# --- _initialize_new_track (direct, white-box) -----------------------------------


def test_initialize_new_track_seeds_state_from_detection(tracker):
    d = _det(10.0, 20.0, 5.0, T0)
    tracker._initialize_new_track(d)

    assert len(tracker.tracks) == 1
    (track,) = tuple(tracker.tracks)
    vec = track.state.state_vector
    assert vec[0, 0] == pytest.approx(10.0)
    assert vec[3, 0] == pytest.approx(20.0)
    assert vec[6, 0] == pytest.approx(5.0)
    assert vec[1, 0] == 0.0  # velocity initialized to zero
    assert track.state.timestamp == T0
