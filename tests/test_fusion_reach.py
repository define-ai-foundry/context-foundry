"""Tests for context_foundry.fusion.reach: the reach check for every track in one array call.

It must answer exactly as the tracker's per-track `_may_gate`, the bound it vectorises,
for any track, detection and time between them, and keep each track's row only as long
as the track's state and motion stay the same.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.array import StateVector
from stonesoup.types.detection import Detection
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track

from context_foundry.fusion import reach
from context_foundry.fusion.tracker import GATE_MAHALANOBIS_DISTANCE, SapientAsynchronousTracker

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _random_track(rng, seconds):
    root = rng.normal(size=(9, 9)) * rng.uniform(1.0, 80.0)
    state_vector = rng.normal(size=9) * np.array([3000, 30, 3] * 3)
    return Track(
        [
            GaussianState(
                StateVector(state_vector),
                root @ root.T,
                timestamp=T0 + timedelta(seconds=seconds),
            )
        ]
    )


def _detection(rng, seconds, mapping=(0, 3, 6)):
    model = LinearGaussian(
        ndim_state=9,
        mapping=mapping,
        noise_covar=np.diag(rng.uniform(4.0, 2500.0, size=len(mapping))),
    )
    return Detection(
        state_vector=np.array([[v] for v in rng.normal(size=len(mapping)) * 3000]),
        measurement_model=model,
        timestamp=T0 + timedelta(seconds=seconds),
    )


def _rows(tracker, tracks):
    return np.stack([reach.track_row(track.state, tracker._axis_noise) for track in tracks])


@pytest.mark.parametrize("seed", range(5))
def test_the_array_check_answers_as_the_per_track_check(seed):
    rng = np.random.default_rng(seed)
    tracker = SapientAsynchronousTracker()
    tracks = [_random_track(rng, rng.uniform(-5.0, 5.0)) for _ in range(300)]
    detections = {_detection(rng, rng.uniform(0.0, 3.0)) for _ in range(6)}
    detections.add(_detection(rng, 1.0, mapping=(0, 3)))

    reached = reach.reachable_rows(_rows(tracker, tracks), detections, GATE_MAHALANOBIS_DISTANCE**2)

    expected = [tracker._may_gate(track, detections) for track in tracks]
    assert reached.tolist() == expected
    assert 0 < sum(expected) < len(tracks)  # the case is not trivially all or none


def test_a_detection_older_than_a_tracks_state_reaches_it():
    rng = np.random.default_rng(1)
    tracker = SapientAsynchronousTracker()
    track = _random_track(rng, 10.0)
    far = _detection(rng, 0.0)

    reached = reach.reachable_rows(_rows(tracker, [track]), {far}, 1.0)

    assert reached.tolist() == [True]


def test_a_detection_that_is_not_a_linear_position_model_reaches_every_track():
    rng = np.random.default_rng(2)
    tracker = SapientAsynchronousTracker()
    tracks = [_random_track(rng, 0.0) for _ in range(3)]
    bearing = _detection(rng, 1.0)
    bearing.measurement_model = object()

    reached = reach.reachable_rows(_rows(tracker, tracks), {bearing}, 1.0)

    assert reached.tolist() == [True, True, True]


def test_no_rows_or_no_detections_reach_nothing():
    rng = np.random.default_rng(3)
    tracker = SapientAsynchronousTracker()

    assert reach.reachable_rows(np.empty((0, 31)), {_detection(rng, 0.0)}, 1.0).size == 0
    assert reach.reachable_rows(_rows(tracker, [_random_track(rng, 0.0)]), set(), 1.0).tolist() == [
        False
    ]


def test_naive_and_aware_times_count_from_the_same_epoch():
    aware = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    assert reach.seconds(aware) == reach.seconds(aware.replace(tzinfo=None))
    later = reach.seconds(aware + timedelta(microseconds=1))
    assert later - reach.seconds(aware) == pytest.approx(1e-6, abs=1e-8)
    assert later > reach.seconds(aware)


def test_a_tracks_row_is_rebuilt_only_when_its_state_changes(monkeypatch):
    tracker = SapientAsynchronousTracker(coast_unmatched=False)
    rng = np.random.default_rng(4)
    track = _random_track(rng, 0.0)
    tracker.tracks = {track}
    built = []
    original = reach.track_row
    monkeypatch.setattr(
        "context_foundry.fusion.tracker.track_row",
        lambda state, noise: built.append(state) or original(state, noise),
    )
    detections = {_detection(rng, 1.0)}

    tracker._reachable({track}, detections)
    tracker._reachable({track}, detections)
    assert len(built) == 1

    track.append(GaussianState(track.state.state_vector, track.state.covar, timestamp=T0))
    tracker._reachable({track}, detections)
    assert len(built) == 2


def test_the_tracker_reaches_the_same_tracks_through_the_array():
    rng = np.random.default_rng(5)
    tracker = SapientAsynchronousTracker()
    tracks = [_random_track(rng, rng.uniform(-5.0, 0.0)) for _ in range(200)]
    detections = {_detection(rng, 1.0) for _ in range(4)}

    reached = tracker._reachable(set(tracks), detections)

    assert reached == {track for track in tracks if tracker._may_gate(track, detections)}
    assert tracker._reachable(set(), detections) == set()
