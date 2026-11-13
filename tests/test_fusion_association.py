"""Tests for context_foundry.fusion.association: the associator's answer in array calls.

It must give every track the same detection, or none, as the tracker's Stone Soup
associator (GNNWith2DAssignment over its hypothesiser) for any tracks, detections, times,
motions and kept-off pairs, in the same order, and leave a detection it cannot model to
the associator.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
from stonesoup.types.array import StateVector
from stonesoup.types.detection import Detection
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track

from context_foundry.fusion import association
from context_foundry.fusion.tracker import SapientAsynchronousTracker

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _track(rng, centre, seconds):
    root = rng.normal(size=(9, 9)) * rng.uniform(0.5, 6.0)
    state_vector = np.array(centre, dtype=float) + rng.normal(size=9) * np.array([40, 3, 0.3] * 3)
    return Track(
        [
            GaussianState(
                StateVector(state_vector),
                root @ root.T + np.eye(9),
                timestamp=T0 + timedelta(seconds=seconds),
            )
        ]
    )


def _detection(rng, near, seconds, mapping=(0, 3, 6)):
    model = LinearGaussian(
        ndim_state=9,
        mapping=mapping,
        noise_covar=np.diag(rng.uniform(4.0, 400.0, size=len(mapping))),
    )
    position = np.array([near[m] for m in mapping]) + rng.normal(size=len(mapping)) * 30
    return Detection(
        state_vector=np.array([[v] for v in position]),
        measurement_model=model,
        timestamp=T0 + timedelta(seconds=seconds),
    )


def _scene(seed, n_tracks=60, n_detections=25):
    """Tracks and detections crowded together, so gates overlap and the assignment matters."""
    rng = np.random.default_rng(seed)
    centres = [rng.normal(size=9) * np.array([300, 10, 1] * 3) for _ in range(8)]
    tracks = {
        _track(rng, centres[i % len(centres)], rng.uniform(-4.0, 0.0)) for i in range(n_tracks)
    }
    detections = set()
    for i in range(n_detections):
        mapping = (0, 3) if i % 7 == 0 else (0, 3, 6)
        stamp = float(rng.choice([0.0, 0.0, 0.4, 1.1]))
        detections.add(_detection(rng, centres[i % len(centres)], stamp, mapping))
    return rng, tracks, detections


def _answer(associations):
    """Each track, in order, with its detection's identity (or None) and distance."""
    return [
        (track, id(h.measurement) if h else None, float(h.distance))
        for track, h in associations.items()
    ]


def _compare(tracker, tracks, detections):
    timestamp = max(det.timestamp for det in detections)
    expected = tracker.data_associator.associate(tracks, detections, timestamp)
    found = association.associate(tracker, tracks, detections, timestamp, tracker._arrays)
    assert _answer(found) == _answer(expected)
    return expected


@pytest.mark.parametrize("seed", range(4))
def test_every_distance_matches_the_measure_to_the_last_bit(seed):
    """Bit for bit, so a tie between identical reports is broken as the associator breaks it."""
    _, tracks, detections = _scene(seed)
    tracker = SapientAsynchronousTracker()
    tracks = list(tracks)
    rows = [tracker._arrays(track) for track in tracks]
    x, p = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    stamps, noise = [r[2] for r in rows], np.array([r[3] for r in rows])
    for det in detections:
        mapping = tuple(det.measurement_model.mapping)
        found = association.distances(x, p, stamps, noise, [det], mapping, gate=1e9)[:, 0]
        for track, distance in zip(tracks, found, strict=True):
            prediction = tracker.predictor.predict(track, timestamp=det.timestamp)
            measured = tracker.updater.predict_measurement(prediction, det.measurement_model)
            assert float(tracker.hypothesiser.measure(measured, det)) == float(distance)


@pytest.mark.parametrize("seed", range(8))
def test_the_array_association_answers_as_the_associator(seed):
    _, tracks, detections = _scene(seed)
    tracker = SapientAsynchronousTracker()

    expected = _compare(tracker, tracks, detections)

    taken = sum(1 for h in expected.values() if h)
    assert 0 < taken < len(tracks)  # the case is not trivially all or none


def test_a_tracks_motion_noise_is_used(monkeypatch):
    _, tracks, detections = _scene(11)
    tracker = SapientAsynchronousTracker()
    motions = list(tracker._motion_noise)
    held = {track: motions[i % len(motions)] for i, track in enumerate(tracks) if i % 2}
    monkeypatch.setattr(tracker, "_motion", held.get)
    tracker.predictor.motion_of = tracker._motion

    _compare(tracker, tracks, detections)


def test_a_kept_off_pair_gets_no_hypothesis(monkeypatch):
    _, tracks, detections = _scene(3)
    tracker = SapientAsynchronousTracker()
    first = _compare(tracker, tracks, detections)
    taken = {track: h.measurement for track, h in first.items() if h}
    kept_off = set(list(taken.items())[:5])
    monkeypatch.setattr(tracker, "_excluded", lambda track, det: (track, det) in kept_off)
    tracker.hypothesiser.excludes = tracker._excluded

    second = _compare(tracker, tracks, detections)

    for track, det in kept_off:
        assert not second[track] or second[track].measurement is not det


def test_no_detections_or_no_tracks():
    _, tracks, detections = _scene(5)
    tracker = SapientAsynchronousTracker()
    timestamp = T0

    assert association.associate(tracker, set(), detections, timestamp, tracker._arrays) == {}
    found = association.associate(tracker, tracks, set(), timestamp, tracker._arrays)
    assert set(found) == tracks and not any(found.values())


def test_a_detection_not_linear_in_position_is_left_to_the_associator():
    rng, tracks, detections = _scene(7)
    tracker = SapientAsynchronousTracker()
    bearing = Detection(
        state_vector=StateVector([[0.1], [500.0]]),
        measurement_model=CartesianToBearingRange(
            ndim_state=9, mapping=(0, 3), noise_covar=np.diag([0.01, 100.0])
        ),
        timestamp=T0,
    )
    velocity = _detection(rng, [0.0] * 9, 0.0, mapping=(1, 4))

    for odd in (bearing, velocity):
        assert (
            association.associate(tracker, tracks, detections | {odd}, T0, tracker._arrays) is None
        )


def test_the_tracker_fuses_the_same_with_either_path():
    """Whole events through the tracker: the same tracks with the array path as without."""

    def run(vector):
        rng, _, _ = _scene(21)
        tracker = SapientAsynchronousTracker()
        tracker.vector_association = vector
        centres = [rng.normal(size=9) * np.array([300, 10, 1] * 3) for _ in range(6)]
        for step in range(30):
            when = T0 + timedelta(seconds=step * 0.5)
            group = {
                _detection(rng, centres[i % len(centres)], step * 0.5)
                for i in range(int(rng.integers(3, 9)))
            }
            tracker.process_async_event(when, group)
        return sorted(
            (
                len(track),
                tuple(np.round(np.asarray(track.state.state_vector, dtype=float).ravel(), 6)),
            )
            for track in tracker.tracks
        )

    assert run(True) == run(False)
