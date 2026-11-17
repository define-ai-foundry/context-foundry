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


def test_a_tracks_row_is_rebuilt_only_when_the_tracker_changes_it(monkeypatch):
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

    # An event that appends a state marks the track changed at its end, as here.
    track.append(GaussianState(track.state.state_vector, track.state.covar, timestamp=T0))
    tracker._reach_dirty.add(track)
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


def _brute_reachable(tracker, tracks, detections):
    """The reach check worked out afresh for each track, as before the kept matrix."""
    tracks = list(tracks)
    if not tracks or not detections:
        return set()
    rows = []
    for track in tracks:
        motion = tracker._motion(track)
        noise = tracker._axis_noise if motion is None else tracker._motion_noise[motion]
        rows.append(reach.track_row(track.state, noise))
    hits = reach.reachable_rows(np.stack(rows), detections, GATE_MAHALANOBIS_DISTANCE**2)
    return {track for track, hit in zip(tracks, hits, strict=True) if hit}


def test_the_kept_matrix_reaches_what_a_fresh_check_would_on_every_call():
    # Targets moving through clutter: tracks start, update, coast and drop, and the
    # matrix fills with dropped tracks' rows until it is rebuilt.
    calls, rebuilt = [], []

    class Checked(SapientAsynchronousTracker):
        def _reachable(self, tracks, detections):
            before = len(self._reach_tracks)
            got = super()._reachable(tracks, detections)
            if len(self._reach_tracks) < before:
                rebuilt.append(1)
            assert got == _brute_reachable(self, tracks, detections)
            calls.append(1)
            return got

    tracker = Checked()
    rng = np.random.default_rng(7)
    model = LinearGaussian(ndim_state=9, mapping=(0, 3, 6), noise_covar=np.diag([400.0] * 3))
    targets = [(rng.uniform(-4e3, 4e3, 3), rng.uniform(-30, 30, 3)) for _ in range(5)]
    for step in range(400):
        when = T0 + timedelta(seconds=step)
        dets = set()
        for start, velocity in targets:
            if rng.random() < 0.7:
                where = start + velocity * step + rng.normal(scale=15.0, size=3)
                dets.add(Detection(StateVector(where), measurement_model=model, timestamp=when))
        for _ in range(rng.integers(0, 4)):
            where = rng.uniform(-6e3, 6e3, 3)
            dets.add(Detection(StateVector(where), measurement_model=model, timestamp=when))
        if dets:
            for det in dets:
                det.metadata = {"classification": "UAS", "swarm_count": 1}
            tracker.process_async_event(when, dets)
    assert len(calls) > 300 and rebuilt, "checked on every call, through a rebuild"


def test_an_ended_declaration_marks_its_tracks_row_changed():
    tracker = SapientAsynchronousTracker()
    rng = np.random.default_rng(8)
    track = _random_track(rng, 0.0)
    tracker.tracks = {track}
    tracker._declarations[track.id] = object()
    tracker._undeclare(track.id)
    assert track in tracker._reach_dirty
