"""Tests for context_foundry.fusion.tracker.SapientAsynchronousTracker.

Association gates each track against each detection and then takes the single
best assignment of detections to tracks, so a detection that gates to an
existing track updates it instead of spawning a new one, and no two tracks are
updated from the same hit. The association-update branch is exercised with a
monkeypatched fake data_associator for the two sensor_geodetic sub-branches;
bootstrap/coasting/unassociated-hit paths use genuine tiny association runs.

Per-event cost has to be bounded by the tracker's own limits and nothing else:
the tests below drive it with the geometry that made the previous joint
associator hang inside one call -- many tracks coasting with kilometre-scale
covariance, so their gates overlap into one cluster -- and with events wider
than the per-event detection cap.

Stale-track pruning is measured from each track's last real detection, not from
track.state.timestamp -- coasting appends a prediction that advances the latter
every event, so a purely coasting track would otherwise never expire. That last
detection comes from the tracker's own per-track bookkeeping, because history is
capped and the Update it would be read from is eventually truncated away. A
track is dropped on its uncertainty as well as on its age, which is the same
horizon measured in metres.
"""

import threading
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.array import StateVector
from stonesoup.types.detection import Detection
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.types.update import GaussianStateUpdate

from context_foundry.fusion import config
from context_foundry.fusion import tracker as tracker_module
from context_foundry.fusion.engine import FusionEngine
from context_foundry.fusion.publish import PublishPolicy
from context_foundry.fusion.tracker import (
    DECLARATION_KEY,
    DEFAULT_CONFIRM_HITS,
    GATE_MAHALANOBIS_DISTANCE,
    SapientAsynchronousTracker,
    declared_until,
)

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


def _wide_track(east, north, sigma, timestamp=T0):
    """A track carrying `sigma` metres of position uncertainty, as coasting builds."""
    covar = np.diag([sigma**2, 50.0, 25.0, sigma**2, 50.0, 25.0, (sigma / 2.0) ** 2, 25.0, 12.5])
    return Track(
        [
            GaussianState(
                state_vector=StateVector([east, 0.0, 0.0, north, 0.0, 0.0, 500.0, 0.0, 0.0]),
                covar=covar,
                timestamp=timestamp,
            )
        ]
    )


def _completes_within(call, seconds):
    """Whether call() returns inside `seconds`, without hanging the whole suite if not.

    A daemon thread rather than a wall-clock assertion afterwards: the failure
    being guarded against is a call that never returns at all, which no
    measurement taken after it can report.
    """
    raised = []

    def _work():
        try:
            call()
        except Exception as exc:
            raised.append(exc)

    thread = threading.Thread(target=_work, daemon=True)
    thread.start()
    thread.join(seconds)
    if raised:
        raise raised[0]
    return not thread.is_alive()


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


def test_bootstrap_creates_one_track_per_det(tracker):
    d1 = _det(0.0, 0.0, 0.0, T0)
    d2 = _det(500.0, 500.0, 0.0, T0)

    result = tracker.process_async_event(T0, {d1, d2})

    assert result == tracker.tracks
    assert len(tracker.tracks) == 2
    for track in tracker.tracks:
        assert len(track.states) == 1
        assert track.state.timestamp == T0


# --- coasting (real, tiny association run: 1 track, 0 detections) ----------------


def test_process_async_event_coasts_track_when_no_detections(tracker):
    d1 = _det(0.0, 0.0, 0.0, T0)
    tracker.process_async_event(T0, {d1})
    (track,) = tuple(tracker.tracks)

    t1 = T0 + timedelta(seconds=2)
    tracker.process_async_event(t1, set())

    assert len(tracker.tracks) == 1
    assert track.state.timestamp == t1
    assert len(track.states) == 2


# --- unassociated hit becomes a new track (real run: 1 track, 1 detection) -------


def test_process_async_event_unassociated_detection_spawns_new_track(tracker):
    d1 = _det(0.0, 0.0, 0.0, T0)
    tracker.process_async_event(T0, {d1})

    t1 = T0 + timedelta(seconds=2)
    d2 = _det(5000.0, 5000.0, 0.0, t1)
    tracker.process_async_event(t1, {d2})

    # The pre-existing track has nothing inside its gate and coasts; the new
    # detection is unassociated -> a second track is spawned.
    assert len(tracker.tracks) == 2
    lengths = sorted(len(t.states) for t in tracker.tracks)
    assert lengths == [1, 2]


# --- nearby detection associates instead of spawning (real association run) ------


def test_process_async_event_associates_nearby_det(tracker):
    """A detection close to an existing track updates it rather than spawning a
    second one. This is the whole point of association, and the property that
    breaks first when a gate is mis-sized in either direction."""
    tracker.process_async_event(T0, {_det(0.0, 0.0, 0.0, T0)})

    t1 = T0 + timedelta(seconds=2)
    tracker.process_async_event(t1, {_det(1.0, 1.0, 0.0, t1)})

    assert len(tracker.tracks) == 1
    (track,) = tuple(tracker.tracks)
    assert track.state.timestamp == t1


# --- association-update branch (fake data_associator, per task brief) -----------


def _fake_truthy_associator(track, measurement):
    """An associator that assigns `measurement` to `track`, as a 2D assignment does:
    one hypothesis per track, not a list of them."""

    class FakeHypothesis:
        distance = 0.5

        def __bool__(self):
            return True

    hyp = FakeHypothesis()
    hyp.measurement = measurement

    class FakeAssociator:
        def associate(self, tracks, detections, timestamp, **kwargs):
            return {track: hyp}

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


def test_process_async_event_skips_bearing_only_ndim1_det(tracker, monkeypatch):
    d0 = _det(0.0, 0.0, 0.0, T0)
    tracker.process_async_event(T0, {d0})

    t1 = T0 + timedelta(seconds=2)

    class AllMissedHypothesis:
        distance = float("inf")

        def __bool__(self):
            return False

    class AllMissedAssociator:
        def associate(self, tracks, detections, timestamp, **kwargs):
            return {t: AllMissedHypothesis() for t in tracks}

    monkeypatch.setattr(tracker, "data_associator", AllMissedAssociator())

    bearing_only = _FakeBearingOnlyDetection(np.array([1.23]), {}, t1)
    real_hit = _det(5000.0, 5000.0, 0.0, t1)

    result = tracker.process_async_event(t1, {bearing_only, real_hit})

    # Original track coasted (1), real_hit spawned a new track (2); bearing_only skipped.
    assert len(result) == 2
    lengths = sorted(len(t.states) for t in result)
    assert lengths == [1, 2]


# --- _prune_stale_tracks (direct, white-box) -------------------------------------


def _gaussian_state(timestamp):
    return GaussianState(
        state_vector=StateVector(np.zeros(9)), covar=np.eye(9), timestamp=timestamp
    )


def _update_state(timestamp):
    """A real detection update (subclass of stonesoup Update)."""
    return GaussianStateUpdate(
        state_vector=StateVector(np.zeros(9)), covar=np.eye(9), hypothesis=None, timestamp=timestamp
    )


def test_prune_stale_tracks_drops_tracks_beyond_timeout(tracker):
    fresh = Track([_gaussian_state(T0)])
    stale = Track([_gaussian_state(T0 - timedelta(seconds=100))])
    tracker.tracks = {fresh, stale}

    tracker._prune_stale_tracks(T0)

    assert tracker.tracks == {fresh}


def test_prune_stale_tracks_keeps_tracks_within_timeout(tracker):
    fresh = Track([_gaussian_state(T0)])
    almost_stale = Track([_gaussian_state(T0 - timedelta(seconds=30))])
    tracker.tracks = {fresh, almost_stale}

    tracker._prune_stale_tracks(T0)

    assert tracker.tracks == {fresh, almost_stale}


def test_prune_stale_tracks_uses_last_detection_not_last_coast(tracker):
    """Regression: a track detected 100s ago but coasted (predictions appended)
    right up to now must still be pruned. The old logic read track.state.timestamp
    -- advanced to 'now' by coasting -- so such a track would never expire."""
    track = Track(
        [
            _update_state(T0 - timedelta(seconds=100)),  # last real detection, stale
            _gaussian_state(T0 - timedelta(seconds=50)),  # coast prediction
            _gaussian_state(T0),  # coast prediction, now
        ]
    )
    tracker.tracks = {track}

    tracker._prune_stale_tracks(T0)

    assert tracker.tracks == set()


def test_prune_stale_tracks_keeps_recently_detected_track(tracker):
    track = Track(
        [
            _update_state(T0 - timedelta(seconds=100)),  # old detection
            _update_state(T0 - timedelta(seconds=10)),  # recent detection
            _gaussian_state(T0),  # coast prediction
        ]
    )
    tracker.tracks = {track}
    tracker._last_update_time[track.id] = T0 - timedelta(seconds=10)

    tracker._prune_stale_tracks(T0)

    assert tracker.tracks == {track}


def test_prune_stale_tracks_forgets_the_pruned_tracks_last_update(tracker):
    """The last-update record must not outlive its track, or a long run leaks one
    entry per track it ever held."""
    fresh = Track([_gaussian_state(T0)])
    stale = Track([_gaussian_state(T0 - timedelta(seconds=100))])
    tracker.tracks = {fresh, stale}
    tracker._last_update_time = {fresh.id: T0, stale.id: T0 - timedelta(seconds=100)}

    tracker._prune_stale_tracks(T0)

    assert tracker._last_update_time == {fresh.id: T0}


# --- bounded per-track history ---------------------------------------------------


def test_track_history_stays_bounded_over_many_events():
    """Every event appends a state (and a metadata copy) to every live track, so
    an uncapped run grows for as long as the process stays up."""
    tracker = SapientAsynchronousTracker(max_track_history=10)
    positions = [(0.0, 0.0, 0.0), (2000.0, 2000.0, 0.0), (-2000.0, 1500.0, 0.0)]

    tracker.process_async_event(T0, {_det(*p, T0) for p in positions})
    assert len(tracker.tracks) == 3

    for step in range(1, 41):
        timestamp = T0 + timedelta(seconds=step)
        tracker.process_async_event(timestamp, {_det(*p, timestamp) for p in positions})

    assert len(tracker.tracks) == 3
    lengths = [len(track.states) for track in tracker.tracks]
    assert max(lengths) == 10  # capped, not the 41 states the run appended
    for track in tracker.tracks:
        assert len(track.metadatas) == len(track.states)


def test_truncated_history_keeps_metadata_aligned_with_the_current_state():
    """states and metadatas are index-aligned and track.metadata is the last
    entry, so truncating one without the other misreports every track's
    classification and swarm count."""
    tracker = SapientAsynchronousTracker(max_track_history=3)
    meta = {"classification": "Quadcopter", "swarm_count": 7}

    tracker.process_async_event(T0, {_det(0.0, 0.0, 0.0, T0, metadata=dict(meta))})
    for step in range(1, 11):
        timestamp = T0 + timedelta(seconds=step)
        tracker.process_async_event(
            timestamp, {_det(0.0, 0.0, 0.0, timestamp, metadata=dict(meta))}
        )

    (track,) = tuple(tracker.tracks)
    assert len(track.states) == 3
    assert len(track.metadatas) == 3
    assert track.metadata["classification"] == "Quadcopter"
    assert track.metadata["swarm_count"] == 7


def test_coasting_track_still_expires_after_history_is_truncated():
    """A track whose last real detection has been truncated out of history must
    still expire. Recovering that timestamp by scanning the retained states finds
    no Update at all once the cap is passed, and the track becomes immortal."""
    tracker = SapientAsynchronousTracker(max_track_history=3)
    tracker.process_async_event(T0, {_det(0.0, 0.0, 0.0, T0)})

    for step in range(1, 41):
        tracker.process_async_event(T0 + timedelta(seconds=step), set())
    # 40s of coasting, under the 45s timeout: still alive, history long truncated.
    (track,) = tuple(tracker.tracks)
    assert len(track.states) == 3

    tracker.process_async_event(T0 + timedelta(seconds=46), set())

    assert tracker.tracks == set()
    assert tracker._last_update_time == {}


# --- the population bounds are per-tracker, with the constants as defaults --------


def test_the_bounds_default_to_the_module_constants():
    """The constants are the documented defaults, so a tracker built without
    arguments has to be the one the documentation describes."""
    tracker = SapientAsynchronousTracker()

    assert tracker.max_live_tracks == tracker_module.MAX_LIVE_TRACKS
    assert tracker.max_track_history == tracker_module.MAX_TRACK_HISTORY


def test_the_default_live_track_bound_keeps_a_pass_inside_the_frame_window():
    """One fusion pass has to finish inside a live source's frame window.

    Measured on one core, per-event cost reaches the 250 ms default window at
    ~120 tracks (247 ms at 112, 421 ms at 140). A bound above that lets the
    population reach a point where a pass outlasts the window, sweeps split into
    single-detection events, association degrades and the population grows
    further -- from which the engine does not recover on its own.
    """
    assert tracker_module.MAX_LIVE_TRACKS <= 120


def test_the_history_bound_is_per_tracker():
    """Two engines on one node may be sized differently; the depth cannot be a
    process-wide constant."""
    shallow = SapientAsynchronousTracker(max_track_history=2)
    deep = SapientAsynchronousTracker(max_track_history=6)

    for step in range(10):
        timestamp = T0 + timedelta(seconds=step)
        for engine in (shallow, deep):
            engine.process_async_event(timestamp, {_det(0.0, 0.0, 0.0, timestamp)})

    assert max(len(t.states) for t in shallow.tracks) == 2
    assert max(len(t.states) for t in deep.tracks) == 6


# --- _initialize_new_track (direct, white-box) -----------------------------------


def test_initialize_new_track_seeds_state_from_det(tracker):
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


def test_the_live_track_population_is_bounded(caplog):
    """Detections that associate with nothing each start a track, so clutter or an
    injected feed grows the population until per-event cost passes the event rate
    -- and once fusion is behind, the backlog keeps it there."""
    tracker = SapientAsynchronousTracker(max_live_tracks=8)

    # Scattered hits, far enough apart that none of them associate.
    with caplog.at_level("WARNING"):
        for i in range(40):
            timestamp = T0 + timedelta(seconds=i)
            detection = _det(20000.0 * i, -20000.0 * i, 500.0, timestamp)
            tracker.process_async_event(timestamp, {detection})

    assert len(tracker.tracks) <= 8
    reports = [r for r in caplog.records if "live-track limit" in r.message]
    assert len(reports) == 1  # rate-limited, not one line per refused detection
    assert "detection(s) have not started a track" in reports[0].message
    # The bound this tracker was built with, not whatever the default happens to be.
    assert "At the 8 live-track limit" in reports[0].message


def test_the_bound_keeps_the_tracks_already_being_followed():
    """Shedding the clutter must not shed the targets already under track."""
    tracker = SapientAsynchronousTracker(max_live_tracks=3)

    established = None
    for i in range(3):
        timestamp = T0 + timedelta(seconds=2 * i)
        # One target moving steadily, so it stays a single associated track.
        tracker.process_async_event(timestamp, {_det(100.0 + 40 * i, 200.0, 300.0, timestamp)})
        established = {t.id for t in tracker.tracks}

    for i in range(10):
        timestamp = T0 + timedelta(seconds=10 + i)
        tracker.process_async_event(timestamp, {_det(50000.0 * (i + 1), 0.0, 300.0, timestamp)})

    assert len(tracker.tracks) <= 3
    # The originally-followed track is still there, not evicted for a newcomer.
    assert established & {t.id for t in tracker.tracks}


# --- one pass is bounded by the tracker's own limits, at any input density -------

# Generous enough that a slow machine does not fail it, short enough that a pass
# which never returns is reported as a failure rather than as a hung suite.
BOUNDED_PASS_SECONDS = 30.0


def _coasting_picture(tracker, count, sigma, spacing=400.0):
    """Seeds `count` tracks whose gates all overlap, as a long coast produces."""
    for index in range(count):
        track = _wide_track(spacing * (index % 8), spacing * (index // 8), sigma)
        tracker.tracks.add(track)
        tracker._last_update_time[track.id] = T0
    return tracker


def test_a_dense_frame_against_coasting_tracks_returns_in_bounded_time():
    """The load that has to stay bounded: a sweep's worth of detections arriving
    while many tracks are coasting with kilometre-scale uncertainty.

    Their validation gates then overlap into a single cluster, which is the input
    that makes exact joint association exponential -- measured hanging inside one
    call and allocating to 9.8 GiB against a live 9-node, 80-object feed. Gating
    plus one assignment is polynomial in both terms, so this returns.
    """
    tracker = SapientAsynchronousTracker()
    _coasting_picture(tracker, count=40, sigma=1000.0)
    t1 = T0 + timedelta(seconds=2)
    detections = {
        _det(400.0 * (i % 8) + 50.0, 400.0 * (i // 8) + 50.0, 500.0, t1) for i in range(40)
    }

    assert _completes_within(
        lambda: tracker.process_async_event(t1, detections), BOUNDED_PASS_SECONDS
    )
    assert len(tracker.tracks) <= tracker.max_live_tracks


def test_repeated_dense_frames_leave_the_population_bounded():
    """The population cannot grow its way back into the same wall: each event's
    unassociated hits start tracks, so a dense feed has to end each pass inside
    the live bound rather than a little above it."""
    tracker = SapientAsynchronousTracker(max_live_tracks=30)
    _coasting_picture(tracker, count=30, sigma=1000.0)

    def _sweeps():
        for step in range(1, 6):
            timestamp = T0 + timedelta(seconds=2 * step)
            tracker.process_async_event(
                timestamp,
                {
                    _det(400.0 * (i % 8) + 50.0, 400.0 * (i // 8) + 50.0, 500.0, timestamp)
                    for i in range(40)
                },
            )

    assert _completes_within(_sweeps, BOUNDED_PASS_SECONDS)
    assert len(tracker.tracks) <= 30


def test_no_two_tracks_are_updated_from_one_detection():
    """A detection is evidence of one object, so exactly one track may take it:
    the assignment gives it to the nearest and leaves the rest to coast. Scored
    per track independently the same hit can update several tracks at once, and
    on two overlapping tracks it was measured updating neither."""
    tracker = SapientAsynchronousTracker()
    # Two tracks metres apart, so a single detection gates to both.
    for east in (0.0, 5.0):
        track = _wide_track(east, 0.0, 20.0)
        tracker.tracks.add(track)
        tracker._last_update_time[track.id] = T0

    t1 = T0 + timedelta(seconds=2)
    tracker.process_async_event(t1, {_det(2.0, 0.0, 500.0, t1)})

    # Coasting appends a prediction stamped with the event too, so it is the state
    # type that says which track was actually updated.
    updated = [t for t in tracker.tracks if isinstance(t.state, GaussianStateUpdate)]
    assert len(updated) == 1
    # The other coasted, and the hit did not also start a third track.
    assert len(tracker.tracks) == 2


# --- the per-event detection cap -------------------------------------------------


def test_an_event_at_the_detection_cap_is_fused_whole():
    tracker = SapientAsynchronousTracker(max_event_detections=5)

    tracker.process_async_event(T0, {_det(1000.0 * i, 0.0, 500.0, T0) for i in range(5)})

    assert len(tracker.tracks) == 5


def test_an_event_past_the_detection_cap_is_shed_and_reported(caplog):
    """Gating is (tracks x detections), so an event wider than the cap is the one
    term of a pass's cost a sender still controls. Shedding it is deliberate; a
    pass that never returns could not report anything at all."""
    tracker = SapientAsynchronousTracker(max_event_detections=5)

    with caplog.at_level("WARNING"):
        for step in range(3):
            timestamp = T0 + timedelta(seconds=2 * step)
            tracker.process_async_event(
                timestamp, {_det(1000.0 * i, 0.0, 500.0, timestamp) for i in range(20)}
            )

    assert len(tracker.tracks) == 5
    reports = [r for r in caplog.records if "past the 5 a single event" in r.message]
    assert len(reports) == 1  # rate-limited, not one line per event
    assert "An event carried 20 detections" in reports[0].message
    assert "15 detection(s) have been shed so far" in reports[0].message


def test_the_detections_kept_at_the_cap_do_not_depend_on_set_ordering():
    """A flood has to fuse the same way twice, or nothing about a run past the cap
    can be reproduced."""
    detections = [_det(1000.0 * i, 0.0, 500.0, T0) for i in range(20)]

    kept = []
    for order in (detections, list(reversed(detections))):
        tracker = SapientAsynchronousTracker(max_event_detections=5)
        tracker.process_async_event(T0, set(order))
        kept.append(sorted(track.state.state_vector[0, 0] for track in tracker.tracks))

    assert kept[0] == kept[1]


def test_the_detection_cap_defaults_to_the_module_constant():
    assert SapientAsynchronousTracker().max_event_detections == tracker_module.MAX_EVENT_DETECTIONS


# --- the coast horizon, in seconds and in metres ---------------------------------


def test_the_coast_horizon_defaults_to_the_module_constant():
    tracker = SapientAsynchronousTracker()

    assert tracker.max_coast_seconds == tracker_module.MAX_COAST_SECONDS
    assert tracker.max_coast_position_sigma == tracker_module.MAX_COAST_POSITION_SIGMA_M


@pytest.mark.parametrize(
    ("coasted", "survives"),
    [(19, True), (21, False)],
)
def test_the_coast_horizon_is_per_tracker(coasted, survives):
    """A network's revisit interval decides how long a track has to coast, so the
    horizon cannot be a process-wide constant."""
    tracker = SapientAsynchronousTracker(max_coast_seconds=20.0)
    track = Track([_gaussian_state(T0 - timedelta(seconds=coasted))])
    tracker.tracks = {track}

    tracker._prune_stale_tracks(T0)

    assert (tracker.tracks == {track}) is survives


@pytest.mark.parametrize(
    ("sigma", "survives"),
    [(999.0, True), (1001.0, False)],
)
def test_a_track_is_dropped_once_its_uncertainty_outgrows_the_gate(sigma, survives):
    """The horizon in metres, which is what holds when the one in seconds is set
    too long: a gate is measured in the track's own sigmas, so a track coasting
    with kilometre covariance accepts most of the picture -- at a smaller distance
    than the track actually following a target reports -- and takes its
    detections away."""
    tracker = SapientAsynchronousTracker(
        max_coast_seconds=10_000.0, max_coast_position_sigma=1000.0
    )
    track = _wide_track(0.0, 0.0, sigma)
    tracker.tracks = {track}
    tracker._last_update_time[track.id] = T0

    tracker._prune_stale_tracks(T0)

    assert (tracker.tracks == {track}) is survives
    # Dropped on geometry alone: the time horizon here is hours away.
    assert (tracker._last_update_time == {}) is not survives


def test_dropping_over_uncertain_tracks_is_reported(caplog):
    """Silently shedding tracks the operator can still see on the feed reads as a
    sensor problem; the warning says which bound did it and what to size."""
    tracker = SapientAsynchronousTracker(
        max_coast_seconds=10_000.0, max_coast_position_sigma=1000.0
    )
    with caplog.at_level("WARNING"):
        for step in range(3):
            track = _wide_track(0.0, 0.0, 4000.0)
            tracker.tracks = {track}
            tracker._last_update_time[track.id] = T0
            tracker._prune_stale_tracks(T0 + timedelta(seconds=step))

    reports = [r for r in caplog.records if "position uncertainty passed" in r.message]
    assert len(reports) == 1  # rate-limited, not one line per pruned track
    assert "Dropped 1 track(s) whose position uncertainty passed 1000 m, 1 so far" in (
        reports[0].message
    )
    assert "--max-coast-seconds (10000 s)" in reports[0].message


def test_a_coasting_track_survives_inside_both_coast_bounds():
    """The default horizon has to remain usable: a track that misses a revisit
    keeps coasting, and only the tracker's bounds end it."""
    tracker = SapientAsynchronousTracker()
    tracker.process_async_event(T0, {_det(0.0, 0.0, 500.0, T0)})

    for step in range(1, 21):
        tracker.process_async_event(T0 + timedelta(seconds=2 * step), set())

    # 40s of coasting under the 45s default, and still inside the metre bound.
    assert len(tracker.tracks) == 1


# --- association by a sensor's stable objectId (Registration tracking_type) -------


def _id_det(e, n, timestamp, object_id, stable=True, node_id="node-A"):
    return _det(
        e,
        n,
        0.0,
        timestamp,
        metadata={
            "classification": "UAS",
            "swarm_count": 1,
            "nodeId": node_id,
            "objectId": object_id,
            "stable_object_id": stable,
        },
    )


def _two_tracks(tracker, stable=True):
    """Tracks A at the origin and B 60 m east, each seeded from its own objectId."""
    tracker.process_async_event(
        T0, {_id_det(0.0, 0.0, T0, "X", stable), _id_det(60.0, 0.0, T0, "Y", stable)}
    )
    a = next(t for t in tracker.tracks if abs(t.state.state_vector[0, 0]) < 1.0)
    b = next(t for t in tracker.tracks if t is not a)
    return a, b


def test_a_stable_object_id_keeps_its_track_over_a_nearer_one(tracker):
    """X's report lands 5 m from track B, yet the sensor says it is X, which it
    last fused into A: A takes it, and B, with nothing, coasts."""
    a, b = _two_tracks(tracker)

    t1 = T0 + timedelta(seconds=2)
    tracker.process_async_event(t1, {_id_det(55.0, 0.0, t1, "X")})

    assert isinstance(a.state, GaussianStateUpdate)
    assert not isinstance(b.state, GaussianStateUpdate)
    assert len(tracker.tracks) == 2


def test_without_a_stable_object_id_the_nearest_track_takes_the_report(tracker):
    a, b = _two_tracks(tracker, stable=False)

    t1 = T0 + timedelta(seconds=2)
    tracker.process_async_event(t1, {_id_det(55.0, 0.0, t1, "X", stable=False)})

    assert isinstance(b.state, GaussianStateUpdate)
    assert not isinstance(a.state, GaussianStateUpdate)


def test_an_object_id_is_followed_per_sensor(tracker):
    """Another sensor's object called X is a different object."""
    a, b = _two_tracks(tracker)

    t1 = T0 + timedelta(seconds=2)
    tracker.process_async_event(t1, {_id_det(55.0, 0.0, t1, "X", node_id="node-B")})

    assert isinstance(b.state, GaussianStateUpdate)
    assert not isinstance(a.state, GaussianStateUpdate)


def test_a_stable_object_id_far_outside_its_gate_is_not_trusted(tracker):
    a, b = _two_tracks(tracker)

    t1 = T0 + timedelta(seconds=2)
    tracker.process_async_event(t1, {_id_det(5000.0, 0.0, t1, "X")})

    assert not isinstance(a.state, GaussianStateUpdate)
    # The report started a track of its own, and X now names that one, not A.
    (new,) = tracker.tracks - {a, b}
    assert tracker._object_tracks[("node-A", "X")] == new.id


def test_a_new_track_is_bound_to_the_object_id_that_started_it(tracker):
    a, b = _two_tracks(tracker)
    assert tracker._object_tracks == {("node-A", "X"): a.id, ("node-A", "Y"): b.id}


def test_reports_without_an_object_id_are_never_bound(tracker):
    tracker.process_async_event(T0, {_id_det(0.0, 0.0, T0, None)})
    assert tracker._object_tracks == {}


def test_a_pruned_track_forgets_its_object_id(tracker):
    _two_tracks(tracker)
    tracker._prune_stale_tracks(T0 + timedelta(seconds=tracker.max_coast_seconds + 1))

    assert tracker.tracks == set()
    assert tracker._object_tracks == {}


def test_position_association_is_skipped_when_every_track_is_bound_by_id(tracker):
    tracker.process_async_event(T0, {_id_det(0.0, 0.0, T0, "X")})
    (track,) = tuple(tracker.tracks)
    tracker.data_associator = MagicMock()

    t1 = T0 + timedelta(seconds=2)
    tracker.process_async_event(t1, {_id_det(3.0, 0.0, t1, "X")})

    tracker.data_associator.associate.assert_not_called()
    assert isinstance(track.state, GaussianStateUpdate)
    assert len(tracker.tracks) == 1


def test_new_tracks_stay_tentative_until_confirmed():
    tracker = SapientAsynchronousTracker(confirm_hits=3)
    t1, t2, t3 = T0, T0 + timedelta(seconds=2), T0 + timedelta(seconds=4)

    assert tracker.process_async_event(t1, {_det(0.0, 0.0, 0.0, t1)}) == set()
    assert tracker.process_async_event(t2, {_det(0.0, 0.0, 0.0, t2)}) == set()
    confirmed = tracker.process_async_event(t3, {_det(0.0, 0.0, 0.0, t3)})

    assert len(confirmed) == 1
    assert tracker.confirmed_tracks() == confirmed


def test_a_tentative_track_that_misses_its_window_is_dropped():
    tracker = SapientAsynchronousTracker(confirm_hits=3, confirm_window_seconds=10)
    tracker.process_async_event(T0, {_det(0.0, 0.0, 0.0, T0)})
    # A stray report elsewhere, 11 s later: the first tentative track never confirmed.
    later = T0 + timedelta(seconds=11)
    tracker.process_async_event(later, {_det(9000.0, 9000.0, 0.0, later)})

    assert len(tracker.tracks) == 1
    assert tracker.confirmed_tracks() == set()


@pytest.mark.parametrize("hits, window", [(0, 15.0), (3, 0.0)])
def test_rejects_bad_confirmation_settings(hits, window):
    with pytest.raises(ValueError):
        SapientAsynchronousTracker(confirm_hits=hits, confirm_window_seconds=window)


def _det_without_height(e, n, timestamp):
    from stonesoup.models.measurement.linear import LinearGaussian

    model = LinearGaussian(ndim_state=9, mapping=(0, 3), noise_covar=np.diag([25.0, 25.0]))
    d = Detection(state_vector=np.array([[e], [n]]), measurement_model=model, timestamp=timestamp)
    d.metadata = {"classification": "UAS", "swarm_count": 1}
    return d


def test_a_track_started_without_height_takes_it_from_the_next_report():
    tracker = SapientAsynchronousTracker()
    tracker.process_async_event(T0, {_det_without_height(0.0, 0.0, T0)})
    (track,) = tracker.tracks
    assert track.state.covar[6, 6] == pytest.approx(1000.0**2)

    later = T0 + timedelta(seconds=2)
    tracker.process_async_event(later, {_det(0.0, 0.0, 400.0, later)})

    (track,) = tracker.tracks
    assert track.state.state_vector[6, 0] == pytest.approx(400.0, abs=5.0)


def test_a_report_without_height_joins_a_track_far_above_the_ground():
    tracker = SapientAsynchronousTracker()
    tracker.process_async_event(T0, {_det(0.0, 0.0, 400.0, T0)})
    later = T0 + timedelta(seconds=2)
    tracker.process_async_event(later, {_det_without_height(3.0, -2.0, later)})

    assert len(tracker.tracks) == 1


def test_the_confirmation_window_holds_three_revisits_of_a_slow_radar():
    # A long-range radar revisiting every 12 s confirms an object on its third report.
    tracker = SapientAsynchronousTracker(confirm_hits=3)
    times = [T0 + timedelta(seconds=12 * i) for i in range(3)]
    for t in times[:2]:
        assert tracker.process_async_event(t, {_det(0.0, 0.0, 0.0, t)}) == set()
    assert len(tracker.process_async_event(times[2], {_det(0.0, 0.0, 0.0, times[2])})) == 1


def test_a_new_track_may_be_moving_fast_horizontally_only():
    tracker = SapientAsynchronousTracker()
    tracker.process_async_event(T0, {_det(0.0, 0.0, 0.0, T0)})
    (track,) = tracker.tracks
    variances = np.diag(track.state.covar)
    assert variances[1] == pytest.approx(tracker_module.DEFAULT_INITIAL_VELOCITY_VARIANCE)
    assert variances[4] == pytest.approx(tracker_module.DEFAULT_INITIAL_VELOCITY_VARIANCE)
    # Acceleration keeps the old, small spread.
    assert variances[2] == pytest.approx(25.0)


def _follow_fast_object(revisit_seconds):
    # 800 m/s east, reported with a stable object id at a fixed revisit.
    tracker = SapientAsynchronousTracker(confirm_hits=3)
    confirmed = set()
    for i in range(5):
        seconds = revisit_seconds * i
        t = T0 + timedelta(seconds=seconds)
        confirmed = tracker.process_async_event(t, {_id_det(800.0 * seconds, 0.0, t, "fast-1")})
    return tracker, confirmed


def test_a_fast_object_with_a_slow_revisit_is_followed():
    # A long-range radar revisiting every 15 s, as the Alakurtti missile is first seen.
    tracker, confirmed = _follow_fast_object(15)
    assert len(tracker.tracks) == 1
    assert len(confirmed) == 1


def test_a_fast_object_with_a_fast_revisit_is_followed():
    # A single radar revisiting every 5 s: 4 km between reports, inside the gate of
    # a track allowed to start at 100 m/s.
    tracker, confirmed = _follow_fast_object(5)
    assert len(tracker.tracks) == 1
    assert len(confirmed) == 1


# --- track ids fixed by the detection that seeds them -------------------------


def _seeded(node="RAD-1", object_id="obj-1", east=0.0, seconds=0.0):
    detection = Detection(
        state_vector=np.array([[east], [0.0], [100.0]]),
        measurement_model=MEAS_MODEL,
        timestamp=T0 + timedelta(seconds=seconds),
    )
    detection.metadata = {"nodeId": node, "objectId": object_id}
    return detection


def test_a_track_seeded_by_the_same_detection_gets_the_same_id():
    first = SapientAsynchronousTracker()
    second = SapientAsynchronousTracker()
    (a,) = first.process_async_event(T0, {_seeded()})
    (b,) = second.process_async_event(T0, {_seeded()})
    assert a.id == b.id


def test_tracks_seeded_by_different_detections_get_different_ids():
    tracker = SapientAsynchronousTracker()
    tracks = tracker.process_async_event(T0, {_seeded(), _seeded(east=5000.0, object_id="obj-2")})
    assert len({track.id for track in tracks}) == 2


def test_two_live_tracks_seeded_alike_are_told_apart():
    tracker = SapientAsynchronousTracker()
    tracker._initialize_new_track(_seeded())
    tracker._initialize_new_track(_seeded())
    assert len({track.id for track in tracker.tracks}) == 2


def test_a_track_id_does_not_depend_on_where_the_tracking_frame_starts():
    # The same report projected about another origin, as after a restart, keeps its id.
    first = SapientAsynchronousTracker()
    second = SapientAsynchronousTracker()
    (a,) = first.process_async_event(T0, {_seeded(east=0.0)})
    (b,) = second.process_async_event(T0, {_seeded(east=1234.5)})
    assert a.id == b.id


def test_two_trackers_in_different_namespaces_mint_different_ids():
    one = SapientAsynchronousTracker(id_namespace="fusion-of-pipeline-a")
    other = SapientAsynchronousTracker(id_namespace="fusion-of-pipeline-b")
    (a,) = one.process_async_event(T0, {_seeded()})
    (b,) = other.process_async_event(T0, {_seeded()})
    assert a.id != b.id


def test_a_reports_own_id_tells_two_reports_of_one_instant_apart():
    tracker = SapientAsynchronousTracker()
    first, second = _seeded(east=0.0), _seeded(east=8000.0)
    first.metadata["reportId"], second.metadata["reportId"] = "r-1", "r-2"
    tracks = tracker.process_async_event(T0, {first, second})
    assert len({track.id for track in tracks}) == 2


def test_a_stable_object_id_names_the_track_whatever_report_started_it():
    # After a restart a track is re-seeded from a later report of the same object.
    first, later = _seeded(seconds=0.0), _seeded(seconds=600.0)
    for det in (first, later):
        det.metadata["stable_object_id"] = True
    a = next(iter(SapientAsynchronousTracker().process_async_event(T0, {first})))
    b = next(
        iter(SapientAsynchronousTracker().process_async_event(T0 + timedelta(seconds=600), {later}))
    )
    assert a.id == b.id


def test_without_a_stable_object_id_a_later_report_starts_a_new_id():
    first, later = _seeded(seconds=0.0), _seeded(seconds=600.0)
    a = next(iter(SapientAsynchronousTracker().process_async_event(T0, {first})))
    b = next(
        iter(SapientAsynchronousTracker().process_async_event(T0 + timedelta(seconds=600), {later}))
    )
    assert a.id != b.id


# --- staleness before association, swarm count over a window (I-64, I-63) --------


def test_a_track_past_its_coast_horizon_does_not_take_a_far_away_detection():
    # Silent longer than max_coast_seconds, the track's uncertainty has grown so wide
    # that matching it first would let it absorb a new object hundreds of km away.
    tracker = SapientAsynchronousTracker(confirm_hits=1)
    tracker.process_async_event(T0, {_det(0.0, 0.0, 0.0, T0)})
    (old,) = tracker.tracks
    later = T0 + timedelta(seconds=tracker.max_coast_seconds + 155)
    tracker.process_async_event(later, {_det(0.0, 333_000.0, 0.0, later)})
    (new,) = tracker.tracks
    assert new.id != old.id, "the new object starts a track of its own"


def test_a_tracks_swarm_count_is_the_largest_seen_within_the_window():
    tracker = SapientAsynchronousTracker(confirm_hits=1)
    tracker.process_async_event(T0, {_det(0.0, 0.0, 0.0, T0, {"swarm_count": 10})})
    soon = T0 + timedelta(seconds=5)
    tracker.process_async_event(soon, {_det(50.0, 0.0, 0.0, soon, {"swarm_count": 1})})
    (track,) = tracker.tracks
    # A sensor that counts single drones does not shrink the swarm another just counted.
    assert track.metadata["swarm_count"] == 10
    late = soon + timedelta(seconds=tracker_module.SWARM_COUNT_WINDOW_SECONDS + 1)
    tracker.process_async_event(late, {_det(100.0, 0.0, 0.0, late, {"swarm_count": 1})})
    (track,) = tracker.tracks
    assert track.metadata["swarm_count"] == 1, "past the window only what was seen since counts"


def test_an_update_without_a_swarm_count_keeps_the_windows_largest():
    # A CoT event carries no swarm count.
    tracker = SapientAsynchronousTracker(confirm_hits=1)
    tracker.process_async_event(T0, {_det(0.0, 0.0, 0.0, T0, {"swarm_count": 6})})
    soon = T0 + timedelta(seconds=3)
    tracker.process_async_event(soon, {_det(30.0, 0.0, 0.0, soon, {"cot_type": "a-u-A"})})
    (track,) = tracker.tracks
    assert track.metadata["swarm_count"] == 6


def test_a_swarm_count_ages_out_when_only_cot_updates_follow():
    # Stone Soup carries a track's metadata forward on every update, so a count left in
    # place would read 10 (and hostile) for as long as CoT events keep the track alive.
    tracker = SapientAsynchronousTracker(confirm_hits=1)
    tracker.process_async_event(T0, {_det(0.0, 0.0, 0.0, T0, {"swarm_count": 10})})
    for step in range(1, 61):
        moment = T0 + timedelta(seconds=5 * step)
        tracker.process_async_event(
            moment, {_det(5.0 * step, 0.0, 0.0, moment, {"cot_type": "a-u-A"})}
        )
        (track,) = tracker.tracks
        if 5 * step <= tracker_module.SWARM_COUNT_WINDOW_SECONDS:
            assert track.metadata["swarm_count"] == 10
    assert "swarm_count" not in track.metadata, "past the window a CoT-only track has no count"


# --- Linear paths -----------------------------------------------------------------------


def test_linear_measurement_prediction_and_update_match_the_unscented_ones():
    from datetime import datetime

    import numpy as np
    from stonesoup.types.detection import Detection
    from stonesoup.types.hypothesis import SingleHypothesis
    from stonesoup.types.prediction import GaussianStatePrediction
    from stonesoup.updater.kalman import UnscentedKalmanUpdater

    from context_foundry.fusion.measurement import DEFAULT_MEASUREMENT_MODEL
    from context_foundry.fusion.tracker import _LinearWhereLinearUpdater

    t = datetime(2026, 10, 27)
    state = GaussianStatePrediction(
        np.arange(9.0).reshape(9, 1) * 10, np.diag(np.arange(1.0, 10)) * 50 + 1, timestamp=t
    )
    det = Detection(
        np.array([[5.0], [33.0], [62.0]]), timestamp=t, measurement_model=DEFAULT_MEASUREMENT_MODEL
    )
    unscented, linear = UnscentedKalmanUpdater(), _LinearWhereLinearUpdater()
    a = unscented.predict_measurement(state, DEFAULT_MEASUREMENT_MODEL)
    b = linear.predict_measurement(state, DEFAULT_MEASUREMENT_MODEL)
    np.testing.assert_allclose(a.state_vector, b.state_vector, atol=1e-9)
    np.testing.assert_allclose(a.covar, b.covar, atol=1e-9)
    np.testing.assert_allclose(a.cross_covar, b.cross_covar, atol=1e-9)
    ua, ub = (
        unscented.update(SingleHypothesis(state, det)),
        linear.update(SingleHypothesis(state, det)),
    )
    np.testing.assert_allclose(ua.state_vector, ub.state_vector, atol=1e-9)
    np.testing.assert_allclose(ua.covar, ub.covar, atol=1e-9)


def test_a_non_linear_measurement_model_keeps_the_unscented_transform():
    from datetime import datetime
    from unittest import mock

    import numpy as np
    from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
    from stonesoup.types.prediction import GaussianStatePrediction
    from stonesoup.updater.kalman import KalmanUpdater

    from context_foundry.fusion.tracker import _LinearWhereLinearUpdater

    model = CartesianToBearingRange(ndim_state=9, mapping=(0, 3), noise_covar=np.diag([0.01, 25.0]))
    state = GaussianStatePrediction(
        np.array([[100.0], [0], [0], [50.0], [0], [0], [10.0], [0], [0]]),
        np.eye(9) * 10,
        timestamp=datetime(2026, 10, 27),
    )
    with mock.patch.object(KalmanUpdater, "predict_measurement") as linear:
        prediction = _LinearWhereLinearUpdater().predict_measurement(state, model)
    linear.assert_not_called()
    assert prediction.state_vector.shape == (2, 1)


def test_the_tracker_predicts_with_the_linear_kalman_predictor():
    from stonesoup.predictor.kalman import KalmanPredictor, UnscentedKalmanPredictor

    from context_foundry.fusion.tracker import SapientAsynchronousTracker

    predictor = SapientAsynchronousTracker().predictor
    assert isinstance(predictor, KalmanPredictor)
    assert not isinstance(predictor, UnscentedKalmanPredictor)


# --- Pre-gate ---------------------------------------------------------------------------


def _random_track(rng, timestamp):
    from stonesoup.types.state import GaussianState
    from stonesoup.types.track import Track

    sv = np.zeros((9, 1))
    for i in (0, 3, 6):
        sv[i, 0] = rng.uniform(-3000, 3000)
        sv[i + 1, 0] = rng.normal(0, 60)
        sv[i + 2, 0] = rng.normal(0, 2)
    root = rng.normal(0, 1, (9, 9)) * rng.uniform(0.5, 60)
    return Track([GaussianState(sv, root @ root.T + np.eye(9), timestamp=timestamp)])


def test_the_pre_gate_never_leaves_out_a_track_inside_the_gate(tracker):
    from stonesoup.types.detection import Detection

    rng = np.random.default_rng(7)
    left_out = 0
    for _ in range(3000):
        track = _random_track(rng, T0)
        when = T0 + timedelta(seconds=float(rng.uniform(0, 6)))
        z = np.array(
            [[rng.uniform(-3500, 3500)], [rng.uniform(-3500, 3500)], [rng.uniform(-500, 500)]]
        )
        det = Detection(z, measurement_model=MEAS_MODEL, timestamp=when)
        prediction = tracker.predictor.predict(track, timestamp=when)
        measurement = tracker.updater.predict_measurement(prediction, MEAS_MODEL)
        distance = tracker.hypothesiser.measure(measurement, det)
        if not tracker._may_gate(track, {det}):
            left_out += 1
            assert distance > GATE_MAHALANOBIS_DISTANCE
    assert left_out > 1000  # the bound does leave far tracks out


def test_the_pre_gate_keeps_a_near_track_and_leaves_out_a_far_one(tracker):
    tracker.process_async_event(T0, {_det(0.0, 0.0, 100.0, T0)})
    (track,) = tracker.tracks
    later = T0 + timedelta(seconds=1)
    assert tracker._may_gate(track, {_det(10.0, -5.0, 100.0, later)})
    assert not tracker._may_gate(track, {_det(50_000.0, 0.0, 100.0, later)})
    assert tracker._may_gate(
        track, {_det(50_000.0, 0.0, 100.0, later), _det(3.0, 0.0, 100.0, later)}
    )
    assert not tracker._may_gate(track, {_det_without_height(80_000.0, 0.0, later)})


def test_the_pre_gate_keeps_what_it_cannot_bound(tracker):
    tracker.process_async_event(T0, {_det(0.0, 0.0, 100.0, T0)})
    (track,) = tracker.tracks
    earlier = T0 - timedelta(seconds=1)
    assert tracker._may_gate(track, {_det(50_000.0, 0.0, 100.0, earlier)})
    assert tracker._may_gate(track, {_FakeBearingOnlyDetection(np.array([1.23]), {}, T0)})


def test_a_track_out_of_reach_still_coasts(tracker):
    tracker.process_async_event(T0, {_det(0.0, 0.0, 100.0, T0)})
    (track,) = tracker.tracks
    later = T0 + timedelta(seconds=2)
    tracker.process_async_event(later, {_det(90_000.0, 0.0, 100.0, later)})
    assert track.state.timestamp == later
    assert len(track.states) == 2


# --- Constant-acceleration predictor -------------------------------------------------------


def test_the_constant_acceleration_predictor_matches_the_general_kalman_predictor():
    from stonesoup.predictor.kalman import KalmanPredictor
    from stonesoup.types.state import GaussianState

    from context_foundry.fusion.models import create_9d_constant_acceleration_model
    from context_foundry.fusion.tracker import _ConstantAccelerationPredictor

    model = create_9d_constant_acceleration_model(q_process_noise=0.2)
    general, fast = KalmanPredictor(model), _ConstantAccelerationPredictor(model)
    rng = np.random.default_rng(11)
    for _ in range(500):
        root = rng.normal(0, 1, (9, 9)) * rng.uniform(0.5, 50)
        prior = GaussianState(rng.normal(0, 500, (9, 1)), root @ root.T + np.eye(9), timestamp=T0)
        # Backwards too: a late event predicts a track back to it, and the noise still grows.
        when = T0 + timedelta(seconds=float(rng.uniform(-2, 8)))
        a, b = general.predict(prior, timestamp=when), fast.predict(prior, timestamp=when)
        np.testing.assert_allclose(b.state_vector, a.state_vector, rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(b.covar, a.covar, rtol=1e-12, atol=1e-9)
        assert b.timestamp == when


def test_the_constant_acceleration_predictor_takes_the_general_path_with_extra_arguments():
    from stonesoup.predictor.kalman import KalmanPredictor
    from stonesoup.types.state import GaussianState

    from context_foundry.fusion.models import create_9d_constant_acceleration_model
    from context_foundry.fusion.tracker import _ConstantAccelerationPredictor

    model = create_9d_constant_acceleration_model()
    prior = GaussianState(np.ones((9, 1)), np.eye(9), timestamp=T0)
    when = T0 + timedelta(seconds=2)
    expected = KalmanPredictor(model).predict(prior, timestamp=when, measurement=None)
    prediction = _ConstantAccelerationPredictor(model).predict(
        prior, timestamp=when, measurement=None
    )
    np.testing.assert_allclose(prediction.state_vector, expected.state_vector)
    np.testing.assert_allclose(prediction.covar, expected.covar)


def _self_report(east, timestamp, uid="ATAK-1", cot_type="a-f-G", how="m-g", stale=None):
    det = Detection(StateVector([[east], [0.0], [0.0]]), measurement_model=MEAS_MODEL)
    det.timestamp = timestamp
    det.metadata = {
        "nodeId": "cot",
        "objectId": uid,
        "stable_object_id": True,
        "type": "CoT",
        "cot_type": cot_type,
        "how": how,
        "stale": stale,
    }
    return det


def _sensor_hit(east, timestamp, object_id="o1"):
    det = Detection(StateVector([[east], [0.0], [0.0]]), measurement_model=MEAS_MODEL)
    det.timestamp = timestamp
    det.metadata = {"nodeId": "RAD-1", "objectId": object_id}
    return det


def _seconds(seconds):
    return T0 + timedelta(seconds=seconds)


def _declared(cot_type, until, east=0.0, radius=50.0):
    """The head of a track's declaration: type, fresh until, where it was made, its radius."""
    return (cot_type, until, east, 0.0, radius)


@pytest.mark.parametrize(
    ("how", "declares"),
    [
        ("m-g", True),
        ("m-g-l", True),
        ("m-f", True),
        ("m-i", True),
        ("m-i-x", True),
        ("h-e", True),
        ("m-r", False),
        ("m-p", False),
        ("m-s", False),
        ("h-g-i-g-o", False),
        ("m-gx", False),
        ("m-ix", False),
        ("m", False),
        (None, False),
    ],
)
def test_only_an_objects_own_position_is_a_self_report(how, declares):
    tracker = SapientAsynchronousTracker()
    (track,) = tracker.process_async_event(T0, {_self_report(0.0, T0, how=how)})
    assert (DECLARATION_KEY in track.metadata) is declares


def test_a_self_report_records_its_declaration_on_the_track():
    tracker = SapientAsynchronousTracker()
    (track,) = tracker.process_async_event(T0, {_self_report(0.0, T0)})
    assert track.metadata[DECLARATION_KEY][:5] == _declared("a-f-G", _seconds(60))


def test_a_surveyed_position_declares_as_an_emplaced_sensor_reports_itself():
    tracker = SapientAsynchronousTracker()
    det = _self_report(0.0, T0, cot_type="a-f-G-E-S", how="m-i", stale="2026-01-01T00:04:00Z")
    (track,) = tracker.process_async_event(T0, {det})
    assert track.metadata[DECLARATION_KEY][:5] == _declared("a-f-G-E-S", _seconds(240))


def test_the_uids_latest_self_report_replaces_its_declaration():
    tracker = SapientAsynchronousTracker()
    tracker.process_async_event(T0, {_self_report(0.0, T0, cot_type="a-h-G")})
    later = _seconds(30)
    (track,) = tracker.process_async_event(later, {_self_report(0.0, later)})
    assert track.metadata[DECLARATION_KEY][:5] == _declared("a-f-G", _seconds(90))
    assert tracker._declarations[track.id].key == ("cot", "ATAK-1")


@pytest.mark.parametrize(
    ("stale", "seconds"),
    [
        ("2026-01-01T00:05:00Z", 300.0),
        ("2026-01-01T00:05:00.250+00:00", 300.25),
        (datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc), 120.0),
        ("2026-01-01T00:00:54Z", 60.0),
        ("2025-12-31T23:59:00Z", 60.0),
        ("2026-01-01T00:30:00Z", 600.0),
        (None, 60.0),
        ("", 60.0),
        ("not a time", 60.0),
        (7, 60.0),
        ("2026-01-01T00:05:00", 60.0),
    ],
)
def test_a_declaration_stands_until_its_stale_clamped_to_one_to_ten_minutes(stale, seconds):
    assert declared_until(stale, T0) == T0 + timedelta(seconds=seconds)


def test_a_self_report_declares_until_its_own_stale():
    tracker = SapientAsynchronousTracker()
    det = _self_report(0.0, T0, stale="2026-01-01T00:06:15Z")
    (track,) = tracker.process_async_event(T0, {det})
    assert track.metadata[DECLARATION_KEY][:5] == _declared("a-f-G", _seconds(375))


def test_two_uids_reporting_themselves_at_one_place_are_two_tracks():
    tracker = SapientAsynchronousTracker()
    tracker.process_async_event(T0, {_self_report(0.0, T0, uid="A")})
    later = _seconds(5)
    tracks = tracker.process_async_event(later, {_self_report(0.0, later, uid="B")})
    assert len(tracks) == 2
    declared = sorted(declared.key for declared in tracker._declarations.values())
    assert declared == [("cot", "A"), ("cot", "B")]


def test_two_uids_reporting_themselves_in_one_event_stay_two_tracks():
    tracker = SapientAsynchronousTracker()
    tracker.process_async_event(T0, {_self_report(0.0, T0, uid="A")})
    tracker.process_async_event(T0, {_self_report(1.0, T0, uid="B")})
    for second in range(5, 60, 5):
        t = _seconds(second)
        tracks = tracker.process_async_event(
            t, {_self_report(0.0, t, uid="A"), _self_report(1.0, t, uid="B")}
        )
        assert len(tracks) == 2
        for track in tracks:
            key = tracker._declarations[track.id].key
            assert tracker._object_tracks[key] == track.id


def test_a_self_report_stays_off_a_track_whose_uid_has_gone_quiet():
    tracker = SapientAsynchronousTracker()
    quiet = _self_report(0.0, T0, uid="A", stale="2026-01-01T00:05:00Z")
    (first,) = tracker.process_async_event(T0, {quiet})
    # A sensor keeps A's track alive while A is quiet; A's declaration lasts to 300 s.
    for second in range(20, 120, 20):
        tracker.process_async_event(_seconds(second), {_sensor_hit(0.0, _seconds(second))})
    later = _seconds(125)
    tracks = tracker.process_async_event(later, {_self_report(0.0, later, uid="B")})
    assert len(tracks) == 2
    assert tracker._declarations[first.id].key == ("cot", "A")


def test_a_sensor_still_joins_a_self_reporters_track():
    tracker = SapientAsynchronousTracker()
    (track,) = tracker.process_async_event(T0, {_self_report(0.0, T0, uid="A")})
    tracks = tracker.process_async_event(_seconds(2), {_sensor_hit(0.0, _seconds(2))})
    assert tracks == {track}


@pytest.mark.parametrize("how", ["h-g-i-g-o", "m-r", "m-g"])
def test_another_uids_event_never_joins_a_self_reporters_track(how):
    tracker = SapientAsynchronousTracker()
    (track,) = tracker.process_async_event(T0, {_self_report(0.0, T0, uid="A")})
    other = _self_report(0.0, _seconds(2), uid="B", how=how)
    tracks = tracker.process_async_event(_seconds(2), {other})
    (own,) = [t for t in tracks if t.id != track.id]
    assert tracker._object_tracks[("cot", "B")] == own.id
    assert {t: bound[0] for t, bound in tracker._cot_uids.items()} == {
        track.id: ("cot", "A"),
        own.id: ("cot", "B"),
    }


def _relayed(east, timestamp, uid):
    det = _self_report(east, timestamp, uid=uid, cot_type="a-h-A", how="m-r")
    det.measurement_model = LinearGaussian(
        ndim_state=9, mapping=(0, 3, 6), noise_covar=np.diag([2500.0, 2500.0, 10000.0])
    )
    return det


@pytest.mark.parametrize(
    "uid_at",
    [
        lambda s: f"C2-{s // 60}",  # a C2 re-initiating its track every minute
        lambda s: f"R-{s}",  # a new uid on every report
        lambda s: "X" if (s // 2) % 2 else "Y",  # two relays of one object
    ],
    ids=["reinitiated", "per-report", "two-relays"],
)
def test_relays_of_one_object_under_churning_uids_stay_one_confirmed_track(uid_at):
    tracker = SapientAsynchronousTracker(confirm_hits=DEFAULT_CONFIRM_HITS)
    confirmed = set()
    for second in range(1, 300, 2):
        t = _seconds(second)
        confirmed |= {
            tr.id
            for tr in tracker.process_async_event(t, {_relayed(20.0 * second, t, uid_at(second))})
        }
    assert len(confirmed) == 1
    assert len(tracker.tracks) == 1


def test_a_self_report_keeps_off_a_relayed_track_until_the_relay_falls_silent():
    tracker = SapientAsynchronousTracker()
    (relayed,) = tracker.process_async_event(T0, {_relayed(0.0, T0, "C2-1")})
    for second in range(10, 120, 10):
        tracker.process_async_event(_seconds(second), {_sensor_hit(0.0, _seconds(second))})
        if second == 30:
            tracker.process_async_event(_seconds(31), {_relayed(0.0, _seconds(31), "C2-1")})
    # 40 s after the relay's last event it still holds the track.
    t = _seconds(71)
    assert len(tracker.process_async_event(t, {_self_report(0.0, t, uid="A")})) == 2
    # A fresh tracker past the coast time from it: the relay's binding has lapsed.
    tracker = SapientAsynchronousTracker()
    (relayed,) = tracker.process_async_event(T0, {_relayed(0.0, T0, "C2-1")})
    for second in range(10, 120, 10):
        tracker.process_async_event(_seconds(second), {_sensor_hit(0.0, _seconds(second))})
    t = _seconds(115)
    assert tracker.process_async_event(t, {_self_report(0.0, t, uid="A")}) == {relayed}


def test_a_long_silent_markers_binding_does_not_outlast_a_rebuild():
    def run(start):
        tracker = SapientAsynchronousTracker(confirm_hits=DEFAULT_CONFIRM_HITS)
        events = [(s, _sensor_hit(5.0 * s, _seconds(s))) for s in range(0, 700, 2)]
        marker = _self_report(0.0, _seconds(1), uid="MARK", cot_type="a-h-G", how="h-g-i-g-o")
        events.append((1, marker))
        events += [(s, _relayed(5.0 * s, _seconds(s), "C2-7")) for s in range(601, 700, 4)]
        for second, det in sorted(events, key=lambda e: (e[0], e[1].metadata["nodeId"])):
            if second >= start:
                tracker.process_async_event(_seconds(second), {det})
        return len(tracker.tracks), len(tracker.confirmed_tracks())

    # The marker's binding lapsed long ago, so the relay joins the sensor's track live as
    # it does in a rebuild that never saw the marker.
    assert run(0) == run(400) == (1, 1)


def test_a_uid_reported_by_another_party_continues_its_own_track_once_it_reports_itself():
    tracker = SapientAsynchronousTracker()
    (track,) = tracker.process_async_event(T0, {_self_report(0.0, T0, uid="A")})
    relayed = _self_report(0.0, _seconds(2), uid="B", how="m-r")
    tracks = tracker.process_async_event(_seconds(2), {relayed})
    (other,) = [t for t in tracks if t.id != track.id]
    own = _self_report(0.0, _seconds(4), uid="B")
    assert tracker.process_async_event(_seconds(4), {own}) == {track, other}
    assert tracker._object_tracks[("cot", "B")] == other.id
    assert tracker._declarations[track.id].key == ("cot", "A")
    assert tracker._declarations[other.id].key == ("cot", "B")


def test_an_object_moving_to_another_track_takes_its_declaration_along():
    tracker = SapientAsynchronousTracker()
    (first,) = tracker.process_async_event(T0, {_self_report(0.0, T0)})
    # A sensor keeps the first track alive; the same uid reappears far out of its gate.
    tracker.process_async_event(_seconds(1), {_sensor_hit(0.0, _seconds(1))})
    later = _seconds(2)
    tracks = tracker.process_async_event(later, {_self_report(50_000.0, later)})
    (second,) = [t for t in tracks if t.id != first.id]
    assert first in tracks
    assert DECLARATION_KEY not in first.metadata
    assert first.id not in tracker._declarations
    assert second.metadata[DECLARATION_KEY][:5] == _declared("a-f-G", _seconds(62), east=50_000.0)
    # The track it left takes another uid's self-reports again.
    t = _seconds(3)
    tracks = tracker.process_async_event(t, {_self_report(0.0, t, uid="B")})
    assert tracker._object_tracks[("cot", "B")] == first.id
    assert tracker._declarations[first.id].key == ("cot", "B")


def test_a_track_dropped_for_coasting_out_takes_its_declarations_along():
    tracker = SapientAsynchronousTracker(max_coast_seconds=10.0)
    (track,) = tracker.process_async_event(T0, {_self_report(0.0, T0)})
    assert track.id in tracker._declarations
    # Another object far away is all that is seen from here; the first coasts out.
    later = _seconds(11)
    tracker.process_async_event(later, {_self_report(50_000.0, later, uid="OTHER")})
    assert track not in tracker.tracks
    assert track.id not in tracker._declarations


def test_a_pruned_track_takes_its_declarations_along():
    tracker = SapientAsynchronousTracker(max_coast_seconds=10.0)
    (track,) = tracker.process_async_event(T0, {_self_report(0.0, T0)})
    tracker._prune_stale_tracks(_seconds(11))
    assert track not in tracker.tracks
    assert track.id not in tracker._declarations


@pytest.mark.parametrize(
    ("make", "published"),
    [
        (lambda t: _self_report(0.0, t), True),
        (lambda t: _self_report(0.0, t, cot_type="a-f-G-E-S", how="m-i"), True),
        (lambda t: _self_report(0.0, t, how="m-r"), False),
        (lambda t: _self_report(0.0, t, how="h-g-i-g-o"), False),
        (lambda t: _sensor_hit(0.0, t), False),
    ],
    ids=["gps", "surveyed", "relayed", "hand-placed", "sensor"],
)
def test_only_a_self_report_publishes_from_its_first_report_under_default_confirmation(
    make, published
):
    config.set_reference_origin(62.9, 29.8, 0.0)
    engine = FusionEngine(
        SapientAsynchronousTracker(confirm_hits=DEFAULT_CONFIRM_HITS),
        PublishPolicy("window", window_seconds=5.0),
    )
    result = engine.process(T0, {make(T0)})
    assert bool(result.published) is published


def test_a_tentative_track_is_confirmed_by_the_first_self_report_it_takes():
    tracker = SapientAsynchronousTracker(confirm_hits=3)
    assert tracker.process_async_event(T0, {_sensor_hit(0.0, T0)}) == set()
    (track,) = tracker.tracks
    tracks = tracker.process_async_event(_seconds(2), {_self_report(0.0, _seconds(2))})
    assert tracks == {track}
    assert track.id not in tracker._tentative


def _camera(east, timestamp, uid="CAM-1", stale=None, sigma=20.0):
    det = _self_report(east, timestamp, uid=uid, cot_type="a-f-G-E-S", how="m-i", stale=stale)
    det.measurement_model = LinearGaussian(
        ndim_state=9, mapping=(0, 3, 6), noise_covar=np.diag([sigma**2, sigma**2, 4 * sigma**2])
    )
    return det


@pytest.mark.parametrize(("how", "tracks"), [("m-i", 2), ("m-g", 1)])
def test_an_emplaced_sensors_track_keeps_a_gate_the_size_of_its_error(how, tracks):
    tracker = SapientAsynchronousTracker()
    first = _camera(0.0, T0)
    first.metadata["how"] = how
    tracker.process_async_event(T0, {first})
    # A sensor's hit 300 m off, half a minute on: a moving track's gate has grown past
    # it, an emplaced sensor's has not.
    later = _seconds(30)
    assert len(tracker.process_async_event(later, {_sensor_hit(300.0, later)})) == tracks


def test_a_surveyed_self_report_holds_the_track_it_joins_still():
    tracker = SapientAsynchronousTracker()
    for second in range(0, 10, 2):
        tracker.process_async_event(_seconds(second), {_sensor_hit(5.0 * second, _seconds(second))})
    (track,) = tracker.tracks
    assert abs(track.state.state_vector[1, 0]) > 1.0
    tracker.process_async_event(_seconds(10), {_camera(50.0, _seconds(10))})
    assert tracker._motion(track) == "still"
    assert all(track.state.state_vector[i, 0] == 0.0 for i in (1, 2, 4, 5, 7, 8))
    coasted = tracker.predictor.predict(track, timestamp=_seconds(40))
    assert np.sqrt(coasted.covar[0, 0]) < 25.0


def test_a_drone_passing_over_a_camera_never_takes_its_track():
    from context_foundry.fusion.augmentor import TacticalContextAugmentor

    config.set_reference_origin(62.9, 29.8, 0.0)
    tracker = SapientAsynchronousTracker()
    augmentor = TacticalContextAugmentor()
    stale = "2026-01-01T00:04:00Z"
    camera_marks, drone_marks = [], []
    for second in range(0, 40):
        t = _seconds(second)
        hits = set()
        if second % 20 == 0:
            hits.add(_camera(0.0, t, stale=stale))
        if second >= 1:
            hits.add(_sensor_hit(-450.0 + 30.0 * second, t, object_id="drone"))
        tracker.process_async_event(t, hits)
        for track in tracker.tracks:
            mark = augmentor.extract_tactical_track(track)
            if track.id in tracker._declarations:
                camera_marks.append((round(track.state.state_vector[0, 0], 1), mark.threat_level))
            elif second >= 3:
                drone_marks.append(mark.threat_level)
    assert set(camera_marks) == {(0.0, "friend")}
    assert set(drone_marks) == {"hostile"}


def _swap(start, end):
    """A's phone reports until 60 s, stale 2 min on; B's, 5 m off, from 70 s; a sensor
    sees the person carrying them every 2 s. Fused from `start`, to `end`."""
    tracker = SapientAsynchronousTracker(id_namespace="ns")
    for second in range(start, end):
        t = _seconds(second)
        stale = (t + timedelta(seconds=120)).isoformat()
        hits = set()
        if second <= 60 and second % 20 == 0:
            hits.add(_self_report(0.0, t, uid="A", stale=stale))
        if second >= 70 and second % 20 == 10:
            hits.add(_self_report(5.0, t, uid="B", stale=stale))
        if second % 2 == 1:
            hits.add(_sensor_hit(0.0, t))
        if hits:
            tracker.process_async_event(t, hits)
    return tracker


def test_a_uid_gone_stale_gives_its_track_up_to_the_uid_reporting_in_its_place():
    before = _swap(0, 175)
    assert len(before.tracks) == 2
    after = _swap(0, 300)
    (track,) = after.tracks
    assert after._declarations[track.id].key == ("cot", "B")
    assert ("cot", "A") not in after._object_tracks
    # A rebuild that never saw A ends the same.
    rebuild = _swap(150, 300)
    (rebuilt,) = rebuild.tracks
    assert rebuild._declarations[rebuilt.id].key == ("cot", "B")


def test_a_released_track_with_no_other_self_reporter_near_stays_undeclared():
    tracker = SapientAsynchronousTracker()
    (track,) = tracker.process_async_event(T0, {_self_report(0.0, T0)})
    for second in range(20, 100, 20):
        tracker.process_async_event(_seconds(second), {_sensor_hit(0.0, _seconds(second))})
    assert tracker.tracks == {track}
    assert track.id not in tracker._declarations
    assert DECLARATION_KEY not in track.metadata
    assert ("cot", "ATAK-1") not in tracker._object_tracks


def test_another_partys_report_of_a_self_reporter_leaves_its_binding():
    tracker = SapientAsynchronousTracker()
    (track,) = tracker.process_async_event(T0, {_self_report(0.0, T0, uid="A")})
    relayed = _self_report(50_000.0, _seconds(2), uid="A", how="m-r")
    tracks = tracker.process_async_event(_seconds(2), {relayed})
    assert len(tracks) == 2
    assert tracker._object_tracks[("cot", "A")] == track.id
    assert tracker._declarations[track.id].key == ("cot", "A")
    assert track.metadata[DECLARATION_KEY][:5] == _declared("a-f-G", _seconds(60))


def test_a_vehicle_gone_quiet_beside_another_reporting_one_keeps_its_track():
    tracker = SapientAsynchronousTracker()
    for second in range(0, 200):
        t = _seconds(second)
        stale = (t + timedelta(seconds=60)).isoformat()
        hits = set()
        if second % 5 == 0:
            if second < 60:
                hits.add(_self_report(15.0 * second, t, uid="A", stale=stale))
            hits.add(_self_report(15.0 * second - 40.0, t, uid="B", stale=stale))
        tracker.process_async_event(t, hits)
        moment = t + timedelta(milliseconds=500)
        sensor = {
            _sensor_hit(15.0 * second, moment, object_id="a"),
            _sensor_hit(15.0 * second - 40.0, moment, object_id="b"),
        }
        tracker.process_async_event(moment, sensor)
        if second == 55:
            a_track = tracker._object_tracks[("cot", "A")]
    # A's declaration ran out at 115 s; B, reporting beside it all along, is another vehicle.
    assert a_track in {track.id for track in tracker.tracks}
    assert len(tracker.tracks) == 2


def _marks_by_uid(events):
    """uid -> set of (threat, whether the record took another uid's report) it was published
    with, fused as the stage does with the default confirmation."""
    from context_foundry.fusion.sapient_raw import cot_object_ulid

    config.set_reference_origin(62.9, 29.8, 0.0)
    engine = FusionEngine(
        SapientAsynchronousTracker(confirm_hits=DEFAULT_CONFIRM_HITS),
        PublishPolicy("window", window_seconds=5.0),
    )
    uids = {cot_object_ulid(det.metadata["objectId"]): det.metadata["objectId"] for det in events}
    owner, marks = {}, {}
    for det in sorted(events, key=lambda d: d.timestamp):
        result = engine.process(det.timestamp, {det})
        for track, tactical in result.published:
            fed = {uids[s["object_id"]] for s in result.sources[str(track.id)]}
            owner.setdefault(track.id, set()).update(fed)
            for uid in fed:
                marks.setdefault(uid, set()).add(tactical.threat_level)
    return owner, marks


def test_a_hand_placed_marker_kilometres_from_standing_phones_never_joins_them():
    events = []
    for second in range(0, 900):
        t = _seconds(second)
        stale = (t + timedelta(minutes=6)).isoformat()
        if second % 90 == 0:
            events.append(_self_report(3200.0, t, uid="PHONE-1", cot_type="a-f-G-U-C", stale=stale))
        if second % 75 == 10:
            events.append(
                _self_report(23_000.0, t, uid="PHONE-2", cot_type="a-f-G-U-C", stale=stale)
            )
        if second % 60 == 30:
            marker = _self_report(
                0.0,
                t,
                uid="MARKER",
                cot_type="a-f-G-U-S-M-T",
                how="h-g-i-g-o",
                stale=(t + timedelta(minutes=10)).isoformat(),
            )
            marker.measurement_model = LinearGaussian(
                ndim_state=9, mapping=(0, 3, 6), noise_covar=np.diag([100.0, 100.0, 400.0])
            )
            events.append(marker)
    owner, marks = _marks_by_uid(events)
    assert all(len(fed) == 1 for fed in owner.values())
    assert marks["PHONE-1"] == marks["PHONE-2"] == {"friend"}


def _phone(east, timestamp, speed, uid="PHONE"):
    det = _self_report(east, timestamp, uid=uid, cot_type="a-f-G-U-C")
    det.metadata["stale"] = (timestamp + timedelta(minutes=6)).isoformat()
    det.metadata["speed"] = speed
    return det


def _every_second(tracker, events, end):
    """Runs the tracker once a second to `end`, as the fusion stage does, coasting every
    track between reports; `events` maps a second to the detections of that second.
    Returns, per second, the confirmed tracks."""
    return {
        second: tracker.process_async_event(_seconds(second), events.get(second, set()))
        for second in range(end)
    }


@pytest.mark.parametrize(("speed", "tracks"), [("0.3", 2), (None, 1), ("12.0", 1)])
def test_a_standing_phones_gate_does_not_reach_a_plot_kilometres_away(speed, tracks):
    tracker = SapientAsynchronousTracker()
    events = {s: {_phone(0.0, _seconds(s), speed)} for s in range(0, 400, 40)}
    events[390] = {_sensor_hit(3000.0, _seconds(390))}
    assert len(_every_second(tracker, events, 391)[390]) == tracks


def test_a_standing_phone_that_drives_off_keeps_its_track():
    tracker = SapientAsynchronousTracker()
    (track,) = tracker.process_async_event(T0, {_phone(0.0, T0, "0.2")})
    assert tracker._motion(track) == "standing"
    events = {40: {_phone(600.0, _seconds(40), "15.0")}}
    assert _every_second(tracker, events, 41)[40] == {track}
    assert tracker._motion(track) is None


@pytest.mark.parametrize(("period", "speed"), [(1, 20.0), (2, 40.0), (5, 20.0), (10, 40.0)])
def test_a_mover_giving_speed_zero_keeps_one_track(period, speed):
    tracker = SapientAsynchronousTracker()
    events = {s: {_phone(speed * s, _seconds(s), "0.0")} for s in range(0, 120, period)}
    _every_second(tracker, events, 120)
    assert len(tracker.tracks) == 1


def test_a_report_away_from_the_previous_one_is_not_held_standing():
    tracker = SapientAsynchronousTracker()
    (track,) = tracker.process_async_event(T0, {_phone(0.0, T0, "0.0")})
    assert tracker._motion(track) == "standing"
    # 10 s on, within 50 m + 10 m: still standing; then 200 m off: moving, whatever it says.
    _every_second(tracker, {10: {_phone(55.0, _seconds(10), "0.0")}}, 11)
    assert tracker._motion(track) == "standing"
    _every_second(tracker, {20: {_phone(255.0, _seconds(20), "0.0")}}, 21)
    assert tracker._motion(track) is None


@pytest.mark.parametrize("speed", ["0.0", "25.0"])
def test_a_standing_phone_that_drives_off_fast_keeps_its_track(speed):
    tracker = SapientAsynchronousTracker()
    (track,) = tracker.process_async_event(T0, {_phone(0.0, T0, "0.0")})
    assert _every_second(tracker, {20: {_phone(375.0, _seconds(20), speed)}}, 21)[20] == {track}


def test_a_standing_tracks_velocity_stays_at_a_walking_pace():
    tracker = SapientAsynchronousTracker()
    # A sensor's hit drags the phone's track off at speed; its next standing report
    # where it was holds it to a walking pace again.
    events = {s: {_phone(0.0, _seconds(s), "0.0")} for s in range(0, 60, 10)}
    events[12] = {_sensor_hit(30.0, _seconds(12))}
    _every_second(tracker, events, 60)
    (track,) = tracker.tracks
    velocity = track.state.state_vector[[1, 4, 7], 0]
    assert np.hypot.reduce(velocity) <= tracker_module.STANDING_SPEED_MPS + 1e-9


@pytest.mark.parametrize("speed", [5.0, 10.0, 15.0, 25.0])
@pytest.mark.parametrize("gap", [10, 20, 30, 40])
@pytest.mark.parametrize("after", ["stops", "moves"])
def test_a_phone_back_from_a_silence_leaves_no_ghost_track(speed, gap, after):
    tracker = SapientAsynchronousTracker()
    events = {s: {_phone(0.0, _seconds(s), "0.0")} for s in range(0, 60, 10)}
    back = 50 + gap
    for k in range(8):
        s = back + 10 * k
        if after == "stops":
            events[s] = {_phone(speed * gap, _seconds(s), "0.0")}
        else:
            events[s] = {_phone(speed * (gap + 10 * k), _seconds(s), str(speed))}
    ids = set()
    for tracks in _every_second(tracker, events, back + 130).values():
        ids |= {track.id for track in tracks}
    assert len(ids) == 1


def test_a_relay_silent_past_the_coast_time_is_associated_by_position_again():
    tracker = SapientAsynchronousTracker()
    tracker.process_async_event(T0, {_sensor_hit(0.0, T0, "o1"), _sensor_hit(100.0, T0, "o2")})
    tracker.process_async_event(_seconds(1), {_relayed(0.0, _seconds(1), "C2-1")})
    first = tracker._object_tracks[("cot", "C2-1")]
    for second in range(5, 65, 5):
        t = _seconds(second)
        tracker.process_async_event(t, {_sensor_hit(0.0, t, "o1"), _sensor_hit(100.0, t, "o2")})
    # 61 s silent: its uid no longer pulls it back to the first track; its position decides.
    t = _seconds(62)
    tracker.process_async_event(t, {_relayed(100.0, t, "C2-1")})
    assert tracker._object_tracks[("cot", "C2-1")] != first
    assert len(tracker.tracks) == 2


@pytest.mark.parametrize("speed", [5.0, 15.0, 25.0])
@pytest.mark.parametrize("gap", [20, 40])
def test_a_phone_that_stops_after_a_silence_stays_where_it_stopped(speed, gap):
    tracker = SapientAsynchronousTracker()
    events = {s: {_phone(0.0, _seconds(s), "0.0")} for s in range(0, 60, 10)}
    back = 50 + gap
    for k in range(8):
        events[back + 10 * k] = {_phone(speed * gap, _seconds(back + 10 * k), "0.0")}
    swing = 0.0
    for second in range(back + 80):
        tracker.process_async_event(_seconds(second), events.get(second, set()))
        if second > back + 10:
            (track,) = tracker.tracks
            east, north = track.state.state_vector[0, 0], track.state.state_vector[3, 0]
            swing = max(swing, float(np.hypot(east - speed * gap, north)))
    assert swing < 50.0


def _follow(truth, speed_of, period, end):
    """Largest distance (m) between the phone's track and the truth, and its top speed,
    with the tracker run every second and the phone reporting every `period` s."""
    tracker = SapientAsynchronousTracker()
    events = {s: {_phone(truth(s), _seconds(s), speed_of(s))} for s in range(0, end, period)}
    error = top = 0.0
    for second in range(end):
        tracker.process_async_event(_seconds(second), events.get(second, set()))
        if second > 2:
            (track,) = tracker.tracks
            error = max(error, abs(track.state.state_vector[0, 0] - truth(second)))
            top = max(top, abs(track.state.state_vector[1, 0]))
    return error, top


def _stop_and_go(second):
    """30 s stopped, 30 s at 10 m/s, over and over."""
    cycle, phase = divmod(second, 60)
    return cycle * 300.0 + (0.0 if phase < 30 else 10.0 * (phase - 30))


@pytest.mark.parametrize(("period", "most"), [(1, 10.0), (10, 150.0), (15, 250.0)])
def test_a_stop_and_go_vehicle_drives_off_without_overshooting(period, most):
    error, top = _follow(_stop_and_go, lambda s: "0.0" if s % 60 < 30 else "10.0", period, 600)
    assert error < most
    assert top < 35.0


@pytest.mark.parametrize(("period", "most"), [(1, 33.0), (2, 35.0), (3, 45.0)])
def test_a_steady_mover_giving_a_standing_speed_is_not_held_for_long(period, most):
    error, _ = _follow(lambda s: 10.0 * s, lambda s: "0.5", period, 300)
    assert error < most
