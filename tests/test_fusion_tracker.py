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

from context_foundry.fusion import tracker as tracker_module
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

    assert result is tracker.tracks
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
