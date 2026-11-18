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


def test_a_clamped_live_report_is_named_at_its_own_time():
    engine = _engine()
    seed = engine.process(T0 + timedelta(seconds=5), [_det(5)], now=T0)
    ((track, _),) = seed.published
    assert seed.sources[str(track.id)][0]["timestamp"] == T0 + timedelta(seconds=5)
    update = engine.process(T0 + timedelta(seconds=9), [_det(9)], now=T0 + timedelta(seconds=6))
    assert update.sources[str(track.id)][0]["timestamp"] == T0 + timedelta(seconds=9)


def test_reset_starts_a_new_scenario():
    engine = _engine()
    engine.process(T0 + timedelta(seconds=60), [_det(60)])
    engine.reset(SapientAsynchronousTracker(confirm_hits=1))
    assert not engine.process(T0, [_det(0)]).dropped


def test_sensor_ids_names_the_sensors_of_an_event():
    assert sensor_ids([_det(0, node="B"), _det(0, node="A")]) == "A, B"
    assert sensor_ids([]) == "an unnamed sensor"


# --- the sensor reports each published record names -------------------------------


def _report(seconds, node="RAD-1", object_id="obj", east=0.0, **metadata):
    detection = _det(seconds, east=east, node=node)
    detection.metadata = {"nodeId": node, "objectId": object_id, **metadata}
    return detection


def _source(node, object_id, seconds):
    return {"node_id": node, "object_id": object_id, "timestamp": T0 + timedelta(seconds=seconds)}


def _only_sources(result):
    ((track, _),) = result.published
    return result.sources[str(track.id)]


def test_a_new_tracks_record_names_the_report_that_started_it():
    engine = _engine()
    result = engine.process(T0, [_report(0)])
    assert _only_sources(result) == [_source("RAD-1", "obj", 0)]


def test_a_tentative_tracks_first_record_names_its_seed_too():
    engine = _engine(confirm_hits=2)
    assert engine.process(T0, [_report(0, node="RAD-1", object_id="a")]).published == []
    result = engine.process(T0 + timedelta(seconds=1), [_report(1, node="RAD-2", object_id="b")])
    assert _only_sources(result) == [_source("RAD-1", "a", 0), _source("RAD-2", "b", 1)]


def test_updates_held_back_by_the_window_are_named_on_the_next_record_then_cleared():
    engine = _engine()
    engine.process(T0, [_report(0)])
    for seconds, node, object_id in ((1, "RAD-2", "a"), (2, "RAD-3", "b")):
        held = engine.process(T0 + timedelta(seconds=seconds), [_report(seconds, node, object_id)])
        assert held.published == []
    result = engine.process(T0 + timedelta(seconds=6), [_report(6)])
    assert _only_sources(result) == [
        _source("RAD-2", "a", 1),
        _source("RAD-3", "b", 2),
        _source("RAD-1", "obj", 6),
    ]
    later = engine.process(T0 + timedelta(seconds=12), [_report(12, "RAD-2", "a")])
    assert _only_sources(later) == [_source("RAD-2", "a", 12)]


def test_a_source_seen_twice_in_a_window_is_named_once_with_its_newest_time():
    engine = _engine()
    engine.process(T0, [_report(0)])
    engine.process(T0 + timedelta(seconds=1), [_report(1, "RAD-2", "a")])
    engine.process(T0 + timedelta(seconds=3), [_report(3, "RAD-2", "a")])
    result = engine.process(T0 + timedelta(seconds=6), [_report(6, "RAD-2", "a")])
    assert _only_sources(result) == [_source("RAD-2", "a", 6)]


def test_a_report_bound_by_its_sensors_stable_object_id_is_named():
    engine = _engine()
    engine.process(T0, [_report(0, object_id="W1", stable_object_id=True)])
    result = engine.process(
        T0 + timedelta(seconds=6), [_report(6, object_id="W1", stable_object_id=True)]
    )
    assert _only_sources(result) == [_source("RAD-1", "W1", 6)]


def test_a_cot_source_is_named_by_derived_ids_and_its_detection_keeps_its_own():
    engine = _engine()
    detection = _report(0, node="cot", object_id="ANDROID-1")
    result = engine.process(T0, [detection])
    # Fixed values: a consumer derives the same ids from a uid, so they must never move.
    assert _only_sources(result) == [
        _source("6e6451cd-0445-5b64-8388-063eab5f2acf", "7PQ5XE0PXMBC2AKKZSWB7TY1C9", 0)
    ]
    assert detection.metadata["nodeId"] == "cot"
    assert detection.metadata["objectId"] == "ANDROID-1"


def test_a_sapient_source_keeps_its_node_id_and_object_id():
    node = "5f7c4ad6-3b2e-4f0b-9b1a-2d6d8e1c0a11"
    engine = _engine()
    result = engine.process(T0, [_report(0, node=node, object_id="01J0000000000000000000000Z")])
    assert _only_sources(result) == [_source(node, "01J0000000000000000000000Z", 0)]


def test_a_predicted_record_with_nothing_measured_since_names_nothing():
    engine = FusionEngine(SapientAsynchronousTracker(confirm_hits=1), PublishPolicy("all"))
    engine.process(T0, [_report(0)])
    result = engine.process(T0 + timedelta(seconds=1), [])
    assert _only_sources(result) == []


def test_a_dropped_event_names_nothing():
    engine = _engine()
    engine.process(T0 + timedelta(seconds=60), [_report(60)])
    assert engine.process(T0, [_report(0)]).sources == {}


def test_pending_sources_of_a_track_no_longer_held_are_dropped():
    engine = FusionEngine(
        SapientAsynchronousTracker(confirm_hits=1, max_coast_seconds=10.0), PublishPolicy("window")
    )
    first = engine.process(T0, [_report(0)])
    ((track, _),) = first.published
    engine.process(T0 + timedelta(seconds=1), [_report(1, "RAD-2", "a")])
    engine.process(T0 + timedelta(seconds=30), [_report(30, "RAD-3", "b", east=5000.0)])
    assert track not in engine.tracker.tracks
    assert engine.pending_sources.take(track.id) == []


def test_reset_forgets_pending_sources():
    engine = _engine()
    first = engine.process(T0, [_report(0)])
    ((track, _),) = first.published
    engine.process(T0 + timedelta(seconds=1), [_report(1, "RAD-2", "a")])
    engine.reset(SapientAsynchronousTracker(confirm_hits=1))
    assert engine.pending_sources.take(track.id) == []


def test_a_track_started_again_under_its_old_id_names_only_its_new_seed():
    engine = _engine()
    first = engine.process(T0, [_report(0, object_id="W1", stable_object_id=True)])
    ((track, _),) = first.published
    engine.process(T0 + timedelta(seconds=1), [_report(1, "RAD-2", "a")])
    # Coasted out, then seeded again by the same sensor object: the same id.
    again = engine.process(
        T0 + timedelta(seconds=60), [_report(60, object_id="W1", stable_object_id=True)]
    )
    ((restarted, _),) = again.published
    assert restarted.id == track.id and restarted is not track
    assert again.sources[str(track.id)] == [_source("RAD-1", "W1", 60)]


@pytest.mark.parametrize(
    ("policy", "coasts"),
    [("window", False), ("updates", False), ("all", True), ("coast", True), ("predicted", True)],
)
def test_tracks_are_coasted_each_event_only_when_a_predicted_state_can_be_sent(policy, coasts):
    engine = FusionEngine(SapientAsynchronousTracker(), PublishPolicy(policy))
    assert engine.tracker.coast_unmatched is coasts

    engine.reset(SapientAsynchronousTracker())
    assert engine.tracker.coast_unmatched is coasts


class _ScanningTracker(SapientAsynchronousTracker):
    """The tracker saying what changed as a scan of every live track would find it:
    every live track touched, the tracks not live before started, the gone ids dropped."""

    def process_async_event(self, timestamp, detection_group):
        before = set(self.tracks)
        result = super().process_async_event(timestamp, detection_group)
        now = set(self.tracks)
        self.touched = now
        self.started = now - before
        self.dropped_ids = {track.id for track in before - now}
        return result


@pytest.mark.parametrize("policy", ["window", "all", "updates"])
def test_sources_from_the_touched_tracks_are_those_of_every_track(policy):
    def run(tracker):
        engine = FusionEngine(tracker, PublishPolicy(policy))
        seen = []
        script = [
            (0, [_report(0, object_id="W1", stable_object_id=True), _report(0, "RAD-2", "a", 900)]),
            (1, [_report(1, "RAD-2", "a", 905)]),
            (2, [_report(2, object_id="W1", stable_object_id=True, east=3)]),
            (3, [_report(3, "RAD-3", "b", 5000)]),
            (4, [_report(4, "RAD-2", "a", 912), _report(4, "RAD-3", "b", 5004)]),
            (60, [_report(60, object_id="W1", stable_object_id=True)]),
            (61, [_report(61, "RAD-2", "c", 2000), _report(61, "RAD-3", "b", 5100)]),
        ]
        for seconds, reports in script:
            result = engine.process(T0 + timedelta(seconds=seconds), reports)
            pending = {k: dict(v) for k, v in engine.pending_sources._pending.items()}
            # As a set: two trackers list their tracks in their own set order.
            published = sorted(track.id for track, _ in result.published)
            sent = dict(engine.publish_policy._last_sent)
            seen.append((result.sources, pending, published, sent))
        return seen

    assert run(SapientAsynchronousTracker(confirm_hits=2)) == run(_ScanningTracker(confirm_hits=2))


@pytest.mark.parametrize("policy", ["window", "all", "updates"])
@pytest.mark.parametrize("confirm_hits", [1, 2])
def test_events_at_one_time_publish_as_a_scan_of_every_track_would(policy, confirm_hits):
    # Several sensors' events at the same second: a track one of them updated has its
    # state at that time while the next event, at the same mark, does not touch it.
    def run(tracker):
        engine = FusionEngine(tracker, PublishPolicy(policy))
        seen = []
        script = [
            (0, [_report(0, "RAD-1", "a", 0)]),
            (0, [_report(0, "RAD-2", "b", 3000)]),
            (0, [_report(0, "RAD-3", "c", 6000)]),
            (1, [_report(1, "RAD-1", "a", 2)]),
            (1, [_report(1, "RAD-2", "b", 3003)]),
            (1, [_report(1, "RAD-1", "a", 4)]),
            (2, [_report(2, "RAD-3", "c", 6004), _report(2, "RAD-2", "b", 3006)]),
            (2, [_report(2, "RAD-1", "a", 6)]),
            (9, [_report(9, "RAD-2", "b", 3020)]),
            (9, [_report(9, "RAD-2", "b", 3020)]),
            (70, [_report(70, "RAD-1", "a", 10)]),
            (70, [_report(70, "RAD-3", "d", 9000)]),
        ]
        for seconds, reports in script:
            result = engine.process(T0 + timedelta(seconds=seconds), reports)
            published = sorted(track.id for track, _ in result.published)
            pending = {k: dict(v) for k, v in engine.pending_sources._pending.items()}
            seen.append(
                (published, result.sources, pending, dict(engine.publish_policy._last_sent))
            )
        return seen

    fast = run(SapientAsynchronousTracker(confirm_hits=confirm_hits))
    assert fast == run(_ScanningTracker(confirm_hits=confirm_hits))
    assert any(published for published, *_ in fast), "something was published"


def test_a_coarse_clock_keeps_no_dropped_track_in_the_engines_time_map():
    # Three marks only, and clutter starting and dropping tracks: what the engine keeps
    # of each track's state time is the live tracks', whatever the number of marks.
    tracker = SapientAsynchronousTracker(confirm_hits=2, max_coast_seconds=2.0)
    engine = FusionEngine(tracker, PublishPolicy("window"))
    rng = np.random.default_rng(3)
    dropped = 0
    for step in range(200):
        when = T0 + timedelta(seconds=3 * (step // 60) + (step % 3))
        reports = [
            _report(0, f"RAD-{k}", f"x{step}-{k}", float(rng.uniform(-9e3, 9e3)))
            for k in range(int(rng.integers(1, 4)))
        ]
        for report in reports:
            report.timestamp = when
        engine.process(when, reports)
        dropped += len(tracker.dropped_ids)
        assert set(engine._time_of) <= {t.id for t in tracker.tracks}
        assert all(engine._at_time.values()), "no empty time is kept"
    assert dropped


def test_a_tracker_fed_before_the_engine_sees_it_publishes_as_a_scan_would():
    def run(tracker):
        # Fed directly first: the engine has not seen these tracks' state times. With
        # coasting off ("window"), the next event at the same mark touches none of them.
        engine = FusionEngine(tracker, PublishPolicy("window"))
        tracker.process_async_event(T0, {_report(0, "RAD-1", "a"), _report(0, "RAD-2", "b", 4000)})
        result = engine.process(T0, [_report(0, "RAD-3", "c", 9000)])
        return sorted(track.id for track, _ in result.published)

    fast = run(SapientAsynchronousTracker())
    assert fast == run(_ScanningTracker())
    assert len(fast) == 3, "the two it was fed at the mark are published too"


def test_a_tracker_assigned_without_reset_forgets_the_old_ones_ids():
    engine = _engine()
    engine.process(T0, [_report(0)])
    assert engine.publish_policy._last_sent
    engine.tracker = SapientAsynchronousTracker()
    engine.process(T0 + timedelta(seconds=1), [_report(1, "RAD-9", "z", 7000)])
    live = {track.id for track in engine.tracker.tracks}
    assert set(engine.publish_policy._last_sent) <= live
    assert set(engine.pending_sources._pending) <= live


def test_a_track_coasted_and_pruned_in_one_event_is_not_placed_again():
    # Pruning's wide-covariance rule runs after coasting: one event can touch a track and
    # drop it. It must not be placed again under its dead id.
    tracker = SapientAsynchronousTracker(max_coast_position_sigma=50.0)
    engine = FusionEngine(tracker, PublishPolicy("coast"))
    engine.process(T0, [_report(0, "RAD-1", "a")])
    for step in range(1, 40):
        when = T0 + timedelta(seconds=step)
        engine.process(when, [_report(step, "RAD-2", f"far-{step}", 9000.0 + 3000.0 * step)])
        assert set(engine._time_of) <= {track.id for track in tracker.tracks}
