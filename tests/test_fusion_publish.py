"""Tests for context_foundry.fusion.publish: which fused tracks an event publishes."""

from datetime import datetime, timedelta, timezone

import pytest
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.prediction import GaussianStatePrediction
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.types.update import GaussianStateUpdate

from context_foundry.fusion.publish import (
    PendingSources,
    PublishPolicy,
    is_predicted,
    recent_sources,
    source_key,
)
from context_foundry.fusion.sapient_raw import cot_node_id, cot_object_ulid

T0 = datetime(2026, 11, 15, 3, 0, 0, tzinfo=timezone.utc)


class _Track:
    def __init__(self, track_id, predicted):
        self.id = track_id
        kind = GaussianStatePrediction if predicted else GaussianState
        self.state = kind([[0.0]], [[1.0]], timestamp=T0)


def _at(seconds):
    return T0 + timedelta(seconds=seconds)


def test_is_predicted():
    assert is_predicted(_Track("a", predicted=True))
    assert not is_predicted(_Track("a", predicted=False))


def test_all_publishes_measured_and_predicted():
    policy = PublishPolicy("all")
    assert policy.should_publish(_Track("a", predicted=False), _at(0))
    assert policy.should_publish(_Track("a", predicted=True), _at(1))


def test_updates_publishes_only_measured():
    policy = PublishPolicy("updates")
    assert policy.should_publish(_Track("a", predicted=False), _at(0))
    assert not policy.should_publish(_Track("a", predicted=True), _at(1))


def test_coast_sends_a_prediction_once_the_track_has_been_quiet_long_enough():
    policy = PublishPolicy("coast", coast_seconds=5)
    assert policy.should_publish(_Track("a", predicted=False), _at(0))
    assert not policy.should_publish(_Track("a", predicted=True), _at(3))
    assert policy.should_publish(_Track("a", predicted=True), _at(5))
    # The prediction just sent restarts the clock.
    assert not policy.should_publish(_Track("a", predicted=True), _at(8))
    assert policy.should_publish(_Track("a", predicted=False), _at(9))


def test_coast_publishes_a_track_it_has_never_sent():
    assert PublishPolicy("coast").should_publish(_Track("new", predicted=True), _at(0))


def test_forget_all_but_drops_tracks_no_longer_live():
    policy = PublishPolicy("coast", coast_seconds=5)
    policy.should_publish(_Track("a", predicted=False), _at(0))
    policy.should_publish(_Track("b", predicted=False), _at(0))
    policy.forget_all_but(["b"])
    # "a" is forgotten, so it counts as never sent.
    assert policy.should_publish(_Track("a", predicted=True), _at(1))
    assert not policy.should_publish(_Track("b", predicted=True), _at(1))


@pytest.mark.parametrize("policy, coast", [("sometimes", 5), ("coast", 0)])
def test_rejects_bad_settings(policy, coast):
    with pytest.raises(ValueError):
        PublishPolicy(policy, coast_seconds=coast)


def test_predicted_publishes_only_predictions():
    policy = PublishPolicy("predicted")
    assert not policy.should_publish(_Track("a", predicted=False), _at(0))
    assert policy.should_publish(_Track("a", predicted=True), _at(1))


def _detection(node, object_id, seconds):
    return Detection(
        [[0.0]], timestamp=_at(seconds), metadata={"nodeId": node, "objectId": object_id}
    )


def _update(detection):
    prediction = GaussianStatePrediction([[0.0]], [[1.0]], timestamp=detection.timestamp)
    return GaussianStateUpdate(
        [[0.0]],
        [[1.0]],
        hypothesis=SingleHypothesis(prediction, detection),
        timestamp=detection.timestamp,
    )


def test_recent_sources_lists_each_sensor_with_its_newest_sighting():
    start = GaussianState([[0.0]], [[1.0]], timestamp=_at(0))
    track = Track(
        [
            start,
            _update(_detection("KOLI", "W2", 6)),
            GaussianStatePrediction([[0.0]], [[1.0]], timestamp=_at(9)),
            _update(_detection("ONTTOLA", "W2", 12)),
            _update(_detection("KOLI", "W2", 18)),
        ],
        init_metadata={"nodeId": "HEINA", "objectId": "W2"},
    )
    assert recent_sources(track) == [
        {"node_id": "HEINA", "object_id": "W2", "timestamp": _at(0)},
        {"node_id": "ONTTOLA", "object_id": "W2", "timestamp": _at(12)},
        {"node_id": "KOLI", "object_id": "W2", "timestamp": _at(18)},
    ]


def test_recent_sources_keeps_two_object_ids_from_one_sensor_apart():
    track = Track(
        [
            GaussianStatePrediction([[0.0]], [[1.0]], timestamp=_at(0)),
            _update(_detection("KOLI", "W1", 6)),
            _update(_detection("KOLI", "W2", 12)),
            _update(_detection(None, "x", 13)),
        ]
    )
    assert [s["object_id"] for s in recent_sources(track)] == ["W1", "W2"]


def test_window_sends_only_measured_states_at_most_once_per_window():
    policy = PublishPolicy("window", window_seconds=5)
    assert policy.should_publish(_Track("a", predicted=False), _at(0))
    # Another report 2 s later is fused into the track but not sent.
    assert not policy.should_publish(_Track("a", predicted=False), _at(2))
    # A prediction is never sent, however long the track has been quiet.
    assert not policy.should_publish(_Track("a", predicted=True), _at(9))
    assert policy.should_publish(_Track("a", predicted=False), _at(10))
    # Each track has its own window.
    assert policy.should_publish(_Track("b", predicted=False), _at(10))


def test_rejects_a_window_that_is_not_positive():
    with pytest.raises(ValueError):
        PublishPolicy("window", window_seconds=0)


def test_recent_sources_names_a_cot_source_by_its_derived_ids():
    track = Track(
        [GaussianState([[0.0]], [[1.0]], timestamp=_at(0)), _update(_detection("cot", "U-1", 4))],
        init_metadata={"nodeId": "cot", "objectId": "U-1"},
    )
    assert recent_sources(track) == [
        {"node_id": cot_node_id(), "object_id": cot_object_ulid("U-1"), "timestamp": _at(4)}
    ]


def test_source_key_maps_cot_and_passes_sapient_through():
    assert source_key({"nodeId": "cot", "objectId": "U-1"}) == (
        cot_node_id(),
        cot_object_ulid("U-1"),
    )
    assert source_key({"nodeId": "cot"}) == (cot_node_id(), None)
    assert source_key({"nodeId": "N", "objectId": "O"}) == ("N", "O")
    assert source_key({"nodeId": "N", "objectId": ""}) == ("N", "")
    assert source_key({"objectId": "O"}) is None
    assert source_key(None) is None


def test_pending_sources_keep_the_newest_sighting_and_clear_on_take():
    pending = PendingSources()
    pending.note("t", {"nodeId": "B", "objectId": "2"}, _at(3))
    pending.note("t", {"nodeId": "A", "objectId": "1"}, _at(5))
    pending.note("t", {"nodeId": "B", "objectId": "2"}, _at(1))
    pending.note("t", {"objectId": "x"}, _at(2))
    pending.note("t", {"nodeId": "C"}, None)
    assert pending.take("t") == [
        {"node_id": "B", "object_id": "2", "timestamp": _at(3)},
        {"node_id": "A", "object_id": "1", "timestamp": _at(5)},
    ]
    assert pending.take("t") == []


def test_pending_sources_forget_tracks_no_longer_held():
    pending = PendingSources()
    pending.note("a", {"nodeId": "N"}, _at(0))
    pending.note("b", {"nodeId": "N"}, _at(0))
    pending.forget_all_but(["b"])
    assert pending.take("a") == []
    assert len(pending.take("b")) == 1
