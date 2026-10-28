"""Tests for context_foundry.fusion.augmentor."""

import math
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.detection import Detection

from context_foundry.fusion import config
from context_foundry.fusion.augmentor import TacticalContextAugmentor, declared_threat
from context_foundry.fusion.engine import FusionEngine
from context_foundry.fusion.publish import PublishPolicy
from context_foundry.fusion.schemas import TacticalTrack
from context_foundry.fusion.serializers import CotSerializer
from context_foundry.fusion.tracker import SapientAsynchronousTracker


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
#
# Hits run through the real tracker, so the track metadata the augmentor reads is what
# Stone Soup's merge of every hit's metadata and the tracker's own bookkeeping leave.

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
_MODEL = LinearGaussian(ndim_state=9, mapping=(0, 3, 6), noise_covar=np.diag([25.0, 25.0, 100.0]))


def _at(seconds):
    return T0 + timedelta(seconds=seconds)


def _sapient(t, east, swarm_count=1):
    det = Detection(np.array([[east], [0.0], [100.0]]), measurement_model=_MODEL, timestamp=t)
    det.metadata = {
        "nodeId": "RAD-1",
        "objectId": f"o-{t.isoformat()}",
        "stable_object_id": False,
        "classification": "UAS",
        "swarm_count": swarm_count,
        "sensor_geodetic": None,
    }
    return det


def _cot(t, east, uid="ATAK-1", cot_type="a-f-G-U-C", how="m-g"):
    det = Detection(np.array([[east], [0.0], [100.0]]), measurement_model=_MODEL, timestamp=t)
    det.metadata = {
        "nodeId": "cot",
        "objectId": uid,
        "stable_object_id": True,
        "type": "CoT",
        "cot_type": cot_type,
        "how": how,
        "sensor_geodetic": None,
    }
    return det


class _Fusion:
    """The tracker and the augmentor, one event at a time."""

    def __init__(self):
        config.set_reference_origin(62.9, 29.8, 0.0)
        self.tracker = SapientAsynchronousTracker()
        self.augmentor = TacticalContextAugmentor()

    def hit(self, det):
        tracks = self.tracker.process_async_event(det.timestamp, {det})
        assert len(tracks) == 1, "every hit here is of one object"
        (track,) = tracks
        return track, self.augmentor.extract_tactical_track(track)


def _cot_type(result):
    """The CoT type the serializer draws a tactical track as."""
    return CotSerializer().serialize(result).split('type="')[1].split('"')[0]


def test_one_friendly_cot_hit_does_not_make_a_hostile_swarm_friend():
    fusion = _Fusion()
    results = []
    for i in range(12):
        t = _at(2 * i)
        if i == 3:
            track, result = fusion.hit(_cot(t, 60.0 * i))
            # Stone Soup merged the friendly type into the track's metadata.
            assert track.metadata["cot_type"] == "a-f-G-U-C"
        else:
            track, result = fusion.hit(_sapient(t, 60.0 * i, swarm_count=5))
        results.append(result)
    assert [r.threat_level for r in results] == ["hostile"] * 12
    # The overruled declaration does not lend the track its ground dimension either.
    assert {_cot_type(r) for r in results} == {"a-h-A-M-F"}


def test_a_fast_friendly_vehicle_one_radar_plot_sees_stays_friend():
    fusion = _Fusion()
    results = []
    for second in range(0, 90, 2):
        if second == 30:
            # A second after its 29 s report: 25 m on, inside the 50 m its report covers.
            det = _sapient(_at(second), 25.0 * second)
            fusion.hit(_cot(_at(second - 1), 25.0 * (second - 1), cot_type="a-f-G-E-V"))
        else:
            det = _cot(_at(second), 25.0 * second, cot_type="a-f-G-E-V")
        results.append(fusion.hit(det)[1])
    assert results[-1].speed_mps > 20.0
    assert {r.threat_level for r in results} == {"friend"}
    assert {_cot_type(r) for r in results} == {"a-f-G"}


def test_a_declaration_stands_only_near_where_it_was_made():
    fusion = _Fusion()
    assert fusion.hit(_cot(_at(0), 0.0))[1].threat_level == "friend"
    # Only a sensor sees it from here, walking off slowly: the rule would say suspect.
    threats = {}
    for second in range(2, 40, 2):
        threats[2.0 * second] = fusion.hit(_sapient(_at(second), 2.0 * second))[1].threat_level
    # Its report's error is 5 m, so it covers the least radius, 50 m.
    assert all(threats[m] == "friend" for m in threats if m <= 45)
    assert all(threats[m] == "suspect" for m in threats if m >= 56)


def test_a_fast_track_beyond_its_declarations_reach_is_judged_by_the_rule():
    fusion = _Fusion()
    fusion.hit(_cot(_at(0), 0.0))
    # Sensor hits alone take the track away at 40 m/s.
    threats = [fusion.hit(_sapient(_at(s), 40.0 * s))[1].threat_level for s in range(1, 6)]
    assert threats == ["friend"] + ["hostile"] * 4
    # The friend reports itself where the track now is: its declaration stands again.
    assert fusion.hit(_cot(_at(6), 240.0))[1].threat_level == "friend"


@pytest.mark.parametrize("speed", [25.0, 15.0])
def test_a_fast_friend_keeps_its_mark_between_its_reports(speed):
    fusion = _Fusion()
    marks = []
    for second in range(0, 60):
        if second % 5 == 0:
            fusion.hit(_cot(_at(second), speed * second, cot_type="a-f-G-E-V"))
        moment = _at(second + 0.5)
        marks.append(fusion.hit(_sapient(moment, speed * (second + 0.5)))[1].threat_level)
    # Its first report gives no speed yet; from its second on it is carried along.
    assert set(marks[5:]) == {"friend"}


def test_a_standing_friend_lends_no_mark_to_a_drone_passing_by():
    config.set_reference_origin(62.9, 29.8, 0.0)
    tracker = SapientAsynchronousTracker()
    augmentor = TacticalContextAugmentor()
    marks = []
    for second in range(0, 40):
        if second % 5 == 0:
            tracker.process_async_event(_at(second), {_cot(_at(second), 0.0)})
        moment = _at(second + 0.5)
        drone = _sapient(moment, -300.0 + 30.0 * (second + 0.5))
        for track in tracker.process_async_event(moment, {drone}):
            east = track.state.state_vector[0, 0]
            marks.append((abs(east) > 60.0, augmentor.extract_tactical_track(track).threat_level))
    assert any(far for far, _ in marks)
    assert "friend" not in {threat for far, threat in marks if far}


def test_a_hand_placed_marker_on_a_self_reporting_user_never_declares():
    fusion = _Fusion()
    threats = []
    for i in range(8):
        t = _at(5 * i)
        if i % 2 == 0:
            det = _cot(t, 20.0, uid="MARKER-1", cot_type="a-h-G", how="h-g-i-g-o")
        else:
            det = _cot(t, 0.0, uid="ATAK-ME", cot_type="a-f-G-U-C", how="m-g")
        _, result = fusion.hit(det)
        threats.append(result.threat_level)
    # The marker came first and declared nothing; the user's own reports do.
    assert threats == ["suspect"] + ["friend"] * 7


def test_a_hand_placed_marker_on_a_sensors_track_declares_nothing():
    fusion = _Fusion()
    threats = []
    for i in range(6):
        t = _at(5 * i)
        if i == 2:
            det = _cot(t, 0.0, uid="MARKER-1", cot_type="a-f-G", how="h-g-i-g-o")
        else:
            det = _sapient(t, 0.0)
        threats.append(fusion.hit(det)[1].threat_level)
    assert threats == ["suspect"] * 6


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("a-f-G", "a-n-G"),
        ("a-n-G", "a-s-G"),
        ("a-h-G", "a-s-G"),
        ("a-f-G", "a-h-G"),
    ],
)
def test_two_uids_reporting_themselves_at_one_place_keep_their_own_marks(first, second):
    config.set_reference_origin(62.9, 29.8, 0.0)
    tracker = SapientAsynchronousTracker()
    augmentor = TacticalContextAugmentor()
    marks = {}
    for i in range(6):
        uid, cot_type = ("ATAK-A", first) if i % 2 == 0 else ("ATAK-B", second)
        tracks = tracker.process_async_event(_at(5 * i), {_cot(_at(5 * i), 0.0, uid, cot_type)})
        for track in tracks:
            if tracker._object_tracks[("cot", uid)] == track.id:
                threat = augmentor.extract_tactical_track(track).threat_level
                marks.setdefault(uid, set()).add(threat)
    assert len(tracker.tracks) == 2
    assert marks == {"ATAK-A": {declared_threat(first)}, "ATAK-B": {declared_threat(second)}}


def _published_marks(hits, start):
    """Second after T0 -> (threat, drawn CoT type) of every record published from `start`."""
    engine = FusionEngine(
        SapientAsynchronousTracker(confirm_hits=3, id_namespace="n"),
        PublishPolicy("window", window_seconds=5.0),
    )
    marks = {}
    for det in hits:
        second = (det.timestamp - T0).total_seconds()
        if second < start:
            continue
        for _, result in engine.process(det.timestamp, {det}).published:
            marks[second] = (result.threat_level, _cot_type(result))
    return marks


def test_a_rebuild_from_a_later_start_publishes_the_same_marks():
    config.set_reference_origin(62.9, 29.8, 0.0)
    # A friend reporting itself, and a hostile seen only by another party's reports.
    hits = []
    for k in range(40):
        hits.append(_cot(_at(5.0 * k), 0.0, uid="BLUE-1", cot_type="a-f-G-U-C"))
        hits.append(_cot(_at(5.0 * k + 2.5), 3.0, uid="RED-1", cot_type="a-h-G", how="m-r"))
    full = _published_marks(hits, 0.0)
    # A rebuild whose window starts on the hostile's report.
    rebuilt = _published_marks(hits, 102.5)
    # The two publish on different window phases, so compare what each draws from then on.
    from_full = {mark for s, mark in full.items() if s >= 112.5}
    from_rebuild = {mark for s, mark in rebuilt.items() if s >= 112.5}
    assert from_rebuild == from_full == {("friend", "a-f-G")}


def test_a_fast_friendly_user_reporting_only_itself_stays_friend():
    fusion = _Fusion()
    results = [fusion.hit(_cot(_at(2 * i), 80.0 * i))[1] for i in range(15)]
    assert results[-1].speed_mps > 20.0
    assert [r.threat_level for r in results] == ["friend"] * 15
    assert all(r.declared_type == "a-f-G-U-C" for r in results)


def test_a_declaration_expires_sixty_seconds_after_its_last_report():
    fusion = _Fusion()
    for second in (0, 5, 10):
        assert fusion.hit(_cot(_at(second), 0.0))[1].threat_level == "friend"
    # Only a sensor sees it from here, slow and alone: the rule would say suspect.
    threats = {}
    for second in range(15, 95, 5):
        threats[second] = fusion.hit(_sapient(_at(second), 0.0))[1].threat_level
    assert all(threats[s] == "friend" for s in threats if s <= 70)
    assert all(threats[s] == "suspect" for s in threats if s > 70)


@pytest.mark.parametrize(("stale_after", "lasts"), [(240, 240), (30, 60), (3600, 600)])
def test_a_declaration_lasts_until_its_events_stale_within_one_to_ten_minutes(stale_after, lasts):
    fusion = _Fusion()
    det = _cot(_at(0), 0.0)
    det.metadata["stale"] = (_at(stale_after)).isoformat().replace("+00:00", "Z")
    assert fusion.hit(det)[1].threat_level == "friend"
    threats = {}
    for second in range(20, 700, 20):
        threats[second] = fusion.hit(_sapient(_at(second), 0.0))[1].threat_level
    assert all(threats[s] == "friend" for s in threats if s <= lasts)
    assert all(threats[s] == "suspect" for s in threats if s > lasts)


def test_a_declared_hostile_is_held_after_the_declaration_ends():
    fusion = _Fusion()
    for second in (0, 5, 10):
        assert fusion.hit(_cot(_at(second), 0.0, cot_type="a-h-G"))[1].threat_level == "hostile"
    # The same user now declares nothing: the hold from its last hostile report stands.
    threats = {}
    for second in range(15, 95, 5):
        threats[second] = fusion.hit(_cot(_at(second), 0.0, cot_type="a-u-G"))[1].threat_level
    assert all(threats[s] == "hostile" for s in threats if s <= 70)
    assert all(threats[s] == "suspect" for s in threats if s > 70)


def test_a_stale_declaration_leaves_a_swarms_hold_in_place():
    fusion = _Fusion()
    fusion.hit(_cot(_at(0), 0.0))
    _, swarm = fusion.hit(_sapient(_at(5), 0.0, swarm_count=5))
    assert swarm.threat_level == "hostile"
    # The swarm count lasts to 65 s and the hold 60 s past that; from 70 s only the hold
    # keeps it hostile, with the friend's declaration stale since 60 s.
    threats = {}
    for second in range(10, 145, 5):
        threats[second] = fusion.hit(_sapient(_at(second), 0.0))[1].threat_level
    assert all(threats[s] == "hostile" for s in threats if s <= 125)
    assert all(threats[s] == "suspect" for s in threats if s > 125)


def test_a_fresh_friend_overrides_a_hold_but_does_not_end_it():
    fusion = _Fusion()
    fusion.hit(_sapient(_at(0), 0.0, swarm_count=5))
    # The user's own reports keep the track; the swarm count lasts to 60 s, the hold past it.
    for second in range(5, 65, 5):
        assert fusion.hit(_cot(_at(second), 0.0, cot_type="a-u-G"))[1].threat_level == "hostile"
    # No sensor for over 60 s: the user's friendly declaration stands.
    assert fusion.hit(_cot(_at(65), 0.0, cot_type="a-f-G"))[1].threat_level == "friend"
    # It stops declaring: the hold, until 120 s, is still there.
    assert fusion.hit(_cot(_at(75), 0.0, cot_type="a-u-G"))[1].threat_level == "hostile"
    assert fusion.hit(_cot(_at(125), 0.0, cot_type="a-u-G"))[1].threat_level == "suspect"


def test_the_judgement_depends_only_on_the_hits():
    def run():
        fusion = _Fusion()
        hits = [_cot(_at(0), 0.0), _sapient(_at(5), 0.0, swarm_count=5)]
        hits += [_sapient(_at(s), 0.0) for s in range(10, 80, 10)]
        return [fusion.hit(det)[1].threat_level for det in hits]

    assert run() == run()


def test_declared_threat_reads_only_a_cot_atom_types_affiliation():
    assert declared_threat("a-f-G-U-C") == "friend"
    assert declared_threat("a-F-G") is None
    assert declared_threat("b-f-G") is None
    assert declared_threat(None) is None
    assert declared_threat(7) is None


@pytest.mark.parametrize(
    ("cot_type", "threat"),
    [
        ("a-f-G", "friend"),
        ("a-a-G", "friend"),
        ("a-h-G", "hostile"),
        ("a-j-G", "suspect"),
        ("a-k-G", "hostile"),
        ("a-s-G", "suspect"),
        ("a-n-G", "neutral"),
        ("a-u-G", "suspect"),
        ("a-p-G", "suspect"),
        ("a-o-G", "suspect"),
        ("a-x-G", "suspect"),
    ],
)
def test_each_affiliation_of_a_self_report_maps_to_its_threat(cot_type, threat):
    fusion = _Fusion()
    assert fusion.hit(_cot(_at(0), 0.0, cot_type=cot_type))[1].threat_level == threat
