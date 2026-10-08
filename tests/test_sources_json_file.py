"""Tests for context_foundry.fusion.sources.json_file."""

import json
import math

import numpy as np
import pytest

from context_foundry.fusion import config
from context_foundry.fusion.sources.json_file import JsonSapientSource


def _msg(node_id, timestamp, object_id, swarm=None, lat=62.9, lon=29.8, alt=100.0):
    detection_report = {
        "objectId": object_id,
        "state": "ACTIVE",
        "location": {
            "x": lon,
            "y": lat,
            "z": alt,
            "coordinateSystem": "LOCATION_COORDINATE_SYSTEM_LAT_LNG_DEG_M",
            "datum": "LOCATION_DATUM_WGS84_E",
        },
        "classification": [{"type": "UAS", "confidence": 0.9}],
    }
    if swarm is not None:
        detection_report["objectInfo"] = [{"type": "estimatedSwarmCount", "value": str(swarm)}]
    return {
        "sapientMessage": {
            "timestamp": timestamp,
            "nodeId": node_id,
            "detectionReport": detection_report,
        }
    }


def test_iter_events_raises_file_not_found(tmp_path):
    source = JsonSapientSource(tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        list(source.iter_events())


def test_iter_events_groups_by_timestamp_and_sensor(tmp_path):
    ts = "2026-01-01T00:00:00.000000Z"
    messages = [
        _msg("node-A", ts, "obj-1"),
        _msg("node-A", ts, "obj-2"),
        _msg("node-B", ts, "obj-3"),
    ]
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(messages), encoding="utf-8")

    source = JsonSapientSource(path)
    events = list(source.iter_events())

    assert len(events) == 2  # one group per (timestamp, node)
    sizes = sorted(len(dets) for _, dets in events)
    assert sizes == [1, 2]


def test_iter_events_is_one_shot(tmp_path):
    """Regression: a replay file is finite and must be consumed exactly once.

    The CLI main loop re-polls every source on each pass and exits a replay-only
    run when a full pass yields nothing. A source that re-reads and re-emits its
    whole file on every call keeps processed_any_events True forever, so the exit
    branch is unreachable and repeated reprocessing spawns unbounded tracks until
    the JPDA associator's cost explodes -- the observed hang. Draining once fixes it.
    """
    ts = "2026-01-01T00:00:00.000000Z"
    messages = [_msg("node-A", ts, "obj-1")]
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(messages), encoding="utf-8")

    source = JsonSapientSource(path)
    assert len(list(source.iter_events())) == 1
    assert list(source.iter_events()) == []


def test_reset_allows_replay_to_be_consumed_again(tmp_path):
    """reset() flips _exhausted back so a drained source re-yields its file."""
    ts = "2026-01-01T00:00:00.000000Z"
    messages = [_msg("node-A", ts, "obj-1")]
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(messages), encoding="utf-8")

    source = JsonSapientSource(path)
    assert len(list(source.iter_events())) == 1
    assert list(source.iter_events()) == []  # drained

    source.reset()
    assert len(list(source.iter_events())) == 1  # re-armed, same events again


def test_iter_events_yields_groups_in_timestamp_order_even_when_file_is_shuffled(tmp_path):
    """Messages in the source file need not be chronological; groups must still
    come out sorted by timestamp (the tracker can't predict backwards, and
    realtime pacing anchors off the first yielded event)."""
    ts_early = "2026-01-01T00:00:00.000000Z"
    ts_mid = "2026-01-01T00:00:05.000000Z"
    ts_late = "2026-01-01T00:00:10.000000Z"
    # Deliberately out of order in the file: late, early, mid.
    messages = [
        _msg("node-A", ts_late, "obj-late"),
        _msg("node-A", ts_early, "obj-early"),
        _msg("node-A", ts_mid, "obj-mid"),
    ]
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(messages), encoding="utf-8")

    source = JsonSapientSource(path)
    events = list(source.iter_events())

    timestamps = [timestamp for timestamp, _ in events]
    assert timestamps == sorted(timestamps)
    assert len(timestamps) == 3


def test_iter_events_drops_invalid_messages(tmp_path):
    ts = "2026-01-01T00:00:00.000000Z"
    invalid = {"sapientMessage": {"timestamp": ts, "nodeId": "node-A"}}  # no detectionReport
    messages = [invalid, _msg("node-A", ts, "obj-1")]
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(messages), encoding="utf-8")

    source = JsonSapientSource(path)
    events = list(source.iter_events())
    assert len(events) == 1
    _, dets = events[0]
    assert len(dets) == 1


def test_iter_events_all_invalid_yields_nothing(tmp_path):
    invalid = {"sapientMessage": {"timestamp": "2026-01-01T00:00:00.000000Z", "nodeId": "node-A"}}
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps([invalid]), encoding="utf-8")

    source = JsonSapientSource(path)
    assert list(source.iter_events()) == []


def test_iter_events_reads_swarm_count_from_the_report(tmp_path):
    ts = "2026-01-01T00:00:00.000000Z"
    messages = [_msg("node-A", ts, "obj-1", swarm=7)]
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(messages), encoding="utf-8")

    source = JsonSapientSource(path)
    ((_, dets),) = list(source.iter_events())
    assert dets[0].metadata["swarm_count"] == 7


def test_iter_events_default_swarm_count_is_one(tmp_path):
    ts = "2026-01-01T00:00:00.000000Z"
    messages = [_msg("node-A", ts, "obj-1")]
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(messages), encoding="utf-8")

    source = JsonSapientSource(path)
    ((_, dets),) = list(source.iter_events())
    assert dets[0].metadata["swarm_count"] == 1


def test_iter_events_sensor_geodetic_present_when_registered(tmp_path):
    config.load_sensor_network(
        sensor_network_list=[{"id": "node-A", "lat": 1.0, "lon": 2.0, "alt": 3.0}]
    )
    ts = "2026-01-01T00:00:00.000000Z"
    messages = [_msg("node-A", ts, "obj-1")]
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(messages), encoding="utf-8")

    source = JsonSapientSource(path)
    ((_, dets),) = list(source.iter_events())
    assert dets[0].metadata["sensor_geodetic"] == {
        "latitude": 1.0,
        "longitude": 2.0,
        "altitude": 3.0,
    }
    assert dets[0].metadata["nodeId"] == "node-A"
    assert dets[0].metadata["objectId"] == "obj-1"
    assert dets[0].metadata["classification"] == "UAS"


def test_iter_events_sensor_geodetic_none_when_unregistered(tmp_path):
    ts = "2026-01-01T00:00:00.000000Z"
    messages = [_msg("node-A", ts, "obj-1")]
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(messages), encoding="utf-8")

    source = JsonSapientSource(path)
    ((_, dets),) = list(source.iter_events())
    assert dets[0].metadata["sensor_geodetic"] is None


def test_iter_events_uses_a_cautious_default_without_geometric_error(tmp_path):
    # A sensor of no known kind that declares no accuracy is assumed 50 m, not the
    # old shared 5 m that split every radar's reports into tracks of their own.
    config.load_sensor_network(
        sensor_network_list=[{"id": "node-A", "lat": 62.9, "lon": 29.8, "alt": 0.0}]
    )
    ts = "2026-01-01T00:00:00.000000Z"
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps([_msg("node-A", ts, "obj-1")]), encoding="utf-8")

    ((_, dets),) = list(JsonSapientSource(path).iter_events())
    assert dets[0].measurement_model.noise_covar[0, 0] == pytest.approx(50.0**2)


def test_iter_events_takes_noise_from_the_sensors_geometric_error(tmp_path):
    config.load_sensor_network(
        sensor_network_list=[
            {
                "id": "node-A",
                "lat": 62.9,
                "lon": 29.8,
                "alt": 0.0,
                "range_m": 20000.0,
                "geometric_error": {
                    "variation_type": "linear_with_range",
                    "base_m": 5.0,
                    "at_max_range_m": 35.0,
                },
            }
        ]
    )
    ts = "2026-01-01T00:00:00.000000Z"
    # About 11.1 km north of the sensor: a little over half its range.
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps([_msg("node-A", ts, "obj-1", lat=63.0, alt=0.0)]), encoding="utf-8")

    ((_, dets),) = list(JsonSapientSource(path).iter_events())
    sigma = math.sqrt(dets[0].measurement_model.covar()[0, 0])
    assert sigma == pytest.approx(5.0 + 30.0 * 11120.0 / 20000.0, rel=0.01)


def test_iter_events_frame_origin_is_still_the_first_detection(tmp_path):
    """The tracking frame's origin is fixed by the first point projected. Placing the
    sensor for its geometric_error must not take that place from the first detection."""
    config.load_sensor_network(
        sensor_network_list=[
            {
                "id": "node-A",
                "lat": 62.9,
                "lon": 29.8,
                "alt": 0.0,
                "range_m": 20000.0,
                "geometric_error": {"variation_type": "constant", "base_m": 5.0},
            }
        ]
    )
    ts = "2026-01-01T00:00:00.000000Z"
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps([_msg("node-A", ts, "obj-1", lat=63.0, alt=0.0)]), encoding="utf-8")

    ((_, dets),) = list(JsonSapientSource(path).iter_events())
    assert (config._origin_lat, config._origin_lon) == (63.0, 29.8)
    assert np.allclose(np.ravel(dets[0].state_vector), 0.0)


def test_iter_events_classification_defaults_to_unknown(tmp_path):
    ts = "2026-01-01T00:00:00.000000Z"
    msg = _msg("node-A", ts, "obj-1")
    del msg["sapientMessage"]["detectionReport"]["classification"]
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps([msg]), encoding="utf-8")

    source = JsonSapientSource(path)
    ((_, dets),) = list(source.iter_events())
    assert dets[0].metadata["classification"] == "Unknown"


@pytest.mark.parametrize(
    ("tracking_type", "stable"),
    [
        (None, False),
        ("TRACKING_TYPE_NONE", False),
        ("TRACKING_TYPE_TRACKLET", True),
        ("TRACKING_TYPE_TRACK", True),
        ("TRACKING_TYPE_TRACK_WITH_RE_ID", True),
    ],
)
def test_iter_events_marks_object_ids_stable_from_the_sensors_tracking_type(
    tmp_path, tracking_type, stable
):
    sensor = {"id": "node-A", "lat": 62.9, "lon": 29.8, "alt": 0.0}
    if tracking_type:
        sensor["tracking_type"] = tracking_type
    config.load_sensor_network(sensor_network_list=[sensor])
    path = tmp_path / "scenario.json"
    path.write_text(
        json.dumps([_msg("node-A", "2026-01-01T00:00:00.000000Z", "obj-1")]), encoding="utf-8"
    )

    ((_, dets),) = list(JsonSapientSource(path).iter_events())
    assert dets[0].metadata["stable_object_id"] is stable


def test_iter_events_unregistered_sensor_object_ids_are_not_stable(tmp_path):
    path = tmp_path / "scenario.json"
    path.write_text(
        json.dumps([_msg("node-A", "2026-01-01T00:00:00.000000Z", "obj-1")]), encoding="utf-8"
    )

    ((_, dets),) = list(JsonSapientSource(path).iter_events())
    assert dets[0].metadata["stable_object_id"] is False
