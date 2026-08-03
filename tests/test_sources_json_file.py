"""Tests for context_foundry.fusion.sources.json_file."""

import json

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


def test_iter_events_classification_defaults_to_unknown(tmp_path):
    ts = "2026-01-01T00:00:00.000000Z"
    msg = _msg("node-A", ts, "obj-1")
    del msg["sapientMessage"]["detectionReport"]["classification"]
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps([msg]), encoding="utf-8")

    source = JsonSapientSource(path)
    ((_, dets),) = list(source.iter_events())
    assert dets[0].metadata["classification"] == "Unknown"
