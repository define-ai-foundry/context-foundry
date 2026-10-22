"""Tests for the standalone engine's offline input and output: records files."""

import json

import pytest

from context_foundry.fusion import config
from context_foundry.fusion.sinks.records_file import RecordsFileSink
from context_foundry.fusion.sources.records_file import RecordsFileSource

LAT_LNG = "LOCATION_COORDINATE_SYSTEM_LAT_LNG_DEG_M"


def _sapient(content_type, body, ts, node="RAD-1"):
    message = {"timestamp": ts, "node_id": node, content_type: body}
    return {"event_time": ts, "content_type": content_type, "node_id": node, "message": message}


def _detection(ts, object_id="obj-1", node="RAD-1"):
    location = {"x": 29.8, "y": 62.95, "z": 300.0, "coordinate_system": LAT_LNG}
    return _sapient("detection_report", {"object_id": object_id, "location": location}, ts, node)


def _cot(ts, uid="uav-1"):
    point = {"lat": 62.95, "lon": 29.8, "hae": 300.0, "ce": 5.0, "le": 2.0}
    return {"event_time": ts, "uid": uid, "type": "a-h-A", "point": point}


def _write(path, records, archived=False):
    lines = [{"offset": i, "value": r} if archived else r for i, r in enumerate(records)]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


def test_detections_of_every_file_are_fused_by_event_time(tmp_path):
    config.reset_registry()
    sapient = _write(
        tmp_path / "sapient.jsonl",
        [_detection("2026-11-15T03:00:05Z"), _detection("2026-11-15T03:00:01Z", "obj-2")],
        archived=True,
    )
    cot = _write(tmp_path / "cot.jsonl", [_cot("2026-11-15T03:00:03Z")])
    source = RecordsFileSource([sapient, cot])
    events = list(source.iter_events())
    assert [t.isoformat()[11:19] for t, _ in events] == ["03:00:01", "03:00:03", "03:00:05"]
    assert source.records_read == 3
    assert list(source.iter_events()) == [], "a file is read once"


def test_one_sensor_instant_is_one_event_and_a_sensor_is_learned_as_it_comes(tmp_path):
    config.reset_registry()
    registration = _sapient(
        "registration",
        {"node_definition": [{"node_type": "NODE_TYPE_RADAR"}], "mode_definition": []},
        "2026-11-15T03:00:00Z",
    )
    ts = "2026-11-15T03:00:01Z"
    path = _write(tmp_path / "in.jsonl", [registration, _detection(ts), _detection(ts, "obj-2")])
    events = list(RecordsFileSource([path]).iter_events())
    assert len(events) == 1 and len(events[0][1]) == 2
    assert config.sensor_profile("RAD-1")["node_type"] == "radar"


def test_lines_that_are_not_records_are_counted(tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_text('not json\n[1]\n{"neither": "kind"}\n')
    source = RecordsFileSource([path])
    assert list(source.iter_events()) == []
    assert source.unreadable == 3


def test_a_fused_record_carries_its_detections_confidences(tmp_path):
    from context_foundry.fusion import sapient_raw
    from context_foundry.fusion.engine import FusionEngine
    from context_foundry.fusion.publish import PublishPolicy
    from context_foundry.fusion.tracker import SapientAsynchronousTracker

    config.reset_registry()
    record = _detection("2026-11-15T03:00:01Z")
    report = record["message"]["detection_report"]
    report["detection_confidence"] = 0.8
    report["classification"] = [
        {
            "type": "Vehicle",
            "confidence": 0.6,
            "sub_class": [
                {
                    "type": "Drone",
                    "confidence": 0.5,
                    "level": 1,
                    "sub_class": [{"type": "Military", "confidence": 0.4, "level": 2}],
                }
            ],
        }
    ]
    path = _write(tmp_path / "in.jsonl", [record])
    engine = FusionEngine(SapientAsynchronousTracker(confirm_hits=1), PublishPolicy("window"))

    ((timestamp, detections),) = list(RecordsFileSource([path]).iter_events())
    ((_, tactical),) = engine.process(timestamp, detections).published
    _, value, _ = sapient_raw.track_record(tactical, "fusion-node")

    fused = json.loads(value)["message"]["detection_report"]
    assert fused["detection_confidence"] == pytest.approx(0.8)
    top = fused["classification"][0]
    assert top["confidence"] == pytest.approx(0.6)
    assert top["sub_class"][0]["confidence"] == pytest.approx(0.5)
    assert top["sub_class"][0]["sub_class"][0]["confidence"] == pytest.approx(0.4)


def test_the_sink_writes_sapient_raw_records(tmp_path):
    from datetime import UTC, datetime

    from context_foundry.fusion.schemas import TacticalTrack

    track = TacticalTrack(
        track_id="t-1",
        timestamp=datetime(2026, 11, 15, 3, 0, 1, tzinfo=UTC),
        latitude=62.95,
        longitude=29.8,
        altitude=300.0,
        speed_mps=10.0,
        heading_deg=90.0,
        classification="UAV",
        threat_level="hostile",
    )
    sink = RecordsFileSink(str(tmp_path / "out.jsonl"), "fusion-node")
    sink.send_track(track)
    sink.close()
    (record,) = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert record["content_type"] == "detection_report"
    assert record["node_id"] == "fusion-node"
    assert record["message"]["detection_report"]["object_id"]
