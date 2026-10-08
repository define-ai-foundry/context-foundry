"""Tests for context_foundry.fusion.sapient_raw: fused tracks as sapient-raw records."""

import json
import re
import uuid
from datetime import datetime, timezone

from context_foundry.fusion import sapient_raw
from context_foundry.fusion.schemas import TacticalTrack

TRACK_ID = "64578864-2254-4a79-b148-e12e1c06ffd8"
NODE = sapient_raw.fusion_node_id("p-00000000a1", "fusion")
ULID = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")


def _track(**overrides):
    fields = {
        "track_id": TRACK_ID,
        "timestamp": datetime(2026, 11, 15, 3, 2, 23, 512000, tzinfo=timezone.utc),
        "latitude": 62.744313,
        "longitude": 30.293088,
        "altitude": 223.6,
        "speed_mps": 30.0,
        "heading_deg": 90.0,
        "classification": "Air vehicle > UAV fixed wing > Military",
        "swarm_count": 10,
        "threat_level": "hostile",
        "velocity": [30.0, 0.5, -1.0],
    }
    fields.update(overrides)
    return TacticalTrack(**fields)


def _record(track=None):
    key, value, event_time = sapient_raw.track_record(track or _track(), NODE)
    return key, json.loads(value), event_time


def test_envelope_follows_the_sapient_raw_schema():
    key, record, event_time = _record()
    assert key == NODE.encode()
    assert set(record) == {"event_time", "content_type", "node_id", "message"}
    assert record["content_type"] == "detection_report"
    assert record["node_id"] == record["message"]["node_id"] == NODE
    # event_time must be the very string the message carries.
    assert record["event_time"] == record["message"]["timestamp"] == event_time
    assert event_time == "2026-11-15T03:02:23.512Z"


def test_location_is_longitude_x_latitude_y():
    _, record, _ = _record()
    location = record["message"]["detection_report"]["location"]
    assert location["x"] == 30.293088
    assert location["y"] == 62.744313
    assert location["z"] == 223.6
    assert location["coordinate_system"] == "LOCATION_COORDINATE_SYSTEM_LAT_LNG_DEG_M"
    assert location["datum"] == "LOCATION_DATUM_WGS84_E"


def test_object_id_is_a_stable_ulid_and_report_id_a_new_one():
    _, first, _ = _record()
    _, second, _ = _record()
    a = first["message"]["detection_report"]
    b = second["message"]["detection_report"]
    assert ULID.match(a["object_id"]) and ULID.match(a["report_id"])
    assert a["object_id"] == b["object_id"]
    assert a["report_id"] != b["report_id"]


def test_object_id_for_a_track_id_that_is_not_a_uuid():
    assert ULID.match(sapient_raw.object_ulid("track-7"))
    assert sapient_raw.object_ulid("track-7") == sapient_raw.object_ulid("track-7")


def test_velocity_classification_and_object_info():
    _, record, _ = _record()
    report = record["message"]["detection_report"]
    assert report["enu_velocity"] == {"east_rate": 30.0, "north_rate": 0.5, "up_rate": -1.0}
    top = report["classification"][0]
    assert top["type"] == "Air vehicle"
    assert top["sub_class"][0] == {
        "type": "UAV fixed wing",
        "level": 1,
        "sub_class": [{"type": "Military", "level": 2}],
    }
    assert {"type": "estimatedSwarmCount", "value": "10"} in report["object_info"]
    assert {"type": "threatLevel", "value": "hostile"} in report["object_info"]


def test_optional_fields_left_out_when_the_track_has_none():
    track = _track(velocity=None, classification=None, altitude=None)
    report = _record(track)[1]["message"]["detection_report"]
    assert "enu_velocity" not in report
    assert "classification" not in report
    assert "z" not in report["location"]


def test_two_component_velocity_and_confidence():
    message = sapient_raw.track_message(_track(velocity=[1.0, 2.0]), NODE, confidence=0.8)
    report = message["detection_report"]
    assert report["enu_velocity"] == {"east_rate": 1.0, "north_rate": 2.0}
    assert report["classification"][0]["confidence"] == 0.8


def test_fusion_node_id_is_fixed_per_pipeline_stage():
    assert sapient_raw.fusion_node_id("p-00000000a1", "fusion") == NODE
    assert sapient_raw.fusion_node_id("p-00000000a2", "fusion") != NODE
    assert len(NODE) <= 64


def test_uuid7_is_version_7_lowercase():
    value = sapient_raw.uuid7()
    parsed = uuid.UUID(value)
    assert parsed.version == 7
    assert parsed.variant == uuid.RFC_4122
    assert value == value.lower()
    assert sapient_raw.uuid7() != value


def test_predicted_label_in_object_info():
    measured = sapient_raw.track_message(_track(), NODE, predicted=False)["detection_report"]
    predicted = sapient_raw.track_message(_track(), NODE, predicted=True)["detection_report"]
    unlabelled = sapient_raw.track_message(_track(), NODE)["detection_report"]
    assert {"type": "fusionUpdate", "value": "measured"} in measured["object_info"]
    assert {"type": "fusionUpdate", "value": "predicted"} in predicted["object_info"]
    assert all(i["type"] != "fusionUpdate" for i in unlabelled["object_info"])


def test_sources_become_associated_detection():
    sources = [
        {
            "node_id": "FI-MIL-RAD-KOLI-01",
            "object_id": "A-01-SWM-W2",
            "timestamp": _track().timestamp,
        },
        {"node_id": "acoustic_array_01", "object_id": None, "timestamp": _track().timestamp},
    ]
    report = sapient_raw.track_message(_track(), NODE, sources=sources)["detection_report"]
    first, second = report["associated_detection"]
    assert first == {
        "timestamp": "2026-11-15T03:02:23.512Z",
        "node_id": "FI-MIL-RAD-KOLI-01",
        "object_id": "A-01-SWM-W2",
        "association_type": "ASSOCIATION_RELATION_CHILD",
    }
    assert "object_id" not in second
    assert (
        "associated_detection" not in sapient_raw.track_message(_track(), NODE)["detection_report"]
    )
