"""Tests for context_foundry.fusion.sapient_input (SapientRecordReader).

Records are sapient-raw as sapient-ingest writes them: proto field names, the decoded
message carried whole. No sensor file is loaded: sensors are learned from their
Registration and status reports, as a pipeline stage learns them.
"""

import pytest

from context_foundry.fusion import config
from context_foundry.fusion.sapient_input import SapientRecordReader

RADAR = "RAD-1"
FUSION = "fusion-node"
LAT_LNG = "LOCATION_COORDINATE_SYSTEM_LAT_LNG_DEG_M"


def _record(content_type, message_body, node=RADAR, ts="2026-11-15T03:00:01Z"):
    message = {"timestamp": ts, "node_id": node, content_type: message_body}
    return {"event_time": ts, "content_type": content_type, "node_id": node, "message": message}


def _registration(node=RADAR, error=None, node_type="NODE_TYPE_RADAR"):
    mode = {"mode_name": "default", "tracking_type": "TRACKING_TYPE_TRACK"}
    if error:
        mode["detection_definition"] = [{"geometric_error": error}]
    body = {"node_definition": [{"node_type": node_type}], "mode_definition": [mode]}
    return _record("registration", body, node)


def _status(lat, lon, alt=50.0, node=RADAR):
    location = {"x": lon, "y": lat, "z": alt, "coordinate_system": LAT_LNG}
    return _record("status_report", {"node_location": location}, node)


def _detection(lat=62.95, lon=29.8, node=RADAR, object_id="obj-1"):
    location = {"x": lon, "y": lat, "z": 300.0, "coordinate_system": LAT_LNG}
    report = {"object_id": object_id, "location": location}
    return _record("detection_report", report, node)


def _linear(base=5.0, far=35.0):
    return {
        "type": "Standard Deviation",
        "units": "m",
        "variation_type": "Linear with Range",
        "performance_value": [
            {"type": "Base", "units": "m", "unit_value": str(base)},
            {"type": "At Max Range", "units": "m", "unit_value": str(far)},
        ],
    }


def test_a_registration_and_a_status_report_describe_the_sensor():
    reader = SapientRecordReader()
    assert reader.learn(_registration(error=_linear()))
    assert reader.learn(_status(62.9, 29.8))
    profile = config.sensor_profile(RADAR)
    assert profile["node_type"] == "radar"
    assert profile["tracking_type"] == "TRACKING_TYPE_TRACK"
    assert profile["geometric_error"]["variation_type"] == "linear_with_range"
    assert (profile["lat"], profile["lon"], profile["alt"]) == (62.9, 29.8, 50.0)


def test_a_detection_report_is_not_learned_from():
    reader = SapientRecordReader()
    assert not reader.learn(_detection())
    assert reader.is_detection(_detection())
    assert not reader.is_detection(_status(62.9, 29.8))


def test_a_detection_is_measured_from_the_sensor_its_status_report_placed():
    reader = SapientRecordReader()
    reader.learn(_registration(error=_linear()))
    reader.learn(_status(62.9, 29.8))
    timestamp, sensor_id, detection = reader.detection(_detection())
    assert sensor_id == RADAR
    assert timestamp.isoformat() == "2026-11-15T03:00:01+00:00"
    assert detection.metadata["stable_object_id"]
    assert detection.metadata["sensor_geodetic"]["latitude"] == 62.9
    # About 5.6 km from the sensor with no range declared: the error at its far end.
    assert detection.measurement_model.noise_covar[0, 0] == pytest.approx(35.0**2)


def test_without_a_status_report_the_error_is_taken_at_its_far_end():
    reader = SapientRecordReader()
    reader.learn(_registration(error=_linear()))
    _, _, detection = reader.detection(_detection())
    assert detection.metadata["sensor_geodetic"] is None
    assert detection.measurement_model.noise_covar[0, 0] == pytest.approx(35.0**2)


def test_a_sensor_that_moves_is_projected_again():
    reader = SapientRecordReader()
    reader.learn(_registration(error=_linear()))
    reader.learn(_status(62.9, 29.8))
    reader.detection(_detection())
    first = reader._sensor_enu[RADAR]
    reader.learn(_status(62.91, 29.8))
    reader.detection(_detection())
    assert reader._sensor_enu[RADAR] != first


def test_fusions_own_records_are_skipped():
    reader = SapientRecordReader(own_node_id=FUSION)
    assert not reader.learn(_registration(node=FUSION))
    assert config.sensor_profile(FUSION) is None
    assert not reader.is_detection(_detection(node=FUSION))
    assert reader.own_records_skipped == 1
    assert reader.is_detection(_detection())


def test_a_report_that_does_not_validate_is_counted():
    reader = SapientRecordReader()
    record = _detection()
    del record["message"]["detection_report"]["location"]
    assert reader.detection(record) is None
    assert reader.records_rejected == 1


def test_a_status_report_without_a_usable_location_teaches_nothing():
    reader = SapientRecordReader()
    assert reader.learn(_record("status_report", {}))
    assert config.sensor_profile(RADAR) is None
