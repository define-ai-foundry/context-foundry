"""Tests for context_foundry.fusion.cot_input (CotRecordReader).

Records are cot-raw as cot-ingest writes them: the CoT event's attributes, its point,
and its detail as parsed.
"""

import copy

import pytest

from context_foundry.fusion.cot_input import (
    COT_NODE,
    CotRecordReader,
    known,
    point_accuracy,
    remarks_node,
)
from context_foundry.fusion.measurement import UNKNOWN_SENSOR_GEOMETRIC_ERROR

UAV = {
    "event_time": "2026-10-05T08:14:03.512Z",
    "uid": "uav-589520ccfcd2",
    "type": "a-h-A-M-H-Q",
    "how": "m-r",
    "version": "2.0",
    "start": "2026-10-05T08:14:03.512Z",
    "stale": "2026-10-05T08:14:33.512Z",
    "point": {"lat": 52.1018, "lon": 4.1279, "hae": 42.8, "ce": 5.0, "le": 2.5},
    "detail": {"contact": {"callsign": "UAV-1"}, "track": {"course": "120.3", "speed": "17.8"}},
}


def _uav(**changes):
    record = copy.deepcopy(UAV)
    for key, value in changes.items():
        record[key] = value
    return record


def test_a_located_event_is_a_detection_of_the_object_its_uid_names():
    timestamp, sensor, detection = CotRecordReader().detection(_uav())
    assert timestamp.isoformat() == "2026-10-05T08:14:03.512000+00:00"
    assert sensor == COT_NODE
    assert detection.metadata["objectId"] == "uav-589520ccfcd2"
    assert detection.metadata["stable_object_id"]
    assert detection.metadata["cot_type"] == "a-h-A-M-H-Q"
    assert "classification" not in detection.metadata
    # Three-dimensional, with the point's own error: 5 m across, 2.5 m in height.
    assert detection.state_vector.shape == (3, 1)
    assert detection.measurement_model.noise_covar[0, 0] == pytest.approx(25.0)
    assert detection.measurement_model.noise_covar[2, 2] == pytest.approx(6.25)


def test_an_unknown_height_and_error_fall_back():
    point = {"lat": 52.1, "lon": 4.1, "hae": 9999999.0, "ce": 9999999.0, "le": 9999999.0}
    _, _, detection = CotRecordReader().detection(_uav(point=point))
    # Measured in east and north only, with the cautious default for an unknown sensor.
    assert detection.state_vector.shape == (2, 1)
    sigma = UNKNOWN_SENSOR_GEOMETRIC_ERROR["base_m"]
    assert detection.measurement_model.noise_covar[0, 0] == pytest.approx(sigma**2)


def test_the_sapient_node_in_the_remarks_is_the_sensor():
    detail = {"remarks": {"_text": "node=RAD-7 det_conf=0.900 class=UAV"}}
    _, sensor, detection = CotRecordReader().detection(_uav(detail=detail))
    assert sensor == "RAD-7"
    assert detection.metadata["nodeId"] == "RAD-7"


def test_fusions_own_events_are_skipped():
    reader = CotRecordReader(own_node_id="fusion-node")
    own = _uav(detail={"remarks": {"_text": "node=fusion-node class=UAV"}})
    assert reader.detection(own) is None
    assert reader.own_records_skipped == 1
    assert reader.detection(_uav()) is not None


@pytest.mark.parametrize(
    "broken",
    [
        {"point": None},
        {"point": {"lon": 4.1}},
        {"point": {"lat": "north", "lon": 4.1}},
        {"point": {"lat": 95.0, "lon": 4.1}},
        {"uid": ""},
        {"event_time": "yesterday"},
        {"event_time": "2026-10-05T08:14:03"},
    ],
)
def test_an_event_fusion_cannot_use_is_counted(broken):
    reader = CotRecordReader()
    assert reader.detection(_uav(**broken)) is None
    assert reader.records_rejected == 1


def test_lineage_is_carried_on_the_detection():
    record = _uav()
    record["_lineage"] = {"trace_id": "abc"}
    _, _, detection = CotRecordReader().detection(record)
    assert detection.metadata["trace_id"] == "abc"


def test_helpers():
    assert known(None) is None
    assert known("12.5") == 12.5
    assert known(9999999.0) is None
    assert point_accuracy(None, None)["geometric_error"]["base_m"] == 50.0
    assert remarks_node({"detail": {"remarks": {"_text": "node= class=x"}}}) is None
    assert remarks_node({}) is None
