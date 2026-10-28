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
    skip_reason,
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
    assert detection.metadata["how"] == "m-r"
    assert detection.metadata["stale"] == "2026-10-05T08:14:33.512Z"
    assert detection.metadata["speed"] == "17.8"
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


def test_a_rendering_of_a_sapient_report_is_not_fused_again():
    # sapient-to-cot names the SAPIENT node in the remarks; fusion reads that report itself.
    reader = CotRecordReader(own_node_id="fusion-node")
    detail = {"remarks": {"_text": "node=RAD-7 det_conf=0.900 class=UAV"}}
    assert reader.detection(_uav(detail=detail)) is None
    assert reader.renderings_skipped == 1
    assert reader.own_records_skipped == 0


def test_a_rendering_of_a_status_report_is_not_fused():
    # A sensor's own marker, which fused would be an object standing at the sensor.
    reader = CotRecordReader(own_node_id="fusion-node")
    remarks = {"_text": "node=RAD-7 system=SYSTEM_OK mode=default info=INFO_NEW"}
    marker = _uav(type="a-f-G-E-S", uid="RAD-7", detail={"remarks": remarks})
    assert reader.detection(marker) is None
    assert reader.renderings_skipped == 1


def test_node_later_in_a_remark_is_the_emitters_own_text():
    reader = CotRecordReader(own_node_id="fusion-node")
    detail = {"remarks": {"_text": "relay at node=north-hill offline"}}
    assert reader.detection(_uav(detail=detail)) is not None
    assert reader.renderings_skipped == 0


def test_fusions_own_events_are_skipped():
    reader = CotRecordReader(own_node_id="fusion-node")
    own = _uav(detail={"remarks": {"_text": "node=fusion-node class=UAV"}})
    assert reader.detection(own) is None
    assert reader.own_records_skipped == 1
    assert reader.detection(_uav()) is not None


@pytest.mark.parametrize("cot_type", ["b-m-p-s-m", "t-x-c", "u-d-f", "", None, "A-f-G"])
def test_an_event_that_is_not_an_object_makes_no_detection(cot_type):
    reader = CotRecordReader(own_node_id="fusion-node")
    marker = _uav(type=cot_type)
    assert skip_reason(marker) == "not_object"
    assert reader.detection(marker) is None
    assert reader.not_objects_skipped == 1
    assert reader.renderings_skipped == 0
    assert reader.records_rejected == 0


def test_a_record_without_a_type_is_not_an_object():
    marker = _uav()
    del marker["type"]
    assert skip_reason(marker) == "not_object"


def test_skip_reason_names_renderings_whatever_their_type():
    rendering = {"_text": "node=RAD-7 class=UAV"}
    assert skip_reason(_uav(detail={"remarks": rendering})) == "rendering"
    assert skip_reason(_uav(type="b-m-p-s-m", detail={"remarks": rendering})) == "rendering"
    own = {"_text": "node=fusion-node class=UAV"}
    assert skip_reason(_uav(detail={"remarks": own})) == "rendering"


def test_skip_reason_is_none_for_an_object():
    assert skip_reason(_uav()) is None
    # Unusable but an object: the reader rejects it, it is not skipped.
    assert skip_reason(_uav(point=None)) is None


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


@pytest.mark.parametrize("bad", [{"lat": float("nan")}, {"lon": float("inf")}])
def test_a_point_at_no_real_position_is_counted(bad):
    reader = CotRecordReader()
    record = _uav(point={**UAV["point"], **bad})
    assert reader.detection(record) is None
    assert reader.records_rejected == 1


def test_a_nan_error_estimate_reads_as_unknown():
    assert known(float("nan")) is None
    assert known("Infinity") is None
