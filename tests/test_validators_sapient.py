"""Tests for context_foundry.fusion.validators.sapient."""

import copy
import math

import pytest

from context_foundry.fusion import config
from context_foundry.fusion.schemas import InternalDetection
from context_foundry.fusion.validators.sapient import SapientValidator


@pytest.fixture
def validator():
    return SapientValidator()


def _range_bearing_message():
    return {
        "sapientMessage": {
            "timestamp": "2026-11-15T02:57:27.016900Z",
            "nodeId": "FI-CIV-ACU-SAVONVOIMA-01",
            "detectionReport": {
                "objectId": "ACO-01_FPV",
                "state": "ACTIVE",
                "rangeBearing": {
                    "elevation": 0.0,
                    "azimuth": 90.0,
                    "range": 100.0,
                    "coordinateSystem": "RANGE_BEARING_COORDINATE_SYSTEM_DEGREES_M",
                    "datum": "RANGE_BEARING_DATUM_TRUE",
                },
                "classification": [{"type": "sapient_core:UAV rotary wing", "confidence": 0.38}],
            },
        }
    }


# --- validate() ----------------------------------------------------------------


def test_validate_missing_root_key(validator):
    is_valid, error = validator.validate({})
    assert is_valid is False
    assert "sapientMessage" in error


def test_validate_accepts_location_detection_report(validator, sapient_detection_report_message):
    is_valid, error = validator.validate(sapient_detection_report_message)
    assert is_valid is True
    assert error == ""


def test_validate_accepts_range_bearing_detection_report(validator):
    is_valid, _error = validator.validate(_range_bearing_message())
    assert is_valid is True


def test_validate_rejects_wrong_content_type(validator):
    payload = {
        "sapientMessage": {
            "timestamp": "2026-01-01T00:00:00Z",
            "nodeId": "node-1",
            "statusReport": {"reportId": "r1"},
        }
    }
    is_valid, error = validator.validate(payload)
    assert is_valid is False
    assert "status_report" in error


def test_validate_rejects_detection_report_missing_location_oneof(validator):
    payload = {
        "sapientMessage": {
            "timestamp": "2026-01-01T00:00:00Z",
            "nodeId": "node-1",
            "detectionReport": {"objectId": "obj-1"},
        }
    }
    is_valid, error = validator.validate(payload)
    assert is_valid is False
    assert "location_oneof" in error


def test_validate_rejects_unparseable_payload(validator):
    payload = {"sapientMessage": {"thisFieldDoesNotExist": "boom"}}
    is_valid, error = validator.validate(payload)
    assert is_valid is False
    assert "Protobuf schema violation" in error


# --- normalize(): location branch ----------------------------------------------


def test_normalize_location_with_altitude(validator, sapient_detection_report_message):
    det = validator.normalize(sapient_detection_report_message)
    assert isinstance(det, InternalDetection)
    assert det.sensor_id == "FI-MIL-RAD-KOLI-01"
    assert det.latitude == pytest.approx(62.184924)
    assert det.longitude == pytest.approx(30.631805)
    assert det.altitude == pytest.approx(1578.0)
    assert det.classification == "sapient_core:UAV rotary wing"
    assert det.raw_metadata["original_envelope"] == sapient_detection_report_message


def test_normalize_location_without_altitude(validator, sapient_detection_report_message):
    payload = copy.deepcopy(sapient_detection_report_message)
    del payload["sapientMessage"]["detectionReport"]["location"]["z"]
    det = validator.normalize(payload)
    assert det.altitude is None


def test_normalize_classification_defaults_to_unknown(validator, sapient_detection_report_message):
    payload = copy.deepcopy(sapient_detection_report_message)
    del payload["sapientMessage"]["detectionReport"]["classification"]
    det = validator.normalize(payload)
    assert det.classification == "Unknown"


def test_normalize_detection_confidence_present(validator, sapient_detection_report_message):
    payload = copy.deepcopy(sapient_detection_report_message)
    payload["sapientMessage"]["detectionReport"]["detectionConfidence"] = 0.75
    det = validator.normalize(payload)
    assert det.confidence == pytest.approx(0.75)


def test_normalize_detection_confidence_absent(validator, sapient_detection_report_message):
    det = validator.normalize(sapient_detection_report_message)
    assert det.confidence is None


# --- normalize(): range_bearing branch ------------------------------------------


def test_normalize_range_bearing_uses_enu_to_wgs84(validator):
    config.set_reference_origin(62.9, 29.8, 0.0)
    det = validator.normalize(_range_bearing_message())
    assert det.sensor_id == "FI-CIV-ACU-SAVONVOIMA-01"
    # azimuth=90deg (east), range=100m -> should land east of the origin.
    assert det.longitude > 29.8
    assert det.latitude == pytest.approx(62.9, abs=1e-3)


def test_normalize_range_bearing_offsets_match_trig():
    config.set_reference_origin(0.0, 0.0, 0.0)
    payload = _range_bearing_message()
    payload["sapientMessage"]["detectionReport"]["rangeBearing"]["azimuth"] = 30.0
    payload["sapientMessage"]["detectionReport"]["rangeBearing"]["range"] = 1000.0

    validator_instance = SapientValidator()
    det = validator_instance.normalize(payload)

    az = math.radians(30.0)
    e_expected = 1000.0 * math.sin(az)
    n_expected = 1000.0 * math.cos(az)
    lat_expected, lon_expected, _ = config.enu_to_wgs84(e_expected, n_expected, 0.0)
    assert det.latitude == pytest.approx(lat_expected)
    assert det.longitude == pytest.approx(lon_expected)


def test_process_message_range_bearing_without_origin_is_rejected(caplog):
    # No reference origin registered anywhere -> enu_to_wgs84 raises -> process_message
    # catches it in normalize() and returns None (logged as a normalization failure).
    assert config.ENU_ORIGIN_LAT is None
    result = SapientValidator().process_message(_range_bearing_message())
    assert result is None


# --- normalize(): defensive neither-location-nor-range_bearing branch -----------


def test_normalize_raises_when_neither_location_nor_range_bearing_present(validator):
    # validate() always filters this out before normalize() is reached in the real
    # pipeline (both are required members of the same protobuf oneof), so this
    # defensive branch can only be exercised by calling normalize() directly.
    payload = {
        "sapientMessage": {
            "timestamp": "2026-01-01T00:00:00Z",
            "nodeId": "node-1",
            "detectionReport": {"objectId": "obj-1"},
        }
    }
    with pytest.raises(ValueError, match="missing both"):
        validator.normalize(payload)


def test_process_message_full_pipeline_success(sapient_detection_report_message):
    det = SapientValidator().process_message(sapient_detection_report_message)
    assert det is not None
    assert det.sensor_id == "FI-MIL-RAD-KOLI-01"
