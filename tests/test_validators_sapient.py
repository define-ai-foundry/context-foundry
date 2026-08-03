"""Tests for context_foundry.fusion.validators.sapient."""

import copy
import logging

import pymap3d as pm
import pytest
from pymap3d.vincenty import vdist

from context_foundry.fusion import config
from context_foundry.fusion.schemas import InternalDetection
from context_foundry.fusion.sources.base import swarm_count
from context_foundry.fusion.validators import sapient as sapient_module
from context_foundry.fusion.validators.sapient import SapientValidator

ANCHOR_NODE = "FI-MIL-RAD-KOLI-01"
NON_ANCHOR_NODE = "acoustic_array_01"


@pytest.fixture
def validator():
    return SapientValidator()


@pytest.fixture(autouse=True)
def _forget_unregistered_nodes(monkeypatch):
    """The once-per-node warning is module state, so it would leak between tests."""
    sapient_module._UNREGISTERED_NODES_SEEN.clear()
    monkeypatch.setattr(sapient_module, "_unregistered_seen_count", 0)
    monkeypatch.setattr(sapient_module, "_next_unregistered_report", 0.0)
    yield
    sapient_module._UNREGISTERED_NODES_SEEN.clear()


@pytest.fixture
def registry(joensuu_sensor_network):
    """Sensor registry with the anchor and one non-anchor (bearing-only) sensor."""
    config.load_sensor_network(
        sensor_network_list=joensuu_sensor_network["sensors"],
        primary_anchor_node=ANCHOR_NODE,
    )
    return joensuu_sensor_network


def _range_bearing_message(
    node_id: str = "FI-CIV-ACU-SAVONVOIMA-01",
    azimuth: float = 90.0,
    rng: float = 100.0,
    elevation: float | None = 0.0,
):
    range_bearing = {
        "azimuth": azimuth,
        "range": rng,
        "coordinateSystem": "RANGE_BEARING_COORDINATE_SYSTEM_DEGREES_M",
        "datum": "RANGE_BEARING_DATUM_TRUE",
    }
    if elevation is not None:
        range_bearing["elevation"] = elevation
    return {
        "sapientMessage": {
            "timestamp": "2026-11-15T02:57:27.016900Z",
            "nodeId": node_id,
            "detectionReport": {
                "objectId": "ACO-01_FPV",
                "state": "ACTIVE",
                "rangeBearing": range_bearing,
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


def test_normalize_location_ignores_the_sensor_registry(
    validator, registry, sapient_detection_report_message
):
    """WGS84 locations are already absolute: neither the anchor nor the reporting node
    may move them, and an unregistered reporter is still perfectly usable."""
    payload = copy.deepcopy(sapient_detection_report_message)
    payload["sapientMessage"]["nodeId"] = "FI-CIV-ACU-SAVONVOIMA-01"

    det = validator.normalize(payload)

    assert config.get_sensor("FI-CIV-ACU-SAVONVOIMA-01") is None
    assert det.latitude == pytest.approx(62.184924)
    assert det.longitude == pytest.approx(30.631805)
    assert det.altitude == pytest.approx(1578.0)


# --- normalize(): range_bearing branch ------------------------------------------


def test_normalize_range_bearing_is_measured_from_the_reporting_sensor(validator, registry):
    """A non-anchor sensor's range/azimuth must resolve about that sensor's own position."""
    sensor = config.get_sensor(NON_ANCHOR_NODE)
    det = validator.normalize(
        _range_bearing_message(node_id=NON_ANCHOR_NODE, azimuth=0.0, rng=2000.0)
    )

    # Independent geodesic check: the detection is 2000 m from the sensor, due north.
    distance_m, bearing_deg = vdist(sensor["lat"], sensor["lon"], det.latitude, det.longitude)
    assert distance_m == pytest.approx(2000.0, abs=1.0)
    assert bearing_deg == pytest.approx(0.0, abs=0.01)

    # ...and nowhere near the anchor, which is where the global-origin conversion put it.
    anchor = config.get_sensor(ANCHOR_NODE)
    from_anchor_m, _ = vdist(anchor["lat"], anchor["lon"], det.latitude, det.longitude)
    assert from_anchor_m > 30_000.0


def test_normalize_range_bearing_azimuth_is_clockwise_from_north(validator, registry):
    """Proto: azimuth is 'in relation to the node's north'; 90 deg must land due east."""
    sensor = config.get_sensor(NON_ANCHOR_NODE)
    det = validator.normalize(
        _range_bearing_message(node_id=NON_ANCHOR_NODE, azimuth=90.0, rng=3000.0)
    )

    distance_m, bearing_deg = vdist(sensor["lat"], sensor["lon"], det.latitude, det.longitude)
    assert distance_m == pytest.approx(3000.0, abs=1.5)
    assert bearing_deg == pytest.approx(90.0, abs=0.05)


def test_normalize_range_bearing_uses_elevation_for_altitude(validator, registry):
    """Elevation is above the node's horizon: it lifts the target and shortens ground range."""
    sensor = config.get_sensor(NON_ANCHOR_NODE)
    det = validator.normalize(
        _range_bearing_message(node_id=NON_ANCHOR_NODE, azimuth=45.0, rng=4000.0, elevation=30.0)
    )

    lat_expected, lon_expected, alt_expected = pm.aer2geodetic(
        45.0, 30.0, 4000.0, sensor["lat"], sensor["lon"], sensor["alt"]
    )
    assert det.latitude == pytest.approx(lat_expected, abs=1e-9)
    assert det.longitude == pytest.approx(lon_expected, abs=1e-9)
    # 4000 m slant range at 30 deg elevation -> 2000 m above the sensor.
    assert det.altitude == pytest.approx(sensor["alt"] + 2000.0, abs=1.0)
    assert det.altitude == pytest.approx(alt_expected, abs=1e-6)


def test_normalize_range_bearing_without_elevation_reports_no_altitude(validator, registry):
    """The message carries no height, so altitude stays unknown rather than being invented."""
    sensor = config.get_sensor(NON_ANCHOR_NODE)
    det = validator.normalize(
        _range_bearing_message(node_id=NON_ANCHOR_NODE, azimuth=180.0, rng=1500.0, elevation=None)
    )

    assert det.altitude is None
    distance_m, bearing_deg = vdist(sensor["lat"], sensor["lon"], det.latitude, det.longitude)
    assert distance_m == pytest.approx(1500.0, abs=1.0)
    assert bearing_deg == pytest.approx(180.0, abs=0.01)


def test_normalize_range_bearing_from_the_anchor_node_is_unchanged(validator, registry):
    """Regression guard: the anchor is the ENU origin, so its reports were already correct."""
    anchor = config.get_sensor(ANCHOR_NODE)
    det = validator.normalize(
        _range_bearing_message(node_id=ANCHOR_NODE, azimuth=225.0, rng=5000.0)
    )

    lat_expected, lon_expected, alt_expected = pm.aer2geodetic(
        225.0, 0.0, 5000.0, anchor["lat"], anchor["lon"], anchor["alt"]
    )
    assert det.latitude == pytest.approx(lat_expected, abs=1e-9)
    assert det.longitude == pytest.approx(lon_expected, abs=1e-9)
    # 5 km along the anchor's tangent plane sits ~2 m above the ellipsoid.
    assert det.altitude == pytest.approx(alt_expected, abs=1e-6)


def test_normalize_range_bearing_ignores_the_dynamic_global_origin(validator, registry):
    """The shared origin is used by the tracker and other sources; it must not be read here."""
    config.set_reference_origin(0.0, 0.0, 0.0)
    sensor = config.get_sensor(NON_ANCHOR_NODE)

    det = validator.normalize(
        _range_bearing_message(node_id=NON_ANCHOR_NODE, azimuth=0.0, rng=1000.0)
    )

    distance_m, _ = vdist(sensor["lat"], sensor["lon"], det.latitude, det.longitude)
    assert distance_m == pytest.approx(1000.0, abs=1.0)
    # And the shared origin is left exactly as the caller set it.
    assert (config._origin_lat, config._origin_lon, config._origin_alt) == (0.0, 0.0, 0.0)


def test_range_bearing_and_location_reports_of_one_target_agree(validator, registry):
    """The property fusion depends on: two sensors reporting one target must co-locate."""
    sensor = config.get_sensor(NON_ANCHOR_NODE)
    # Ground truth: 2500 m north-east of the bearing-only sensor.
    truth_lat, truth_lon, _ = pm.aer2geodetic(
        45.0, 0.0, 2500.0, sensor["lat"], sensor["lon"], sensor["alt"]
    )

    rb_det = validator.normalize(
        _range_bearing_message(node_id=NON_ANCHOR_NODE, azimuth=45.0, rng=2500.0)
    )
    loc_det = validator.normalize(
        {
            "sapientMessage": {
                "timestamp": "2026-11-15T02:57:27.016900Z",
                "nodeId": ANCHOR_NODE,
                "detectionReport": {
                    "objectId": "RAD-01_FPV",
                    "location": {
                        "x": truth_lon,
                        "y": truth_lat,
                        "coordinateSystem": "LOCATION_COORDINATE_SYSTEM_LAT_LNG_DEG_M",
                        "datum": "LOCATION_DATUM_WGS84_E",
                    },
                },
            }
        }
    )

    separation_m, _ = vdist(rb_det.latitude, rb_det.longitude, loc_det.latitude, loc_det.longitude)
    assert separation_m == pytest.approx(0.0, abs=1.0)


# --- normalize(): range/bearing from an unregistered node ------------------------


def test_normalize_range_bearing_from_unregistered_node_is_rejected(validator, registry):
    with pytest.raises(ValueError, match="unregistered node 'FI-CIV-ACU-SAVONVOIMA-01'"):
        validator.normalize(_range_bearing_message())


def test_process_message_range_bearing_from_unregistered_node_returns_none(registry):
    assert SapientValidator().process_message(_range_bearing_message()) is None


def test_unregistered_node_is_reported_once_per_node(validator, registry, caplog):
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            with pytest.raises(ValueError):
                validator.normalize(_range_bearing_message(node_id="unknown-node-a"))
        with pytest.raises(ValueError):
            validator.normalize(_range_bearing_message(node_id="unknown-node-b"))

    warnings = [r.message for r in caplog.records if "not in the sensor registry" in r.message]
    assert len(warnings) == 2
    assert "unknown-node-a" in warnings[0]
    assert "unknown-node-b" in warnings[1]


def test_process_message_range_bearing_with_empty_registry_returns_none():
    assert config.list_sensors() == []
    assert SapientValidator().process_message(_range_bearing_message(node_id=ANCHOR_NODE)) is None


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


def test_normalize_keeps_the_report_for_a_snake_case_envelope():
    """ParseDict accepts either spelling, so picking the report out of the raw
    dict by camelCase key silently lost objectId and the swarm count again."""
    validator = SapientValidator()
    payload = {
        "sapientMessage": {
            "timestamp": "2026-01-01T00:00:00Z",
            "node_id": "node-A",
            "detection_report": {
                "object_id": "obj-1",
                "location": {
                    "x": 29.8,
                    "y": 62.9,
                    "z": 100.0,
                    "coordinateSystem": "LOCATION_COORDINATE_SYSTEM_LAT_LNG_DEG_M",
                    "datum": "LOCATION_DATUM_WGS84_E",
                },
                "object_info": [{"type": "estimatedSwarmCount", "value": "9"}],
            },
        }
    }

    det = validator.normalize(payload)

    report = det.raw_metadata["original_report"]
    assert report["objectId"] == "obj-1"
    assert swarm_count(report) == 9


def test_unregistered_node_bookkeeping_is_bounded(validator, registry, monkeypatch, caplog):
    """The node id comes off an unauthenticated socket, so remembering every one
    is a set entry per spoofed datagram in memory and a warning per datagram in
    the log."""
    monkeypatch.setattr(sapient_module, "MAX_UNREGISTERED_NODES_TRACKED", 8)
    monkeypatch.setattr(sapient_module, "UNREGISTERED_REPORT_SECONDS", 3600.0)

    with caplog.at_level("WARNING"):
        for i in range(500):
            payload = _range_bearing_message(node_id=f"spoofed-{i}", azimuth=0.0, rng=2000.0)
            assert validator.process_message(payload) is None

    assert len(sapient_module._UNREGISTERED_NODES_SEEN) <= 8
    named = [r for r in caplog.records if "is not in the sensor registry" in r.message]
    flood = [r for r in caplog.records if "unregistered node ids seen" in r.message]
    # The first few are named individually; the rest collapse into one rate-limited
    # line carrying the count, which is what says "flood" rather than "stray node".
    assert len(named) == 8
    assert len(flood) == 1
    assert "9 so far" in flood[0].message  # fired on the first id past the bound


def test_a_known_unregistered_node_is_reported_once(validator, registry, caplog):
    """One misconfigured sensor should say so once, not once per datagram."""
    with caplog.at_level("WARNING"):
        for _ in range(20):
            payload = _range_bearing_message(node_id="not-in-manifest", azimuth=0.0, rng=2000.0)
            assert validator.process_message(payload) is None

    assert len([r for r in caplog.records if "not in the sensor registry" in r.message]) == 1
