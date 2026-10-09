"""Tests for context_foundry.fusion.registration and the registry it feeds."""

import pytest

from context_foundry.fusion import config
from context_foundry.fusion.measurement import measurement_model_for
from context_foundry.fusion.registration import (
    geometric_error_from,
    node_location,
    range_from,
    sensor_capabilities,
)


def _value(kind, number):
    return {"type": kind, "units": "m", "unit_value": str(number)}


def _registration(tracking_type=None, error=None):
    mode = {"mode_name": "default"}
    if tracking_type:
        mode["tracking_type"] = tracking_type
    if error:
        mode["detection_definition"] = [{"geometric_error": error}]
    return {"node_id": "RAD-9", "registration": {"mode_definition": [mode]}}


def test_linear_geometric_error_is_read():
    error = {
        "type": "Standard Deviation",
        "units": "m",
        "variation_type": "Linear with Range",
        "performance_value": [_value("Base", 5), _value("At Max Range", 35)],
    }
    assert geometric_error_from({"geometric_error": error}) == {
        "variation_type": "linear_with_range",
        "base_m": 5.0,
        "at_max_range_m": 35.0,
    }


def test_a_single_value_without_a_known_variation_is_constant():
    error = {"variation_type": "", "performance_value": [_value("Standard Deviation", 12)]}
    assert geometric_error_from({"geometric_error": error}) == {
        "variation_type": "constant",
        "base_m": 12.0,
    }


@pytest.mark.parametrize(
    "error",
    [
        None,
        {"variation_type": "Linear with Range", "performance_value": [_value("Base", 5)]},
        {"variation_type": "Logarithmic", "performance_value": [_value("Base", 5)]},
        {"variation_type": "Constant", "performance_value": [_value("Base", "about five")]},
    ],
)
def test_unreadable_geometric_error_is_left_out(error):
    assert geometric_error_from({"geometric_error": error} if error else {}) is None


def test_sensor_capabilities_takes_the_first_declaration_of_each():
    error = {"variation_type": "Constant", "performance_value": [_value("Base", 8)]}
    found = sensor_capabilities(_registration("TRACKING_TYPE_TRACK", error))
    assert found == {
        "tracking_type": "TRACKING_TYPE_TRACK",
        "geometric_error": {"variation_type": "constant", "base_m": 8.0},
    }
    assert sensor_capabilities(_registration()) == {}
    assert sensor_capabilities({}) == {}


def test_registration_wins_over_the_manifest():
    config.load_sensor_network(
        sensor_network_list=[
            {
                "id": "RAD-9",
                "type": "radar",
                "lat": 62.9,
                "lon": 29.8,
                "alt": 100.0,
                "geometric_error": {"variation_type": "constant", "base_m": 40.0},
            }
        ]
    )
    config.apply_registration(
        "RAD-9",
        {
            "tracking_type": "TRACKING_TYPE_TRACK",
            "geometric_error": {"variation_type": "constant", "base_m": 8.0},
        },
    )
    profile = config.sensor_profile("RAD-9")
    assert profile["lat"] == pytest.approx(62.9)
    assert profile["geometric_error"]["base_m"] == 8.0
    assert config.has_stable_object_ids(profile)
    model = measurement_model_for(profile, (0.0, 0.0, 0.0), (10.0, 0.0, 0.0))
    assert model.noise_covar[0, 0] == pytest.approx(64.0)


def test_a_sensor_known_only_from_its_registration_has_no_position():
    config.apply_registration("NEW-1", {"tracking_type": "TRACKING_TYPE_TRACK"})
    profile = config.sensor_profile("NEW-1")
    assert profile == {"tracking_type": "TRACKING_TYPE_TRACK"}
    assert config.sensor_profile("NOBODY") is None


def test_an_empty_registration_changes_nothing():
    config.apply_registration("RAD-9", {})
    assert config.sensor_profile("RAD-9") is None


# --- node type, range and position ------------------------------------------


def test_node_type_is_read_as_a_word():
    message = {"registration": {"node_definition": [{"node_type": "NODE_TYPE_CAMERA"}]}}
    assert sensor_capabilities(message) == {"node_type": "camera"}


@pytest.mark.parametrize("node_type", ["NODE_TYPE_UNSPECIFIED", "NODE_TYPE_OTHER", None])
def test_an_unspecified_node_type_is_left_out(node_type):
    message = {"registration": {"node_definition": [{"node_type": node_type}]}}
    assert "node_type" not in sensor_capabilities(message)


@pytest.mark.parametrize(
    ("kind", "units", "value", "expected"),
    [
        ("Max Range", "m", "5000", 5000.0),
        ("Detection Range", "km", "12.5", 12500.0),
        ("Range", "metres", "800", 800.0),
    ],
)
def test_range_is_read_from_detection_performance(kind, units, value, expected):
    definition = {"detection_performance": [{"type": kind, "units": units, "unit_value": value}]}
    assert range_from(definition) == expected


@pytest.mark.parametrize(
    "performance",
    [
        {"type": "Max Range", "units": "furlongs", "unit_value": "5"},
        {"type": "Max Range", "units": "m", "unit_value": "far"},
        {"type": "Max Range", "units": "m", "unit_value": "0"},
        {"type": "Probability of Detection", "units": "%", "unit_value": "90"},
    ],
)
def test_an_unreadable_range_is_left_out(performance):
    assert range_from({"detection_performance": [performance]}) is None


def test_sensor_capabilities_reads_the_range():
    definition = {
        "detection_performance": [{"type": "Max Range", "units": "km", "unit_value": "3"}]
    }
    message = {"registration": {"mode_definition": [{"detection_definition": [definition]}]}}
    assert sensor_capabilities(message) == {"range_m": 3000.0}


def test_node_location_is_read_from_a_status_report():
    location = {
        "x": 29.8,
        "y": 62.9,
        "z": 120.0,
        "coordinate_system": "LOCATION_COORDINATE_SYSTEM_LAT_LNG_DEG_M",
    }
    assert node_location({"status_report": {"node_location": location}}) == {
        "lat": 62.9,
        "lon": 29.8,
        "alt": 120.0,
    }


@pytest.mark.parametrize(
    "location",
    [
        None,
        {"x": 500000.0, "y": 6900000.0, "coordinate_system": "LOCATION_COORDINATE_SYSTEM_UTM_M"},
        {"x": 29.8, "coordinate_system": "LOCATION_COORDINATE_SYSTEM_LAT_LNG_DEG_M"},
        {"x": 29.8, "y": 95.0, "coordinate_system": "LOCATION_COORDINATE_SYSTEM_LAT_LNG_DEG_M"},
    ],
)
def test_an_unusable_node_location_is_left_out(location):
    assert node_location({"status_report": {"node_location": location}}) is None
