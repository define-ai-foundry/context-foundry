"""Tests for context_foundry.fusion.config."""

import json
import logging

import pytest

from context_foundry.fusion import config

# --- load_sensor_network: direct list input ---------------------------------


def test_load_from_list_sets_registry_and_origin(joensuu_sensor_network):
    config.load_sensor_network(sensor_network_list=joensuu_sensor_network["sensors"])
    assert set(config.list_sensors()) == {"FI-MIL-RAD-KOLI-01", "acoustic_array_01"}
    # No explicit primary_anchor_node and no sensor marked "primary_anchor" -> fallback to first.
    assert config.ENU_ORIGIN_NODE_ID == "FI-MIL-RAD-KOLI-01"
    assert config.ENU_ORIGIN_LAT == 62.95
    assert config.ENU_ORIGIN_LON == 29.8
    assert config.ENU_ORIGIN_ALT == 250.0


def test_load_from_list_with_explicit_primary_anchor_node(joensuu_sensor_network):
    config.load_sensor_network(
        sensor_network_list=joensuu_sensor_network["sensors"],
        primary_anchor_node="acoustic_array_01",
    )
    assert config.ENU_ORIGIN_NODE_ID == "acoustic_array_01"


def test_load_from_list_with_primary_anchor_flag_on_sensor():
    sensors = [
        {"id": "a", "lat": 1.0, "lon": 2.0},
        {"id": "b", "lat": 3.0, "lon": 4.0, "primary_anchor": True},
    ]
    config.load_sensor_network(sensor_network_list=sensors)
    assert config.ENU_ORIGIN_NODE_ID == "b"


def test_load_from_list_unknown_primary_anchor_falls_back_with_warning(caplog):
    sensors = [{"id": "a", "lat": 1.0, "lon": 2.0}]
    with caplog.at_level(logging.WARNING):
        config.load_sensor_network(
            sensor_network_list=sensors, primary_anchor_node="does-not-exist"
        )
    assert config.ENU_ORIGIN_NODE_ID == "a"
    assert any("not found" in record.message for record in caplog.records)


def test_load_from_list_sensor_missing_id_is_skipped_with_warning(caplog):
    sensors = [{"lat": 1.0, "lon": 2.0}, {"id": "a", "lat": 5.0, "lon": 6.0}]
    with caplog.at_level(logging.WARNING):
        config.load_sensor_network(sensor_network_list=sensors)
    assert config.list_sensors() == ["a"]
    assert any("Skipping sensor entry" in record.message for record in caplog.records)


def test_load_from_list_defaults_and_extra_fields_preserved():
    sensors = [{"id": "a", "lat": 1.0, "lon": 2.0, "special_field": "kept"}]
    config.load_sensor_network(sensor_network_list=sensors)
    sensor = config.get_sensor("a")
    assert sensor["alt"] == 0.0
    assert sensor["range_m"] == 10000.0
    assert sensor["update_rate_sec"] == 10.0
    assert sensor["capabilities"] == []
    assert sensor["status"] == "operational"
    assert sensor["type"] == "unknown"
    assert sensor["subtype"] is None
    assert sensor["special_field"] == "kept"


def test_load_sensor_network_no_data_raises():
    with pytest.raises(ValueError, match="No sensor data provided"):
        config.load_sensor_network()


# --- load_sensor_network: file input -----------------------------------------


def test_load_from_file_new_format(tmp_path, joensuu_sensor_network):
    path = tmp_path / "sensors.json"
    path.write_text(json.dumps(joensuu_sensor_network), encoding="utf-8")
    config.load_sensor_network(sensor_config_path=path)
    assert config.ENU_ORIGIN_NODE_ID == "FI-MIL-RAD-KOLI-01"


def test_load_from_file_legacy_format(tmp_path):
    legacy = {"sensor_network": [{"id": "legacy-1", "lat": 10.0, "lon": 20.0}]}
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    config.load_sensor_network(sensor_config_path=path)
    assert config.list_sensors() == ["legacy-1"]
    assert config.ENU_ORIGIN_NODE_ID == "legacy-1"


def test_load_from_file_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        config.load_sensor_network(sensor_config_path=tmp_path / "nope.json")


def test_load_from_file_invalid_format_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"nonsense": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid sensor config format"):
        config.load_sensor_network(sensor_config_path=path)


def test_load_from_file_string_path_accepted(tmp_path, joensuu_sensor_network):
    path = tmp_path / "sensors.json"
    path.write_text(json.dumps(joensuu_sensor_network), encoding="utf-8")
    config.load_sensor_network(sensor_config_path=str(path))
    assert config.ENU_ORIGIN_NODE_ID == "FI-MIL-RAD-KOLI-01"


# --- registry accessors -------------------------------------------------------


def test_get_sensor_unknown_returns_none():
    assert config.get_sensor("nope") is None


def test_get_all_sensors_returns_independent_copy(joensuu_sensor_network):
    config.load_sensor_network(sensor_network_list=joensuu_sensor_network["sensors"])
    snapshot = config.get_all_sensors()
    snapshot["extra"] = {}
    assert "extra" not in config.SENSOR_REGISTRY


def test_reset_registry_clears_everything(joensuu_sensor_network):
    config.load_sensor_network(sensor_network_list=joensuu_sensor_network["sensors"])
    config.reset_registry()
    assert config.list_sensors() == []
    assert config.ENU_ORIGIN_LAT is None
    assert config.ENU_ORIGIN_LON is None
    assert config.ENU_ORIGIN_ALT is None
    assert config.ENU_ORIGIN_NODE_ID is None


def test_load_blue_sensor_network_deprecated_wrapper(caplog):
    sensors = [{"id": "FI-MIL-RAD-KOLI-01", "lat": 1.0, "lon": 2.0}]
    with caplog.at_level(logging.WARNING):
        config.load_blue_sensor_network(sensors)
    assert config.ENU_ORIGIN_NODE_ID == "FI-MIL-RAD-KOLI-01"
    assert any("deprecated" in record.message for record in caplog.records)


# --- dynamic ENU origin: set_reference_origin / wgs84_to_enu / enu_to_wgs84 --


def test_wgs84_to_enu_auto_initializes_origin_from_first_call():
    e, n, u = config.wgs84_to_enu(62.9, 29.8, 100.0)
    assert e == pytest.approx(0.0, abs=1e-6)
    assert n == pytest.approx(0.0, abs=1e-6)
    assert u == pytest.approx(0.0, abs=1e-6)
    assert config._origin_lat == 62.9


def test_wgs84_to_enu_uses_explicit_origin():
    config.set_reference_origin(62.9, 29.8, 0.0)
    e, n, u = config.wgs84_to_enu(62.9, 29.8, 0.0)
    assert e == pytest.approx(0.0, abs=1e-6)
    assert n == pytest.approx(0.0, abs=1e-6)
    assert u == pytest.approx(0.0, abs=1e-6)


def test_enu_to_wgs84_round_trip_with_explicit_origin():
    config.set_reference_origin(62.9, 29.8, 100.0)
    e, n, u = config.wgs84_to_enu(62.901, 29.802, 105.0)
    lat, lon, alt = config.enu_to_wgs84(e, n, u)
    assert lat == pytest.approx(62.901, abs=1e-6)
    assert lon == pytest.approx(29.802, abs=1e-6)
    assert alt == pytest.approx(105.0, abs=1e-2)


def test_enu_to_wgs84_raises_when_no_origin_anywhere():
    with pytest.raises(ValueError, match="Reference origin was never set"):
        config.enu_to_wgs84(0.0, 0.0, 0.0)


def test_enu_to_wgs84_about_uses_the_given_origin_and_touches_no_global_state():
    config.set_reference_origin(0.0, 0.0, 0.0)

    lat, lon, alt = config.enu_to_wgs84_about(0.0, 0.0, 0.0, 62.593, 29.836, 180.0)

    assert lat == pytest.approx(62.593, abs=1e-9)
    assert lon == pytest.approx(29.836, abs=1e-9)
    assert alt == pytest.approx(180.0, abs=1e-6)
    # The shared dynamic origin is untouched, so concurrent readers are unaffected.
    assert (config._origin_lat, config._origin_lon, config._origin_alt) == (0.0, 0.0, 0.0)


def test_enu_to_wgs84_about_round_trips_against_wgs84_to_enu():
    origin = (62.593, 29.836, 180.0)
    config.set_reference_origin(*origin)
    e, n, u = config.wgs84_to_enu(62.60, 29.85, 300.0)

    lat, lon, alt = config.enu_to_wgs84_about(e, n, u, *origin)

    assert lat == pytest.approx(62.60, abs=1e-9)
    assert lon == pytest.approx(29.85, abs=1e-9)
    assert alt == pytest.approx(300.0, abs=1e-3)


def test_enu_to_wgs84_about_needs_no_origin_to_have_been_set():
    assert config._origin_lat is None
    lat, lon, _alt = config.enu_to_wgs84_about(0.0, 1000.0, 0.0, 62.0, 29.0, 0.0)
    assert lat > 62.0
    assert lon == pytest.approx(29.0, abs=1e-9)
    assert config._origin_lat is None


def test_enu_to_wgs84_falls_back_to_registry_origin(joensuu_sensor_network):
    config.load_sensor_network(sensor_network_list=joensuu_sensor_network["sensors"])
    # Dynamic origin was never explicitly set; enu_to_wgs84 should sync from the registry.
    lat, lon, alt = config.enu_to_wgs84(0.0, 0.0, 0.0)
    assert lat == pytest.approx(62.95, abs=1e-6)
    assert lon == pytest.approx(29.8, abs=1e-6)
    assert alt == pytest.approx(250.0, abs=1e-2)
    assert config._origin_lat == 62.95
