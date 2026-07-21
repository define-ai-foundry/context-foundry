"""Shared fixtures and determinism/isolation guards for the context_foundry test suite."""

import random

import numpy as np
import pytest

from context_foundry.fusion import config


@pytest.fixture(autouse=True)
def _determinism():
    """Seed RNG and reset all module-level global state before AND after every test.

    config.py holds two independent pieces of global state that leak across tests
    otherwise: the sensor registry / ENU_ORIGIN_* set by load_sensor_network(), and
    the dynamic _origin_lat/_origin_lon/_origin_alt used by wgs84_to_enu/enu_to_wgs84.
    """
    random.seed(1234)
    np.random.seed(1234)
    config.reset_registry()
    config._origin_lat = None
    config._origin_lon = None
    config._origin_alt = None
    yield
    config.reset_registry()
    config._origin_lat = None
    config._origin_lon = None
    config._origin_alt = None


@pytest.fixture
def joensuu_sensor_network():
    """A tiny, representative Blue Team sensor network (mirrors config/sensors/joensuu.json)."""
    return {
        "primary_anchor_node": "FI-MIL-RAD-KOLI-01",
        "sensors": [
            {
                "id": "FI-MIL-RAD-KOLI-01",
                "type": "radar",
                "subtype": "long_range_surveillance",
                "lat": 62.95,
                "lon": 29.8,
                "alt": 250.0,
                "range_m": 15000,
                "capabilities": ["position_3d", "velocity", "classification"],
            },
            {
                "id": "acoustic_array_01",
                "type": "acoustic",
                "subtype": "passive_ring_array",
                "lat": 62.593,
                "lon": 29.836,
                "alt": 180.0,
                "capabilities": ["bearing_only"],
            },
        ],
    }


@pytest.fixture
def sapient_detection_report_message():
    """One representative, proto-compliant SAPIENT detectionReport message dict.

    Modeled on data/generated_input/joensuu_messages.json (not read at test time).
    """
    return {
        "sapientMessage": {
            "timestamp": "2026-11-15T02:45:00.000018Z",
            "nodeId": "FI-MIL-RAD-KOLI-01",
            "detectionReport": {
                "objectId": "RAD-01_DECOY",
                "state": "ACTIVE",
                "location": {
                    "x": 30.631805,
                    "y": 62.184924,
                    "z": 1578.0,
                    "coordinateSystem": "LOCATION_COORDINATE_SYSTEM_LAT_LNG_DEG_M",
                    "datum": "LOCATION_DATUM_WGS84_E",
                },
                "objectInfo": [{"type": "estimatedSwarmCount", "value": "12"}],
                "classification": [{"type": "sapient_core:UAV rotary wing", "confidence": 0.63}],
            },
        }
    }
