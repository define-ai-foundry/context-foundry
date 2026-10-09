# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# src/context_foundry/fusion/config.py

"""
Sensor Registry and Global Coordinate Configuration for the Fusion Engine.

This module maintains the known Blue Team sensor network and establishes
the local ENU (East-North-Up) coordinate frame origin.

Supports:
- New dedicated sensor manifest format (recommended for real/live use)
- Legacy simulator scenario format (for backward compatibility)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pymap3d as pm

# Global sensor registry: node_id -> sensor metadata
SENSOR_REGISTRY: dict[str, dict[str, Any]] = {}

# Global ENU origin (set by load_sensor_network)
ENU_ORIGIN_LAT: float | None = None
ENU_ORIGIN_LON: float | None = None
ENU_ORIGIN_ALT: float | None = None
ENU_ORIGIN_NODE_ID: str | None = None

logger = logging.getLogger(__name__)


def load_sensor_network(
    sensor_config_path: str | Path | None = None,
    sensor_network_list: list[dict[str, Any]] | None = None,
    primary_anchor_node: str | None = None,
) -> None:
    """
    Unified loader for Blue Team sensor network.

    Accepts either:
      1. Path to a JSON config file (new or legacy format)
      2. Direct list of sensor dictionaries (for runtime registration or tests)

    New format (recommended):
        {
          "primary_anchor_node": "...",
          "sensors": [ ... ]
        }

    Legacy simulator format:
        {
          "sensor_network": [ ... ]
        }
    """
    global ENU_ORIGIN_LAT, ENU_ORIGIN_LON, ENU_ORIGIN_ALT, ENU_ORIGIN_NODE_ID

    # Load from file if path is provided
    if sensor_config_path is not None:
        path = Path(sensor_config_path)
        if not path.exists():
            raise FileNotFoundError(f"Sensor configuration file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # New dedicated sensor manifest format
        if "sensors" in data:
            sensor_network_list = data["sensors"]
            primary_anchor_node = primary_anchor_node or data.get("primary_anchor_node")
        # Legacy simulator scenario format
        elif "sensor_network" in data:
            sensor_network_list = data["sensor_network"]
            # Simulator files may or may not specify primary anchor
        else:
            raise ValueError(
                "Invalid sensor config format. Expected 'sensors' (new recommended) "
                "or 'sensor_network' (legacy simulator) key."
            )

    if not sensor_network_list:
        raise ValueError("No sensor data provided (neither path nor list)")

    # Clear previous registry
    SENSOR_REGISTRY.clear()

    # Populate registry
    for sensor in sensor_network_list:
        node_id = sensor.get("id")
        if not node_id:
            logger.warning("Skipping sensor entry without 'id' field")
            continue

        SENSOR_REGISTRY[node_id] = {
            "type": sensor.get("type", "unknown"),
            "subtype": sensor.get("subtype"),
            "lat": float(sensor["lat"]),
            "lon": float(sensor["lon"]),
            "alt": float(sensor.get("alt", 0.0)),
            "range_m": float(sensor.get("range_m", 10000.0)),
            "update_rate_sec": float(sensor.get("update_rate_sec", 10.0)),
            "capabilities": sensor.get("capabilities", []),
            "status": sensor.get("status", "operational"),
            # Preserve any additional fields
            **{
                k: v
                for k, v in sensor.items()
                if k
                not in {
                    "id",
                    "type",
                    "subtype",
                    "lat",
                    "lon",
                    "alt",
                    "range_m",
                    "update_rate_sec",
                    "capabilities",
                    "status",
                }
            },
        }

    # Determine primary anchor node
    if not primary_anchor_node:
        # Look for explicitly marked primary anchor
        primary_anchor_node = next(
            (sid for sid, s in SENSOR_REGISTRY.items() if s.get("primary_anchor")), None
        )

    if primary_anchor_node and primary_anchor_node in SENSOR_REGISTRY:
        anchor_id = primary_anchor_node
    else:
        # Fallback to first registered sensor
        anchor_id = next(iter(SENSOR_REGISTRY.keys()))
        if primary_anchor_node:
            logger.warning(
                f"Requested anchor node '{primary_anchor_node}' not found. "
                f"Using fallback: {anchor_id}"
            )

    anchor = SENSOR_REGISTRY[anchor_id]

    # Set global ENU origin
    ENU_ORIGIN_NODE_ID = anchor_id
    ENU_ORIGIN_LAT = anchor["lat"]
    ENU_ORIGIN_LON = anchor["lon"]
    ENU_ORIGIN_ALT = anchor["alt"]

    logger.info("=" * 70)
    logger.info("✅ BLUE TEAM SENSOR FUSION REGISTRY INITIALIZED")
    logger.info(f"   Anchor Node          : {ENU_ORIGIN_NODE_ID}")
    logger.info(
        f"   Origin (lat, lon, alt): ({ENU_ORIGIN_LAT:.6f}, {ENU_ORIGIN_LON:.6f}, {ENU_ORIGIN_ALT:.1f}m)"
    )
    logger.info(f"   Registered Sensors   : {len(SENSOR_REGISTRY)}")
    logger.info("=" * 70)


def get_sensor(node_id: str) -> dict[str, Any] | None:
    """Retrieve metadata for a specific sensor by ID."""
    return SENSOR_REGISTRY.get(node_id)


# Registration TrackingType values under which a sensor keeps one object_id for one
# object from detection to detection: TRACKLET persists it between detections,
# TRACK also across broken tracks, TRACK_WITH_RE_ID by re-identifying the object.
# NONE, or no declaration at all, mints an id per detection.
STABLE_OBJECT_ID_TRACKING_TYPES = frozenset(
    {"TRACKING_TYPE_TRACKLET", "TRACKING_TYPE_TRACK", "TRACKING_TYPE_TRACK_WITH_RE_ID"}
)


def has_stable_object_ids(sensor: dict[str, Any] | None) -> bool:
    """Whether a sensor's tracking_type says its object_id names the same object over time."""
    return bool(sensor) and sensor.get("tracking_type") in STABLE_OBJECT_ID_TRACKING_TYPES


# What sensors declared about themselves in SAPIENT, by node id: from their
# Registration (tracking_type, geometric_error, range_m, node_type) and from their
# StatusReport's node_location (lat, lon, alt). Kept apart from the operator's
# manifest, which is optional, so that a sensor which is not in it gains only what
# it declared and no made-up position.
REGISTRATION_CAPABILITIES: dict[str, dict[str, Any]] = {}


def apply_registration(node_id: str, capabilities: dict[str, Any]) -> None:
    """Records what a sensor declared about itself; what it declares later wins."""
    if not node_id or not capabilities:
        return
    previous = REGISTRATION_CAPABILITIES.get(node_id)
    merged = {**(previous or {}), **capabilities}
    if merged != previous:
        REGISTRATION_CAPABILITIES[node_id] = merged
        logger.info("Sensor %s registered: %s", node_id, ", ".join(sorted(capabilities)))


def sensor_profile(node_id: str) -> dict[str, Any] | None:
    """A sensor's manifest entry with what its Registration declared on top.

    The Registration wins where both say something: it is the sensor speaking for
    itself, and the manifest is the operator's earlier guess. None for a sensor
    that is in neither.
    """
    entry = SENSOR_REGISTRY.get(node_id)
    declared = REGISTRATION_CAPABILITIES.get(node_id)
    if entry is None and declared is None:
        return None
    return {**(entry or {}), **(declared or {})}


def list_sensors() -> list[str]:
    """Return list of all registered sensor IDs."""
    return list(SENSOR_REGISTRY.keys())


def get_all_sensors() -> dict[str, dict[str, Any]]:
    """Return a copy of the full sensor registry."""
    return SENSOR_REGISTRY.copy()


def reset_registry() -> None:
    """Clear registry and origin (mainly for testing)."""
    global ENU_ORIGIN_LAT, ENU_ORIGIN_LON, ENU_ORIGIN_ALT, ENU_ORIGIN_NODE_ID
    SENSOR_REGISTRY.clear()
    REGISTRATION_CAPABILITIES.clear()
    ENU_ORIGIN_LAT = ENU_ORIGIN_LON = ENU_ORIGIN_ALT = None
    ENU_ORIGIN_NODE_ID = None


# Legacy compatibility
def load_blue_sensor_network(
    sensor_network_list: list[dict[str, Any]], primary_anchor_node: str = "FI-MIL-RAD-KOLI-01"
) -> None:
    """Deprecated. Use load_sensor_network() instead."""
    logger.warning(
        "load_blue_sensor_network() is deprecated. "
        "Use load_sensor_network() for better flexibility."
    )
    load_sensor_network(
        sensor_network_list=sensor_network_list, primary_anchor_node=primary_anchor_node
    )


# Internal state variables for the dynamic origin
_origin_lat = None
_origin_lon = None
_origin_alt = None


def set_reference_origin(lat: float, lon: float, alt: float) -> None:
    """
    Explicitly set the reference origin.
    Call this if your data contains a sensor Registration message with its exact location.
    """
    global _origin_lat, _origin_lon, _origin_alt
    _origin_lat = lat
    _origin_lon = lon
    _origin_alt = alt


def wgs84_to_enu(lat: float, lon: float, alt: float) -> tuple[float, float, float]:
    """
    Converts WGS84 coordinates to local East, North, Up vectors.
    Auto-initializes the origin to the first received coordinate if not explicitly set.
    """
    global _origin_lat, _origin_lon, _origin_alt

    # Auto-initialize origin from the very first data point if it is currently empty
    if _origin_lat is None:
        _origin_lat = lat
        _origin_lon = lon
        _origin_alt = alt

    e, n, u = pm.geodetic2enu(lat, lon, alt, _origin_lat, _origin_lon, _origin_alt)
    return e, n, u


def enu_to_wgs84(e: float, n: float, u: float) -> tuple[float, float, float]:
    """
    Converts local East, North, Up vectors back to WGS84 global coordinates.
    """
    global _origin_lat, _origin_lon, _origin_alt

    # If the dynamic origin isn't set, take it from the initialized registry
    if _origin_lat is None and ENU_ORIGIN_LAT is not None:
        set_reference_origin(ENU_ORIGIN_LAT, ENU_ORIGIN_LON, ENU_ORIGIN_ALT)

    if _origin_lat is None:
        raise ValueError("Reference origin was never set. Cannot convert ENU back to WGS84.")

    lat, lon, alt = pm.enu2geodetic(e, n, u, _origin_lat, _origin_lon, _origin_alt)
    return lat, lon, alt


def enu_to_wgs84_about(
    e: float,
    n: float,
    u: float,
    origin_lat: float,
    origin_lon: float,
    origin_alt: float,
) -> tuple[float, float, float]:
    """
    Converts local East, North, Up vectors to WGS84 about a caller-supplied origin.

    Sensor-relative measurements (range/bearing) must be resolved against the sensor
    that made them, not the network anchor. This is stateless on purpose: the shared
    _origin_* triple is read by every source thread and by the tracker, so temporarily
    repointing it would corrupt unrelated conversions.
    """
    lat, lon, alt = pm.enu2geodetic(e, n, u, origin_lat, origin_lon, origin_alt)
    return lat, lon, alt
