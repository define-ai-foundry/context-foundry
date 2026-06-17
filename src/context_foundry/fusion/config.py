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
from typing import Any, Dict, List, Optional, Union

# Global sensor registry: node_id -> sensor metadata
SENSOR_REGISTRY: Dict[str, Dict[str, Any]] = {}

# Global ENU origin (set by load_sensor_network)
ENU_ORIGIN_LAT: Optional[float] = None
ENU_ORIGIN_LON: Optional[float] = None
ENU_ORIGIN_ALT: Optional[float] = None
ENU_ORIGIN_NODE_ID: Optional[str] = None

logger = logging.getLogger(__name__)


def load_sensor_network(
    sensor_config_path: Optional[Union[str, Path]] = None,
    sensor_network_list: Optional[List[Dict[str, Any]]] = None,
    primary_anchor_node: Optional[str] = None,
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

        with open(path, 'r', encoding='utf-8') as f:
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
            **{k: v for k, v in sensor.items()
               if k not in {"id", "type", "subtype", "lat", "lon", "alt",
                           "range_m", "update_rate_sec", "capabilities", "status"}}
        }

    # Determine primary anchor node
    if not primary_anchor_node:
        # Look for explicitly marked primary anchor
        primary_anchor_node = next(
            (sid for sid, s in SENSOR_REGISTRY.items() if s.get("primary_anchor")),
            None
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
    logger.info(f"   Origin (lat, lon, alt): ({ENU_ORIGIN_LAT:.6f}, {ENU_ORIGIN_LON:.6f}, {ENU_ORIGIN_ALT:.1f}m)")
    logger.info(f"   Registered Sensors   : {len(SENSOR_REGISTRY)}")
    logger.info("=" * 70)


def get_sensor(node_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve metadata for a specific sensor by ID."""
    return SENSOR_REGISTRY.get(node_id)


def list_sensors() -> List[str]:
    """Return list of all registered sensor IDs."""
    return list(SENSOR_REGISTRY.keys())


def get_all_sensors() -> Dict[str, Dict[str, Any]]:
    """Return a copy of the full sensor registry."""
    return SENSOR_REGISTRY.copy()


def reset_registry() -> None:
    """Clear registry and origin (mainly for testing)."""
    global ENU_ORIGIN_LAT, ENU_ORIGIN_LON, ENU_ORIGIN_ALT, ENU_ORIGIN_NODE_ID
    SENSOR_REGISTRY.clear()
    ENU_ORIGIN_LAT = ENU_ORIGIN_LON = ENU_ORIGIN_ALT = None
    ENU_ORIGIN_NODE_ID = None


# Legacy compatibility
def load_blue_sensor_network(
    sensor_network_list: List[Dict[str, Any]],
    primary_anchor_node: str = "FI-MIL-RAD-KOLI-01"
) -> None:
    """Deprecated. Use load_sensor_network() instead."""
    logger.warning(
        "load_blue_sensor_network() is deprecated. "
        "Use load_sensor_network() for better flexibility."
    )
    load_sensor_network(
        sensor_network_list=sensor_network_list,
        primary_anchor_node=primary_anchor_node
    )