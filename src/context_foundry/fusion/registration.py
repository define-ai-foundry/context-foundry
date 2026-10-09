# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# src/context_foundry/fusion/registration.py

"""What a sensor says about itself in SAPIENT, in the terms fusion uses.

A sensor that registers can tell fusion two things that decide how its reports are
associated: how far off its positions may be (DetectionDefinition.geometric_error)
and whether its object_id keeps naming the same object (ModeDefinition.tracking_type);
and two more that refine the first: what kind of sensor it is (NodeType), and how far
it sees (detection_performance). Where it stands comes from its StatusReport's
node_location, so fusion needs no sensor file.
Both are optional in BSI Flex 335, and geometric_error is loosely typed -- its
variation is free text and its performance values are named by the sensor -- so
this reads the common spellings and leaves out what it cannot read. What is left
out falls back to the operator's sensor config, then to a cautious default for the
kind of sensor (see measurement.default_geometric_error).

The message is taken in the proto3 JSON mapping with the proto field names, as
sapient-raw carries it.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Free-text variation types, normalised to letters only, and what they mean here.
_VARIATIONS = {
    "constant": "constant",
    "fixed": "constant",
    "linearwithrange": "linear_with_range",
    "linear": "linear_with_range",
    "quadraticwithrange": "quadratic_with_range",
    "quadratic": "quadratic_with_range",
    "bearing": "bearing",
}

# Performance value names, normalised to letters only, and the geometric_error key
# each one fills.
_VALUE_KEYS = {
    "base": "base_m",
    "atsensor": "base_m",
    "minrange": "base_m",
    "minimum": "base_m",
    "value": "base_m",
    "standarddeviation": "base_m",
    "atmaxrange": "at_max_range_m",
    "maxrange": "at_max_range_m",
    "maximum": "at_max_range_m",
    "vertical": "vertical_m",
    "height": "vertical_m",
    "bearing": "bearing_deg",
    "azimuth": "bearing_deg",
    "range": "range_sigma_m",
}


def _letters(text) -> str:
    return re.sub(r"[^a-z]", "", str(text or "").lower())


def geometric_error_from(definition: dict) -> dict | None:
    """A DetectionDefinition.geometric_error as fusion's geometric_error, or None."""
    error = (definition or {}).get("geometric_error")
    if not error:
        return None
    values = {}
    for value in error.get("performance_value", []):
        key = _VALUE_KEYS.get(_letters(value.get("type")))
        try:
            number = float(value.get("unit_value"))
        except (TypeError, ValueError):
            continue
        if key and key not in values:
            values[key] = number
    stated = _letters(error.get("variation_type"))
    # No variation stated with a single value is a constant error; a variation this
    # does not know is not guessed at.
    variation = _VARIATIONS.get(stated) if stated else ("constant" if "base_m" in values else None)
    needs = {
        "constant": ("base_m",),
        "linear_with_range": ("base_m", "at_max_range_m"),
        "quadratic_with_range": ("base_m", "at_max_range_m"),
        "bearing": ("bearing_deg",),
        None: ("never",),
    }[variation]
    if not all(key in values for key in needs):
        return None
    return {"variation_type": variation, **values}


# detection_performance value names, normalised to letters only, that state how far
# the sensor sees, and the units they may be given in, in metres.
_RANGE_NAMES = frozenset({"range", "maxrange", "maximumrange", "detectionrange"})
_RANGE_UNITS = {"m": 1.0, "metre": 1.0, "metres": 1.0, "meter": 1.0, "meters": 1.0}
_RANGE_UNITS |= {"km": 1000.0, "kilometre": 1000.0, "kilometres": 1000.0}
_RANGE_UNITS |= {"kilometer": 1000.0, "kilometers": 1000.0}


def range_from(definition: dict) -> float | None:
    """How far a DetectionDefinition says its sensor sees, in metres, or None."""
    for value in (definition or {}).get("detection_performance", []):
        if _letters(value.get("type")) not in _RANGE_NAMES:
            continue
        scale = _RANGE_UNITS.get(_letters(value.get("units")))
        try:
            number = float(value.get("unit_value"))
        except (TypeError, ValueError):
            continue
        if scale and number > 0:
            return number * scale
    return None


def node_kind(registration: dict) -> str | None:
    """The first declared NodeType as a lower-case word ("radar", "camera"), or None."""
    for definition in registration.get("node_definition", []):
        node_type = str(definition.get("node_type") or "")
        if node_type and node_type not in ("NODE_TYPE_UNSPECIFIED", "NODE_TYPE_OTHER"):
            return node_type.removeprefix("NODE_TYPE_").lower()
    return None


def sensor_capabilities(registration_message: dict) -> dict:
    """What a Registration message declares about its sensor, as found.

    {"tracking_type", "geometric_error", "range_m", "node_type"}: the first mode that
    declares a tracking type gives it, the first detection definition with a readable
    geometric_error gives that, the first with a readable range gives range_m, and
    the first node definition with a type gives node_type. A key is left out when
    nothing declares it.
    """
    registration = (registration_message or {}).get("registration") or {}
    found = {}
    kind = node_kind(registration)
    if kind:
        found["node_type"] = kind
    for mode in registration.get("mode_definition", []):
        if "tracking_type" not in found and mode.get("tracking_type"):
            found["tracking_type"] = mode["tracking_type"]
        for definition in mode.get("detection_definition", []):
            if "geometric_error" not in found:
                geometric_error = geometric_error_from(definition)
                if geometric_error:
                    found["geometric_error"] = geometric_error
            if "range_m" not in found:
                range_m = range_from(definition)
                if range_m:
                    found["range_m"] = range_m
    return found


def node_location(status_message: dict) -> dict | None:
    """{"lat", "lon", "alt"} of a StatusReport's node_location in degrees, or None.

    Where a sensor is comes from its status reports, not its Registration. Only a
    latitude/longitude location is read; a UTM one is left out rather than guessed.
    """
    location = ((status_message or {}).get("status_report") or {}).get("node_location")
    if not location:
        return None
    if "LAT_LNG" not in str(location.get("coordinate_system", "")):
        return None
    try:
        lat, lon = float(location["y"]), float(location["x"])
        alt = float(location.get("z") or 0.0)
    except (KeyError, TypeError, ValueError):
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return {"lat": lat, "lon": lon, "alt": alt}
