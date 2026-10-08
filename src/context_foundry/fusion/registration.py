# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# src/context_foundry/fusion/registration.py

"""What a sensor says about itself in its SAPIENT Registration, in the terms fusion uses.

A sensor that registers can tell fusion two things that decide how its reports are
associated: how far off its positions may be (DetectionDefinition.geometric_error)
and whether its object_id keeps naming the same object (ModeDefinition.tracking_type).
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


def sensor_capabilities(registration_message: dict) -> dict:
    """{"tracking_type": ..., "geometric_error": ...} from a Registration message, as found.

    The first mode that declares a tracking type gives it, and the first detection
    definition with a readable geometric_error gives that; either key is left out
    when nothing declares it.
    """
    registration = (registration_message or {}).get("registration") or {}
    found = {}
    for mode in registration.get("mode_definition", []):
        if "tracking_type" not in found and mode.get("tracking_type"):
            found["tracking_type"] = mode["tracking_type"]
        if "geometric_error" not in found:
            for definition in mode.get("detection_definition", []):
                geometric_error = geometric_error_from(definition)
                if geometric_error:
                    found["geometric_error"] = geometric_error
                    break
    return found
