# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# src/context_foundry/fusion/measurement.py

import math

import numpy as np
from stonesoup.models.measurement.linear import LinearGaussian

# Position noise every detection gets from a sensor whose manifest entry carries no
# geometric_error: 5 m horizontally and 10 m vertically, 1-sigma.
DEFAULT_NOISE_COVAR = np.diag([25.0, 25.0, 100.0])
DEFAULT_VERTICAL_SIGMA_M = 10.0

# Floor on any sigma a geometric_error yields, so a detection at the sensor's own
# position never claims a zero-variance, and therefore singular, measurement.
MIN_SIGMA_M = 0.5

# The state components a 3D position measurement observes in the 9D track state.
POSITION_MAPPING = (0, 3, 6)

DEFAULT_MEASUREMENT_MODEL = LinearGaussian(
    ndim_state=9, mapping=POSITION_MAPPING, noise_covar=DEFAULT_NOISE_COVAR
)

# The state components a report without a height observes: east and north only.
HORIZONTAL_MAPPING = (0, 3)


def noise_covariance(geometric_error, sensor_enu, detection_enu, range_m):
    """3x3 ENU position covariance of one detection, from its sensor's geometric_error.

    Mirrors the Registration message's DetectionDefinition.geometric_error: a
    standard deviation in metres and how it varies across the sensor's coverage.

    - `constant`: `base_m` everywhere.
    - `linear_with_range` / `quadratic_with_range`: `base_m` at the sensor, growing
      to `at_max_range_m` at `range_m`, linearly or with the square of the fraction.
    - `bearing`: a direction-finding sensor, accurate in range to `range_sigma_m`
      but only to `bearing_deg` across it, so the error ellipse is long across the
      line of sight and grows with distance.

    Height is `vertical_m` in every case.
    """
    d_east = detection_enu[0] - sensor_enu[0]
    d_north = detection_enu[1] - sensor_enu[1]
    distance = math.hypot(d_east, d_north)
    vertical = float(geometric_error.get("vertical_m", DEFAULT_VERTICAL_SIGMA_M))
    variation = geometric_error["variation_type"]

    if variation == "bearing":
        along = max(float(geometric_error.get("range_sigma_m", MIN_SIGMA_M)), MIN_SIGMA_M)
        across = max(distance * math.tan(math.radians(geometric_error["bearing_deg"])), MIN_SIGMA_M)
        # Unit vectors along and across the sensor's line of sight to the detection.
        # At the sensor itself the line of sight is undefined; any direction will do.
        u = np.array([d_east, d_north]) / distance if distance > 0 else np.array([0.0, 1.0])
        v = np.array([u[1], -u[0]])
        horizontal = along**2 * np.outer(u, u) + across**2 * np.outer(v, v)
    else:
        base = float(geometric_error["base_m"])
        if variation == "constant":
            sigma = base
        else:
            fraction = min(distance / range_m, 1.0) if range_m else 0.0
            if variation == "quadratic_with_range":
                fraction **= 2
            elif variation != "linear_with_range":
                raise ValueError(f"unknown geometric_error variation_type {variation!r}")
            sigma = base + (float(geometric_error["at_max_range_m"]) - base) * fraction
        horizontal = max(sigma, MIN_SIGMA_M) ** 2 * np.eye(2)

    covar = np.zeros((3, 3))
    covar[:2, :2] = horizontal
    covar[2, 2] = max(vertical, MIN_SIGMA_M) ** 2
    return covar


# What a sensor that declares no geometric_error is assumed to be, by the kind of
# sensor its manifest type names. Deliberately cautious, and so constant at the far
# end of what that kind of sensor does: assuming a sensor is better than it is
# splits one object into several tracks -- what the old shared 5 m did to every
# radar, and what a default that shrinks near the sensor does to a camera whose
# real error is larger there -- while assuming it a little worse only widens its
# gates. Matched on lower-case substrings of the type, the first match winning.
TYPE_DEFAULT_GEOMETRIC_ERRORS = (
    ("doppler", {"variation_type": "constant", "base_m": 5.0}),
    ("radar", {"variation_type": "constant", "base_m": 35.0}),
    (
        "acoustic",
        {
            "variation_type": "bearing",
            "bearing_deg": 5.0,
            "range_sigma_m": 10.0,
            "vertical_m": 1000.0,
        },
    ),
    ("cam", {"variation_type": "constant", "base_m": 17.0}),
    ("optical", {"variation_type": "constant", "base_m": 17.0}),
)
# For a sensor of an unknown kind, or one missing from the manifest altogether.
UNKNOWN_SENSOR_GEOMETRIC_ERROR = {"variation_type": "constant", "base_m": 50.0, "vertical_m": 50.0}


def default_geometric_error(sensor_meta):
    """The geometric_error assumed for a sensor that declares none."""
    kind = str((sensor_meta or {}).get("type", "")).lower()
    for keyword, geometric_error in TYPE_DEFAULT_GEOMETRIC_ERRORS:
        if keyword in kind:
            return geometric_error
    return UNKNOWN_SENSOR_GEOMETRIC_ERROR


def measurement_model_for(sensor_meta, sensor_enu, detection_enu):
    """The position measurement model of one detection from the sensor that reported it.

    The sensor's declared geometric_error when it has one; otherwise a cautious
    default for its kind of sensor (default_geometric_error). A sensor missing from
    the manifest has no position to measure range from, so it gets the unknown-kind
    default, which does not depend on range.
    """
    if sensor_meta is None:
        geometric_error = UNKNOWN_SENSOR_GEOMETRIC_ERROR
    else:
        geometric_error = sensor_meta.get("geometric_error") or default_geometric_error(sensor_meta)
    if sensor_enu is None:
        # Without the sensor's position nothing can depend on range or bearing.
        sensor_enu = detection_enu
    covar = noise_covariance(
        geometric_error, sensor_enu, detection_enu, float((sensor_meta or {}).get("range_m", 0.0))
    )
    return LinearGaussian(ndim_state=9, mapping=POSITION_MAPPING, noise_covar=covar)


def position_measurement(sensor_meta, sensor_enu, detection_enu, has_altitude):
    """(state vector, measurement model) of one detection's position.

    A detection that carries no height is measured in east and north only, with
    its sensor's horizontal noise. Placing it at some made-up height instead --
    the ground, say -- puts it hundreds of metres from where the object flies, so
    it cannot associate with the track the other sensors keep, and a track it
    starts would sit at that made-up height.
    """
    model = measurement_model_for(sensor_meta, sensor_enu, detection_enu)
    e, n, u = detection_enu
    if has_altitude:
        return np.array([[e], [n], [u]]), model
    horizontal = LinearGaussian(
        ndim_state=9, mapping=HORIZONTAL_MAPPING, noise_covar=model.noise_covar[:2, :2]
    )
    return np.array([[e], [n]]), horizontal
