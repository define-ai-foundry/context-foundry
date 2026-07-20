# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# src/context_foundry/fusion/models.py

import math

import numpy as np
from stonesoup.models.measurement.base import MeasurementModel
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantAcceleration,
)
from stonesoup.types.array import StateVector

from . import config

# --- WGS84 ELLIPSOID GEODETIC CONSTANTS ---
WGS84_A = 6378137.0           # Semi-major axis (meters)
WGS84_F = 1.0 / 298.257223563 # Flattening factor
WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2 # Square of first eccentricity

def wgs84_to_ecef(lat, lon, alt):
    """Converts WGS84 Geodetic coordinates to Earth-Centered, Earth-Fixed (ECEF) Cartesian frame."""
    rad_lat = math.radians(lat)
    rad_lon = math.radians(lon)

    sin_lat = math.sin(rad_lat)
    cos_lat = math.cos(rad_lat)
    sin_lon = math.sin(rad_lon)
    cos_lon = math.cos(rad_lon)

    prime_vertical = WGS84_A / math.sqrt(1.0 - WGS84_E2 * (sin_lat ** 2))

    x = (prime_vertical + alt) * cos_lat * cos_lon
    y = (prime_vertical + alt) * cos_lat * sin_lon
    z = (prime_vertical * (1.0 - WGS84_E2) + alt) * sin_lat
    return x, y, z

def ecef_to_wgs84(x, y, z):
    """Converts ECEF Cartesian coordinates back to WGS84 Geodetic bounds."""
    # Bowring's closed-form geodetic inversion algorithm
    p = math.sqrt(x**2 + y**2)
    if p < 1e-6: # Handle pole case safely
        lat = 90.0 if z > 0 else -90.0
        return lat, 0.0, abs(z) - WGS84_A

    theta = math.atan2(z * WGS84_A, p * (WGS84_A * (1.0 - WGS84_F)))

    lat = math.atan2(
        z + WGS84_E2 * (WGS84_A / (1.0 - WGS84_F)) * (math.sin(theta)**3),
        p - WGS84_E2 * WGS84_A * (math.cos(theta)**3)
    )

    lon = math.atan2(y, x)
    prime_vertical = WGS84_A / math.sqrt(1.0 - WGS84_E2 * (math.sin(lat)**2))
    alt = p / math.cos(lat) - prime_vertical

    return math.degrees(lat), math.degrees(lon), alt

def ecef_to_enu(x, y, z, lat0, lon0, alt0):
    """Transforms ECEF coordinates into a local East-North-Up (ENU) Cartesian vector relative to an origin."""
    x0, y0, z0 = wgs84_to_ecef(lat0, lon0, alt0)

    dx = x - x0
    dy = y - y0
    dz = z - z0

    rad_lat0 = math.radians(lat0)
    rad_lon0 = math.radians(lon0)

    sin_lat0 = math.sin(rad_lat0)
    cos_lat0 = math.cos(rad_lat0)
    sin_lon0 = math.sin(rad_lon0)
    cos_lon0 = math.cos(rad_lon0)

    east = -sin_lon0 * dx + cos_lon0 * dy
    north = -sin_lat0 * cos_lon0 * dx - sin_lat0 * sin_lon0 * dy + cos_lat0 * dz
    up = cos_lat0 * cos_lon0 * dx + cos_lat0 * sin_lon0 * dy + sin_lat0 * dz
    return east, north, up

def enu_to_ecef(east, north, up, lat0, lon0, alt0):
    """Transforms local ENU coordinates back into ECEF Cartesian coordinates."""
    x0, y0, z0 = wgs84_to_ecef(lat0, lon0, alt0)

    rad_lat0 = math.radians(lat0)
    rad_lon0 = math.radians(lon0)

    s_lat, c_lat = math.sin(rad_lat0), math.cos(rad_lat0)
    s_lon, c_lon = math.sin(rad_lon0), math.cos(rad_lon0)

    dx = -s_lon * east - s_lat * c_lon * north + c_lat * c_lon * up
    dy =  c_lon * east - s_lat * s_lon * north + c_lat * s_lon * up
    dz =  c_lat * north + s_lat * up

    return x0 + dx, y0 + dy, z0 + dz

def project_geodetic_to_local_enu(lat, lon, alt):
    """
    Projects raw WGS84 coordinates into localized metric ENU coordinates
    using the globally registered theater reference anchor.
    """
    if config.ENU_ORIGIN_LAT is None or config.ENU_ORIGIN_LON is None:
        raise ValueError("CRITICAL: Spatial transformation executed before registering global ENU reference origin.")

    x, y, z = wgs84_to_ecef(lat, lon, alt)
    return ecef_to_enu(x, y, z, config.ENU_ORIGIN_LAT, config.ENU_ORIGIN_LON, config.ENU_ORIGIN_ALT)

# --- HIGH-FIDELITY STONE SOUP KINEMATICS MATRIX FACTORY ---

def create_9d_constant_acceleration_model(q_process_noise=0.1):
    """
    Constructs a unified 3D Constant Acceleration transition model (9D State Vector).
    State space: [x, vx, ax, y, vy, ay, z, vz, az]^T
    """
    return CombinedLinearGaussianTransitionModel([
        ConstantAcceleration(q_process_noise),  # X axis tracking
        ConstantAcceleration(q_process_noise),  # Y axis tracking
        ConstantAcceleration(q_process_noise)   # Z axis tracking
    ])

# --- NON-LINEAR ASYNCHRONOUS ACOUSTIC MEASUREMENT ENGINE ---

class AcousticAzimuthMeasurementModel(MeasurementModel):
    """
    Custom Non-Linear Measurement Model for 1D Passive Acoustic Ring Arrays.
    Maps a 9D ENU target state vector down to a 1D bearing angle (radians).
    """
    @property
    def ndim_meas(self):
        return 1

    def function(self, state, noise=False, **kwargs):
        """
        Maps target state vector to expected azimuth tracking line from the sensor's origin point.
        Mathematical expression evaluated via the Unscented Transform:
            theta = arctan2(E_target - E_sensor, N_target - N_sensor)
        """
        # Extract local projected Cartesian positions from our 9D state layout
        target_east = state.state_vector[0, 0]
        target_north = state.state_vector[3, 0]

        # Pull sensor's geodetic base position from tracking frame metadata
        sensor_geo = kwargs.get('sensor_geodetic')
        if not sensor_geo:
            raise ValueError("Acoustic model requires 'sensor_geodetic' bounds passed via tracking context.")

        # Dynamically project the acoustic array station into the shared local metric canvas
        s_east, s_north, _ = project_geodetic_to_local_enu(
            sensor_geo["latitude"],
            sensor_geo["longitude"],
            sensor_geo.get("altitude", 0.0)
        )

        delta_east = target_east - s_east
        delta_north = target_north - s_north

        # Calculate azimuth angle measured clockwise from true north (radians)
        azimuth = math.atan2(delta_east, delta_north) % (2 * math.pi)

        if noise:
            raw_noise = self.noise_covar @ np.random.randn(self.ndim_meas, 1)
            azimuth = (azimuth + raw_noise[0, 0]) % (2 * math.pi)

        return StateVector([azimuth])
