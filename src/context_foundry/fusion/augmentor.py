# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

import math
from .schemas import TacticalState
from . import config

# WGS84 Ellipsoid Constants
WGS84_A = 6378137.0
WGS84_E2 = 0.00669437999014

class TacticalContextAugmentor:
    """
    Acts as the format-independent bridge between the mathematical tracking 
    engine (Stone Soup) and the downstream serializers.
    """

    def __init__(self):
        pass

    def ecef_to_wgs84(self, x, y, z):
        """Converts ECEF Cartesian coordinates back to WGS84 Geodetic bounds."""
        p = math.sqrt(x**2 + y**2)
        if p < 1e-6: # Handle pole case safely
            return (90.0 if z > 0 else -90.0), 0.0, abs(z) - WGS84_A
            
        theta = math.atan2(z * WGS84_A, p * (WGS84_A * (1.0 - 0.0033528106647474805)))
        
        lat = math.atan2(
            z + WGS84_E2 * (WGS84_A / (1.0 - 0.0033528106647474805)) * (math.sin(theta)**3),
            p - WGS84_E2 * WGS84_A * (math.cos(theta)**3)
        )
        lon = math.atan2(y, x)
        
        prime_vertical = WGS84_A / math.sqrt(1.0 - WGS84_E2 * (math.sin(lat)**2))
        alt = p / math.cos(lat) - prime_vertical
        
        return math.degrees(lat), math.degrees(lon), alt

    def enu_to_wgs84(self, east, north, up):
        """Projects local ENU metric tracking coordinates back to global WGS84."""
        # 1. ENU to ECEF (using the configured theater reference center)
        lat0 = config.ENU_ORIGIN_LAT
        lon0 = config.ENU_ORIGIN_LON
        alt0 = config.ENU_ORIGIN_ALT
        
        x0, y0, z0 = config.wgs84_to_ecef(lat0, lon0, alt0)
        
        rad_lat0, rad_lon0 = math.radians(lat0), math.radians(lon0)
        s_lat, c_lat = math.sin(rad_lat0), math.cos(rad_lat0)
        s_lon, c_lon = math.sin(rad_lon0), math.cos(rad_lon0)
        
        dx = -s_lon * east - s_lat * c_lon * north + c_lat * c_lon * up
        dy =  c_lon * east - s_lat * s_lon * north + c_lat * s_lon * up
        dz =  c_lat * north + s_lat * up
        
        # 2. ECEF to WGS84
        return self.ecef_to_wgs84(x0 + dx, y0 + dy, z0 + dz)

    def extract_tactical_state(self, track) -> TacticalState:
        """
        Extracts position, absolute speed, and true heading from the 9D state 
        vector and returns a standardized TacticalState data model.
        """
        state = track.latest_state
        vec = state.state_vector
        
        # Extract 9D elements: [East, vEast, aEast, North, vNorth, aNorth, Up, vUp, aUp]
        e, ve, n, vn, u, vu = vec[0,0], vec[1,0], vec[3,0], vec[4,0], vec[6,0], vec[7,0]
        
        # Calculate WGS84 Position
        lat, lon, alt = self.enu_to_wgs84(e, n, u)
        
        # Calculate Kinematics
        speed_m_s = math.sqrt(ve**2 + vn**2 + vu**2)
        # Note: atan2(x, y) vs atan2(y, x) maps math angles to compass navigation angles
        heading_deg = (math.degrees(math.atan2(ve, vn))) % 360.0 
        
        # Recover context passed through the tracker from the JSON source
        metadata = getattr(state, 'metadata', {})
        classification = metadata.get("classification", "Unknown")
        swarm_count = metadata.get("swarm_count", 1)
        
        # Core Rules Engine: Threat Assessment Logic
        is_fast = speed_m_s > 20.0
        is_swarm = swarm_count > 3
        threat_level = "hostile" if (is_fast or is_swarm) else "suspect"
        
        # Return the universal state object for the serializers to consume
        return TacticalState(
            track_id=track.id,
            timestamp=state.timestamp,
            lat=lat,
            lon=lon,
            alt=alt,
            speed_m_s=speed_m_s,
            heading_deg=heading_deg,
            classification=classification,
            swarm_count=swarm_count,
            threat_level=threat_level
        )