# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

import math
import re
from datetime import timedelta

from . import config
from .schemas import TacticalTrack

# WGS84 Ellipsoid Constants
WGS84_A = 6378137.0
WGS84_E2 = 0.00669437999014

# How long a track judged hostile stays hostile: a track flying near the speed limit, or
# one whose swarm count briefly reads small, would otherwise flip its affiliation between
# hostile and suspect from one update to the next.
HOSTILE_HOLD_SECONDS = 60.0

# The affiliation letter of a CoT atom type, `a-<letter>-...`.
COT_AFFILIATION = re.compile(r"^a-([a-z])-")

# A CoT sender's declared affiliation as a threat level. Unknown, pending, assumed
# none and exercise-unknown are not a declaration and leave the threat to the rule.
DECLARED_THREAT = {
    "f": "friend",
    "a": "friend",
    "h": "hostile",
    "j": "hostile",
    "k": "hostile",
    "s": "suspect",
    "n": "neutral",
}


def declared_threat(cot_type) -> str | None:
    """The threat level a CoT type's affiliation letter declares, or None when it declares none."""
    match = COT_AFFILIATION.match(cot_type) if isinstance(cot_type, str) else None
    return DECLARED_THREAT.get(match.group(1)) if match else None


class TacticalContextAugmentor:
    """
    Acts as the format-independent bridge between the mathematical tracking
    engine (Stone Soup) and the downstream serializers.
    """

    def __init__(self):
        # Track id -> until when it stays hostile, from the last time it was judged so.
        self._hostile_until = {}

    def _held_threat(self, track_id, timestamp, hostile_now: bool) -> str:
        """The more severe of now's judgement and a hostile one within HOSTILE_HOLD_SECONDS."""
        if hostile_now:
            self._hostile_until[track_id] = timestamp + timedelta(seconds=HOSTILE_HOLD_SECONDS)
            # Bounded by the tracks still published: one expired long ago is forgotten.
            if len(self._hostile_until) > 1000:
                self._hostile_until = {
                    k: until for k, until in self._hostile_until.items() if until >= timestamp
                }
            return "hostile"
        until = self._hostile_until.get(track_id)
        return "hostile" if until is not None and timestamp <= until else "suspect"

    def ecef_to_wgs84(self, x, y, z):
        """Converts ECEF Cartesian coordinates back to WGS84 Geodetic bounds."""
        p = math.sqrt(x**2 + y**2)
        if p < 1e-6:  # Handle pole case safely
            return (90.0 if z > 0 else -90.0), 0.0, abs(z) - WGS84_A

        theta = math.atan2(z * WGS84_A, p * (WGS84_A * (1.0 - 0.0033528106647474805)))

        lat = math.atan2(
            z + WGS84_E2 * (WGS84_A / (1.0 - 0.0033528106647474805)) * (math.sin(theta) ** 3),
            p - WGS84_E2 * WGS84_A * (math.cos(theta) ** 3),
        )
        lon = math.atan2(y, x)

        prime_vertical = WGS84_A / math.sqrt(1.0 - WGS84_E2 * (math.sin(lat) ** 2))
        alt = p / math.cos(lat) - prime_vertical

        return math.degrees(lat), math.degrees(lon), alt

    def extract_tactical_track(self, track) -> TacticalTrack:
        """
        Extracts position, absolute speed, and true heading from the 9D state
        vector and returns a standardized TacticalTrack data model.
        """
        state = track.state
        vec = state.state_vector

        # Extract 9D elements: [East, vEast, aEast, North, vNorth, aNorth, Up, vUp, aUp]
        e, ve, n, vn, u, vu = vec[0, 0], vec[1, 0], vec[3, 0], vec[4, 0], vec[6, 0], vec[7, 0]

        # Calculate WGS84 Position
        lat, lon, alt = config.enu_to_wgs84(e, n, u)

        # Calculate Kinematics
        speed_m_s = math.sqrt(ve**2 + vn**2 + vu**2)
        # Note: atan2(x, y) vs atan2(y, x) maps math angles to compass navigation angles
        heading_deg = (math.degrees(math.atan2(ve, vn))) % 360.0

        # Recover the source context the tracker carried through. Stone Soup
        # accumulates detection metadata on the Track, not on its states, so
        # reading the state yields nothing and every track looks unclassified.
        metadata = getattr(track, "metadata", None) or {}
        # SAPIENT sources write the literal "Unknown" when the sensor gave no
        # classification, so treat it as absent: a CoT hit's 2525 type code is
        # the only label such a track has.
        classification = metadata.get("classification")
        class_confidences = metadata.get("class_confidences") or []
        if not classification or classification == "Unknown":
            classification = metadata.get("cot_type") or "Unknown"
            if classification != "Unknown":
                class_confidences = []
        swarm_count = metadata.get("swarm_count", 1)

        # A CoT sender's declared affiliation wins over the rule and ends any hostile hold.
        threat_level = declared_threat(metadata.get("cot_type"))
        if threat_level is not None:
            self._hostile_until.pop(track.id, None)
        else:
            is_fast = speed_m_s > 20.0
            is_swarm = swarm_count > 3
            threat_level = self._held_threat(track.id, state.timestamp, is_fast or is_swarm)

        # Return the universal state object for the serializers to consume
        return TacticalTrack(
            track_id=track.id,
            timestamp=state.timestamp,
            latitude=lat,
            longitude=lon,
            altitude=alt,
            speed_mps=speed_m_s,
            heading_deg=heading_deg,
            classification=classification,
            detection_confidence=metadata.get("detection_confidence"),
            class_confidences=class_confidences,
            swarm_count=swarm_count,
            threat_level=threat_level,
            velocity=[float(ve), float(vn), float(vu)],
        )
