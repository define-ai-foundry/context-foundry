# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

import math
import re
from datetime import timedelta

from . import config
from .schemas import TacticalTrack
from .tracker import DECLARATION_FRESH_SECONDS, DECLARATIONS_KEY, SENSOR_HIT_AT_KEY

# WGS84 Ellipsoid Constants
WGS84_A = 6378137.0
WGS84_E2 = 0.00669437999014

# How long a track judged hostile stays hostile: a track flying near the speed limit, or
# one whose swarm count briefly reads small, would otherwise flip its affiliation between
# hostile and suspect from one update to the next.
HOSTILE_HOLD_SECONDS = 60.0

# How recent a sensor's (non-CoT) hit must be for a swarm to overrule a declaration.
SENSOR_HIT_RECENT_SECONDS = 60.0

# Which of several fresh declarations on one track stands: the most severe.
DECLARED_SEVERITY = {"hostile": 0, "suspect": 1, "neutral": 2, "friend": 3}

# The affiliation letter of a CoT atom type, `a-<letter>-...`.
COT_AFFILIATION = re.compile(r"^a-([a-z])-")

# A CoT sender's declared affiliation as a threat level. Unknown, pending, assumed
# none and exercise-unknown are not a declaration and leave the threat to the rule.
DECLARED_THREAT = {
    "f": "friend",
    "a": "friend",
    "h": "hostile",
    "j": "suspect",
    "k": "hostile",
    "s": "suspect",
    "n": "neutral",
}


def declared_threat(cot_type) -> str | None:
    """The threat level a CoT type's affiliation letter declares, or None when it declares none."""
    match = COT_AFFILIATION.match(cot_type) if isinstance(cot_type, str) else None
    return DECLARED_THREAT.get(match.group(1)) if match else None


def _within(timestamp, moment, seconds: float) -> bool:
    """Whether `moment` is no more than `seconds` before `timestamp`."""
    return (timestamp - moment).total_seconds() <= seconds


def _standing_declaration(declarations, timestamp):
    """(threat, CoT type) of the most severe declaration fresh at `timestamp`, or (None, None).

    Between equally severe ones the latest report stands, then the type, so the choice
    depends only on the reports and not on the order the uids first came in.
    """
    fresh = [
        (DECLARED_SEVERITY[threat], -(timestamp - at).total_seconds(), cot_type, threat)
        for cot_type, at in declarations or ()
        if (threat := declared_threat(cot_type)) is not None
        and _within(timestamp, at, DECLARATION_FRESH_SECONDS)
    ]
    if not fresh:
        return None, None
    *_, cot_type, threat = min(fresh)
    return threat, cot_type


class TacticalContextAugmentor:
    """
    Acts as the format-independent bridge between the mathematical tracking
    engine (Stone Soup) and the downstream serializers.
    """

    def __init__(self):
        # Track id -> until when it stays hostile, from the last time it was judged so.
        self._hostile_until = {}
        # Track id -> until when a swarm overrules a declaration, from its last swarm count.
        self._swarm_until = {}

    @staticmethod
    def _hold(holds: dict, track_id, timestamp) -> dict:
        """`holds` with the track's hold renewed for HOSTILE_HOLD_SECONDS from `timestamp`."""
        holds[track_id] = timestamp + timedelta(seconds=HOSTILE_HOLD_SECONDS)
        # Bounded by the tracks still published: one expired long ago is forgotten.
        if len(holds) > 1000:
            holds = {k: until for k, until in holds.items() if until >= timestamp}
        return holds

    @staticmethod
    def _holding(holds: dict, track_id, timestamp) -> bool:
        """Whether the track's hold in `holds` still runs at `timestamp`."""
        until = holds.get(track_id)
        return until is not None and timestamp <= until

    def _held_threat(self, track_id, timestamp, hostile_now: bool) -> str:
        """The more severe of now's judgement and a hostile one within HOSTILE_HOLD_SECONDS."""
        if hostile_now:
            self._hostile_until = self._hold(self._hostile_until, track_id, timestamp)
            return "hostile"
        return "hostile" if self._holding(self._hostile_until, track_id, timestamp) else "suspect"

    def _threat(self, track_id, timestamp, metadata, is_fast: bool, is_swarm: bool):
        """(threat level, declared CoT type or None) from the rule and the fresh declarations.

        Without a fresh declaration the rule decides: fast or a swarm is hostile, held. The
        most severe fresh declaration otherwise stands, a hostile one held like the rule's.
        A friend, neutral or suspect one is overruled only by a swarm a sensor other than
        CoT backs: a non-CoT hit within SENSOR_HIT_RECENT_SECONDS and a swarm count now or
        within its hold. Speed never overrules it. An overruled one gives no declared type.
        """
        if is_swarm:
            self._swarm_until = self._hold(self._swarm_until, track_id, timestamp)
        declared, declared_type = _standing_declaration(metadata.get(DECLARATIONS_KEY), timestamp)
        if declared is None:
            return self._held_threat(track_id, timestamp, is_fast or is_swarm), None
        if declared == "hostile":
            return self._held_threat(track_id, timestamp, True), declared_type
        sensor_at = metadata.get(SENSOR_HIT_AT_KEY)
        sensed = sensor_at is not None and _within(timestamp, sensor_at, SENSOR_HIT_RECENT_SECONDS)
        if sensed and self._holding(self._swarm_until, track_id, timestamp):
            self._held_threat(track_id, timestamp, is_swarm)
            return "hostile", None
        return declared, declared_type

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

        is_fast = speed_m_s > 20.0
        is_swarm = swarm_count > 3
        threat_level, declared_type = self._threat(
            track.id, state.timestamp, metadata, is_fast, is_swarm
        )

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
            declared_type=declared_type,
            velocity=[float(ve), float(vn), float(vu)],
        )
