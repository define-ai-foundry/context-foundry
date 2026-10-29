# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

import math
import re
from datetime import timedelta

from . import config
from .schemas import TacticalTrack
from .tracker import CLASS_KIND_KEY, DECLARATION_KEY, SENSOR_HIT_AT_KEY

# WGS84 Ellipsoid Constants
WGS84_A = 6378137.0
WGS84_E2 = 0.00669437999014

# How long a track judged hostile stays hostile: a track flying near the speed limit, or
# one whose swarm count briefly reads small, would otherwise flip its affiliation between
# hostile and suspect from one update to the next.
HOSTILE_HOLD_SECONDS = 60.0

# The fastest an animal is taken to fly. A track the sensors class as an animal that goes
# faster may be a drone they classed wrongly, and is judged by the rule like any other.
ANIMAL_MAX_SPEED_MPS = 35.0
# How many of a track's updates in a row must be faster than ANIMAL_MAX_SPEED_MPS before an
# animal is judged by the rule. A report landing far from where the track expected gives one
# update an absurd speed (a sensor's bird track on the exercise read 86 to 434 m/s for an
# update), which a bird never flies and a drone keeps up over several.
ANIMAL_FAST_UPDATES = 3

# How recent a sensor's (non-CoT) hit must be for a swarm to overrule a declaration.
SENSOR_HIT_RECENT_SECONDS = 60.0

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


def _standing_declaration(declaration, timestamp, east, north):
    """(threat, CoT type) of the track's declaration if it stands, else (None, None).

    `declaration` is the tracker's (CoT type, fresh until, east, north, radius, made at,
    velocity east, velocity north). It stands while fresh at `timestamp` and while the
    track, at (`east`, `north`), is within `radius` metres of where the report placed it
    carried on at the track's velocity then, and only if its type declares an affiliation.
    So a vehicle keeps its mark between reports, while an object that takes over a track
    whose reporter stood still, or that turns away from where it was going, does not.
    """
    if not declaration:
        return None, None
    cot_type, until, at_east, at_north, radius, made, v_east, v_north = declaration
    threat = declared_threat(cot_type)
    if threat is None or timestamp > until:
        return None, None
    elapsed = (timestamp - made).total_seconds()
    expected_east, expected_north = at_east + v_east * elapsed, at_north + v_north * elapsed
    if math.hypot(east - expected_east, north - expected_north) > radius:
        return None, None
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
        # Track ids whose hostile hold was last renewed while the track was already an
        # animal (one faster than ANIMAL_MAX_SPEED_MPS): an animal ends only a hold from
        # before it was one, or a speed jittering about that limit would flip its mark.
        self._animal_holds = set()
        # Track id -> how many of its updates in a row were of an animal faster than
        # ANIMAL_MAX_SPEED_MPS; a track with none has no entry.
        self._animal_fast_run = {}

    def _fast_run(self, track_id, speed_m_s) -> int:
        """The track's run of updates faster than ANIMAL_MAX_SPEED_MPS, this one included."""
        if speed_m_s <= ANIMAL_MAX_SPEED_MPS:
            self._animal_fast_run.pop(track_id, None)
            return 0
        run = self._animal_fast_run.get(track_id, 0) + 1
        # Bounded like the holds: a track gone mid-run is forgotten with the rest.
        if len(self._animal_fast_run) > 1000:
            self._animal_fast_run = {}
        self._animal_fast_run[track_id] = run
        return run

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

    def _held_threat(self, track_id, timestamp, hostile_now: bool, animal_kind=False) -> str:
        """The more severe of now's judgement and a hostile one within HOSTILE_HOLD_SECONDS.

        A hold renewed while the track is an animal (`animal_kind`) is marked as such.
        """
        if hostile_now:
            self._hostile_until = self._hold(self._hostile_until, track_id, timestamp)
            if animal_kind:
                self._animal_holds.add(track_id)
            else:
                self._animal_holds.discard(track_id)
            self._animal_holds &= self._hostile_until.keys()
            return "hostile"
        return "hostile" if self._holding(self._hostile_until, track_id, timestamp) else "suspect"

    def _threat(
        self,
        track_id,
        timestamp,
        metadata,
        position,
        is_fast: bool,
        is_swarm: bool,
        animal=False,
        animal_kind=False,
    ):
        """(threat level, declared CoT type or None) from the rule and the declaration.

        An `animal` (see extract_tactical_track) is neither fast nor a swarm to the rule,
        and ends the hostile and swarm holds it is under, unless they were set while the
        track was already an animal (`animal_kind`, then over ANIMAL_MAX_SPEED_MPS): those
        run their course, so a speed about that limit does not flip the mark.

        Without a standing declaration (fresh, and the track at `position`, its east and
        north, near where it was made) the rule decides: fast or a swarm is hostile, held.
        A standing declaration otherwise stands, a hostile one held like the rule's.
        A friend, neutral or suspect one is overruled only by a swarm a sensor other than
        CoT backs: a non-CoT hit within SENSOR_HIT_RECENT_SECONDS and a swarm count now or
        within its hold. Speed never overrules it. An overruled one gives no declared type.
        """
        if animal and track_id not in self._animal_holds:
            self._hostile_until.pop(track_id, None)
            self._swarm_until.pop(track_id, None)
        if is_swarm:
            self._swarm_until = self._hold(self._swarm_until, track_id, timestamp)
        declared, declared_type = _standing_declaration(
            metadata.get(DECLARATION_KEY), timestamp, *position
        )
        if declared is None:
            return self._held_threat(track_id, timestamp, is_fast or is_swarm, animal_kind), None
        if declared == "hostile":
            return self._held_threat(track_id, timestamp, True, animal_kind), declared_type
        sensor_at = metadata.get(SENSOR_HIT_AT_KEY)
        sensed = sensor_at is not None and _within(timestamp, sensor_at, SENSOR_HIT_RECENT_SECONDS)
        if sensed and self._holding(self._swarm_until, track_id, timestamp):
            self._held_threat(track_id, timestamp, is_swarm, animal_kind)
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

        # What the track's hits agree it is (the tracker's CLASS_KIND_KEY) tempers the rule:
        # an animal is neither fast nor a swarm (a flock) until ANIMAL_FAST_UPDATES of its
        # updates in a row are faster than ANIMAL_MAX_SPEED_MPS, and something on the ground
        # is not fast, only a swarm.
        kind = metadata.get(CLASS_KIND_KEY)
        animal = kind == "animal" and self._fast_run(track.id, speed_m_s) < ANIMAL_FAST_UPDATES
        is_fast = speed_m_s > 20.0 and not animal and kind != "ground"
        is_swarm = swarm_count > 3 and not animal
        threat_level, declared_type = self._threat(
            track.id,
            state.timestamp,
            metadata,
            (float(e), float(n)),
            is_fast,
            is_swarm,
            animal,
            kind == "animal",
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
