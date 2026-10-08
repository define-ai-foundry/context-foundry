# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# src/context_foundry/fusion/sapient_raw.py

"""Fused tracks as sapient-raw records.

Each fused track is published as one sapient-raw record holding a SAPIENT
DetectionReport from the fusion stage's own node: the same format sapient-ingest
writes for a sensor, so every consumer of sapient-raw can read fused tracks
without knowing they were fused. The record is built in the proto3 JSON mapping
with the proto field names, and is round-tripped through the SapientMessage
definition so it can only hold what the definition allows.
"""

import json
import os
import time
import uuid
from datetime import datetime

from google.protobuf.json_format import MessageToDict, ParseDict

from sapient_msg.bsi_flex_335_v2_0.sapient_message_pb2 import SapientMessage

from .schemas import TacticalTrack
from .timeutil import as_utc
from .validators.sapient import CLASS_PATH_SEPARATOR

SCHEMA = "sapient-raw:1.0"
CONTENT_TYPE = "detection_report"

LAT_LNG = "LOCATION_COORDINATE_SYSTEM_LAT_LNG_DEG_M"
WGS84 = "LOCATION_DATUM_WGS84_E"
_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def ulid_text(value: int) -> str:
    """A 128-bit integer as a 26-character Crockford base32 ULID."""
    return "".join(_CROCKFORD[(value >> (5 * i)) & 31] for i in range(25, -1, -1))


def report_ulid(moment: datetime) -> str:
    """A new ULID: the 48-bit millisecond time of `moment`, then 80 random bits."""
    ms = int(as_utc(moment).timestamp() * 1000) & ((1 << 48) - 1)
    return ulid_text((ms << 80) | int.from_bytes(os.urandom(10), "big"))


def object_ulid(track_id: str) -> str:
    """The track's SAPIENT object_id: a ULID derived from its id, the same on every report.

    Consumers key on object_id (sapient-to-cot uses it as the map marker's uid), so
    it has to stay put for the life of the track.
    """
    try:
        value = uuid.UUID(str(track_id)).int
    except ValueError:
        value = uuid.uuid5(uuid.NAMESPACE_URL, f"context-foundry-track:{track_id}").int
    return ulid_text(value)


def fusion_node_id(pipeline_id: str, stage: str) -> str:
    """The fusion stage's SAPIENT node_id, fixed for a pipeline's stage."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"context-foundry-fusion:{pipeline_id}:{stage}"))


def uuid7() -> str:
    """A version 7 UUID in canonical lowercase form, the platform's message_id."""
    ms = time.time_ns() // 1_000_000
    rand = int.from_bytes(os.urandom(10), "big")
    value = (ms & ((1 << 48) - 1)) << 80 | (0x7 << 76) | ((rand >> 62) & 0xFFF) << 64
    value |= (0b10 << 62) | (rand & ((1 << 62) - 1))
    return str(uuid.UUID(int=value))


def _classification(path: str | None, confidence: float | None) -> list[dict]:
    """A flattened class path ("Air vehicle > UAV rotary wing") as nested sub_class levels."""
    parts = [p for p in (path or "").split(CLASS_PATH_SEPARATOR) if p]
    if not parts:
        return []
    top = {"type": parts[0]}
    if confidence is not None:
        top["confidence"] = confidence
    node = top
    for level, part in enumerate(parts[1:], start=1):
        child = {"type": part, "level": level}
        node["sub_class"] = [child]
        node = child
    return [top]


# object_info type naming how a published track state was reached: "measured" when
# a detection updated (or started) the track, "predicted" when it only coasted.
UPDATE_TYPE = "fusionUpdate"


def _timestamp(moment: datetime) -> str:
    return as_utc(moment).isoformat().replace("+00:00", "Z")


def _associated(source: dict) -> dict:
    """One sensor report behind a fused track: the track is made of it, so it is a child."""
    entry = {
        "timestamp": _timestamp(source["timestamp"]),
        "node_id": source["node_id"],
        "association_type": "ASSOCIATION_RELATION_CHILD",
    }
    if source.get("object_id"):
        entry["object_id"] = source["object_id"]
    return entry


def track_message(
    track: TacticalTrack,
    node_id: str,
    confidence: float | None = None,
    predicted: bool | None = None,
    sources: list[dict] | None = None,
) -> dict:
    """The SapientMessage, in the JSON mapping with proto field names, for one fused track.

    `predicted` labels the state as coasted or measured in object_info; None leaves
    the label out. `sources` (node_id, object_id, timestamp each) lists the sensor
    reports behind the track as associated_detection.
    """
    location = {
        "x": track.longitude,
        "y": track.latitude,
        "coordinate_system": LAT_LNG,
        "datum": WGS84,
    }
    if track.altitude is not None:
        location["z"] = track.altitude

    report = {
        "report_id": report_ulid(track.timestamp),
        "object_id": object_ulid(track.track_id),
        "location": location,
        "object_info": [
            {"type": "estimatedSwarmCount", "value": str(track.swarm_count)},
            {"type": "threatLevel", "value": track.threat_level},
        ],
    }
    if predicted is not None:
        report["object_info"].append(
            {"type": UPDATE_TYPE, "value": "predicted" if predicted else "measured"}
        )
    if track.velocity is not None and len(track.velocity) >= 2:
        velocity = {"east_rate": track.velocity[0], "north_rate": track.velocity[1]}
        if len(track.velocity) >= 3:
            velocity["up_rate"] = track.velocity[2]
        report["enu_velocity"] = velocity
    if sources:
        report["associated_detection"] = [_associated(source) for source in sources]
    classification = _classification(track.classification, confidence)
    if classification:
        report["classification"] = classification

    message = {
        "timestamp": _timestamp(track.timestamp),
        "node_id": node_id,
        "detection_report": report,
    }
    # Round-trip: rejects anything the definition does not declare and writes the
    # timestamp the way the JSON mapping does, the form event_time must repeat.
    return MessageToDict(ParseDict(message, SapientMessage()), preserving_proto_field_name=True)


def track_record(
    track: TacticalTrack,
    node_id: str,
    confidence: float | None = None,
    predicted: bool | None = None,
    sources: list[dict] | None = None,
) -> tuple[bytes, bytes, str]:
    """(key, value, event_time) of the sapient-raw record for one fused track."""
    message = track_message(track, node_id, confidence, predicted, sources)
    record = {
        "event_time": message["timestamp"],
        "content_type": CONTENT_TYPE,
        "node_id": node_id,
        "message": message,
    }
    value = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return node_id.encode("utf-8"), value.encode("utf-8"), record["event_time"]
