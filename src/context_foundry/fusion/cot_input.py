# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# src/context_foundry/fusion/cot_input.py

"""One cot-raw record into what the tracker takes.

A cot-raw record is one Cursor-on-Target event as cot-ingest parses it from the XML:
uid, type, how, start and stale, the point with its own error estimates, and the detail
carried through as parsed. Every located event is a detection of the object its uid
names. CoT says how accurate each point is -- ce across the ground, le in height -- so
that is the detection's accuracy; an unknown one falls back to the cautious default for
a sensor of unknown kind.

A CoT event names the object, not the sensor that saw it, so the uid is the object id,
and it is stable: a CoT emitter keeps one uid for one object, which is how TAK updates
it in place. The sensor is the node `sapient-to-cot` names in the remarks
(`node=<node_id>`) when the event came from a SAPIENT node, and otherwise "cot".

Events fusion produced itself, routed back through sapient-to-cot, carry fusion's own
node in the remarks and are skipped, so its tracks are not fused again.
"""

import logging
from datetime import datetime

from stonesoup.types.detection import Detection

from . import config
from .measurement import UNKNOWN_SENSOR_GEOMETRIC_ERROR, position_measurement

logger = logging.getLogger(__name__)

# The value CoT uses for "unknown" in hae, ce and le.
COT_UNKNOWN = 9999999.0

# The sensor of a CoT event that does not name its SAPIENT node.
COT_NODE = "cot"


def known(value) -> float | None:
    """A CoT point value as a float, or None when absent or marked unknown."""
    if value is None:
        return None
    number = float(value)
    return None if number >= COT_UNKNOWN else number


def point_accuracy(ce: float | None, le: float | None) -> dict:
    """A pseudo sensor entry carrying a CoT point's own error as its geometric_error."""
    unknown = UNKNOWN_SENSOR_GEOMETRIC_ERROR
    return {
        "type": "cot",
        "geometric_error": {
            "variation_type": "constant",
            "base_m": ce if ce is not None else unknown["base_m"],
            "vertical_m": le if le is not None else unknown["vertical_m"],
        },
    }


def remarks_node(record: dict) -> str | None:
    """The node_id a `node=<node_id>` remark names, as sapient-to-cot writes it, or None."""
    remarks = ((record.get("detail") or {}).get("remarks") or {}).get("_text") or ""
    for token in str(remarks).split():
        if token.startswith("node=") and len(token) > len("node="):
            return token[len("node=") :]
    return None


class CotRecordReader:
    """Turns cot-raw records into (timestamp, sensor id, Detection), one at a time.

    `own_node_id`, when given, is fusion's own node: events that name it are skipped.
    """

    def __init__(self, own_node_id: str | None = None):
        self.own_node_id = own_node_id
        self.records_rejected = 0
        self.own_records_skipped = 0

    def detection(self, record: dict):
        """(timestamp, sensor id, Detection) of a located cot-raw event, or None.

        None for fusion's own events and for an event fusion cannot use: no uid, no
        time, or no point with a latitude and longitude.
        """
        node = remarks_node(record)
        if self.own_node_id and node == self.own_node_id:
            self.own_records_skipped += 1
            return None
        try:
            uid = str(record["uid"])
            timestamp = datetime.fromisoformat(str(record["event_time"]).replace("Z", "+00:00"))
            point = record["point"]
            lat, lon = float(point["lat"]), float(point["lon"])
            height = known(point.get("hae"))
            ce, le = known(point.get("ce")), known(point.get("le"))
        except (KeyError, TypeError, ValueError) as e:
            self.records_rejected += 1
            logger.warning("Skipping a cot-raw event fusion cannot read: %s", e)
            return None
        if not uid or timestamp.tzinfo is None or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            self.records_rejected += 1
            logger.warning("Skipping a cot-raw event without a usable uid, time or point")
            return None

        # A point without a height is projected at 0 m but measured in east and north only.
        e, n, u = config.wgs84_to_enu(lat, lon, height if height is not None else 0.0)
        state_vector, model = position_measurement(
            point_accuracy(ce, le), None, (e, n, u), height is not None
        )
        detection = Detection(
            state_vector=state_vector, measurement_model=model, timestamp=timestamp
        )
        sensor = node or COT_NODE
        # The CoT type is the emitter's own label, kept under its own key: Stone Soup merges
        # a hit's metadata into the track, and a 2525 code in "classification" would
        # overwrite the real classification a SAPIENT sensor gave the same track.
        detection.metadata = {
            "nodeId": sensor,
            "objectId": uid,
            "stable_object_id": True,
            "type": "CoT",
            "cot_type": record.get("type"),
            "sensor_geodetic": None,
            **record.get("_lineage", {}),
        }
        return timestamp, sensor, detection
