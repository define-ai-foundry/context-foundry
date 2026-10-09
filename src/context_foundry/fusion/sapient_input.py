# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# src/context_foundry/fusion/sapient_input.py

"""One sapient-raw record into what the tracker takes, with no Kafka and no sensor file.

A sapient-raw record is one decoded SapientMessage: a sensor's Registration, a status
report, or a detection report. Fusion learns each sensor from the first two -- what
kind it is, how accurate it is, whether its object_id is stable, how far it sees,
where it stands -- and turns the third into a Detection measured with that sensor's
accuracy. A pipeline stage hands records over one at a time; the Kafka source does the
same after merging partitions in time order.

Records fusion wrote itself are skipped: a pipeline may route fused tracks back to
fusion's input, and fusing them again would make each its own track's twin.
"""

import logging
from datetime import datetime

from stonesoup.types.detection import Detection

from . import config
from .measurement import position_measurement
from .registration import node_location, sensor_capabilities
from .sources.base import swarm_count
from .validators.sapient import SapientValidator

logger = logging.getLogger(__name__)


def event_time(record: dict) -> datetime:
    """A sapient-raw record's event_time."""
    return datetime.fromisoformat(record["event_time"].replace("Z", "+00:00"))


class SapientRecordReader:
    """Turns sapient-raw records into (timestamp, sensor id, Detection), one at a time.

    `own_node_id`, when given, is fusion's own node: its records are skipped.
    """

    def __init__(self, own_node_id: str | None = None):
        self.own_node_id = own_node_id
        self.validator = SapientValidator()
        # Each sensor's position in the tracking frame, by the geodetic position it
        # was projected from, so a sensor that moves is projected again.
        self._sensor_enu: dict[str, tuple[tuple, tuple]] = {}
        self.records_rejected = 0
        self.own_records_skipped = 0

    def learn(self, record: dict) -> bool:
        """Learn from a Registration or status report; True when the record was one.

        Skips fusion's own records and records of other content, returning False.
        """
        node_id = record.get("node_id")
        if self.own_node_id and node_id == self.own_node_id:
            return False
        content_type = record.get("content_type")
        if content_type == "registration":
            # What the sensor says about itself decides how its reports associate.
            config.apply_registration(node_id, sensor_capabilities(record.get("message")))
            return True
        if content_type == "status_report":
            location = node_location(record.get("message"))
            if location:
                config.apply_registration(node_id, location)
            return True
        return False

    def is_detection(self, record: dict) -> bool:
        """Whether a record is a detection report fusion should fuse."""
        if self.own_node_id and record.get("node_id") == self.own_node_id:
            self.own_records_skipped += 1
            return False
        return record.get("content_type") == "detection_report"

    def detection(self, record: dict):
        """(timestamp, sensor id, Detection) of a detection report record, or None.

        None when the report does not validate or carries no position fusion can use.
        """
        clean_det = self.validator.process_message({"sapientMessage": record.get("message")})
        if clean_det is None:
            self.records_rejected += 1
            return None

        # Without a height the position is projected at 0 m, but only its east and
        # north are used; see position_measurement.
        alt = clean_det.altitude if clean_det.altitude is not None else 0.0
        e, n, u = config.wgs84_to_enu(clean_det.latitude, clean_det.longitude, alt)

        original_report = clean_det.raw_metadata.get("original_report", {})
        sensor_meta = config.sensor_profile(clean_det.sensor_id)
        sensor_enu = self._sensor_position(clean_det.sensor_id, sensor_meta)

        state_vector, model = position_measurement(
            sensor_meta, sensor_enu, (e, n, u), clean_det.altitude is not None
        )
        detection = Detection(
            state_vector=state_vector,
            measurement_model=model,
            timestamp=clean_det.timestamp,
        )
        detection.metadata = {
            "nodeId": clean_det.sensor_id,
            "objectId": original_report.get("objectId"),
            # Whether the tracker may follow this objectId from report to report.
            "stable_object_id": config.has_stable_object_ids(sensor_meta),
            "classification": clean_det.classification or "Unknown",
            "swarm_count": swarm_count(original_report),
            "sensor_geodetic": {
                "latitude": sensor_meta["lat"],
                "longitude": sensor_meta["lon"],
                "altitude": sensor_meta["alt"],
            }
            if sensor_meta and "lat" in sensor_meta
            else None,
            # Lineage for the fused track this detection updates; see the Kafka sink.
            **record.get("_lineage", {}),
        }
        return clean_det.timestamp, clean_det.sensor_id, detection

    def _sensor_position(self, sensor_id: str, sensor_meta: dict | None):
        """The sensor's position in the tracking frame, or None when it is not known.

        A geometric_error grows with distance from the sensor, so the sensor is needed
        in the same frame. Projected only after a detection: the frame's origin is
        fixed by the first point ever projected, which has to stay the first detection.
        """
        if not sensor_meta or "lat" not in sensor_meta:
            return None
        geodetic = (sensor_meta["lat"], sensor_meta["lon"], sensor_meta.get("alt", 0.0))
        cached = self._sensor_enu.get(sensor_id)
        if cached is None or cached[0] != geodetic:
            cached = (geodetic, config.wgs84_to_enu(*geodetic))
            self._sensor_enu[sensor_id] = cached
        return cached[1]
