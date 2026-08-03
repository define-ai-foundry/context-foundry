# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

import logging
import math
from datetime import timezone
from typing import Any

from google.protobuf.json_format import MessageToDict, ParseDict, ParseError

from context_foundry.fusion import config
from context_foundry.fusion.schemas import InternalDetection
from context_foundry.fusion.validators.base import ProtocolValidator
from sapient_msg.bsi_flex_335_v2_0.range_bearing_pb2 import RangeBearing

# Now we import the newly compiled root message
from sapient_msg.bsi_flex_335_v2_0.sapient_message_pb2 import SapientMessage

logger = logging.getLogger(__name__)

# Node IDs already reported as unregistered, so the explanation is logged once per node
# instead of once per datagram.
_UNREGISTERED_NODES_SEEN: set[str] = set()


class SapientValidator(ProtocolValidator):
    def validate(self, raw_payload: dict[str, Any]) -> tuple[bool, str]:
        if "sapientMessage" not in raw_payload:
            return False, "Missing 'sapientMessage' root dictionary key."

        try:
            msg = SapientMessage()
            ParseDict(raw_payload["sapientMessage"], msg, ignore_unknown_fields=False)

            # Check the 'oneof' field instead of checking for the message type directly
            if msg.WhichOneof("content") != "detection_report":
                return (
                    False,
                    f"Valid SAPIENT message, but content is {msg.WhichOneof('content')} (expected detection_report).",
                )

            if not msg.detection_report.HasField("location_oneof"):
                return False, "DetectionReport missing required location_oneof block."

            return True, ""

        except ParseError as e:
            return False, f"Protobuf schema violation: {e}"

    def _range_bearing_to_wgs84(
        self, node_id: str, rb: RangeBearing
    ) -> tuple[float, float, float | None]:
        """Place a range/bearing detection using the reporting sensor as the origin.

        Range and azimuth are measured from the sensor (proto: "Range from the node's
        location", "Azimuth in relation to the node's north"), so resolving them against
        the network ENU anchor displaces every detection from a non-anchor sensor by the
        anchor-to-sensor baseline, which is tens of kilometres in a real deployment.
        """
        sensor = config.get_sensor(node_id)
        if sensor is None:
            if node_id not in _UNREGISTERED_NODES_SEEN:
                _UNREGISTERED_NODES_SEEN.add(node_id)
                logger.warning(
                    f"Node '{node_id}' is not in the sensor registry; its range/bearing "
                    "detections cannot be geolocated and will be dropped. Add it to the "
                    "sensor manifest to ingest them."
                )
            raise ValueError(f"range_bearing detection from unregistered node '{node_id}'.")

        # Azimuth is clockwise from the node's north; elevation is above its horizon,
        # so the slant range projects onto the ground plane before splitting East/North.
        has_elevation = rb.HasField("elevation")
        az = math.radians(rb.azimuth)
        elev = math.radians(rb.elevation) if has_elevation else 0.0
        ground_range = rb.range * math.cos(elev)

        e_offset = ground_range * math.sin(az)
        n_offset = ground_range * math.cos(az)
        u_offset = rb.range * math.sin(elev)

        lat, lon, alt = config.enu_to_wgs84_about(
            e_offset, n_offset, u_offset, sensor["lat"], sensor["lon"], sensor["alt"]
        )
        # Without an elevation the message carries no height information, and the sensor's
        # own altitude would be a fabricated target altitude.
        return lat, lon, alt if has_elevation else None

    def normalize(self, raw_payload: dict[str, Any]) -> InternalDetection:
        msg = SapientMessage()
        ParseDict(raw_payload["sapientMessage"], msg)
        report = msg.detection_report
        dt = msg.timestamp.ToDatetime().replace(tzinfo=timezone.utc)

        lat, lon, alt = None, None, None

        # 1. Prioritize Cartesian 'location'
        if report.HasField("location"):
            lat = report.location.y
            lon = report.location.x
            alt = report.location.z if report.location.HasField("z") else None

        # 2. Resolve polar coordinates against the reporting sensor
        elif report.HasField("range_bearing"):
            lat, lon, alt = self._range_bearing_to_wgs84(msg.node_id, report.range_bearing)

        else:
            raise ValueError("Detection missing both 'location' and 'range_bearing' fields.")

        # --- Classification and Return ---
        primary_class = (
            report.classification[0].type if len(report.classification) > 0 else "Unknown"
        )

        return InternalDetection(
            sensor_id=msg.node_id,
            timestamp=dt,
            latitude=lat,
            longitude=lon,
            altitude=alt,
            speed_mps=None,
            heading_deg=None,
            classification=primary_class,
            confidence=report.detection_confidence
            if report.HasField("detection_confidence")
            else None,
            # Sources read the detection report out of "original_report" for the
            # fields the universal schema does not carry (objectId, swarm count).
            # Rendered back from the parsed message rather than picked out of the
            # raw dict, which ParseDict also accepts in snake_case.
            raw_metadata={
                "original_envelope": raw_payload,
                "original_report": MessageToDict(report, preserving_proto_field_name=False),
            },
        )
