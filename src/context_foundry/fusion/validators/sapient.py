# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

import logging
import math
from datetime import timezone
from typing import Any

from google.protobuf.json_format import ParseDict, ParseError

from context_foundry.fusion import config
from context_foundry.fusion.schemas import InternalDetection
from context_foundry.fusion.validators.base import ProtocolValidator

# Now we import the newly compiled root message
from sapient_msg.bsi_flex_335_v2_0.sapient_message_pb2 import SapientMessage

logger = logging.getLogger(__name__)


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

        # 2. Use 'azimuth' for polar coordinates
        elif report.HasField("range_bearing"):
            rng = report.range_bearing.range
            # Corrected: Accessing 'azimuth' instead of 'bearing'
            az = math.radians(report.range_bearing.azimuth)

            # Calculate offsets in meters (East, North)
            e_offset = rng * math.sin(az)
            n_offset = rng * math.cos(az)

            # Use the global stateful origin
            lat, lon, alt = config.enu_to_wgs84(e_offset, n_offset, 0.0)

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
            raw_metadata={
                "original_envelope": raw_payload,
                "original_report": raw_payload["sapientMessage"].get("detectionReport", {}),
            },
        )
