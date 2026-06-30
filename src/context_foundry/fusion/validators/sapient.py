# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Any, Tuple
import logging
from datetime import datetime

from google.protobuf.json_format import ParseDict, ParseError

# Now we import the newly compiled root message
from sapient_msg.bsi_flex_335_v2_0.sapient_message_pb2 import SapientMessage
from context_foundry.fusion.schemas import InternalDetection
from context_foundry.fusion.validators.base import ProtocolValidator

logger = logging.getLogger(__name__)

class SapientValidator(ProtocolValidator):
    def validate(self, raw_payload: Dict[str, Any]) -> Tuple[bool, str]:
        # 1. Catch the outer JSON wrapper expected from your stream
        if "sapientMessage" not in raw_payload:
            return False, "Missing 'sapientMessage' root dictionary key."

        try:
            # 2. Parse the ENTIRE message at once. 
            # If a timestamp is malformed, a field is missing, or a type is wrong,
            # ParseDict will instantly throw a ParseError here.
            msg = SapientMessage()
            ParseDict(raw_payload["sapientMessage"], msg, ignore_unknown_fields=False)
            
            # 3. We only want to track Detections, ignore registrations/status reports
            if not msg.HasField("detectionReport"):
                return False, "Valid SAPIENT message, but not a DetectionReport."

            # 4. Ensure spatial data exists for Stone Soup
            if not msg.detectionReport.HasField("location"):
                 return False, "DetectionReport missing required location block."

            return True, ""
            
        except ParseError as e:
            return False, f"Protobuf schema violation: {e}"

    def normalize(self, raw_payload: Dict[str, Any]) -> InternalDetection:
        # Re-parse (this is lightning fast in memory)
        msg = SapientMessage()
        ParseDict(raw_payload["sapientMessage"], msg)

        # Look how clean this is! Full dot-notation, zero dictionary `.get()` lookups.
        report = msg.detectionReport
        
        # Parse UTC Timestamp
        ts_str = msg.header.timestamp.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts_str)

        return InternalDetection(
            sensor_id=str(msg.header.sourceNode.nodeId),
            timestamp=dt,
            latitude=report.location.latitude,
            longitude=report.location.longitude,
            
            altitude=report.location.altitude if report.location.HasField("altitude") else None,
            speed_mps=report.kinematics.speed if report.HasField("kinematics") and report.kinematics.HasField("speed") else None,
            heading_deg=report.kinematics.heading if report.HasField("kinematics") and report.kinematics.HasField("heading") else None,
            classification=str(report.objectClass.classType) if report.HasField("objectClass") else None,
            confidence=report.confidence if report.HasField("confidence") else None,
            
            raw_metadata={"original_envelope": raw_payload} 
        )