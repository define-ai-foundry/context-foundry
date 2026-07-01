# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from abc import ABC, abstractmethod

from .schemas import TacticalTrack

class BaseSerializer(ABC):
    @abstractmethod
    def serialize(self, state: TacticalTrack, node_id: str) -> str:
        pass

class CotSerializer(BaseSerializer):
    """Formats tactical state for ATAK/WinTAK networks."""
    
    def serialize(self, state: TacticalTrack, node_id: str = "FUSION-NODE") -> str:
        now = state.timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        stale = datetime.fromtimestamp(state.timestamp.timestamp() + 15.0, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        
        identity = "h" if state.threat_level == "hostile" else "s"
        sidc = f"a-{identity}-A-M-F" 

        event = ET.Element("event", {
            "version": "2.0",
            "uid": f"TRK-{state.track_id}",
            "type": sidc,
            "time": now, "start": now, "stale": stale,
            "how": "m-g"
        })
        
        ET.SubElement(event, "point", {
            "lat": f"{state.latitude:.6f}", "lon": f"{state.longitude:.6f}",
            "hae": f"{state.altitude:.1f}", "ce": "10.0", "le": "10.0"
        })
        
        detail = ET.SubElement(event, "detail")
        ET.SubElement(detail, "track", {"speed": f"{state.speed_mps:.2f}", "course": f"{state.heading_deg:.1f}"})
        ET.SubElement(detail, "contact", {"callsign": f"SWM({state.swarm_count}) {state.classification}"})
        
        return ET.tostring(event, encoding="utf-8").decode("utf-8")

class SapientSerializer(BaseSerializer):
    """Formats tactical state back into a SAPIENT BSI Flex 335 message."""
    
    def serialize(self, state: TacticalTrack, node_id: str) -> str:
        # Construct the valid Pydantic model and output JSON
        from .schemas import SapientMessage, DetectionReport, SapientLocation, SapientClassification, TrackObjectInfo
        
        report = DetectionReport(
            objectId=f"TRK-{state.track_id}",
            state="Active",
            location=SapientLocation(x=state.latitude, y=state.longitude, z=state.altitude),
            classification=[SapientClassification(type=state.classification, confidence=0.95)],
            object_info=[
                TrackObjectInfo(type="estimatedSwarmCount", value=str(state.swarm_count)),
                TrackObjectInfo(type="threatLevel", value=state.threat_level),
                TrackObjectInfo(type="speed", value=f"{state.speed_mps:.2f}"),
                TrackObjectInfo(type="heading", value=f"{state.heading_deg:.2f}")
            ]
        )
        
        msg = SapientMessage(
            timestamp=state.timestamp,
            nodeId=node_id,
            detectionReport=report
        )
        
        # Pydantic natively exports to standard JSON
        return msg.model_dump_json(exclude_none=True)