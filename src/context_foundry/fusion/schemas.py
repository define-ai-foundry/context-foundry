# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field, model_validator, RootModel
from typing import Optional, List
from datetime import datetime

# --- SAPIENT Sub-Components ---
class SapientLocation(BaseModel):
    # Per BSI Flex 335: x is typically Latitude, y is Longitude
    x: float
    y: float
    z: Optional[float] = 0.0

class SapientClassification(BaseModel):
    type: str
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)

class TrackObjectInfo(BaseModel):
    type: str
    value: str

# --- Payload Definitions ---
class DetectionReport(BaseModel):
    objectId: str = Field(default="UNKNOWN_OBJ")
    state: Optional[str] = None
    location: SapientLocation
    classification: Optional[List[SapientClassification]] = []
    object_info: Optional[List[TrackObjectInfo]] = []

class StatusReport(BaseModel):
    system: str
    info: str

# --- Top-Level Message Envelope ---
class SapientMessage(BaseModel):
    timestamp: datetime
    nodeId: str
    
    # Payloads
    detectionReport: Optional[DetectionReport] = None
    statusReport: Optional[StatusReport] = None
    
    @model_validator(mode='after')
    def verify_single_payload(self) -> 'SapientMessage':
        payloads = [self.detectionReport, self.statusReport]
        active = sum(1 for p in payloads if p is not None)
        
        if active == 0:
            raise ValueError("Invalid Message: Missing a valid payload block.")
        if active > 1:
            raise ValueError("Invalid Message: Multiple payload blocks detected.")
        return self

# --- Stream Validator (For JSON Arrays) ---
class SapientMessageStream(RootModel):
    root: List[SapientMessage]

# --- Internal Fused State Representation ---
class TacticalState(BaseModel):
    """The format-independent internal state of a fused track."""
    track_id: str
    timestamp: datetime
    lat: float
    lon: float
    alt: float
    speed_m_s: float
    heading_deg: float
    classification: str
    swarm_count: int
    threat_level: str