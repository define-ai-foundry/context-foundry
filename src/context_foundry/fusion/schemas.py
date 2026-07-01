# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any

class InternalDetection(BaseModel):
    """
    The universal, flattened detection format used internally by the Fusion Engine.
    All Protocol Gateways MUST convert their specific formats to this model.
    """
    # Origin & Timing
    sensor_id: str = Field(..., description="Unique identifier of the reporting sensor or node")
    timestamp: datetime = Field(..., description="UTC Time of the detection")
    
    # Kinematics / Spatial (Required for basic Stone Soup tracking)
    latitude: float = Field(..., description="WGS84 Latitude in decimal degrees")
    longitude: float = Field(..., description="WGS84 Longitude in decimal degrees")
    altitude: Optional[float] = Field(None, description="Altitude in meters")
    
    # Optional velocity vector (if provided by smart sensors)
    speed_mps: Optional[float] = Field(None, description="Speed in meters per second")
    heading_deg: Optional[float] = Field(None, description="Heading in degrees from True North")
    
    # Metadata & Uncertainty
    classification: Optional[str] = Field(None, description="Object classification (e.g., 'UAS', 'Vehicle')")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Detection confidence (0.0 to 1.0)")
    
    # Escape Hatch for protocol-specific data that serializers might need later
    raw_metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Preserved protocol-specific data (e.g., original SAPIENT task ID)"
    )

    class Config:
        # Ensures that any extra fields accidentally passed in are dropped, 
        # keeping the internal state pristine.
        extra = "ignore"

class TacticalTrack(BaseModel):
    """
    Contextualized fused track representation.
    Consumed by TAK/SAPIENT serializers.
    """

    track_id: str = Field(...)
    timestamp: datetime = Field(...)

    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None

    # Added fields to resolve the AttributeError
    speed_mps: Optional[float] = None
    heading_deg: Optional[float] = None
    classification: Optional[str] = None
    swarm_count: int = 1
    threat_level: str = "unknown"

    velocity: Optional[list[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "ignore"