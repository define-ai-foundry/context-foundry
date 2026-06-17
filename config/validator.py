# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# src/context_foundry/config/validator.py

from typing import List, Dict, Literal
from pydantic import BaseModel, Field, FilePath
import yaml

class GatingConfig(BaseModel):
    max_mahalanobis_distance: float = Field(..., gt=0)
    missed_updates_limit: int = Field(..., ge=1)
    min_initiation_confidence: float = Field(..., ge=0.0, le=1.0)

class MotionModelConfig(BaseModel):
    q_noise_xyz: List[float] = Field(..., min_items=3, max_items=3)

class SensorProfile(BaseModel):
    measurement_variance_range: float | None = Field(None, ge=0)
    measurement_variance_bearing: float = Field(..., ge=0)
    measurement_variance_elevation: float | None = Field(None, ge=0)
    max_allowable_latency_sec: float = Field(..., gt=0)
    weight: float = Field(..., ge=0.0, le=1.0)

class TrackerCoreConfig(BaseModel):
    coordinate_system: Literal["ENU", "NED", "WGS84"]
    update_interval_sec: float = Field(..., gt=0)
    max_track_history: int = Field(..., gt=0)

class AppConfig(BaseModel):
    tracker: TrackerCoreConfig
    gating: GatingConfig
    motion_models: Dict[str, MotionModelConfig]
    sensor_profiles: Dict[str, SensorProfile]

    @classmethod
    def load_from_yaml(cls, file_path: str) -> "AppConfig":
        with open(file_path, "r") as f:
            raw_data = yaml.safe_load(f)
        return cls(**raw_data)