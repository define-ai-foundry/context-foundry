# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# src/context_foundry/fusion/sources/json_file.py

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from stonesoup.types.detection import Detection
from stonesoup.models.measurement.linear import LinearGaussian
from pydantic import ValidationError

from .. import config
from ..schemas import SapientMessageStream
from .base import SapientSource

logger = logging.getLogger(__name__)


class JsonSapientSource(SapientSource):

    def __init__(self, json_path: Path):

        self.json_path = Path(json_path)

        self.cartesian_meas_model = LinearGaussian(
            ndim_state=9,
            mapping=(0, 3, 6),
            noise_covar=np.diag([
                25.0,
                25.0,
                100.0
            ])
        )

    def iter_events(self):

        if not self.json_path.exists():
            raise FileNotFoundError(self.json_path)

        # 1. Load the raw JSON file
        with open(self.json_path, "r", encoding="utf-8") as f:
            messages = json.load(f)

        # 2. STRICT VALIDATION LAYER
        # Validates against the BSI Flex 335 schema defined in schemas.py
        try:
            validated_stream = SapientMessageStream.model_validate(messages)
        except ValidationError as e:
            logger.error(f"CRITICAL: Generated JSON failed SAPIENT schema validation!\n{e}")
            raise

        sensor_frames = defaultdict(list)

        # 3. GROUP ASYNCHRONOUS FRAMES
        for packet in validated_stream.root:
            # We only forward detection reports to the tracking engine
            if not packet.detectionReport:
                continue

            # Group simultaneous observations from the same sensor node
            # Note: Pydantic automatically converts packet.timestamp to a datetime object
            sensor_frames[(packet.timestamp, packet.nodeId)].append(packet)

        # 4. YIELD STONE SOUP DETECTIONS
        for (timestamp, node_id), reports in sensor_frames.items():

            detections = []
            
            # Look up sensor baseline coordinates to inject into non-linear measurement models
            sensor_meta = config.get_sensor(node_id)

            for packet in reports:
                
                # Utilize dot-notation access thanks to Pydantic
                report = packet.detectionReport
                loc = report.location
                
                # BSI Flex 335 spatial mappings
                lat = loc.x
                lon = loc.y
                alt = loc.z if loc.z is not None else 0.0

                # Project to local metric Cartesian tracking frame
                e, n, u = config.wgs84_to_enu(lat, lon, alt)

                detection = Detection(
                    state_vector=np.array([[e], [n], [u]]),
                    measurement_model=self.cartesian_meas_model,
                    timestamp=timestamp
                )

                # Safe classification extraction
                primary_class = "Unknown"
                if report.classification:
                    primary_class = report.classification[0].type

                # Dynamic extraction of swarm attributes from object_info
                swarm_count = 1
                if report.object_info:
                    for info in report.object_info:
                        if info.type == "estimatedSwarmCount":
                            swarm_count = int(info.value)

                # Bind complete contextual payload to Stone Soup observation
                detection.metadata = {
                    "nodeId": node_id,
                    "objectId": report.objectId,
                    "classification": primary_class,
                    "swarm_count": swarm_count,
                    "sensor_geodetic": {
                        "latitude": sensor_meta["lat"],
                        "longitude": sensor_meta["lon"],
                        "altitude": sensor_meta["alt"]
                    } if sensor_meta else None
                }

                detections.append(detection)

            if detections:
                yield timestamp, detections