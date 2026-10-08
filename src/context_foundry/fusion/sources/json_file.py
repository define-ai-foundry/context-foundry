# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# src/context_foundry/fusion/sources/json_file.py

import json
import logging
from collections import defaultdict
from pathlib import Path

from stonesoup.types.detection import Detection

from .. import config
from ..measurement import DEFAULT_MEASUREMENT_MODEL, position_measurement

# Import the new Gatekeeper
from ..validators.sapient import SapientValidator
from .base import SapientSource, swarm_count

logger = logging.getLogger(__name__)


class JsonSapientSource(SapientSource):
    def __init__(self, json_path: Path):
        self.json_path = Path(json_path)
        # The shared model before per-sensor accuracy; detections use measurement_model_for.
        self.cartesian_meas_model = DEFAULT_MEASUREMENT_MODEL
        # Instantiate the Gatekeeper once
        self.validator = SapientValidator()
        # A replay file is finite: it is drained on the first pass and empty after
        self._exhausted = False

    def reset(self):
        # Re-arm the finite file so a subsequent iter_events() re-reads and re-yields it.
        self._exhausted = False

    def iter_events(self):
        # The main loop re-polls every source each pass; a finite file must be
        # consumed exactly once, otherwise the replay-exit condition is never met
        if self._exhausted:
            return

        if not self.json_path.exists():
            raise FileNotFoundError(self.json_path)

        self._exhausted = True

        # 1. Load the raw JSON array
        with open(self.json_path, encoding="utf-8") as f:
            messages = json.load(f)

        valid_detections = []

        # 2. STRICT VALIDATION LAYER (The Gatekeeper)
        for raw_dict in messages:
            clean_det = self.validator.process_message(raw_dict)
            if clean_det is not None:
                valid_detections.append(clean_det)

        # 3. GROUP ASYNCHRONOUS FRAMES
        # We group valid InternalDetection objects by timestamp and sensor
        sensor_frames = defaultdict(list)
        for det in valid_detections:
            sensor_frames[(det.timestamp, det.sensor_id)].append(det)

        # 4. TRANSLATE TO STONE SOUP DETECTIONS
        # Chronological order, independent of the order messages appear in the file:
        # the tracker cannot predict backwards, and realtime pacing needs it too.
        for (timestamp, node_id), reports in sorted(sensor_frames.items(), key=lambda kv: kv[0][0]):
            detections = []
            sensor_meta = config.sensor_profile(node_id)
            sensor_enu = None

            for det in reports:
                # Project WGS84 Geodetic to local metric Cartesian tracking frame
                # Without a height the position is projected at 0 m, but only its east and
                # north are used; see position_measurement.
                alt = det.altitude if det.altitude is not None else 0.0
                e, n, u = config.wgs84_to_enu(det.latitude, det.longitude, alt)

                # A geometric_error grows with distance from the sensor, so the sensor
                # is needed in the same frame. Projected for any registered sensor, and only
                # after a detection: the frame's origin is fixed by the
                # first point ever projected, which has to stay the first detection.
                if sensor_enu is None and sensor_meta and "lat" in sensor_meta:
                    sensor_enu = config.wgs84_to_enu(
                        sensor_meta["lat"], sensor_meta["lon"], sensor_meta["alt"]
                    )

                state_vector, model = position_measurement(
                    sensor_meta, sensor_enu, (e, n, u), det.altitude is not None
                )
                detection = Detection(
                    state_vector=state_vector,
                    measurement_model=model,
                    timestamp=timestamp,
                )

                # Extract protocol-specific metadata saved by the validator
                # This allows us to access obscure fields without cluttering the universal schema
                original_report = det.raw_metadata.get("original_report", {})

                # Bind complete contextual payload to Stone Soup observation
                detection.metadata = {
                    "nodeId": node_id,
                    "objectId": original_report.get("objectId"),
                    # Whether the tracker may follow this objectId from report to report.
                    "stable_object_id": config.has_stable_object_ids(sensor_meta),
                    "classification": det.classification or "Unknown",
                    "swarm_count": swarm_count(original_report),
                    "sensor_geodetic": {
                        "latitude": sensor_meta["lat"],
                        "longitude": sensor_meta["lon"],
                        "altitude": sensor_meta["alt"],
                    }
                    if sensor_meta and "lat" in sensor_meta
                    else None,
                }

                detections.append(detection)

            if detections:
                yield timestamp, detections
