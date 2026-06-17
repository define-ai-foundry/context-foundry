# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

import socket
import logging
from datetime import datetime, timezone
import numpy as np

from stonesoup.types.detection import Detection
from stonesoup.models.measurement.linear import LinearGaussian

from .. import config
from .base import SapientSource

# Import the compiled Protobuf classes (from Step 1 of the Protobuf integration)
from sapient_msg.bsi_flex_335_v2_0 import detection_report_pb2

logger = logging.getLogger(__name__)

class NetworkSapientStream(SapientSource):
    """
    Live ingress adapter. Binds to a UDP network socket, listens for binary 
    SAPIENT Protobuf packets, and streams them into the fusion engine in real-time.
    """

    def __init__(self, ip: str = "0.0.0.0", port: int = 5000):
        self.ip = ip
        self.port = port
        
        self.cartesian_meas_model = LinearGaussian(
            ndim_state=9,
            mapping=(0, 3, 6),
            noise_covar=np.diag([25.0, 25.0, 100.0])
        )
        
        # Initialize the live UDP listener
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))
        logger.info(f"Live SAPIENT Stream listening for ASMs on {self.ip}:{self.port}")

    def iter_events(self):
        """
        A blocking generator. Pauses the fusion loop until a packet arrives, 
        then instantly yields it to the tracker.
        """
        while True:
            try:
                # 1. Block and wait for a network packet
                payload_bytes, address = self.sock.recvfrom(4096)
                
                # 2. Fast Binary Deserialization
                report = detection_report_pb2.DetectionReport()
                report.ParseFromString(payload_bytes)
                
                # We only process messages that contain spatial data
                if not report.HasField("location"):
                    continue
                    
                # Extract strict types
                lat = report.location.x
                lon = report.location.y
                alt = report.location.z if report.location.HasField("z") else 0.0
                
                # Project to local metric tracking frame
                e, n, u = config.wgs84_to_enu(lat, lon, alt)
                
                # Extract custom object info attributes safely
                swarm_count = 1
                for info in report.object_info:
                    if info.type == "estimatedSwarmCount":
                        swarm_count = int(info.value)
                        
                primary_class = report.classification[0].type if report.classification else "Unknown"
                
                # The Protobuf Timestamp -> Python Datetime
                # If no timestamp is provided, stamp it with time-of-arrival
                if report.HasField("timestamp"):
                    timestamp = report.timestamp.ToDatetime().replace(tzinfo=timezone.utc)
                else:
                    timestamp = datetime.now(timezone.utc)
                
                # Assuming node_id is part of the derived payload or envelope
                node_id = getattr(report, "node_id", "UNKNOWN_NODE")
                sensor_meta = config.get_sensor(node_id)
                
                # 3. Build the Stone Soup Object
                detection = Detection(
                    state_vector=np.array([[e], [n], [u]]),
                    measurement_model=self.cartesian_meas_model,
                    timestamp=timestamp
                )
                
                # Inject Metadata
                detection.metadata = {
                    "nodeId": node_id,
                    "objectId": report.object_id,
                    "classification": primary_class,
                    "swarm_count": swarm_count,
                    "sensor_geodetic": {
                        "latitude": sensor_meta["lat"],
                        "longitude": sensor_meta["lon"],
                        "altitude": sensor_meta["alt"]
                    } if sensor_meta else None
                }
                
                # Yield as a list to match the identical signature of JsonSapientSource
                yield timestamp, [detection]
                
            except Exception as e:
                logger.warning(f"Failed to parse incoming SAPIENT packet: {e}")