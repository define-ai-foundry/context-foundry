# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

import logging
import socket

import numpy as np
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.detection import Detection

from .. import config
from ..validators.cot import CotValidator
from .base import SapientSource

logger = logging.getLogger(__name__)


class CotNetworkStream(SapientSource):
    """
    Live ingress adapter for CoT XML. Listens for CoT packets,
    validates them against MITRE schemas, and yields internal detections.
    """

    def __init__(self, ip: str = "0.0.0.0", port: int = 6969):  # noqa: S104 edge node listens on all interfaces
        self.ip = ip
        self.port = port
        self.validator = CotValidator()  # The Gatekeeper we defined earlier

        self.cartesian_meas_model = LinearGaussian(
            ndim_state=9, mapping=(0, 3, 6), noise_covar=np.diag([25.0, 25.0, 100.0])
        )

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))
        logger.info(f"Live CoT Stream listening on UDP {self.ip}:{self.port}")

    def iter_events(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(65535)
                raw_xml = data.decode("utf-8")

                # 1. Validate using our Gatekeeper
                clean_det = self.validator.process_message(raw_xml)

                if clean_det is None:
                    continue  # Drop invalid CoT

                # 2. Translate to Stone Soup Detection
                e, n, u = config.wgs84_to_enu(
                    clean_det.latitude, clean_det.longitude, clean_det.altitude or 0.0
                )

                detection = Detection(
                    state_vector=np.array([[e], [n], [u]]),
                    measurement_model=self.cartesian_meas_model,
                    timestamp=clean_det.timestamp,
                )

                # The CoT event type is the sender's own label. Keep it under its
                # own key: Stone Soup merges a hit's metadata into the track, so
                # putting a 2525 type code in "classification" would overwrite the
                # real classification a SAPIENT sensor gave the same track.
                detection.metadata = {
                    "nodeId": clean_det.sensor_id,
                    "type": "CoT",
                    "cot_type": clean_det.classification,
                }

                yield clean_det.timestamp, [detection]

            except Exception as e:
                logger.error(f"Failed to process CoT packet: {e}")
