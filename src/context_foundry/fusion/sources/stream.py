# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# src/context_foundry/fusion/sources/stream.py

import logging
import socket

import numpy as np
from google.protobuf.json_format import MessageToDict
from google.protobuf.message import DecodeError
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.detection import Detection

# Import the root SapientMessage wrapper and our Gatekeeper
from sapient_msg.bsi_flex_335_v2_0.sapient_message_pb2 import SapientMessage

from .. import config
from ..validators.sapient import SapientValidator
from .base import SapientSource, swarm_count
from .frames import DEFAULT_FRAME_WINDOW_SECONDS, FrameAssembler

logger = logging.getLogger(__name__)


class NetworkSapientStream(SapientSource):
    """
    Live ingress adapter. Binds to a UDP network socket, listens for binary
    SAPIENT Protobuf packets, and streams them into the fusion engine.

    Datagrams are assembled into frames so one sensor-instant is one event, the
    same contract JsonSapientSource replays; see FrameAssembler.
    """

    def __init__(
        self,
        ip: str = "0.0.0.0",  # noqa: S104 edge node listens on all interfaces
        port: int = 5000,
        frame_window_seconds: float = DEFAULT_FRAME_WINDOW_SECONDS,
    ):
        self.ip = ip
        self.port = port

        self.cartesian_meas_model = LinearGaussian(
            ndim_state=9, mapping=(0, 3, 6), noise_covar=np.diag([25.0, 25.0, 100.0])
        )

        self.validator = SapientValidator()
        # A SAPIENT report names its sending node, so an instant is one node's
        # sweep -- exactly the (timestamp, sensor) grouping the replay source uses.
        self.frames = FrameAssembler(window_seconds=frame_window_seconds)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))
        logger.info(f"Live SAPIENT Stream listening on UDP {self.ip}:{self.port}")

    def iter_events(self):
        while True:
            completed = []
            try:
                completed = self._read_into_frames()
            except TimeoutError:
                # A frame's window ran out: expected, and the whole point of the
                # timeout, so nothing is logged.
                pass
            except DecodeError:
                logger.warning("Received malformed binary Protobuf packet over UDP. Dropping.")
            except Exception as e:
                logger.error(f"Unexpected error in live stream ingestion: {e}")

            yield from completed
            # Once per read, whatever it produced. A rejected datagram never
            # reaches the assembler, and recvfrom() does not time out while the
            # receive buffer is non-empty, so a flood of heartbeats, status
            # updates or garbage would otherwise hold every open frame for as
            # long as it lasts and the engine would go silent.
            yield from self.frames.take_expired()

    def _read_into_frames(self):
        """Read one datagram and buffer its detection; returns any events that it completes."""
        # Block while no frame is open, otherwise expire with the shortest window.
        self.sock.settimeout(self.frames.next_timeout())

        # 1. Block and wait for a network packet. Sized to the maximum UDP
        # payload: a truncated read is a DecodeError, i.e. a silently lost
        # detection whenever an ASM sends a large report.
        payload_bytes, _ = self.sock.recvfrom(65535)

        # 2. Parse binary bytes into the full Protobuf Envelope
        msg = SapientMessage()
        msg.ParseFromString(payload_bytes)

        # 3. Convert to Dictionary to pass to our unified Gatekeeper.
        # camelCase keys ('nodeId'), matching the JSON scenario spec the
        # validator parses.
        msg_dict = MessageToDict(msg, preserving_proto_field_name=False)

        # Re-wrap in the root key to match JSON spec
        raw_payload = {"sapientMessage": msg_dict}

        # 4. Pass through the Gatekeeper
        clean_det = self.validator.process_message(raw_payload)

        if clean_det is None:
            return []  # Invalid message, Heartbeat, or Status update. Ignore it.

        # 5. Build Stone Soup Object
        alt = clean_det.altitude if clean_det.altitude is not None else 0.0
        e, n, u = config.wgs84_to_enu(clean_det.latitude, clean_det.longitude, alt)

        original_report = clean_det.raw_metadata.get("original_report", {})
        sensor_meta = config.get_sensor(clean_det.sensor_id)

        detection = Detection(
            state_vector=np.array([[e], [n], [u]]),
            measurement_model=self.cartesian_meas_model,
            timestamp=clean_det.timestamp,
        )

        detection.metadata = {
            "nodeId": clean_det.sensor_id,
            "objectId": original_report.get("objectId"),
            "classification": clean_det.classification or "Unknown",
            "swarm_count": swarm_count(original_report),
            "sensor_geodetic": {
                "latitude": sensor_meta["lat"],
                "longitude": sensor_meta["lon"],
                "altitude": sensor_meta["alt"],
            }
            if sensor_meta
            else None,
        }

        # 6. Buffer into this node's frame for the instant. One sweep is one
        # event, and two nodes' overlapping sweeps are assembled independently.
        return self.frames.add(
            (clean_det.timestamp, clean_det.sensor_id), clean_det.timestamp, detection
        )
