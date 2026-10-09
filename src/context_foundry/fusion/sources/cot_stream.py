# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

import logging
import socket

import numpy as np
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.detection import Detection

from .. import config
from ..cot_input import point_accuracy
from ..measurement import position_measurement
from ..validators.cot import CotValidator
from .base import SapientSource
from .frames import DEFAULT_FRAME_WINDOW_SECONDS, FrameAssembler

logger = logging.getLogger(__name__)


def _point_accuracy(raw_metadata):
    """A pseudo sensor entry carrying the CoT point's own error as its geometric_error.

    CoT says how accurate each point is (ce horizontally, le vertically); an
    unknown one falls back to the cautious default for a sensor of unknown kind.
    """
    raw_metadata = raw_metadata or {}
    return point_accuracy(raw_metadata.get("ce"), raw_metadata.get("le"))


class CotNetworkStream(SapientSource):
    """
    Live ingress adapter for CoT XML. Listens for CoT packets,
    validates them against MITRE schemas, and yields internal detections.

    Datagrams are assembled into frames so one instant is one event, the same
    contract JsonSapientSource replays; see FrameAssembler.
    """

    def __init__(
        self,
        ip: str = "0.0.0.0",  # noqa: S104 edge node listens on all interfaces
        port: int = 6969,
        frame_window_seconds: float = DEFAULT_FRAME_WINDOW_SECONDS,
    ):
        self.ip = ip
        self.port = port
        self.validator = CotValidator()  # The Gatekeeper we defined earlier

        self.cartesian_meas_model = LinearGaussian(
            ndim_state=9, mapping=(0, 3, 6), noise_covar=np.diag([25.0, 25.0, 100.0])
        )

        # Frames are keyed by (timestamp, sender IP). A CoT event's uid identifies
        # the object it describes, not the sender, so keying on it would put every
        # object of one instant in a frame of its own and batch nothing; keying on
        # the timestamp alone would merge two senders reporting the same object at
        # the same instant into one event, where the associator takes one of the
        # two hits and initiates a duplicate track from the other. The peer address
        # names the sending host, which is the closest the base CoT event schema
        # gets to sender identity -- so emitters behind one NAT address, or
        # forwarded by one relay, do still merge.
        self.frames = FrameAssembler(window_seconds=frame_window_seconds)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))
        logger.info(f"Live CoT Stream listening on UDP {self.ip}:{self.port}")

    def iter_events(self):
        while True:
            completed = []
            try:
                completed = self._read_into_frames()
            except TimeoutError:
                # A frame's window ran out: expected, and the whole point of the
                # timeout, so nothing is logged.
                pass
            except Exception as e:
                logger.error(f"Failed to process CoT packet: {e}")

            yield from completed
            # Once per read, whatever it produced. A dropped packet never reaches
            # the assembler, and recvfrom() does not time out while the receive
            # buffer is non-empty, so a flood of schema-invalid or malformed XML
            # would otherwise hold every open frame for as long as it lasts and
            # the engine would go silent.
            yield from self.frames.take_expired()

    def _read_into_frames(self):
        """Read one datagram and buffer its detection; returns any events that it completes."""
        # Block while no frame is open, otherwise expire with the shortest window.
        self.sock.settimeout(self.frames.next_timeout())

        data, sender = self.sock.recvfrom(65535)
        raw_xml = data.decode("utf-8")

        # 1. Validate using our Gatekeeper
        clean_det = self.validator.process_message(raw_xml)

        if clean_det is None:
            return []  # Drop invalid CoT

        # 2. Translate to Stone Soup Detection. A point without a height is
        # projected at 0 m but measured in east and north only.
        has_height = clean_det.altitude is not None
        e, n, u = config.wgs84_to_enu(
            clean_det.latitude, clean_det.longitude, clean_det.altitude if has_height else 0.0
        )
        state_vector, model = position_measurement(
            _point_accuracy(clean_det.raw_metadata), None, (e, n, u), has_height
        )
        detection = Detection(
            state_vector=state_vector,
            measurement_model=model,
            timestamp=clean_det.timestamp,
        )

        # The CoT event type is the sender's own label. Keep it under its own key:
        # Stone Soup merges a hit's metadata into the track, so putting a 2525
        # type code in "classification" would overwrite the real classification a
        # SAPIENT sensor gave the same track.
        detection.metadata = {
            "nodeId": clean_det.sensor_id,
            "type": "CoT",
            "cot_type": clean_det.classification,
        }

        # The port is left out: an emitter may send from a fresh ephemeral port
        # per datagram, which would split one sender's instant across frames.
        sender_ip = sender[0]
        return self.frames.add((clean_det.timestamp, sender_ip), clean_det.timestamp, detection)
