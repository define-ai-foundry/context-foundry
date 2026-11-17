# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

import logging
import math
import time
from datetime import timezone
from typing import Any

from google.protobuf.json_format import ParseDict, ParseError

from context_foundry.fusion import config
from context_foundry.fusion.schemas import InternalDetection
from context_foundry.fusion.validators import base
from context_foundry.fusion.validators.base import ProtocolValidator
from sapient_msg.bsi_flex_335_v2_0.range_bearing_pb2 import RangeBearing

# Now we import the newly compiled root message
from sapient_msg.bsi_flex_335_v2_0.sapient_message_pb2 import SapientMessage

logger = logging.getLogger(__name__)

# Node ids already reported as unregistered, so the explanation is logged once per
# node instead of once per datagram. Bounded, and the reporting rate-limited with
# it: the id comes off an unauthenticated socket, so an unbounded set is a node id
# per spoofed datagram in memory and a warning per datagram in the log.
_UNREGISTERED_NODES_SEEN: set[str] = set()
MAX_UNREGISTERED_NODES_TRACKED = 64
UNREGISTERED_REPORT_SECONDS = 60.0
_unregistered_seen_count = 0
_next_unregistered_report = 0.0


def _report_unregistered_node(node_id: str) -> None:
    """Name an unregistered node once, without letting the wire size the bookkeeping.

    A handful of misconfigured sensors is the ordinary case and each is worth
    naming, so ids are remembered and reported individually while there is room.
    Past that the sender is inventing ids rather than misconfigured, and both the
    set and the log switch to a rate-limited count -- otherwise an unauthenticated
    socket sizes both.
    """
    global _unregistered_seen_count, _next_unregistered_report

    if node_id in _UNREGISTERED_NODES_SEEN:
        return
    _unregistered_seen_count += 1

    if len(_UNREGISTERED_NODES_SEEN) < MAX_UNREGISTERED_NODES_TRACKED:
        _UNREGISTERED_NODES_SEEN.add(node_id)
        logger.warning(
            f"Node '{node_id}' is not in the sensor registry; its range/bearing "
            "detections cannot be geolocated and will be dropped. Add it to the "
            "sensor manifest to ingest them."
        )
        return

    if time.monotonic() < _next_unregistered_report:
        return
    _next_unregistered_report = time.monotonic() + UNREGISTERED_REPORT_SECONDS
    logger.warning(
        "More than %d unregistered node ids seen (%d so far), most recently '%s'. Something is "
        "sending under ids that are not in the sensor manifest; their range/bearing detections "
        "are dropped.",
        MAX_UNREGISTERED_NODES_TRACKED,
        _unregistered_seen_count,
        node_id,
    )


# Separator between taxonomy levels of a flattened classification path.
CLASS_PATH_SEPARATOR = " > "


def flatten_classification(classification) -> str:
    """Join a DetectionReportClassification's type and its nested SubClass chain into one path.

    BSI Flex 335 carries only the top-level taxonomy class in `type`; every finer level
    hangs off a recursively nested `sub_class`. Both `classification` and `sub_class` are
    repeated, i.e. alternative hypotheses; the internal schema holds a single string, so
    the first (primary) entry is followed at every level, matching how the reports are built.
    """
    if not classification:
        return "Unknown"

    return CLASS_PATH_SEPARATOR.join(node.type for node in _primary_chain(classification))


def class_confidences(classification) -> list[float | None]:
    """The confidence of each level `flatten_classification` names, None where a level has none
    or one that is not a number in [0, 1]."""
    return [
        _bounded(node.confidence) if node.HasField("confidence") else None
        for node in _primary_chain(classification)
    ]


def _bounded(confidence: float) -> float | None:
    return confidence if math.isfinite(confidence) and 0.0 <= confidence <= 1.0 else None


def _primary_chain(classification) -> list:
    """The first entry at every level of a classification, the levels without a type left out."""
    if not classification:
        return []

    node = classification[0]
    chain = [node]
    while node.sub_class:
        node = node.sub_class[0]
        chain.append(node)

    return [node for node in chain if node.type]


class SapientValidator(ProtocolValidator):
    def validate(self, raw_payload: dict[str, Any]) -> tuple[bool, str]:
        msg, error = self.parse(raw_payload)
        return msg is not None, error

    def process_message(self, raw_payload: dict[str, Any]) -> InternalDetection | None:
        """The base pipeline, validating and normalizing from one parse of the message.

        The JSON-to-protobuf parse is most of a report's cost on the way in, and
        `validate` and `normalize` would each make the same one.
        """
        msg, error = self.parse(raw_payload)
        if msg is None:
            base.logger.warning(f"Message rejected by {self.__class__.__name__}: {error}")
            return None
        return self.process_parsed(raw_payload, msg)

    def process_parsed(
        self, raw_payload: dict[str, Any], msg: SapientMessage
    ) -> InternalDetection | None:
        """`process_message` of a payload `parse` has already accepted, as `msg`.

        For a caller that validates a report on arrival and normalizes it later: the
        parse it kept stands for the one this would make, so it is not made again.
        """
        try:
            return self._normalize(raw_payload, msg)
        except Exception as e:
            base.logger.error(f"Normalization failed in {self.__class__.__name__}: {e!s}")
            return None

    def parse(self, raw_payload: dict[str, Any]) -> tuple[SapientMessage | None, str]:
        """The parsed detection report, or None and why the payload is refused.

        What `validate` checks: a SAPIENT message the schema takes, unknown fields
        refused, whose content is a detection report with a location.
        """
        if "sapientMessage" not in raw_payload:
            return None, "Missing 'sapientMessage' root dictionary key."

        try:
            msg = SapientMessage()
            ParseDict(raw_payload["sapientMessage"], msg, ignore_unknown_fields=False)

            # Check the 'oneof' field instead of checking for the message type directly
            if msg.WhichOneof("content") != "detection_report":
                return (
                    None,
                    f"Valid SAPIENT message, but content is {msg.WhichOneof('content')} (expected detection_report).",
                )

            if not msg.detection_report.HasField("location_oneof"):
                return None, "DetectionReport missing required location_oneof block."

            return msg, ""

        except ParseError as e:
            return None, f"Protobuf schema violation: {e}"

    def _range_bearing_to_wgs84(
        self, node_id: str, rb: RangeBearing
    ) -> tuple[float, float, float | None]:
        """Place a range/bearing detection using the reporting sensor as the origin.

        Range and azimuth are measured from the sensor (proto: "Range from the node's
        location", "Azimuth in relation to the node's north"), so resolving them against
        the network ENU anchor displaces every detection from a non-anchor sensor by the
        anchor-to-sensor baseline, which is tens of kilometres in a real deployment.
        """
        # The manifest's position or, without one, where the sensor's status reports
        # placed it.
        sensor = config.sensor_profile(node_id)
        if sensor is None or "lat" not in sensor:
            _report_unregistered_node(node_id)
            raise ValueError(
                f"range_bearing detection from unregistered node '{node_id}' (position unknown)."
            )

        # Azimuth is clockwise from the node's north; elevation is above its horizon,
        # so the slant range projects onto the ground plane before splitting East/North.
        has_elevation = rb.HasField("elevation")
        az = math.radians(rb.azimuth)
        elev = math.radians(rb.elevation) if has_elevation else 0.0
        ground_range = rb.range * math.cos(elev)

        e_offset = ground_range * math.sin(az)
        n_offset = ground_range * math.cos(az)
        u_offset = rb.range * math.sin(elev)

        lat, lon, alt = config.enu_to_wgs84_about(
            e_offset, n_offset, u_offset, sensor["lat"], sensor["lon"], sensor["alt"]
        )
        # Without an elevation the message carries no height information, and the sensor's
        # own altitude would be a fabricated target altitude.
        return lat, lon, alt if has_elevation else None

    def normalize(self, raw_payload: dict[str, Any]) -> InternalDetection:
        msg = SapientMessage()
        ParseDict(raw_payload["sapientMessage"], msg)
        return self._normalize(raw_payload, msg)

    def _normalize(self, raw_payload: dict[str, Any], msg: SapientMessage) -> InternalDetection:
        """`normalize` of a message already parsed from `raw_payload`."""
        report = msg.detection_report
        dt = msg.timestamp.ToDatetime().replace(tzinfo=timezone.utc)

        lat, lon, alt = None, None, None

        # 1. Prioritize Cartesian 'location'
        if report.HasField("location"):
            lat = report.location.y
            lon = report.location.x
            alt = report.location.z if report.location.HasField("z") else None

        # 2. Resolve polar coordinates against the reporting sensor
        elif report.HasField("range_bearing"):
            lat, lon, alt = self._range_bearing_to_wgs84(msg.node_id, report.range_bearing)

        else:
            raise ValueError("Detection missing both 'location' and 'range_bearing' fields.")

        # A NaN or infinite double reaches here as such ("NaN" in the JSON mapping), and
        # fused it turns the tracker's state into NaN for good; a position off the world
        # is no better.
        if not (_finite(lat) and _finite(lon) and (alt is None or _finite(alt))):
            raise ValueError("Detection location is not a finite number.")
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            raise ValueError(f"Detection location {lat}, {lon} is off the world.")

        # --- Classification and Return ---
        primary_class = flatten_classification(report.classification)

        return InternalDetection(
            sensor_id=msg.node_id,
            timestamp=dt,
            latitude=lat,
            longitude=lon,
            altitude=alt,
            speed_mps=None,
            heading_deg=None,
            classification=primary_class,
            confidence=report.detection_confidence
            if report.HasField("detection_confidence")
            else None,
            class_confidences=class_confidences(report.classification),
            # Sources read the detection report out of "original_report" for the
            # fields the universal schema does not carry (objectId, swarm count).
            # Rendered back from the parsed message rather than picked out of the
            # raw dict, which ParseDict also accepts in snake_case.
            raw_metadata={
                "original_envelope": raw_payload,
                "original_report": _report_fields(report),
            },
        )


def _report_fields(report) -> dict[str, Any]:
    """What sources read of a detection report, as `MessageToDict` renders it.

    The object id, the report id and the object info (where a swarm count is), each
    only when the report sets it, under their JSON names. Rendering the whole report
    was a large part of reading one, for fields nothing reads.
    """
    out: dict[str, Any] = {}
    if report.HasField("object_id"):
        out["objectId"] = report.object_id
    if report.HasField("report_id"):
        out["reportId"] = report.report_id
    if report.object_info:
        out["objectInfo"] = [
            {
                name: getattr(info, name)
                for name in ("type", "value", "error")
                if info.HasField(name)
            }
            for info in report.object_info
        ]
    return out


def _finite(value) -> bool:
    """Whether a coordinate is a real number: not None, NaN or infinite."""
    return value is not None and math.isfinite(value)
