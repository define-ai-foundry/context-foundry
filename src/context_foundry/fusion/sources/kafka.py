# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# src/context_foundry/fusion/sources/kafka.py

import json
import logging
from collections import deque
from datetime import datetime

from stonesoup.types.detection import Detection

from .. import config
from ..measurement import DEFAULT_MEASUREMENT_MODEL, position_measurement
from ..registration import sensor_capabilities
from ..validators.sapient import SapientValidator
from .base import SapientSource, swarm_count

logger = logging.getLogger(__name__)

# How long one poll waits for the broker when nothing is buffered.
POLL_SECONDS = 0.5

# Records buffered per partition before that partition is paused. Reading a
# backlog, the broker hands over one partition's records in bulk while the others
# have not arrived yet; the merge has to hold them until it can order them.
MAX_BUFFERED_PER_PARTITION = 5000


def _lineage(headers) -> dict:
    """The platform headers a fused track carries forward from the detection behind it."""
    found = dict(headers or [])
    return {
        name: found[name].decode("utf-8", "replace")
        for name in ("trace_id", "ingest_time")
        if found.get(name)
    }


def _event_time(record: dict) -> datetime:
    return datetime.fromisoformat(record["event_time"].replace("Z", "+00:00"))


class KafkaSapientSource(SapientSource):
    """
    Pipeline ingress adapter. Subscribes to a sapient-raw topic -- the records
    sapient-ingest writes, one decoded SapientMessage each -- and streams its
    detection reports into the fusion engine.

    The topic is keyed by node_id, so its partitions each carry some sensors in
    order, but not one another's: a backlog arrives a partition at a time. The
    tracker cannot predict backwards, so records are merged across partitions in
    event_time order before they become events. A partition only holds the merge
    back while it may still deliver something older: once it has a record buffered,
    or the consumer has reached its end, it no longer does.

    One sensor-instant is one event, the same (timestamp, sensor) grouping the
    replay source uses. Offsets are not committed yet: a restart re-reads the topic.
    """

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        group_id: str,
        security_protocol: str = "PLAINTEXT",
        consumer=None,
    ):
        self.topic = topic
        # The shared model before per-sensor accuracy; detections use measurement_model_for.
        self.cartesian_meas_model = DEFAULT_MEASUREMENT_MODEL
        # Each sensor's position in the tracking frame, for its geometric_error.
        self._sensor_enu = {}
        self.validator = SapientValidator()

        self._buffers: dict[int, deque] = {}
        self._at_end: set[int] = set()
        self._paused: set[int] = set()
        self.records_read = 0
        self.records_rejected = 0

        if consumer is None:
            # Imported here so the rest of the engine runs without the Kafka client.
            from confluent_kafka import Consumer

            consumer = Consumer(
                {
                    "bootstrap.servers": bootstrap_servers,
                    "security.protocol": security_protocol,
                    "group.id": group_id,
                    "auto.offset.reset": "earliest",
                    "enable.auto.commit": False,
                    "enable.partition.eof": True,
                }
            )
        self.consumer = consumer
        self.consumer.subscribe([topic], on_assign=self._on_assign, on_revoke=self._on_revoke)
        logger.info(f"Kafka SAPIENT source subscribed to {topic} as group {group_id}")

    # --- partition bookkeeping -------------------------------------------------

    def _on_assign(self, _consumer, partitions):
        for tp in partitions:
            self._buffers.setdefault(tp.partition, deque())
        logger.info(
            "Assigned partitions %s of %s",
            sorted(tp.partition for tp in partitions),
            self.topic,
        )

    def _on_revoke(self, _consumer, partitions):
        for tp in partitions:
            self._buffers.pop(tp.partition, None)
            self._at_end.discard(tp.partition)
            self._paused.discard(tp.partition)

    def _poll(self):
        """Fetch what the broker has into the per-partition buffers."""
        from confluent_kafka import KafkaError

        timeout = 0 if any(self._buffers.values()) else POLL_SECONDS
        for msg in self.consumer.consume(num_messages=500, timeout=timeout):
            error = msg.error()
            if error is not None:
                if error.code() == KafkaError._PARTITION_EOF:
                    self._at_end.add(msg.partition())
                else:
                    logger.warning("Kafka consumer error: %s", error)
                continue
            self.records_read += 1
            record = self._parse(msg.value())
            if record is None:
                continue
            record["_lineage"] = _lineage(msg.headers())
            self._at_end.discard(msg.partition())
            self._buffers.setdefault(msg.partition(), deque()).append(record)

        self._pause_full_partitions()

    def _pause_full_partitions(self):
        from confluent_kafka import TopicPartition

        for partition, buffer in self._buffers.items():
            full = len(buffer) >= MAX_BUFFERED_PER_PARTITION
            if full and partition not in self._paused:
                self.consumer.pause([TopicPartition(self.topic, partition)])
                self._paused.add(partition)
            elif not full and partition in self._paused:
                self.consumer.resume([TopicPartition(self.topic, partition)])
                self._paused.discard(partition)

    def _parse(self, value: bytes):
        """A sapient-raw record holding a detection report, or None."""
        try:
            record = json.loads(value)
            _event_time(record)
        except (TypeError, ValueError, KeyError, AttributeError) as e:
            self.records_rejected += 1
            logger.warning("Skipping a record that is not sapient-raw: %s", e)
            return None
        if record.get("content_type") == "registration":
            # What the sensor says about itself decides how its reports associate.
            config.apply_registration(
                record.get("node_id"), sensor_capabilities(record.get("message"))
            )
            return None
        if record.get("content_type") != "detection_report":
            return None  # Status reports and alerts carry no detection.
        return record

    def _next_in_order(self):
        """The oldest buffered record, once no partition can still deliver an older one."""
        if not self._buffers:
            return None
        heads = []
        for partition, buffer in self._buffers.items():
            if buffer:
                heads.append((_event_time(buffer[0]), partition))
            elif partition not in self._at_end:
                return None  # This partition may still hold something older.
        if not heads:
            return None
        _, partition = min(heads)
        return self._buffers[partition].popleft()

    # --- records into detections ---------------------------------------------

    def _detection(self, record: dict):
        clean_det = self.validator.process_message({"sapientMessage": record.get("message")})
        if clean_det is None:
            self.records_rejected += 1
            return None

        # Without a height the position is projected at 0 m, but only its east and
        # north are used; see position_measurement.
        alt = clean_det.altitude if clean_det.altitude is not None else 0.0
        e, n, u = config.wgs84_to_enu(clean_det.latitude, clean_det.longitude, alt)

        original_report = clean_det.raw_metadata.get("original_report", {})
        sensor_meta = config.sensor_profile(clean_det.sensor_id)

        # A geometric_error grows with distance from the sensor, so the sensor is
        # needed in the same frame. Projected only after a detection: the frame's
        # origin is fixed by the first point ever projected, which has to stay the
        # first detection.
        sensor_enu = None
        if sensor_meta and "lat" in sensor_meta:
            sensor_enu = self._sensor_enu.get(clean_det.sensor_id)
            if sensor_enu is None:
                sensor_enu = config.wgs84_to_enu(
                    sensor_meta["lat"], sensor_meta["lon"], sensor_meta["alt"]
                )
                self._sensor_enu[clean_det.sensor_id] = sensor_enu

        state_vector, model = position_measurement(
            sensor_meta, sensor_enu, (e, n, u), clean_det.altitude is not None
        )
        detection = Detection(
            state_vector=state_vector,
            measurement_model=model,
            timestamp=clean_det.timestamp,
        )
        detection.metadata = {
            "nodeId": clean_det.sensor_id,
            "objectId": original_report.get("objectId"),
            # Whether the tracker may follow this objectId from report to report.
            "stable_object_id": config.has_stable_object_ids(sensor_meta),
            "classification": clean_det.classification or "Unknown",
            "swarm_count": swarm_count(original_report),
            "sensor_geodetic": {
                "latitude": sensor_meta["lat"],
                "longitude": sensor_meta["lon"],
                "altitude": sensor_meta["alt"],
            }
            if sensor_meta and "lat" in sensor_meta
            else None,
            # Lineage for the fused track this detection updates; see the Kafka sink.
            **record.get("_lineage", {}),
        }
        return clean_det.timestamp, clean_det.sensor_id, detection

    def iter_events(self):
        key, frame = None, []
        while True:
            self._poll()
            while (record := self._next_in_order()) is not None:
                made = self._detection(record)
                if made is None:
                    continue
                timestamp, sensor_id, detection = made
                if frame and (timestamp, sensor_id) != key:
                    yield key[0], frame
                    frame = []
                key = (timestamp, sensor_id)
                frame.append(detection)
            # Nothing more can be ordered right now. If that is only because a fetch
            # ended mid-backlog, the rest of the open sensor-instant may be in the
            # next one, so it is held; once every partition is caught up nothing
            # already written can still belong to it.
            if frame and self._caught_up():
                yield key[0], frame
                key, frame = None, []

    def _caught_up(self):
        """Every assigned partition is read to its end and fully merged."""
        return all(
            not buffer and partition in self._at_end for partition, buffer in self._buffers.items()
        )

    def close(self):
        self.consumer.close()
