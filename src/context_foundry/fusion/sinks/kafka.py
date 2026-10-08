# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# src/context_foundry/fusion/sinks/kafka.py

import logging
import uuid
from datetime import datetime, timezone

from ..sapient_raw import SCHEMA, fusion_node_id, track_record, uuid7
from ..schemas import TacticalTrack

logger = logging.getLogger(__name__)


def _now_rfc3339() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class KafkaSapientRawSink:
    """
    Pipeline egress adapter. Publishes each fused track as a sapient-raw record
    to a Kafka topic, keyed by the fusion stage's node_id, with the seven platform
    headers.

    A fused track is a derived record, so it gets a new message_id; trace_id and
    ingest_time are carried forward from the detection that triggered the update
    (handed in as `lineage`), and minted only when there is none, as for a replay.
    """

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        pipeline_id: str,
        stage: str,
        security_protocol: str = "PLAINTEXT",
        producer=None,
    ):
        self.topic = topic
        self.pipeline_id = pipeline_id
        self.stage = stage
        self.node_id = fusion_node_id(pipeline_id, stage)
        self.sent = 0
        self.failed = 0

        if producer is None:
            # Imported here so the rest of the engine runs without the Kafka client.
            from confluent_kafka import Producer

            producer = Producer(
                {
                    "bootstrap.servers": bootstrap_servers,
                    "security.protocol": security_protocol,
                    "acks": "all",
                    "enable.idempotence": True,
                    "partitioner": "murmur2_random",
                }
            )
        self.producer = producer
        logger.info(f"Publishing fused tracks as sapient-raw to {topic} as node {self.node_id}")

    def _delivered(self, error, _msg):
        if error is not None:
            self.failed += 1
            logger.warning("A fused track was not delivered to %s: %s", self.topic, error)
        else:
            self.sent += 1

    def send_track(
        self,
        track: TacticalTrack,
        lineage: dict | None = None,
        predicted: bool | None = None,
        sources: list[dict] | None = None,
    ):
        lineage = lineage or {}
        key, value, event_time = track_record(
            track, self.node_id, predicted=predicted, sources=sources
        )
        headers = {
            "message_id": uuid7(),
            "trace_id": lineage.get("trace_id") or uuid.uuid4().hex,
            "event_time": event_time,
            "ingest_time": lineage.get("ingest_time") or _now_rfc3339(),
            "pipeline_id": self.pipeline_id,
            "producer_stage": self.stage,
            "schema": SCHEMA,
        }
        self.producer.produce(
            self.topic,
            value=value,
            key=key,
            headers=[(name, text.encode("utf-8")) for name, text in headers.items()],
            on_delivery=self._delivered,
        )
        # Serve delivery callbacks without blocking the fusion loop.
        self.producer.poll(0)

    def close(self):
        left = self.producer.flush(10)
        if left:
            logger.warning("%d fused track(s) not delivered to %s at shutdown", left, self.topic)
        logger.info("Published %d fused track(s) to %s", self.sent, self.topic)
