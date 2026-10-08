"""Tests for context_foundry.fusion.sinks.kafka (KafkaSapientRawSink).

The producer is a fake that records what it is handed and serves delivery
callbacks on poll()/flush(), so nothing here reaches a broker.
"""

import json
import re
from datetime import datetime, timezone

from context_foundry.fusion.sapient_raw import fusion_node_id
from context_foundry.fusion.schemas import TacticalTrack
from context_foundry.fusion.sinks.kafka import KafkaSapientRawSink

HEADERS = {
    "message_id",
    "trace_id",
    "event_time",
    "ingest_time",
    "pipeline_id",
    "producer_stage",
    "schema",
}


class _FakeProducer:
    def __init__(self, fail=False, left=0):
        self.produced = []
        self._pending = []
        self._fail = fail
        self._left = left

    def produce(self, topic, value, key, headers, on_delivery):
        self.produced.append({"topic": topic, "value": value, "key": key, "headers": headers})
        self._pending.append(on_delivery)

    def poll(self, _timeout):
        while self._pending:
            self._pending.pop(0)("broker down" if self._fail else None, None)

    def flush(self, _timeout):
        self.poll(0)
        return self._left


def _track():
    return TacticalTrack(
        track_id="64578864-2254-4a79-b148-e12e1c06ffd8",
        timestamp=datetime(2026, 11, 15, 3, 2, 23, tzinfo=timezone.utc),
        latitude=62.7,
        longitude=30.3,
        altitude=200.0,
        classification="Air vehicle",
        swarm_count=4,
        threat_level="hostile",
    )


def _sink(producer):
    return KafkaSapientRawSink("unused:9092", "fused", "p-00000000a1", "fusion", producer=producer)


def test_publishes_one_record_with_the_seven_headers():
    producer = _FakeProducer()
    sink = _sink(producer)
    sink.send_track(_track(), {"trace_id": "ab" * 16, "ingest_time": "2026-10-06T13:06:21.359Z"})

    (sent,) = producer.produced
    headers = {name: value.decode() for name, value in sent["headers"]}
    assert set(headers) == HEADERS
    assert headers["trace_id"] == "ab" * 16
    assert headers["ingest_time"] == "2026-10-06T13:06:21.359Z"
    assert headers["pipeline_id"] == "p-00000000a1"
    assert headers["producer_stage"] == "fusion"
    assert headers["schema"] == "sapient-raw:1.0"
    record = json.loads(sent["value"])
    assert headers["event_time"] == record["event_time"]
    assert sent["key"] == fusion_node_id("p-00000000a1", "fusion").encode()
    assert sent["topic"] == "fused"
    assert sink.sent == 1


def test_mints_trace_id_and_ingest_time_without_lineage():
    producer = _FakeProducer()
    _sink(producer).send_track(_track())
    headers = {name: value.decode() for name, value in producer.produced[0]["headers"]}
    assert re.fullmatch(r"[0-9a-f]{32}", headers["trace_id"])
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", headers["ingest_time"])


def test_each_record_gets_a_new_message_id():
    producer = _FakeProducer()
    sink = _sink(producer)
    sink.send_track(_track())
    sink.send_track(_track())
    ids = [dict(p["headers"])["message_id"] for p in producer.produced]
    assert ids[0] != ids[1]


def test_failed_delivery_is_counted_and_logged(caplog):
    sink = _sink(_FakeProducer(fail=True))
    with caplog.at_level("WARNING"):
        sink.send_track(_track())
    assert sink.failed == 1
    assert any("not delivered" in r.message for r in caplog.records)


def test_close_flushes_and_reports_what_was_left(caplog):
    sink = _sink(_FakeProducer(left=2))
    with caplog.at_level("WARNING"):
        sink.close()
    assert any("2 fused track(s) not delivered" in r.message for r in caplog.records)


def test_predicted_label_reaches_the_record():
    producer = _FakeProducer()
    _sink(producer).send_track(_track(), predicted=True)
    report = json.loads(producer.produced[0]["value"])["message"]["detection_report"]
    assert {"type": "fusionUpdate", "value": "predicted"} in report["object_info"]
