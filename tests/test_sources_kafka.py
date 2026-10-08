"""Tests for context_foundry.fusion.sources.kafka (KafkaSapientSource).

We never reach a broker: the consumer is a fake whose consume() hands out
scripted batches, terminated by a sentinel BaseException that breaks the
infinite iter_events() loop. Records are sapient-raw as sapient-ingest writes
them: proto field names, the decoded message carried whole.
"""

import json

import pytest
from confluent_kafka import KafkaError

from context_foundry.fusion import config
from context_foundry.fusion.sources.kafka import KafkaSapientSource

NODE_A = "FI-MIL-RAD-KOLI-01"
NODE_B = "acoustic_array_01"


@pytest.fixture(autouse=True)
def _sensor_network():
    config.load_sensor_network(
        sensor_network_list=[{"id": NODE_A, "lat": 62.9, "lon": 29.8, "alt": 100.0}]
    )


class _Stop(BaseException):
    """Sentinel used to break the infinite iter_events() loop from within a test."""


class _Error:
    def __init__(self, code):
        self._code = code

    def code(self):
        return self._code


class _Msg:
    def __init__(self, partition, value=None, eof=False, headers=None):
        self._partition = partition
        self._value = value
        self._headers = headers
        self._error = _Error(KafkaError._PARTITION_EOF) if eof else None

    def partition(self):
        return self._partition

    def value(self):
        return self._value

    def error(self):
        return self._error

    def headers(self):
        return self._headers


class _TP:
    def __init__(self, partition):
        self.partition = partition


class _FakeConsumer:
    def __init__(self, batches, partitions=(0, 1)):
        self._batches = iter(batches)
        self._partitions = partitions
        self.closed = False

    def subscribe(self, topics, on_assign, on_revoke):
        on_assign(self, [_TP(p) for p in self._partitions])

    def consume(self, num_messages, timeout):
        batch = next(self._batches)
        if isinstance(batch, BaseException):
            raise batch
        return batch

    def pause(self, partitions):
        pass

    def resume(self, partitions):
        pass

    def close(self):
        self.closed = True


def _record(ts, node=NODE_A, object_id="obj-1", content_type="detection_report"):
    message = {"timestamp": ts, "node_id": node}
    if content_type == "detection_report":
        message["detection_report"] = {
            "object_id": object_id,
            "location": {
                "x": 29.8,
                "y": 62.9,
                "z": 100.0,
                "coordinate_system": "LOCATION_COORDINATE_SYSTEM_LAT_LNG_DEG_M",
                "datum": "LOCATION_DATUM_WGS84_E",
            },
            "object_info": [{"type": "estimatedSwarmCount", "value": "12"}],
            "classification": [{"type": "Air vehicle", "confidence": 0.6}],
        }
    else:
        message[content_type] = {}
    record = {"event_time": ts, "content_type": content_type, "node_id": node, "message": message}
    return json.dumps(record).encode()


def _ts(second):
    return f"2026-11-15T03:00:{second:02d}Z"


def _source(batches, partitions=(0, 1)):
    return KafkaSapientSource(
        "unused:9092", "topic", "group", consumer=_FakeConsumer(batches, partitions)
    )


def _drain(source):
    events = []
    gen = source.iter_events()
    with pytest.raises(_Stop):
        while True:
            events.append(next(gen))
    return events


def test_merges_partitions_in_event_time_order():
    # Partition 0 arrives whole before partition 1, as a backlog does.
    batches = [
        [_Msg(0, _record(_ts(2))), _Msg(0, _record(_ts(4))), _Msg(0, eof=True)],
        [_Msg(1, _record(_ts(1), NODE_B)), _Msg(1, _record(_ts(3), NODE_B)), _Msg(1, eof=True)],
        _Stop(),
    ]
    events = _drain(_source(batches))
    assert [ts.second for ts, _ in events] == [1, 2, 3, 4]


def test_waits_for_a_partition_that_has_not_reached_its_end():
    batches = [[_Msg(0, _record(_ts(5)))], _Stop()]
    assert _drain(_source(batches)) == []


def test_a_partition_at_its_end_does_not_hold_the_merge_back():
    batches = [[_Msg(0, _record(_ts(5))), _Msg(0, _record(_ts(6))), _Msg(1, eof=True)], _Stop()]
    events = _drain(_source(batches))
    assert [ts.second for ts, _ in events] == [5]


def test_a_sensor_instant_split_across_fetches_stays_one_event():
    # The first fetch ends mid-instant, before the partition is read to its end.
    batches = [
        [_Msg(0, _record(_ts(1), object_id="a"))],
        [_Msg(0, _record(_ts(1), object_id="b")), _Msg(0, eof=True)],
        _Stop(),
    ]
    events = _drain(_source(batches, partitions=(0,)))
    assert [len(dets) for _, dets in events] == [2]


def test_one_sensor_instant_is_one_event():
    batches = [
        [
            _Msg(0, _record(_ts(1), object_id="a")),
            _Msg(0, _record(_ts(1), object_id="b")),
            _Msg(0, _record(_ts(2), object_id="a")),
            _Msg(0, eof=True),
        ],
        _Stop(),
    ]
    events = _drain(_source(batches, partitions=(0,)))
    assert [len(dets) for _, dets in events] == [2, 1]
    ids = [d.metadata["objectId"] for d in events[0][1]]
    assert ids == ["a", "b"]


def test_detection_metadata_comes_from_the_record():
    batches = [[_Msg(0, _record(_ts(1))), _Msg(0, eof=True)], _Stop()]
    ((_, [det]),) = _drain(_source(batches, partitions=(0,)))
    assert det.metadata["nodeId"] == NODE_A
    assert det.metadata["swarm_count"] == 12
    assert det.metadata["classification"] == "Air vehicle"


def test_skips_registrations_and_records_that_are_not_sapient_raw(caplog):
    batches = [
        [
            _Msg(0, _record(_ts(1), content_type="registration")),
            _Msg(0, b"not json"),
            _Msg(0, _record(_ts(2))),
            _Msg(0, eof=True),
        ],
        _Stop(),
    ]
    source = _source(batches, partitions=(0,))
    with caplog.at_level("WARNING"):
        events = _drain(source)
    assert len(events) == 1
    assert source.records_read == 3
    assert source.records_rejected == 1
    assert any("not sapient-raw" in r.message for r in caplog.records)


def test_close_closes_the_consumer():
    source = _source([_Stop()])
    source.close()
    assert source.consumer.closed


def test_a_registration_record_sets_the_sensors_accuracy_and_tracking():
    registration = {
        "event_time": _ts(0),
        "content_type": "registration",
        "node_id": NODE_A,
        "message": {
            "timestamp": _ts(0),
            "node_id": NODE_A,
            "registration": {
                "mode_definition": [
                    {
                        "tracking_type": "TRACKING_TYPE_TRACK",
                        "detection_definition": [
                            {
                                "geometric_error": {
                                    "variation_type": "Constant",
                                    "performance_value": [{"type": "Base", "unit_value": "8"}],
                                }
                            }
                        ],
                    }
                ]
            },
        },
    }
    batches = [
        [_Msg(0, json.dumps(registration).encode()), _Msg(0, _record(_ts(1))), _Msg(0, eof=True)],
        _Stop(),
    ]
    ((_, [det]),) = _drain(_source(batches, partitions=(0,)))
    assert det.metadata["stable_object_id"] is True
    assert det.measurement_model.noise_covar[0, 0] == pytest.approx(64.0)
