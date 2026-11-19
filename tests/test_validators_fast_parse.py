# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

"""parse_dict gives what ParseDict gives: the same message, or the same ParseError."""

import copy

import pytest
from google.protobuf.json_format import ParseDict, ParseError

from context_foundry.fusion.validators import fast_parse
from sapient_msg.bsi_flex_335_v2_0.sapient_message_pb2 import SapientMessage


def _report(**detection):
    report = {
        "objectId": "RAD-01",
        "location": {
            "x": 30.63,
            "y": 62.18,
            "z": 1578.0,
            "coordinateSystem": "LOCATION_COORDINATE_SYSTEM_LAT_LNG_DEG_M",
            "datum": "LOCATION_DATUM_WGS84_E",
        },
        "objectInfo": [{"type": "estimatedSwarmCount", "value": "12"}],
        "classification": [
            {
                "type": "Air vehicle",
                "confidence": 0.63,
                "subClass": [{"type": "UAV", "confidence": 0.5, "level": 1}],
            }
        ],
    }
    report.update(detection)
    return {
        "timestamp": "2026-11-15T02:45:00.000018Z",
        "nodeId": "FI-MIL-RAD-KOLI-01",
        "detectionReport": report,
    }


def _outcome(read, payload):
    try:
        msg = SapientMessage()
        read(copy.deepcopy(payload), msg)
        return "message", msg.SerializeToString(deterministic=True)
    except ParseError as e:
        return "refused", str(e)


def _slow(payload, msg):
    ParseDict(payload, msg, ignore_unknown_fields=False)


@pytest.fixture
def fallbacks(monkeypatch):
    """How many payloads parse_dict handed to ParseDict."""
    calls = []
    real = fast_parse.ParseDict
    monkeypatch.setattr(fast_parse, "ParseDict", lambda *a, **k: calls.append(1) or real(*a, **k))
    return calls


def _snake(value):
    names = {
        "objectId": "object_id",
        "nodeId": "node_id",
        "detectionReport": "detection_report",
        "coordinateSystem": "coordinate_system",
        "objectInfo": "object_info",
        "subClass": "sub_class",
    }
    if isinstance(value, dict):
        return {names.get(k, k): _snake(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_snake(v) for v in value]
    return value


FAST = {
    "json names": _report(),
    "proto names": _snake(_report()),
    "an integer for a float": _report(location={"x": 30, "y": 62, "z": 0}),
    "an empty message": _report(location={}),
    "an empty list": _report(objectInfo=[]),
    "a few fields": {"detectionReport": {"objectId": "a"}, "nodeId": "n"},
    "a bool and repeated strings": {
        "nodeId": "n",
        "registrationAck": {"acceptance": True, "ackResponseReason": ["a", "b"]},
    },
}

SLOW = {
    "an unknown field": _report(colourOfSky="blue"),
    "a null": _report(colour=None),
    "a field twice": _report(object_id="again"),
    "two of one oneof": {**_report(), "registration": {}},
    "an enum by number": _report(location={"x": 1.0, "y": 1.0, "coordinateSystem": 2}),
    "an unknown enum name": _report(location={"x": 1.0, "y": 1.0, "datum": "NOT_A_DATUM"}),
    "an integer as a string": _report(
        classification=[{"type": "a", "subClass": [{"type": "b", "level": "2"}]}]
    ),
    "a float as a string": _report(detectionConfidence="0.5"),
    "NaN": _report(detectionConfidence="NaN"),
    "a bool for a float": _report(detectionConfidence=True),
    "a float32 out of range": _report(detectionConfidence=1e39),
    "a float for an integer": _report(
        classification=[{"type": "a", "subClass": [{"type": "b", "level": 2.5}]}]
    ),
    "a bool for an integer": _report(
        classification=[{"type": "a", "subClass": [{"type": "b", "level": True}]}]
    ),
    "a number for a string": _report(objectId=7),
    "a lone surrogate": _report(objectId="a\ud800b"),
    "a list that is not": _report(objectInfo={"type": "a"}),
    "a message that is not": _report(location=[1, 2]),
    "a bad timestamp": {**_report(), "timestamp": "yesterday"},
    "a timestamp that is not a string": {**_report(), "timestamp": 12},
    "not an object": ["detectionReport"],
    "a NaN number": _report(detectionConfidence=float("nan")),
    "an infinite number": _report(detectionConfidence=float("inf")),
    "a string for a bool": {"nodeId": "n", "registrationAck": {"acceptance": "true"}},
    "repeated strings not in a list": {"registrationAck": {"ackResponseReason": "a"}},
    "a number among repeated strings": {"registrationAck": {"ackResponseReason": ["a", 1]}},
}


@pytest.mark.parametrize("payload", FAST.values(), ids=FAST.keys())
def test_plain_shapes_are_read_without_parsedict_and_read_the_same(payload, fallbacks):
    assert _outcome(fast_parse.parse_dict, payload) == _outcome(_slow, payload)
    assert fallbacks == []


@pytest.mark.parametrize("payload", SLOW.values(), ids=SLOW.keys())
def test_anything_else_is_parsedicts_to_read_and_ends_as_parsedict_ends(payload, fallbacks):
    assert _outcome(fast_parse.parse_dict, payload) == _outcome(_slow, payload)
    assert fallbacks == [1]


def test_a_fallback_starts_from_an_empty_message():
    # The fast path has already filled nodeId when it meets the unknown field; ParseDict
    # must not find it there.
    payload = {"nodeId": "n", "detectionReport": {"objectId": "a", "nope": 1}}
    msg = SapientMessage()
    with pytest.raises(ParseError):
        fast_parse.parse_dict(payload, msg)
    assert _outcome(fast_parse.parse_dict, {**payload, "detectionReport": {"objectId": "a"}}) == (
        _outcome(_slow, {**payload, "detectionReport": {"objectId": "a"}})
    )


def test_field_plans_are_made_once_per_message_type():
    fast_parse.parse_dict(_report(), SapientMessage())
    made = dict(fast_parse._PLANS)
    fast_parse.parse_dict(_report(), SapientMessage())
    assert made == fast_parse._PLANS
    assert all(fast_parse._PLANS[d] is made[d] for d in made)


TIMESTAMPS = [
    "2026-11-15T02:45:00Z",
    "2026-11-15T02:45:00.1Z",
    "2026-11-15T02:45:00.000018Z",
    "2026-11-15T02:45:00.123456789Z",
    "2026-10-23T13:52:11.269699800Z",
    "1970-01-01T00:00:00Z",
    "0001-01-01T00:00:00Z",
    "9999-12-31T23:59:59.999999999Z",
    "2024-02-29T12:00:00.5Z",
]
ODD_TIMESTAMPS = [
    "2026-11-15T02:45:00+02:00",
    "2026-11-15T02:45:00.5-01:30",
    "2026-11-15t02:45:00Z",
    "2026-2-5T02:45:00Z",
    "2026-02-30T02:45:00Z",
    "2026-11-15T24:00:00Z",
    "2026-11-15T02:45:60Z",
    "2026-11-15T02:45:00.1234567891Z",
    "2026-11-15T02:45:00Zjunk",
    "2026-11-15 02:45:00Z",
    "yesterday",
]


@pytest.mark.parametrize("stamp", TIMESTAMPS)
def test_a_utc_timestamp_is_read_by_the_fast_path_as_protobuf_reads_it(stamp, fallbacks):
    payload = {**_report(), "timestamp": stamp}
    assert _outcome(fast_parse.parse_dict, payload) == _outcome(_slow, payload)
    assert fallbacks == []


@pytest.mark.parametrize("stamp", ODD_TIMESTAMPS)
def test_any_other_timestamp_ends_as_protobuf_ends_it(stamp):
    payload = {**_report(), "timestamp": stamp}
    assert _outcome(fast_parse.parse_dict, payload) == _outcome(_slow, payload)


def test_nanoseconds_are_rounded_as_protobuf_rounds_them():
    import random

    from google.protobuf.timestamp_pb2 import Timestamp

    rng = random.Random(3)
    for _ in range(20000):
        digits = "".join(rng.choice("0123456789") for _ in range(rng.randint(1, 9)))
        stamp = f"2026-11-15T02:45:00.{digits}Z"
        mine, theirs = Timestamp(), Timestamp()
        fast_parse._timestamp(mine, stamp)
        theirs.FromJsonString(stamp)
        assert (mine.seconds, mine.nanos) == (theirs.seconds, theirs.nanos), stamp


def _outcome_of(kind, read, payload):
    try:
        msg = kind()
        read(copy.deepcopy(payload), msg)
        return "message", msg.SerializeToString(deterministic=True)
    except ParseError as e:
        return "refused", str(e)


@pytest.mark.parametrize(
    "kind, payload",
    [
        ("UninterpretedOption", {"stringValue": "aGk="}),
        ("UninterpretedOption", {"identifierValue": "x", "positiveIntValue": "7"}),
        ("Struct", {"fields": {"a": 1}}),
        ("Struct", {"a": [1, "b", None]}),
        (
            "Api",
            {
                "name": "n",
                "options": [
                    {"name": "o", "value": {"@type": "type.googleapis.com/google.protobuf.Empty"}}
                ],
            },
        ),
        ("Api", {"name": "n", "sourceContext": {"fileName": "f"}}),
    ],
)
def test_bytes_maps_and_other_well_known_types_end_as_parsedict_ends_them(kind, payload, fallbacks):
    from google.protobuf import api_pb2, descriptor_pb2, struct_pb2

    cls = {
        "UninterpretedOption": descriptor_pb2.UninterpretedOption,
        "Struct": struct_pb2.Struct,
        "Api": api_pb2.Api,
    }[kind]
    assert _outcome_of(cls, fast_parse.parse_dict, payload) == _outcome_of(
        cls, lambda p, m: ParseDict(p, m, ignore_unknown_fields=False), payload
    )
    assert fallbacks == [1]
