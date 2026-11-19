# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

"""JSON to protobuf for the plain shapes SAPIENT reports take, as `ParseDict` would.

`ParseDict` is pure Python and checks every value through several layers; on a busy feed
it is most of what reading a report costs. `parse_dict` fills the message straight from
the descriptor for the shapes it knows: fields by their proto or JSON name, nested
messages, repeated fields, enums by name, strings, bools, integers given as integers,
floats given as numbers, and `Timestamp` in UTC with Z. Anything else, and anything `ParseDict` might
refuse or read differently (an unknown field, a null, a field given twice, two members of
one oneof, a number where a string belongs, a map, bytes, another well-known type, a
lone surrogate, a float out of range), sends the whole payload to `ParseDict` instead, so
the message, or the `ParseError`, is always the one `ParseDict` gives.
"""

from __future__ import annotations

import datetime
import math
import re
from typing import Any

from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.json_format import ParseDict
from google.protobuf.message import Message

_INT = frozenset(
    {
        FieldDescriptor.CPPTYPE_INT32,
        FieldDescriptor.CPPTYPE_INT64,
        FieldDescriptor.CPPTYPE_UINT32,
        FieldDescriptor.CPPTYPE_UINT64,
    }
)
# The largest finite float32, the bound ParseDict holds a float field to.
_FLOAT32_MAX = 3.4028234663852886e38
# A lone surrogate, which ParseDict refuses in a string.
_SURROGATE = re.compile("[\ud800-\udfff]")
_TIMESTAMP = "google.protobuf.Timestamp"


class _Unsure(Exception):
    """A value this reader does not vouch for: ParseDict reads the payload instead."""


# What each field of a message type takes, made once per type: by proto and JSON name,
# (field, attribute, oneof, kind, convert), so a value is placed without asking the
# descriptor again.
_PLANS: dict[Any, dict[str, tuple]] = {}
_MESSAGE, _REPEATED_MESSAGE, _TIMESTAMP_FIELD, _SCALAR, _REPEATED_SCALAR, _UNSURE = range(6)


def parse_dict(value: Any, message: Message) -> Message:
    """`ParseDict(value, message)` with unknown fields refused, faster for plain shapes.

    `message` must be empty. On anything the fast path is not sure of, it is cleared and
    `ParseDict` reads `value` into it, so the result or the `ParseError` raised is
    `ParseDict`'s.
    """
    try:
        _fill(message, value)
    except (_Unsure, TypeError, ValueError, OverflowError):
        message.Clear()
        ParseDict(value, message, ignore_unknown_fields=False)
    return message


def _plan(descriptor) -> dict[str, tuple]:
    found = _PLANS.get(descriptor)
    if found is None:
        found = {}
        for field in descriptor.fields:
            kind, convert = _UNSURE, None
            if field.cpp_type == FieldDescriptor.CPPTYPE_MESSAGE:
                sub = field.message_type
                if sub.GetOptions().map_entry or sub.full_name.startswith("google.protobuf."):
                    if not field.is_repeated and sub.full_name == _TIMESTAMP:
                        kind = _TIMESTAMP_FIELD
                else:
                    kind = _REPEATED_MESSAGE if field.is_repeated else _MESSAGE
            else:
                convert = _converter(field)
                if convert is not None:
                    kind = _REPEATED_SCALAR if field.is_repeated else _SCALAR
            step = (field, field.name, field.containing_oneof, kind, convert)
            found[field.name] = step
            found[field.json_name] = step
        _PLANS[descriptor] = found
    return found


def _fill(message: Message, value: Any) -> None:
    if not isinstance(value, dict):
        raise _Unsure
    plan = _plan(message.DESCRIPTOR)
    taken: set = set()
    for name, item in value.items():
        step = plan.get(name)
        if step is None or item is None:
            raise _Unsure
        field, attr, oneof, kind, convert = step
        if field in taken:
            raise _Unsure
        taken.add(field)
        if oneof is not None:
            if oneof in taken:
                raise _Unsure
            taken.add(oneof)
        if kind == _SCALAR:
            setattr(message, attr, convert(item))
        elif kind == _MESSAGE:
            sub = getattr(message, attr)
            sub.SetInParent()
            _fill(sub, item)
        elif kind == _REPEATED_MESSAGE:
            if not isinstance(item, list):
                raise _Unsure
            target = getattr(message, attr)
            for entry in item:
                _fill(target.add(), entry)
        elif kind == _REPEATED_SCALAR:
            if not isinstance(item, list):
                raise _Unsure
            getattr(message, attr).extend([convert(entry) for entry in item])
        elif kind == _TIMESTAMP_FIELD:
            if not isinstance(item, str):
                raise _Unsure
            _timestamp(getattr(message, attr), item)
        else:
            raise _Unsure


# The one timestamp form read here: UTC with Z and up to nine fractional digits. Anything
# else goes to protobuf's own FromJsonString.
_TS = re.compile(r"(\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d):(\d\d)(?:\.(\d{1,9}))?Z")
_EPOCH = datetime.datetime(1970, 1, 1)


def _timestamp(message: Message, text: str) -> None:
    """`message.FromJsonString(text)`, by its own arithmetic, for the plain UTC form."""
    match = _TS.fullmatch(text)
    if match is None:
        message.FromJsonString(text)
        return
    year, month, day, hour, minute, second, fraction = match.groups()
    try:
        moment = datetime.datetime(
            int(year), int(month), int(day), int(hour), int(minute), int(second)
        )
    except ValueError as e:
        raise _Unsure from e
    span = moment - _EPOCH
    message.seconds = span.seconds + span.days * 86400
    # As protobuf rounds them, not as the digits read: the same nanos, whatever they are.
    message.nanos = round(float("0." + fraction) * 1e9) if fraction else 0


def _converter(field):
    """How a scalar field's JSON value becomes its Python value, or None if not read here."""
    kind = field.cpp_type
    if kind == FieldDescriptor.CPPTYPE_STRING:
        return None if field.type == FieldDescriptor.TYPE_BYTES else _string
    if kind == FieldDescriptor.CPPTYPE_FLOAT:
        return _float32
    if kind == FieldDescriptor.CPPTYPE_DOUBLE:
        return _float64
    if kind in _INT:
        return _integer
    if kind == FieldDescriptor.CPPTYPE_BOOL:
        return _boolean
    names = field.enum_type.values_by_name

    def enum(value):
        if not isinstance(value, str):
            raise _Unsure
        known = names.get(value)
        if known is None:
            raise _Unsure
        return known.number

    return enum


def _string(value):
    if not isinstance(value, str) or _SURROGATE.search(value):
        raise _Unsure
    return value


def _float64(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _Unsure
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        raise _Unsure
    return value


def _float32(value):
    value = _float64(value)
    if abs(value) > _FLOAT32_MAX:
        raise _Unsure
    return value


def _integer(value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise _Unsure
    return value


def _boolean(value):
    if not isinstance(value, bool):
        raise _Unsure
    return value
