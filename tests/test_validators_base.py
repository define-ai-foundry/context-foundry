"""Tests for context_foundry.fusion.validators.base."""

import logging

import pytest

from context_foundry.fusion.schemas import InternalDetection
from context_foundry.fusion.validators.base import ProtocolValidator


def test_protocol_validator_is_abstract():
    with pytest.raises(TypeError):
        ProtocolValidator()


class _AlwaysRejects(ProtocolValidator):
    def validate(self, raw_payload):
        return False, "nope"

    def normalize(self, raw_payload):  # pragma: no cover - never reached
        raise AssertionError("normalize should not be called when invalid")


class _NormalizeBlowsUp(ProtocolValidator):
    def validate(self, raw_payload):
        return True, ""

    def normalize(self, raw_payload):
        raise ValueError("boom")


class _HappyPath(ProtocolValidator):
    def validate(self, raw_payload):
        return True, ""

    def normalize(self, raw_payload):
        return InternalDetection(
            sensor_id="s1",
            timestamp="2026-01-01T00:00:00Z",
            latitude=1.0,
            longitude=2.0,
        )


def test_process_message_returns_none_and_logs_on_invalid(caplog):
    with caplog.at_level(logging.WARNING):
        result = _AlwaysRejects().process_message({})
    assert result is None
    assert any("rejected" in r.message for r in caplog.records)


def test_process_message_returns_none_and_logs_on_normalize_failure(caplog):
    with caplog.at_level(logging.ERROR):
        result = _NormalizeBlowsUp().process_message({})
    assert result is None
    assert any("Normalization failed" in r.message for r in caplog.records)


def test_process_message_returns_normalized_detection_on_success():
    result = _HappyPath().process_message({})
    assert isinstance(result, InternalDetection)
    assert result.sensor_id == "s1"


def test_abstract_validate_and_normalize_bodies_are_reachable_via_super():
    """validate()/normalize() are abstractmethods with a bare `pass` body; the
    only way to execute that body at all is a concrete override that still
    calls super()."""

    class CallsSuper(ProtocolValidator):
        def validate(self, raw_payload):
            return super().validate(raw_payload)

        def normalize(self, raw_payload):
            return super().normalize(raw_payload)

    validator = CallsSuper()
    assert validator.validate({}) is None
    assert validator.normalize({}) is None
