"""Tests for the CotSink abstract base.

CotSink cannot be instantiated directly; a concrete subclass need only implement
send(), and inherits a no-op close().
"""

import pytest

from context_foundry.fusion.sinks.base import CotSink


def test_cot_sink_cannot_be_instantiated():
    with pytest.raises(TypeError):
        CotSink()


def test_concrete_subclass_only_needs_send_and_inherits_noop_close():
    sent = []

    class _Sink(CotSink):
        def send(self, cot_payload):
            sent.append(cot_payload)

    sink = _Sink()
    sink.send("<event/>")
    # Inherited default close() is a no-op and must not raise.
    sink.close()

    assert sent == ["<event/>"]
