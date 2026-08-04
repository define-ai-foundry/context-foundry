"""Tests for context_foundry.fusion.sources.base."""

import pytest

from context_foundry.fusion.sources import base
from context_foundry.fusion.sources.base import SapientSource


def test_sapient_source_is_abstract():
    with pytest.raises(TypeError):
        SapientSource()


def test_sapient_source_subclass_must_implement_iter_events():
    class Incomplete(SapientSource):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_base_iter_events_raises_not_implemented_when_invoked_via_super():
    class Passthrough(SapientSource):
        def iter_events(self):
            return super().iter_events()

    with pytest.raises(NotImplementedError):
        Passthrough().iter_events()


def test_swarm_count_reads_the_reported_value():
    report = {"objectInfo": [{"type": "estimatedSwarmCount", "value": "12"}]}
    assert base.swarm_count(report) == 12


def test_swarm_count_defaults_to_one_when_absent():
    assert base.swarm_count({}) == 1
    assert base.swarm_count({"objectInfo": [{"type": "somethingElse", "value": "7"}]}) == 1


def test_swarm_count_accepts_a_decimal_value():
    """A swarm of 12 reported as "12.0" must not read as 1 -- that flips the
    track's threat level from hostile to suspect."""
    report = {"objectInfo": [{"type": "estimatedSwarmCount", "value": "12.0"}]}
    assert base.swarm_count(report) == 12


def test_swarm_count_survives_an_unparseable_value(caplog):
    """objectInfo values are free text on the wire; one bad sensor must not take
    the ingest path down."""
    report = {"objectInfo": [{"type": "estimatedSwarmCount", "value": "many"}]}

    with caplog.at_level("WARNING"):
        assert base.swarm_count(report) == 1

    assert any("Unparseable estimatedSwarmCount" in r.message for r in caplog.records)
