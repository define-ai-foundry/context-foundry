"""Tests for context_foundry.fusion.serializers."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from lxml import etree

from context_foundry.fusion.schemas import TacticalTrack
from context_foundry.fusion.serializers import BaseSerializer, CotSerializer
from context_foundry.fusion.validators.cot import CotValidator

REPO_ROOT = Path(__file__).resolve().parents[1]
XSD_PATH = REPO_ROOT / "protos" / "cot" / "CoT Base-Event Schema  (PUBLIC RELEASE).xsd"


def _track(**overrides):
    defaults = {
        "track_id": "abc-123",
        "timestamp": datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        "latitude": 62.9,
        "longitude": 29.8,
        "altitude": 250.0,
        "speed_mps": 5.5,
        "heading_deg": 90.0,
        "classification": "UAS",
        "swarm_count": 2,
        "threat_level": "suspect",
    }
    defaults.update(overrides)
    return TacticalTrack(**defaults)


def test_base_serializer_is_abstract():
    with pytest.raises(TypeError):
        BaseSerializer()


def test_base_serializer_abstract_body_is_reachable_via_super():
    class CallsSuper(BaseSerializer):
        def serialize(self, state, node_id="FUSION-NODE"):
            return super().serialize(state, node_id)

    assert CallsSuper().serialize(_track()) is None


def test_cot_serializer_produces_expected_fields():
    xml_str = CotSerializer().serialize(_track())
    root = etree.fromstring(xml_str.encode("utf-8"))

    assert root.get("uid") == "TRK-abc-123"
    assert root.get("type") == "a-s-A-M-F"  # suspect -> "s"
    assert root.get("version") == "2.0"
    assert root.get("how") == "m-g"

    point = root.find("point")
    assert point.get("lat") == "62.900000"
    assert point.get("lon") == "29.800000"
    assert point.get("hae") == "250.0"

    track_el = root.find("detail/track")
    assert track_el.get("speed") == "5.50"
    assert track_el.get("course") == "90.0"

    contact = root.find("detail/contact")
    assert contact.get("callsign") == "SWM(2) UAS"


def test_cot_serializer_stale_defaults_to_fifteen_seconds_after_the_event():
    root = etree.fromstring(CotSerializer().serialize(_track()).encode("utf-8"))

    assert root.get("time") == "2026-01-01T12:00:00.000Z"
    assert root.get("start") == "2026-01-01T12:00:00.000Z"
    assert root.get("stale") == "2026-01-01T12:00:15.000Z"


def test_cot_serializer_stale_window_is_configurable():
    root = etree.fromstring(CotSerializer(stale_seconds=90).serialize(_track()).encode("utf-8"))

    assert root.get("stale") == "2026-01-01T12:01:30.000Z"


def test_cot_serializer_converts_a_non_utc_timestamp_rather_than_relabelling_it():
    """The `Z` suffix is asserted, so the value has to actually be UTC."""
    helsinki = timezone(timedelta(hours=3))
    track = _track(timestamp=datetime(2026, 1, 1, 15, 0, 0, tzinfo=helsinki))

    root = etree.fromstring(CotSerializer().serialize(track).encode("utf-8"))

    assert root.get("time") == "2026-01-01T12:00:00.000Z"
    assert root.get("stale") == "2026-01-01T12:00:15.000Z"


def test_cot_serializer_reads_a_naive_timestamp_as_utc():
    track = _track(timestamp=datetime(2026, 1, 1, 12, 0, 0))

    root = etree.fromstring(CotSerializer().serialize(track).encode("utf-8"))

    assert root.get("time") == "2026-01-01T12:00:00.000Z"


def test_cot_serializer_hostile_identity():
    xml_str = CotSerializer().serialize(_track(threat_level="hostile"))
    root = etree.fromstring(xml_str.encode("utf-8"))
    assert root.get("type") == "a-h-A-M-F"


def test_cot_serializer_default_node_id_argument_accepted():
    # node_id has a default, so calling with a single positional arg (as cli.py does) works.
    xml_str = CotSerializer().serialize(_track())
    assert "TRK-abc-123" in xml_str


def test_cot_serializer_output_validates_against_real_cot_xsd():
    xml_str = CotSerializer().serialize(_track())
    validator = CotValidator(xsd_path=str(XSD_PATH))
    is_valid, error = validator.validate(xml_str)
    assert is_valid, error
