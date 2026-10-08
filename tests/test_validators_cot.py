"""Tests for context_foundry.fusion.validators.cot."""

from pathlib import Path

import pytest
from lxml import etree

from context_foundry.fusion.schemas import InternalDetection
from context_foundry.fusion.validators.cot import CotValidator

REPO_ROOT = Path(__file__).resolve().parents[1]
XSD_PATH = str(REPO_ROOT / "protos" / "cot" / "CoT Base-Event Schema  (PUBLIC RELEASE).xsd")

VALID_COT_XML = (
    '<event version="2.0" uid="TEST-UID-1" type="a-f-A-M-F" '
    'time="2026-01-01T00:00:00.000Z" start="2026-01-01T00:00:00.000Z" '
    'stale="2026-01-01T00:00:15.000Z" how="m-g">'
    '<point lat="62.900000" lon="29.800000" hae="100.0" ce="10.0" le="10.0"/>'
    "</event>"
)

MISSING_REQUIRED_ATTR_XML = (
    '<event version="2.0" uid="TEST-UID-1" type="a-f-A-M-F" '
    'time="2026-01-01T00:00:00.000Z" start="2026-01-01T00:00:00.000Z" '
    'stale="2026-01-01T00:00:15.000Z" how="m-g">'
    '<point lat="62.900000" lon="29.800000" ce="10.0" le="10.0"/>'
    "</event>"
)


@pytest.fixture
def validator():
    return CotValidator(xsd_path=XSD_PATH)


def test_default_xsd_path_resolves_from_repo_root():
    # Relies on pytest being invoked from the repo root (testpaths=["tests"]).
    validator = CotValidator()
    is_valid, error = validator.validate(VALID_COT_XML)
    assert is_valid, error


def test_validate_accepts_schema_compliant_xml(validator):
    is_valid, error = validator.validate(VALID_COT_XML)
    assert is_valid is True
    assert error == ""


def test_validate_rejects_malformed_xml(validator):
    is_valid, error = validator.validate("<event this is not valid xml")
    assert is_valid is False
    assert "Malformed XML" in error


def test_validate_rejects_schema_violation(validator):
    is_valid, error = validator.validate(MISSING_REQUIRED_ATTR_XML)
    assert is_valid is False
    assert "CoT Schema Violation" in error


def test_normalize_extracts_internal_detection(validator):
    det = validator.normalize(VALID_COT_XML)
    assert isinstance(det, InternalDetection)
    assert det.sensor_id == "TEST-UID-1"
    assert det.latitude == pytest.approx(62.9)
    assert det.longitude == pytest.approx(29.8)
    assert det.altitude == pytest.approx(100.0)
    assert det.classification == "a-f-A-M-F"
    assert det.raw_metadata == {"original_xml": VALID_COT_XML, "ce": 10.0, "le": 10.0}
    assert det.timestamp.year == 2026


def test_process_message_full_pipeline_success(validator):
    det = validator.process_message(VALID_COT_XML)
    assert det is not None
    assert det.sensor_id == "TEST-UID-1"


def test_process_message_returns_none_for_invalid_xml(validator):
    assert validator.process_message("<not xml") is None


def test_init_loads_schema_from_explicit_path():
    v = CotValidator(xsd_path=XSD_PATH)
    assert isinstance(v.schema, etree.XMLSchema)


def test_normalize_treats_cot_unknown_values_as_absent(validator):
    xml = VALID_COT_XML.replace(
        'hae="100.0" ce="10.0" le="10.0"', 'hae="9999999.0" ce="9999999.0" le="9999999.0"'
    )
    det = validator.normalize(xml)
    assert det.altitude is None
    assert det.raw_metadata["ce"] is None
    assert det.raw_metadata["le"] is None
