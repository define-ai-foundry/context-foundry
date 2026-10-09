# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0


from lxml import etree

from context_foundry.fusion.cot_input import known
from context_foundry.fusion.schemas import InternalDetection
from context_foundry.fusion.validators.base import ProtocolValidator


class CotValidator(ProtocolValidator):
    def __init__(self, xsd_path: str = "protos/cot/CoT Base-Event Schema  (PUBLIC RELEASE).xsd"):
        # Load the official MITRE schema once on startup
        with open(xsd_path, "rb") as f:
            schema_root = etree.XML(f.read())
            self.schema = etree.XMLSchema(schema_root)

    def validate(self, raw_xml_string: str) -> tuple[bool, str]:
        try:
            # Parse the incoming XML string
            doc = etree.fromstring(raw_xml_string.encode("utf-8"))

            # The Magic Line: Validate against the MITRE XSD
            self.schema.assertValid(doc)
            return True, ""

        except etree.XMLSyntaxError as e:
            return False, f"Malformed XML: {e}"
        except etree.DocumentInvalid as e:
            return False, f"CoT Schema Violation: {e}"

    def normalize(self, raw_xml_string: str) -> InternalDetection:
        # Since we know it's valid, we can safely extract the data
        root = etree.fromstring(raw_xml_string.encode("utf-8"))
        point = root.find("point")

        return InternalDetection(
            sensor_id=root.get("uid"),
            timestamp=root.get("time"),
            latitude=float(point.get("lat")),
            longitude=float(point.get("lon")),
            altitude=known(point.get("hae")),  # Height Above Ellipsoid
            classification=root.get("type"),
            raw_metadata={
                "original_xml": raw_xml_string,
                # The point's own circular (horizontal) and linear (vertical) error,
                # in metres; None where the sender marked it unknown.
                "ce": known(point.get("ce")),
                "le": known(point.get("le")),
            },
        )
