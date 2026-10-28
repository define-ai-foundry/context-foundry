# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from .schemas import TacticalTrack
from .timeutil import as_utc

# How long a marker stays live in TAK after the event it was built from.
DEFAULT_STALE_SECONDS = 15.0

# A threat level's CoT affiliation letter; any other level is drawn as suspect.
THREAT_IDENTITY = {"hostile": "h", "suspect": "s", "friend": "f", "neutral": "n"}

# The type after the affiliation for a track whose dimension nothing declared.
DEFAULT_COT_FUNCTION = "A-M-F"


def _cot_function(declared_type) -> str:
    """The dimension a declared CoT type gives (`a-f-G-U-C` gives `G`), else air fixed-wing."""
    parts = declared_type.split("-") if isinstance(declared_type, str) else []
    if len(parts) < 3 or not parts[2] or parts[2] == "A":
        return DEFAULT_COT_FUNCTION
    return parts[2]


def _cot_time(moment: datetime) -> str:
    """CoT wants UTC, millisecond precision, `Z`-suffixed."""
    return as_utc(moment).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class BaseSerializer(ABC):
    @abstractmethod
    def serialize(self, state: TacticalTrack, node_id: str) -> str:
        pass


class CotSerializer(BaseSerializer):
    """Formats tactical state for ATAK/WinTAK networks."""

    def __init__(self, stale_seconds: float = DEFAULT_STALE_SECONDS):
        self.stale_seconds = stale_seconds

    def serialize(self, state: TacticalTrack, node_id: str = "FUSION-NODE") -> str:
        now = _cot_time(state.timestamp)
        stale = _cot_time(state.timestamp + timedelta(seconds=self.stale_seconds))

        identity = THREAT_IDENTITY.get(state.threat_level, "s")
        sidc = f"a-{identity}-{_cot_function(state.declared_type)}"

        event = ET.Element(
            "event",
            {
                "version": "2.0",
                "uid": f"TRK-{state.track_id}",
                "type": sidc,
                "time": now,
                "start": now,
                "stale": stale,
                "how": "m-g",
            },
        )

        ET.SubElement(
            event,
            "point",
            {
                "lat": f"{state.latitude:.6f}",
                "lon": f"{state.longitude:.6f}",
                "hae": f"{state.altitude:.1f}",
                "ce": "10.0",
                "le": "10.0",
            },
        )

        detail = ET.SubElement(event, "detail")
        ET.SubElement(
            detail,
            "track",
            {"speed": f"{state.speed_mps:.2f}", "course": f"{state.heading_deg:.1f}"},
        )
        ET.SubElement(
            detail, "contact", {"callsign": f"SWM({state.swarm_count}) {state.classification}"}
        )

        return ET.tostring(event, encoding="utf-8").decode("utf-8")
