# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# src/context_foundry/fusion/sources/base.py

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


def swarm_count(original_report: dict) -> int:
    """Read estimatedSwarmCount out of a detection report.

    objectInfo values are free text on the wire. A sensor reporting "12.0" means
    twelve -- reading it as one would drop the track's threat level -- and one
    reporting "many" must not take the ingest path down with it.
    """
    for info in original_report.get("objectInfo", []):
        if info.get("type") != "estimatedSwarmCount":
            continue
        try:
            return int(float(info.get("value", 1)))
        except (TypeError, ValueError, OverflowError):
            logger.warning("Unparseable estimatedSwarmCount %r; treating as 1", info.get("value"))
            return 1
    return 1


class SapientSource(ABC):
    """
    Common interface for all SAPIENT ingestion sources.

    Implementations:
    - Generated JSON replay
    - Live stream adapters
    """

    @abstractmethod
    def iter_events(self):
        """
        Yield asynchronous sensor updates:

            timestamp, sensor_update

        timestamp:
            datetime

        sensor_update:
            iterable of Stone Soup Detection objects
        """
        raise NotImplementedError

    def reset(self):  # noqa: B027  intentional concrete no-op; replay sources override
        """Rewind the source so iter_events() can be consumed again.

        No-op for live streams; finite replay sources override this.
        """
