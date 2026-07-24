# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# src/context_foundry/fusion/sources/base.py

from abc import ABC, abstractmethod


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
