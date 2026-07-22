# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# src/context_foundry/fusion/sinks/base.py

from abc import ABC, abstractmethod


class CotSink(ABC):
    """
    Common interface for all CoT dissemination sinks.

    Implementations:
    - File logging for offline validation
    - UDP broadcast for ATAK mesh clients
    - TCP+TLS stream to a TAK Server
    """

    @abstractmethod
    def send(self, cot_payload: str) -> None:
        """Send one serialized CoT XML document to the destination."""
        raise NotImplementedError

    def close(self) -> None:  # noqa: B027 optional lifecycle hook; default no-op is intentional
        """Release any held resources. No-op by default; override if needed."""
