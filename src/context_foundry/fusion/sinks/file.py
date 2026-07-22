# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# src/context_foundry/fusion/sinks/file.py

from .base import CotSink


class FileCotSink(CotSink):
    """Appends fused CoT payloads to a file, one <event> per line, for offline validation."""

    def __init__(self, path: str):
        self.path = path

    def send(self, cot_payload: str) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(cot_payload + "\n")
