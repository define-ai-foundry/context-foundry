# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# src/context_foundry/fusion/sinks/records_file.py

"""Fused tracks as sapient-raw records, one per line, to a file: the Kafka sink's records
without a broker, so an offline run can be compared with a pipeline's output."""

import logging

from ..sapient_raw import track_record
from ..schemas import TacticalTrack

logger = logging.getLogger(__name__)


class RecordsFileSink:
    """Writes each fused track as the sapient-raw record the Kafka sink would produce."""

    def __init__(self, path: str, node_id: str):
        self.path = path
        self.node_id = node_id
        self.sent = 0
        self._file = open(path, "w", encoding="utf-8")  # noqa: SIM115 -- closed in close()

    def send_track(
        self,
        track: TacticalTrack,
        lineage: dict | None = None,
        predicted: bool | None = None,
        sources: list[dict] | None = None,
    ):
        _, value, _ = track_record(track, self.node_id, predicted=predicted, sources=sources)
        self._file.write(value.decode("utf-8") + "\n")
        self.sent += 1

    def close(self):
        self._file.close()
        logger.info("Wrote %d fused track(s) to %s", self.sent, self.path)
