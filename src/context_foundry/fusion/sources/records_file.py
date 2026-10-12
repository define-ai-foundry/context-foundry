# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# src/context_foundry/fusion/sources/records_file.py

"""sapient-raw and cot-raw records from files, fused as the pipeline would have fused them.

Each file holds one record per line: the record itself, or an archive line carrying it
under `value` (the archiver's form, with its offset, key and headers), so a team's
archive of a pipeline's input runs through the standalone engine offline. A line whose
record has a `content_type` is sapient-raw, one with a `uid` and a `point` cot-raw.

The whole input is known, so it is fused in the order a pipeline aims for: Registrations
and status reports teach the sensors as they come, file by file in line order, and the
detections are then taken by event_time, a tie keeping its line order, each sensor-instant
one event as the Kafka source groups them. Fusion's own records are skipped.
"""

import json
import logging
from pathlib import Path

from ..cot_input import CotRecordReader
from ..measurement import DEFAULT_MEASUREMENT_MODEL
from ..sapient_input import SapientRecordReader, event_time
from .base import SapientSource

logger = logging.getLogger(__name__)


class RecordsFileSource(SapientSource):
    """Detections of sapient-raw and cot-raw records read from files, by event time."""

    def __init__(self, paths: list[Path], own_node_id: str | None = None):
        self.paths = [Path(p) for p in paths]
        self.cartesian_meas_model = DEFAULT_MEASUREMENT_MODEL
        self.sapient = SapientRecordReader(own_node_id)
        self.cot = CotRecordReader(own_node_id)
        self.records_read = 0
        self.unreadable = 0
        self._exhausted = False

    def reset(self):
        self._exhausted = False

    def _records(self):
        for path in self.paths:
            with open(path, encoding="utf-8") as lines:
                for number, line in enumerate(lines, start=1):
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                    except ValueError as e:
                        self.unreadable += 1
                        logger.warning("%s:%d is not JSON: %s", path, number, e)
                        continue
                    record = entry.get("value", entry) if isinstance(entry, dict) else None
                    if not isinstance(record, dict):
                        self.unreadable += 1
                        continue
                    self.records_read += 1
                    yield record

    def iter_events(self):
        if self._exhausted:
            return
        self._exhausted = True
        held = []
        for record in self._records():
            if "content_type" in record:
                try:
                    moment = event_time(record)
                except (TypeError, ValueError, KeyError, AttributeError):
                    self.unreadable += 1
                    continue
                # A sensor is learned as it arrives, ahead of the order of detections.
                if self.sapient.learn(record) or not self.sapient.is_detection(record):
                    continue
                held.append((moment, len(held), self.sapient, record))
            elif "uid" in record and "point" in record:
                try:
                    moment = event_time(record)
                except (TypeError, ValueError, KeyError, AttributeError):
                    self.unreadable += 1
                    continue
                held.append((moment, len(held), self.cot, record))
            else:
                self.unreadable += 1
        held.sort(key=lambda item: (item[0], item[1]))

        key, frame = None, []
        for _, _, reader, record in held:
            made = reader.detection(record)
            if made is None:
                continue
            timestamp, sensor_id, detection = made
            if frame and (timestamp, sensor_id) != key:
                yield key[0], frame
                frame = []
            key = (timestamp, sensor_id)
            frame.append(detection)
        if frame:
            yield key[0], frame
        logger.info(
            "Read %d record(s) from %d file(s); %d unreadable, %d rejected",
            self.records_read,
            len(self.paths),
            self.unreadable,
            self.sapient.records_rejected + self.cot.records_rejected,
        )
