# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

import logging
from datetime import datetime, timezone

from ..timeutil import as_utc
from .base import SapientSource

logger = logging.getLogger(__name__)


class OffsetReplaySource(SapientSource):
    """Shifts a replay's timestamps so its first event lands at the present.

    Wraps any SapientSource and adds one constant offset to every timestamp,
    chosen so the first event of each iteration is stamped `now`. Later events
    keep their original spacing, so the scenario's own intervals -- and the
    inter-event dt the tracker's motion model depends on -- are unchanged.

    A recorded scenario therefore surfaces as if it were happening now, which is
    what downstream consumers assume: TAK clients hide or expire CoT whose time
    is far from the present, and a file dated months away from `now` renders as
    events nobody is looking at.

    The offset is independent of emission speed. `RealtimeReplaySource` controls
    how fast events are handed on; this changes what they are stamped with. At a
    factor other than 1.0 the two deliberately diverge: timestamps keep the
    scenario's spacing while emission is compressed or stretched.

    Each iteration re-anchors, so a looping replay restarts at the new present
    rather than repeating the first pass's timestamps.
    """

    def __init__(self, inner: SapientSource):
        self.inner = inner

    def reset(self):
        self.inner.reset()

    def iter_events(self):
        offset = None

        for timestamp, detections in self.inner.iter_events():
            if offset is None:
                offset = datetime.now(timezone.utc) - as_utc(timestamp)
                logger.info(
                    "Offsetting replay timestamps by %+.1fs so the scenario starts now "
                    "(--use-scenario-timestamps to keep the file's own clock)",
                    offset.total_seconds(),
                )

            shifted = as_utc(timestamp) + offset
            # The tracker keys a track's state off detection.timestamp, and the
            # broadcast gate compares that against the event timestamp, so both
            # have to move by the same offset or nothing is ever broadcast.
            for detection in detections:
                detection.timestamp = as_utc(detection.timestamp) + offset

            yield shifted, detections
