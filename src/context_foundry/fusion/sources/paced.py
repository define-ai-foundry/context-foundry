# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

import logging
import time

from .base import SapientSource

logger = logging.getLogger(__name__)

# How far behind the scenario clock the engine may fall before it is reported.
LAG_WARN_SECONDS = 1.0


class RealtimeReplaySource(SapientSource):
    """Paces a finite replay source so events surface at their scenario cadence.

    Wraps any SapientSource and sleeps before each event until its scenario
    offset has elapsed in wall time, divided by `factor` (2.0 = twice as fast).
    Fusion output is unaffected: the tracker and serializers key off the
    scenario timestamp, which is passed through untouched.
    """

    def __init__(self, inner: SapientSource, factor: float = 1.0):
        if factor <= 0:
            raise ValueError(f"factor must be > 0, got {factor}")
        self.inner = inner
        self.factor = factor

    def reset(self):
        self.inner.reset()

    def iter_events(self):
        # Both anchors are taken from the first event, so sleeps are computed
        # against the scenario start rather than the previous event: per-event
        # processing cost cannot accumulate into drift.
        anchor_scenario = None
        anchor_wall = 0.0
        lag_reported = False

        for timestamp, detections in self.inner.iter_events():
            if anchor_scenario is None:
                anchor_scenario, anchor_wall = timestamp, time.monotonic()

            offset = (timestamp - anchor_scenario).total_seconds() / self.factor
            lag = time.monotonic() - (anchor_wall + offset)

            if lag < 0:
                time.sleep(-lag)
            elif lag > LAG_WARN_SECONDS and not lag_reported:
                # Fusion is slower than the requested rate: emit as fast as
                # possible until the scenario clock is caught up again. Reported
                # once, so a run that briefly stalls logs one line, not thousands.
                lag_reported = True
                logger.warning(
                    "Fell %.1fs behind the scenario clock at --realtime-factor %g; emitting as "
                    "fast as events are fused until caught up. Further lag is not reported.",
                    lag,
                    self.factor,
                )

            yield timestamp, detections
