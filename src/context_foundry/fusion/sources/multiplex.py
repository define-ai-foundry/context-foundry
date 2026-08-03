# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

import logging
import queue
import threading

from .base import SapientSource

logger = logging.getLogger(__name__)

# Events buffered between the reader threads and the fusion loop. Deep enough to
# absorb a burst while a slow event is fused, shallow enough that a persistently
# overloaded engine reports it instead of growing without bound.
DEFAULT_QUEUE_SIZE = 1000


class _EndOfSource:
    """Sentinel queued by a reader thread when its source is done."""


class MultiplexedSource(SapientSource):
    """Reads several sources concurrently and yields their events as they arrive.

    A live source blocks in recvfrom() until its own next packet, so draining
    sources one after another in a single loop starves every source but the
    first: with `--enable-sapient --enable-cot` the CoT socket is never read.
    Each inner source therefore gets a daemon reader thread feeding one shared
    queue, and this yields whatever arrives first.

    Events are yielded in arrival order, not timestamp order. Two sensors whose
    clocks differ can hand the tracker an event older than the one before it;
    the tracker takes it, so a late packet pulls its track's timestamp back
    rather than being rejected.
    """

    def __init__(self, sources, queue_size: int = DEFAULT_QUEUE_SIZE):
        if not sources:
            raise ValueError("MultiplexedSource needs at least one source")
        self.sources = list(sources)
        self.queue_size = queue_size

    def reset(self):
        for source in self.sources:
            source.reset()

    def _drain(self, source, events):
        try:
            for event in source.iter_events():
                try:
                    events.put_nowait(event)
                except queue.Full:
                    # Fusion is slower than the sensors; the newest event is
                    # dropped rather than stalling every other source behind it.
                    logger.warning(
                        "Event queue full (%d); dropping an event from %s. Fusion is not "
                        "keeping up with the incoming rate.",
                        self.queue_size,
                        type(source).__name__,
                    )
        except Exception:
            logger.exception("Source %s stopped with an error", type(source).__name__)
        finally:
            events.put(_EndOfSource)

    def iter_events(self):
        events = queue.Queue(maxsize=self.queue_size)

        for source in self.sources:
            thread = threading.Thread(
                target=self._drain,
                args=(source, events),
                name=f"source-{type(source).__name__}",
                daemon=True,
            )
            thread.start()

        live = len(self.sources)
        while live:
            event = events.get()
            if event is _EndOfSource:
                live -= 1
                continue
            yield event
