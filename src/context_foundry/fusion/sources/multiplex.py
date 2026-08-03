# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

import logging
import queue
import threading
import time

from .base import SapientSource

logger = logging.getLogger(__name__)

# Events buffered between the reader threads and the fusion loop. Deep enough to
# absorb a burst while a slow event is fused, shallow enough that a persistently
# overloaded engine reports it instead of growing without bound.
DEFAULT_QUEUE_SIZE = 1000

# How long a reader waits on a full queue before re-checking whether the
# consumer has gone away. Bounds how long a thread outlives its consumer.
PUT_TIMEOUT_SECONDS = 0.5

# How often a reader reports that it is waiting on the fusion loop. A saturated
# queue makes every event wait, so this is a rate limit: without it the log is
# one line per packet. "Recovered" is not a usable trigger either -- under
# sustained overload each event does get through eventually, so the stall looks
# like thousands of momentary ones rather than the one it is.
STALL_REPORT_SECONDS = 30.0


class _EndOfSource:
    """Sentinel queued by a reader thread when its source is done."""


class _ReaderState:
    """Per-reader stall bookkeeping, so one stall is one log line."""

    def __init__(self, name):
        self.name = name
        self.held_up = 0
        self.next_report = 0.0


class MultiplexedSource(SapientSource):
    """Reads several sources concurrently and yields their events as they arrive.

    A live source blocks in recvfrom() until its own next packet, so draining
    sources one after another in a single loop starves every source but the
    first: with `--enable-sapient --enable-cot` the CoT socket is never read.
    Each inner source therefore gets a daemon reader thread feeding one shared
    queue, and this yields whatever arrives first.

    A full queue back-pressures the readers rather than discarding events, so a
    finite source is never silently truncated. Sustained overload then stalls
    the readers and packets are dropped by the kernel's socket buffer, where
    the loss is at least visible and counted.

    Events are yielded in arrival order, not timestamp order. Two sensors whose
    clocks differ can hand on an event older than the one before it, which the
    tracker cannot predict through; the fusion loop gates on that.
    """

    def __init__(self, sources, queue_size: int = DEFAULT_QUEUE_SIZE):
        if not sources:
            raise ValueError("MultiplexedSource needs at least one source")
        self.sources = list(sources)
        self.queue_size = queue_size

    def reset(self):
        for source in self.sources:
            source.reset()

    def _put(self, events, item, stop, state):
        """Hand one item to the consumer, giving up if it has gone away."""
        waited = False
        while not stop.is_set():
            try:
                events.put(item, timeout=PUT_TIMEOUT_SECONDS)
                return
            except queue.Full:
                if not waited:
                    waited = True  # count events held up, not waits
                    state.held_up += 1
                if time.monotonic() >= state.next_report:
                    state.next_report = time.monotonic() + STALL_REPORT_SECONDS
                    logger.warning(
                        "Event queue full (%d); %s is waiting on the fusion loop, which is not "
                        "keeping up with the incoming rate. %d event(s) held up so far.",
                        self.queue_size,
                        state.name,
                        state.held_up,
                    )

    def _drain(self, source, events, stop):
        state = _ReaderState(type(source).__name__)
        try:
            for event in source.iter_events():
                if stop.is_set():
                    return
                self._put(events, event, stop, state)
        except Exception:
            logger.exception("Source %s stopped with an error", state.name)
        finally:
            self._put(events, _EndOfSource, stop, state)

    def iter_events(self):
        events = queue.Queue(maxsize=self.queue_size)
        # Readers outlive a consumer that stops early (a live one is parked in
        # recvfrom() until its next packet), so they are daemons and this only
        # stops them from blocking forever on a queue nobody drains.
        stop = threading.Event()

        for source in self.sources:
            threading.Thread(
                target=self._drain,
                args=(source, events, stop),
                name=f"source-{type(source).__name__}",
                daemon=True,
            ).start()

        try:
            live = len(self.sources)
            while live:
                event = events.get()
                if event is _EndOfSource:
                    live -= 1
                    continue
                yield event
        finally:
            stop.set()
