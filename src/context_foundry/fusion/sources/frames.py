# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# src/context_foundry/fusion/sources/frames.py

import logging
import time

logger = logging.getLogger(__name__)

# How long a frame stays open after its first detection, waiting for the rest of
# the same sensor-instant. It is really two bounds at once: how spread out one
# sweep's datagrams may be between the sensor and this socket, and how long one
# fusion pass may take -- a pass blocks the read loop, so the frames open across
# it age by its duration. Exceed either and the sweep splits into events that
# share a timestamp, which the fusion loop then fuses as separate zero-dt
# updates: the per-datagram degradation frames exist to remove, and silent. Sized
# for a routed network and tens of tracks rather than a loopback burst. The cost
# is this much latency on every live detection, which is small against the
# interval between a track's updates (~18s in the reference scenario).
DEFAULT_FRAME_WINDOW_SECONDS = 0.25

# Floor for the socket timeout of an almost-expired frame. settimeout(0) puts the
# socket in non-blocking mode, where an empty buffer raises BlockingIOError
# instead of timing out, so the window would end in an error rather than a frame.
MIN_FRAME_TIMEOUT_SECONDS = 0.001

# Cap on frames open at once. Only sensors reporting within one window hold a
# frame, so a handful is the honest expectation (the reference network has 9
# nodes) and this is far above it. It is a bound, not a target: keys come off the
# wire, so a stream of datagrams with distinct timestamps -- a broken clock, or a
# spoofer -- would otherwise grow the table for as long as it lasts.
MAX_OPEN_FRAMES = 64

# How often to report frames handed on early because the table was full. It only
# happens when something is stamping an instant per detection, which is per
# datagram, so this is a rate limit rather than a one-shot.
EVICTION_REPORT_SECONDS = 60.0


class _Frame:
    """One sensor-instant being assembled."""

    __slots__ = ("detections", "opened_at", "timestamp")

    def __init__(self, timestamp, detection, opened_at):
        self.timestamp = timestamp
        self.detections = [detection]
        self.opened_at = opened_at


class FrameAssembler:
    """Collects the detections of one sensor-instant into a single event.

    A source's contract is one event per sensor report instant, and one instant
    may carry several detections: JsonSapientSource groups the replay file by
    (timestamp, sensor) and yields a sweep as one multi-detection event. A UDP
    socket sees one datagram at a time, and yielding each on its own changes the
    fusion result -- the associator matches a single hit per track per event and
    coasts every other track in between -- and emits one CoT per detection instead of one per
    sweep.

    Several frames are open at once, one per key. Closing on a key change instead
    would only batch a sweep whose datagrams happen to arrive contiguously: two
    sensors sharing the port interleave whenever their sweeps overlap, which
    degrades every event to a single detection and, because the timestamps then
    alternate, makes the fusion loop's watermark gate drop the ones that arrive
    behind the other sensor's.

    A frame is bounded by a wall-clock window from its first detection, since the
    next datagram for its key is a whole sweep interval away on a live feed (~18s
    in the reference scenario) and may never come at all. The caller arms the
    socket timeout with `next_timeout()` for the quiet case; for the busy one it
    calls `take_expired()` once per read, whatever that read produced, because a
    non-empty receive buffer means recvfrom() never times out.

    The key is the caller's choice, because what identifies an instant is
    protocol-specific: SAPIENT reports carry the sending node, CoT's uid names
    the observed object and not the sender.
    """

    def __init__(self, window_seconds: float = DEFAULT_FRAME_WINDOW_SECONDS):
        self.window_seconds = window_seconds
        # Keyed by the caller's key, in the order frames were opened.
        self._frames = {}
        self._evicted = 0
        self._next_eviction_report = 0.0

    def next_timeout(self) -> float | None:
        """The socket timeout for the next read: None to block, else the shortest remaining window.

        With no frame open there is nothing to hand on when a read would expire,
        so the socket blocks indefinitely as an idle live source must, rather
        than polling. With several open, the read must not outlast the frame that
        expires first.
        """
        if not self._frames:
            return None
        now = time.monotonic()
        remaining = min(self._remaining(frame, now) for frame in self._frames.values())
        return max(remaining, MIN_FRAME_TIMEOUT_SECONDS)

    def add(self, key, timestamp, detection):
        """Buffer a detection under its key; returns the events completed by this call.

        A detection joins the open frame for its key, or opens one. Frames are
        never closed by a key change -- an instant is done when its window is up,
        and nothing else about the traffic says so -- so a detection for an
        instant already emitted opens a frame of its own rather than being
        merged into whatever is open. No record of emitted keys is kept: it would
        grow without bound on a live feed, and the fusion loop already gates and
        reports timestamps that are out of step.

        Expired frames are closed here as well as in `take_expired()`, so a
        detection never joins a frame past its window. A sweep too long to fit
        the window therefore splits across events, which is what the source did
        before frames existed.
        """
        closed = self._expired()
        frame = self._frames.get(key)
        if frame is not None:
            frame.detections.append(detection)
            return _events(closed)
        if len(self._frames) >= MAX_OPEN_FRAMES:
            # Hand the oldest on early rather than let the table grow.
            closed.append(self._pop_oldest())
            self._report_eviction()
        self._frames[key] = _Frame(timestamp, detection, time.monotonic())
        return _events(closed)

    def take_expired(self):
        """The events whose window has elapsed, oldest-opened first; possibly none.

        The caller drives this once per read so a read that produced no detection
        -- a rejected datagram, a decode failure, a timeout -- still lets the
        open frames out.

        Oldest-opened first, which with one window for every frame is both expiry
        order and the order the datagrams arrived in -- but not necessarily
        ascending event timestamps. Two sensors whose clocks disagree can still
        hand on a frame the fusion loop's watermark gate then drops; that is the
        clocks' problem, and the gate reports it.
        """
        return _events(self._expired())

    def _remaining(self, frame, now) -> float:
        return self.window_seconds - (now - frame.opened_at)

    def _expired(self) -> list:
        now = time.monotonic()
        keys = [k for k, f in self._frames.items() if self._remaining(f, now) <= 0]
        return [self._frames.pop(k) for k in keys]

    def _report_eviction(self) -> None:
        self._evicted += 1
        if time.monotonic() < self._next_eviction_report:
            return
        self._next_eviction_report = time.monotonic() + EVICTION_REPORT_SECONDS
        logger.warning(
            "More than %d instants open at once; %d frame(s) handed on part-assembled so far. "
            "A sensor is stamping the detections of one sweep separately, or something is "
            "flooding distinct timestamps -- either way sweeps are being split.",
            MAX_OPEN_FRAMES,
            self._evicted,
        )

    def _pop_oldest(self) -> _Frame:
        oldest = min(self._frames, key=lambda k: self._frames[k].opened_at)
        return self._frames.pop(oldest)


def _events(closed: list) -> list:
    return [(f.timestamp, f.detections) for f in sorted(closed, key=lambda f: f.opened_at)]
