# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# src/context_foundry/fusion/engine.py

"""One fusion step: an event's detections in, the tracks to publish out.

What the standalone application and a pipeline stage share. The caller reads its
input -- a file, sockets, Kafka -- into events of (timestamp, detections), hands each to
`FusionEngine.process`, and writes what comes back wherever it writes: CoT to TAK,
sapient-raw to Kafka. The engine keeps the event-time gate, the tracker and the publish
rule, so both fuse the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .augmentor import TacticalContextAugmentor
from .publish import PublishPolicy
from .schemas import TacticalTrack
from .tracker import SapientAsynchronousTracker

# How far ahead of now a live event may be stamped. Live sensors report the
# present, so anything beyond this is a broken clock or a spoofed datagram --
# and without the bound one such packet would become the high-water mark that
# every healthy event afterwards is measured against and dropped.
FUTURE_HORIZON_SECONDS = 10.0

# How far behind the newest event seen a live event may still be fused. A sensor
# stamps before it transmits, so every event arrives already a little in the past
# -- flight time, frame assembly, one fusion pass -- while the newest timestamp a
# live run will accept is pinned to the present. A gate with no tolerance
# therefore drops the ordinary case: measured, healthy sensors landed 0.3s behind
# a mark an out-of-step sender had pulled up to now, and every one was refused.
# Beyond this it is a clock that disagrees rather than latency. This is also the
# whole bound on how far an accepted event can rewind the tracker, because the
# tracker is fed the same clamped timestamp the mark is made of: the mark never
# runs ahead of the present, so an event that passes this gate is at most this far
# behind what was last fused.
LATE_EVENT_TOLERANCE_SECONDS = 2.0


def sensor_ids(detections) -> str:
    """The sensors an event came from, for reporting which one is out of step."""
    ids = {getattr(det, "metadata", {}).get("nodeId") for det in detections}
    return ", ".join(sorted(str(node_id) for node_id in ids if node_id)) or "an unnamed sensor"


@dataclass
class EventResult:
    """What one event did.

    `skew` says why it was dropped, None when it was fused. `published` holds each
    track to publish with its tactical form; `active` every live track after it.
    """

    mark: datetime | None = None
    skew: str | None = None
    published: list[tuple[Any, TacticalTrack]] = field(default_factory=list)
    active: list[Any] = field(default_factory=list)

    @property
    def dropped(self) -> bool:
        return self.skew is not None


class FusionEngine:
    """The event-time gate, the tracker and the publish rule, one event at a time.

    `future_horizon_seconds` is for a live run, measured against the `now` its caller
    passes; a replay, or a stream of records carrying their sensors' own event_time,
    passes no `now` and keeps its own timestamps throughout.
    """

    def __init__(
        self,
        tracker: SapientAsynchronousTracker,
        publish_policy: PublishPolicy,
        augmentor: TacticalContextAugmentor | None = None,
        future_horizon_seconds: float = FUTURE_HORIZON_SECONDS,
        late_tolerance_seconds: float = LATE_EVENT_TOLERANCE_SECONDS,
    ):
        self.tracker = tracker
        self.publish_policy = publish_policy
        self.augmentor = augmentor or TacticalContextAugmentor()
        self.future_horizon_seconds = future_horizon_seconds
        self.late_tolerance_seconds = late_tolerance_seconds
        self.latest_timestamp: datetime | None = None
        # Which sensor's event set the current watermark, so a drop report can name
        # the sensor that caused it and not only the one that suffered it.
        self.watermark_sensor = "no event accepted yet"

    def reset(self, tracker: SapientAsynchronousTracker) -> None:
        """Start again with a new tracker: a new scenario, not a continuation."""
        self.tracker = tracker
        self.latest_timestamp = None

    def process(self, timestamp: datetime, detections, now: datetime | None = None) -> EventResult:
        """Fuse one event, or drop it when its timestamp is out of step with the rest.

        Stone Soup cannot predict backwards: feeding it an event older than the last
        one rewinds every track's timestamp and re-broadcasts the lot with a time that
        moves back, which TAK draws as markers jumping into the past -- for every
        track, not just the late sensor's. So events are gated to the highest
        timestamp seen so far, within `late_tolerance_seconds`.

        That makes the gate only as good as the highest timestamp, which is why a live
        event stamped in the future is refused the mark: one bad clock or spoofed
        datagram would otherwise set a watermark every healthy sensor then falls
        behind, and the engine would go quiet for good.
        """
        skew = None
        if now is not None:
            ahead = (timestamp - now).total_seconds()
            if ahead > self.future_horizon_seconds:
                skew = f"{ahead:.1f}s ahead of the present"
        if skew is None and self.latest_timestamp is not None:
            behind = (self.latest_timestamp - timestamp).total_seconds()
            if behind > self.late_tolerance_seconds:
                skew = f"{behind:.1f}s behind the newest event seen"
        if skew is not None:
            return EventResult(skew=skew)

        # On a live run the event is fused at the present rather than at its own
        # timestamp, and that clamped time is also the watermark it sets. A sensor a
        # few seconds fast is inside the future horizon, so it is fused -- and with
        # its raw timestamp it would push every track's state that far ahead, while
        # the next healthy event, inside the lateness tolerance and so accepted,
        # rewound the tracker by the whole offset and re-broadcast every track with a
        # time in the past. Clamping bounds any rewind by the tolerance alone. Letting
        # it mark the future would also put the gate ahead of what the healthy sensors
        # report, so their events -- not its -- get dropped.
        mark = timestamp if now is None else min(timestamp, now)
        if self.latest_timestamp is None or mark > self.latest_timestamp:
            self.latest_timestamp = mark
            self.watermark_sensor = sensor_ids(detections)

        if mark != timestamp:
            # The detections move with the event. Stone Soup predicts each hypothesis
            # to its detection's own timestamp, so a track updated by one ends up
            # stamped with it -- leave them raw and the publish gate below matches
            # nothing, which publishes nothing for that sensor and says nothing.
            for detection in detections:
                detection.timestamp = mark

        active = list(self.tracker.process_async_event(mark, set(detections)))
        published = []
        for track in active:
            # Only a track this event updated, and only when the publish rule lets it
            # out (by default: a state a sensor report made, at most once per track per
            # window), so tracks that have not changed are not sent again.
            if track.state.timestamp == mark and self.publish_policy.should_publish(track, mark):
                published.append((track, self.augmentor.extract_tactical_track(track)))
        self.publish_policy.forget_all_but(t.id for t in active)
        return EventResult(mark=mark, published=published, active=active)
