# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# src/context_foundry/fusion/publish.py

"""Which fused tracks an event publishes.

Every event moves every live track to the event's time: the tracks a detection
matched are updated from it, the rest are predicted forward. Publishing all of
them re-sends the whole picture on every sensor report; most of it is then
predicted positions that read like sightings. The policy picks what goes out:

  all      every track the event moved (the standalone engine's behaviour)
  updates  only tracks a detection updated or started
  coast    updates, plus a predicted state for a track that has not been
           published for `coast_seconds`, so a track no sensor is seeing still
           shows it is alive
  predicted  only predicted states: for studying the predictions on their own,
           not for a live pipeline
  window   only measured states, and at most one per track per `window_seconds`:
           every report that lands on a track inside the window is fused into it,
           and one message carries the result, so fewer messages go out than
           reports come in and no prediction is ever sent
"""

from stonesoup.types.prediction import Prediction
from stonesoup.types.update import Update

POLICIES = ("all", "updates", "coast", "predicted", "window")
DEFAULT_COAST_SECONDS = 5.0
DEFAULT_WINDOW_SECONDS = 5.0


def is_predicted(track) -> bool:
    """The track's newest state was predicted, not measured."""
    return isinstance(track.state, Prediction)


class PublishPolicy:
    def __init__(
        self,
        policy: str = "all",
        coast_seconds: float = DEFAULT_COAST_SECONDS,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
    ):
        if policy not in POLICIES:
            raise ValueError(f"unknown publish policy {policy!r}; one of {', '.join(POLICIES)}")
        if coast_seconds <= 0:
            raise ValueError("coast_seconds must be > 0")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        self.policy = policy
        self.coast_seconds = coast_seconds
        self.window_seconds = window_seconds
        self._last_sent = {}

    def should_publish(self, track, moment) -> bool:
        """Whether the track's state at `moment` goes out; records it when it does."""
        if self.policy == "predicted":
            send = is_predicted(track)
        elif self.policy == "window":
            last = self._last_sent.get(track.id)
            send = not is_predicted(track) and (
                last is None or (moment - last).total_seconds() >= self.window_seconds
            )
        elif self.policy == "all" or not is_predicted(track):
            send = True
        elif self.policy == "updates":
            send = False
        else:
            last = self._last_sent.get(track.id)
            send = last is None or (moment - last).total_seconds() >= self.coast_seconds
        if send:
            self._last_sent[track.id] = moment
        return send

    def forget_all_but(self, live_track_ids):
        """Drop the memory of tracks that are no longer live."""
        live = set(live_track_ids)
        for track_id in [t for t in self._last_sent if t not in live]:
            del self._last_sent[track_id]


def recent_sources(track) -> list[dict]:
    """The sensor reports behind a track, from the history it still holds.

    One entry per (sensor, sensor object id), with its newest sighting, oldest
    first. Read from the track's retained states -- each Update carries the
    detection that made it, and the first state, while still held, the detection
    that started the track -- so it covers the recent past, not the whole life.
    """
    found = {}

    def note(metadata, moment):
        node_id = (metadata or {}).get("nodeId")
        if not node_id or moment is None:
            return
        key = (node_id, (metadata or {}).get("objectId"))
        if key not in found or moment > found[key]:
            found[key] = moment

    states = list(track.states)
    for index, state in enumerate(states):
        if isinstance(state, Update):
            detection = getattr(state.hypothesis, "measurement", None)
            if detection is not None:
                note(detection.metadata, detection.timestamp)
        elif index == 0 and not isinstance(state, Prediction):
            note(getattr(track, "init_metadata", None), state.timestamp)

    return [
        {"node_id": node_id, "object_id": object_id, "timestamp": moment}
        for (node_id, object_id), moment in sorted(found.items(), key=lambda kv: kv[1])
    ]
