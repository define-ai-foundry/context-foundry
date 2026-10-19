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

from datetime import datetime

from stonesoup.types.prediction import Prediction
from stonesoup.types.update import Update

from .cot_input import COT_NODE
from .sapient_raw import cot_node_id, cot_object_ulid

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


def source_key(metadata) -> tuple[str, str | None] | None:
    """(node_id, object_id) naming the sensor report a detection came from, or None.

    A SAPIENT report is named by its own node_id and object_id. A CoT event has no
    SAPIENT identity, so it is named by ids derived from it: `cot_node_id()` for every
    CoT source and `cot_object_ulid(uid)` for its object. The detection's own metadata
    keeps "cot" and the uid, which the tracker and the sensor registry key on.
    """
    metadata = metadata or {}
    node_id = metadata.get("nodeId")
    if not node_id:
        return None
    object_id = metadata.get("objectId")
    if node_id == COT_NODE:
        return cot_node_id(), cot_object_ulid(str(object_id)) if object_id else None
    return node_id, object_id


def _note(found: dict, metadata, moment) -> None:
    """Record one sighting in `found`, keeping each source's newest time."""
    key = source_key(metadata)
    if key is None or moment is None:
        return
    if key not in found or moment > found[key]:
        found[key] = moment


def _listed(found: dict) -> list[dict]:
    """Sources as associated_detection entries, oldest first."""
    return [
        {"node_id": node_id, "object_id": object_id, "timestamp": moment}
        for (node_id, object_id), moment in sorted(found.items(), key=lambda kv: kv[1])
    ]


def recent_sources(track) -> list[dict]:
    """The sensor reports behind a track, from the history it still holds.

    One entry per (sensor, sensor object id), with its newest sighting, oldest
    first. Read from the track's retained states -- each Update carries the
    detection that made it, and the first state, while still held, the detection
    that started the track -- so it covers the recent past, not the whole life.
    """
    found = {}
    states = list(track.states)
    for index, state in enumerate(states):
        if isinstance(state, Update):
            detection = getattr(state.hypothesis, "measurement", None)
            if detection is not None:
                _note(found, detection.metadata, detection.timestamp)
        elif index == 0 and not isinstance(state, Prediction):
            _note(found, getattr(track, "init_metadata", None), state.timestamp)
    return _listed(found)


class PendingSources:
    """The sensor reports each track took since its previous published record.

    A record within the publish window stands for every update since the last one, so
    it names the sources of all of them, not only the latest; taking a track's list
    for a record starts the next one empty.
    """

    def __init__(self):
        self._pending: dict[str, dict[tuple[str, str | None], datetime]] = {}

    def note(self, track_id, metadata, moment) -> None:
        """A report `track_id` took at `moment`, from the detection's `metadata`."""
        _note(self._pending.setdefault(track_id, {}), metadata, moment)

    def take(self, track_id) -> list[dict]:
        """The track's pending sources, oldest first, and clear them."""
        return _listed(self._pending.pop(track_id, {}))

    def forget(self, track_id) -> None:
        """Drop one track's pending sources, as for a track started again under its id."""
        self._pending.pop(track_id, None)

    def forget_all_but(self, live_track_ids) -> None:
        """Drop the pending sources of tracks that are no longer held."""
        live = set(live_track_ids)
        for track_id in [t for t in self._pending if t not in live]:
            del self._pending[track_id]
