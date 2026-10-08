# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# src/fusion/tracker.py

import logging
import time

import numpy as np
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.predictor.kalman import UnscentedKalmanPredictor
from stonesoup.types.array import StateVector
from stonesoup.types.hypothesis import SingleDistanceHypothesis
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.updater.kalman import UnscentedKalmanUpdater

from .models import create_9d_constant_acceleration_model

logger = logging.getLogger(__name__)

# States kept per track, and the default for the tracker's max_track_history.
# Every event appends one state (and one metadata copy) to every live track, so
# an uncapped run grows for as long as it stays up -- hundreds of MiB an hour at
# a few dozen tracks. Nothing downstream reads history: the augmentor reads
# track.state and the serializers read the tactical track built from it, so this
# only has to be deep enough to inspect a track's recent past. The operating
# point it encodes is the memory ceiling: at ~7.6 KiB a state it is this times
# max_live_tracks that has to fit the limit, and it is the term to trade away
# when a picture needs more tracks than the default allows.
MAX_TRACK_HISTORY = 20

# Live tracks held at once, and the default for the tracker's max_live_tracks.
# Detections that do not associate each start a track, so clutter -- weather,
# birds, a sensor whose clock disagrees, a spoofer -- grows the population until
# per-event association cost passes the event rate, and once fusion is behind the
# backlog keeps it there. Refusing new tracks past this bound sheds the clutter
# and keeps the ones already being followed; the population then recovers as
# coasting tracks are pruned.
#
# The operating point: this many tracks is roughly where one fusion pass (fuse +
# augment + serialize every broadcast track) reaches the live sources' frame
# window on one core -- measured 247 ms at 112 tracks against a 250 ms window,
# 421 ms at 140. Past it a pass outlasts the window, sweeps split into
# single-detection events, association degrades and the population grows further.
# Raise it only alongside the frame window, or with cores to fuse on.
MAX_LIVE_TRACKS = 120

# Detections one event is associated with, and the default for the tracker's
# max_event_detections. Per-event cost is (live tracks x detections) gate
# evaluations plus one assignment over that matrix, so with the track population
# bounded this is the only term of it a sender still controls. Without a cap a
# single frame carrying a flood -- a mis-set sensor reporting its whole history,
# an injected feed, a replay drained unpaced -- makes one pass arbitrarily long
# and its hypothesis set arbitrarily large, which is a stall the engine cannot
# report because it never returns from the pass.
#
# The operating point: the reference network's densest documented instant is one
# node's sweep of ~80 objects. This carries nearly four times that, measured at
# 1.27 s an association pass against 0.44 s for 80, both at the default
# live-track bound. Raise it only with the frame window raised to match.
MAX_EVENT_DETECTIONS = 300

# How long a track may coast on prediction alone before it is dropped, and the
# default for the tracker's max_coast_seconds. See the constructor for how to
# size it against a sensor network's revisit interval.
MAX_COAST_SECONDS = 45.0

# Validation gate, in Mahalanobis distance (i.e. standard deviations of the
# measurement prediction, measurement noise included). A true hit exceeds 5 in
# three position dimensions about 1.5 times in 100,000, so the gate is loose
# enough that what falls outside it is a different object rather than a noisy
# report of this one -- which is what it has to be, since a detection that gates
# to nothing starts a track and a track that gates to nothing coasts.
GATE_MAHALANOBIS_DISTANCE = 5.0

# Gate, in the same units, for a detection carrying an objectId its sensor declares
# stable (Registration tracking_type) and already bound to a live track. The sensor
# has said which object this is, so the track it was last fused into takes it even
# a little outside the position gate -- that is the point of following the id. It
# is still a gate: past it the id is not trusted, since a sensor that reuses or
# confuses ids should cost an association, not drag a track across the picture.
OBJECT_ID_GATE_MAHALANOBIS_DISTANCE = 3 * GATE_MAHALANOBIS_DISTANCE

# Track confirmation. A detection that associates with nothing starts a track, and
# a single stray report -- clutter, a false alarm, a ghost -- would otherwise be a
# real track at once: published, and kept coasting for max_coast_seconds while its
# gate widens until it pulls in other stray reports kilometres away. A new track is
# therefore tentative until it has gathered CONFIRM_HITS detections (its first one
# included) within CONFIRM_WINDOW_SECONDS of starting; only confirmed tracks are
# returned to the caller, and a tentative track that misses the window is dropped.
# A real object is reported again and again, so it confirms within a few revisits;
# a stray report never does. The tracker's own default is 1 (no confirmation), which
# keeps its behaviour for callers that do not ask for it; the CLI asks for 3.
DEFAULT_CONFIRM_HITS = 3
CONFIRM_WINDOW_SECONDS = 15.0

# Height uncertainty (1-sigma, metres) of a track started by a detection that
# carries no height.
UNKNOWN_HEIGHT_SIGMA_M = 1000.0

# Position uncertainty (1-sigma, metres, worst axis) a track may carry before it
# is dropped, and the default for the tracker's max_coast_position_sigma. This is
# the coast horizon expressed as geometry instead of time, and it is the bound
# that survives a mis-sized horizon: under the shipped process noise a coasting
# track reaches ~1 km at 20 s, ~5.5 km at 45 s and ~40 km at 2 minutes, and the
# gate above is measured in the track's own sigmas -- so the wider a track's
# covariance grows the more of the picture it accepts, and the *smaller* its
# distance to any of it. Past a few kilometres such a track no longer says where
# its target is and can only take detections from tracks that do. Sized just
# above the default horizon so that at the default it is the horizon that binds.
MAX_COAST_POSITION_SIGMA_M = 6000.0

# How often to report each of the bounds being hit. They are reached by
# conditions that persist, so these are rate limits rather than one-shots, and
# each bound keeps its own deadline so a busy one cannot mask a quiet one.
LIMIT_REPORT_SECONDS = 60.0


def _object_key(detection):
    """(nodeId, objectId) of a detection whose sensor keeps stable ids, else None."""
    metadata = detection.metadata or {}
    object_id = metadata.get("objectId")
    if not metadata.get("stable_object_id") or not object_id:
        return None
    return metadata.get("nodeId"), object_id


def _position_sigma(state):
    """Worst-axis 1-sigma position uncertainty of a 9D state, in metres.

    The worst axis rather than an average: it is the widest reach of the state's
    validation gate that decides what it can pull in, and up is modelled with its
    own, smaller noise.
    """
    covar = state.covar
    return float(np.sqrt(max(covar[0, 0], covar[3, 3], covar[6, 6])))


class SapientAsynchronousTracker:
    """Fuses asynchronous multi-sensor detections into 9D tracks.

    Every bound here is a constructor parameter rather than a fixed constant
    because they trade against each other on a given node: the live-track limit
    is what keeps one fusion pass inside the frame window, history depth is what
    that limit's memory cost is spent on, the per-event detection cap is the
    other half of a pass's cost, and the two coast bounds are one horizon in two
    units. An operator moving one has to be able to move the others.

    Per-event work is bounded by max_live_tracks x max_event_detections gate
    evaluations plus one assignment of that size, and the track population by
    max_live_tracks -- so no input density makes a pass unbounded in time or in
    allocation. What that costs is association quality: past the caps detections
    are shed and tracks are refused, both reported.
    """

    def __init__(
        self,
        q_noise=0.2,
        p_init_variance=50.0,
        max_live_tracks: int = MAX_LIVE_TRACKS,
        max_track_history: int = MAX_TRACK_HISTORY,
        max_coast_seconds: float = MAX_COAST_SECONDS,
        max_event_detections: int = MAX_EVENT_DETECTIONS,
        max_coast_position_sigma: float = MAX_COAST_POSITION_SIGMA_M,
        confirm_hits: int = 1,
        confirm_window_seconds: float = CONFIRM_WINDOW_SECONDS,
    ):
        """Builds the fusion engine.

        max_coast_seconds is how long a track survives without a detection. It
        has to exceed the revisit interval of the sensors that see it -- two or
        three revisits to ride out sweeps that miss -- or tracks die between
        sweeps and every sweep re-initiates them. Every second above that is a
        track coasting on prediction alone with a gate that keeps widening, so
        max_coast_position_sigma bounds the same horizon in metres and is what
        holds when this one is set too long.
        """
        self.max_live_tracks = max_live_tracks
        self.max_track_history = max_track_history
        self.max_coast_seconds = max_coast_seconds
        self.max_event_detections = max_event_detections
        self.max_coast_position_sigma = max_coast_position_sigma
        if confirm_hits < 1:
            raise ValueError("confirm_hits must be >= 1")
        if confirm_window_seconds <= 0:
            raise ValueError("confirm_window_seconds must be > 0")
        self.confirm_hits = confirm_hits
        self.confirm_window_seconds = confirm_window_seconds
        # Tentative track id -> [detections gathered, time of its first detection].
        self._tentative = {}

        # 1. Instantiate the high-order Constant Acceleration transition model
        self.transition_model = create_9d_constant_acceleration_model(q_process_noise=q_noise)

        # 2. Setup Unscented Kalman Components to manage non-linear kinematics safely
        self.predictor = UnscentedKalmanPredictor(self.transition_model)
        self.updater = UnscentedKalmanUpdater(
            measurement_model=None
        )  # Assigned dynamically via Detections

        # 3. Setup data association: one validation gate per track-detection pair,
        # then one globally best assignment of detections to tracks.
        #
        # This engine updates a track with exactly one detection -- it picks a
        # hypothesis and hard-updates -- so the joint association probabilities a
        # JPDA computes are paid for and then discarded. Paid for without a
        # bound, at that: a gate is measured in the track's own sigmas, a
        # coasting track's covariance grows without limit, and once the widened
        # gates make every track and detection one connected cluster, exact JPDA
        # marginals (EHM2 included) are exponential in that cluster's size. It
        # was measured hanging inside a single call, and allocating to 9.8 GiB,
        # at 9 nodes x 80 objects -- while returning one detection per track.
        #
        # A Mahalanobis gate plus a 2D assignment answers the same question for
        # cost that is polynomial in both terms: tracks x detections gate
        # evaluations, and one rectangular assignment over that matrix.
        # GNNWith2DAssignment, not GlobalNearestNeighbour: the latter enumerates
        # the joint hypotheses with itertools.product, which is the same
        # exponential wall under a different name.
        #
        # What a hard assignment costs: inside a cluster tight enough that
        # several objects share a gate, the nearest detection is not reliably
        # the right one, so tracks trade identities. Measured against ground
        # truth on 10 objects within a 100 m radius: 36 tracks and 105 identity
        # swaps, versus 120 tracks (the live-track bound, saturated) under the
        # probabilistic associator. Object count and position stay usable and
        # per-object identity does not, so treat a track id inside a swarm as
        # provisional.
        self.hypothesiser = DistanceHypothesiser(
            predictor=self.predictor,
            updater=self.updater,
            measure=Mahalanobis(),
            missed_distance=GATE_MAHALANOBIS_DISTANCE,
        )
        self.data_associator = GNNWith2DAssignment(self.hypothesiser)

        # Track storage manifest
        self.tracks = set()
        self.p_init_val = p_init_variance
        # Last real detection per track id. Pruning needs it and track history is
        # capped, so it cannot be recovered by scanning states; see
        # _prune_stale_tracks.
        self._last_update_time = {}
        # (nodeId, objectId) -> id of the track that sensor's object was last fused
        # into, for sensors whose objectId is stable. Pruned with the tracks.
        self._object_tracks = {}
        self._refused_tracks = 0
        self._next_track_limit_report = 0.0
        self._shed_detections = 0
        self._next_detection_limit_report = 0.0
        self._dropped_wide_tracks = 0
        self._next_wide_coast_report = 0.0

    def process_async_event(self, timestamp, detection_group):
        """
        Ingests a completely asynchronous multi-target tracking frame from a single sensor node,
        calculates dynamic time deltas, maps associations, and updates state history.
        """
        # Bound the width of this pass before anything iterates over it.
        detection_group = self._limit_detections(detection_group)

        # If no tracks exist yet, bootstrap the tracking engine using current observations
        if not self.tracks:
            for det in detection_group:
                self._initialize_new_track(det)
            return self.confirmed_tracks()

        # A detection whose sensor's stable objectId is already bound to a live
        # track goes back to that track; only what is left competes on position.
        associations = self._associate_by_object_id(detection_group)
        bound_detections = {hypothesis.measurement for hypothesis in associations.values()}
        remaining_tracks = self.tracks - associations.keys()

        # Gate every remaining track against every remaining detection, then take the
        # assignment with the lowest total distance. Each track dynamically
        # calculates its own unique Delta-t prediction internally.
        if remaining_tracks:
            associations.update(
                self.data_associator.associate(
                    remaining_tracks, detection_group - bound_detections, timestamp
                )
            )

        associated_detections = set()

        for track, joint_hypothesis in associations.items():
            # One hypothesis per track: the assignment already chose it, and it is
            # falsy when the track was assigned its own missed-detection column.
            # A detection is assigned to at most one track, so no two tracks are
            # updated from the same hit.
            if not joint_hypothesis:
                # Track fell outside validation gates; coast forward using kinematics prediction
                prediction = self.predictor.predict(track.state, timestamp=timestamp)
                track.append(prediction)
            else:
                # High-Fidelity Step: Extract the measurement model embedded inside the SAPIENT detection
                det = joint_hypothesis.measurement
                associated_detections.add(det)

                # Wire the updater's internal mathematical model directly to the sensor specification
                self.updater.measurement_model = det.measurement_model

                # Execute Unscented Transform correction step
                # Special contextual kwargs handle dynamic acoustic location coordinate injection
                update_context = {}
                if "sensor_geodetic" in det.metadata:
                    update_context["sensor_geodetic"] = det.metadata["sensor_geodetic"]

                updated_state = self.updater.update(joint_hypothesis, **update_context)
                track.append(updated_state)
                self._last_update_time[track.id] = timestamp
                self._bind_object_id(det, track)
                self._count_confirmation_hit(track)

        # This event appended a state to every track above; cap what is retained.
        for live_track in self.tracks:
            self._trim_history(live_track)

        # Track Initiation: Handle unassociated hits to capture newly emerging threat vectors
        unassociated_hits = detection_group - associated_detections
        for raw_det in unassociated_hits:
            # Avoid seeding tracks off of partial 1D bearing lines to prevent false horizons
            if raw_det.state_vector.ndim == 1:
                continue
            self._initialize_new_track(raw_det)

        # Track Pruning: Drop stale tracks that have spent too long coasting without verification
        self._prune_stale_tracks(timestamp)

        return self.confirmed_tracks()

    def confirmed_tracks(self):
        """The live tracks that have passed confirmation; tentative ones are held back."""
        return {track for track in self.tracks if track.id not in self._tentative}

    def _count_confirmation_hit(self, track):
        """Counts a detection towards a tentative track's confirmation."""
        pending = self._tentative.get(track.id)
        if pending is None:
            return
        pending[0] += 1
        if pending[0] >= self.confirm_hits:
            del self._tentative[track.id]

    def _associate_by_object_id(self, detection_group):
        """Binds detections to the tracks their sensor's stable objectId already names.

        Returns track -> hypothesis, built the way the DistanceHypothesiser builds
        one, so the update path cannot tell the two kinds of association apart. A
        detection is bound only if its sensor declares stable ids, its id is
        bound to a live track that no other detection of this event has claimed,
        and it falls inside OBJECT_ID_GATE_MAHALANOBIS_DISTANCE of that track.
        Everything else is left to position-based association.
        """
        tracks_by_id = {track.id: track for track in self.tracks}
        bound = {}
        for det in detection_group:
            key = _object_key(det)
            if key is None:
                continue
            track = tracks_by_id.get(self._object_tracks.get(key))
            if track is None or track in bound:
                continue
            prediction = self.predictor.predict(track, timestamp=det.timestamp)
            measurement_prediction = self.updater.predict_measurement(
                prediction, det.measurement_model
            )
            distance = self.hypothesiser.measure(measurement_prediction, det)
            if distance > OBJECT_ID_GATE_MAHALANOBIS_DISTANCE:
                continue
            bound[track] = SingleDistanceHypothesis(
                prediction, det, distance, measurement_prediction
            )
        return bound

    def _bind_object_id(self, detection, track):
        """Remembers which track a stable objectId was fused into."""
        key = _object_key(detection)
        if key is not None:
            self._object_tracks[key] = track.id

    def _limit_detections(self, detection_group):
        """Caps how many detections one event is associated with.

        Both terms of a pass's cost have to be bounded for the pass to be
        bounded, and this is the one a sender controls: gating is tracks x
        detections, so a single frame carrying thousands of hits -- a mis-set
        sensor, an injected feed -- makes one pass arbitrarily long and its
        hypothesis set arbitrarily large. That is the worst failure available
        here, because a pass that never returns cannot report anything.

        Which detections survive is decided by position rather than by set
        iteration order, so a flood fuses the same way twice and the shed can be
        reasoned about after the fact. There is nothing better to sort on: at a
        flood the engine has no way to tell which of the hits are real.
        """
        if len(detection_group) <= self.max_event_detections:
            return detection_group

        self._report_detection_limit(len(detection_group))
        ordered = sorted(detection_group, key=lambda det: tuple(np.ravel(det.state_vector)))
        return set(ordered[: self.max_event_detections])

    def _initialize_new_track(self, detection):
        """Seeds a brand new 9D Constant Acceleration Gaussian state around a hit.

        Refused once max_live_tracks are live: the tracks already being followed
        are worth more than another one built from a hit that associated with
        nothing, and an unbounded population takes per-event cost past the event
        rate.
        """
        if len(self.tracks) >= self.max_live_tracks:
            self._report_track_limit()
            return
        e, n = detection.state_vector[0, 0], detection.state_vector[1, 0]
        # A detection without a height (east and north only) leaves the track's
        # height unknown: it starts at the frame's origin height with a sigma wide
        # enough that the first detection that does measure height sets it.
        has_height = detection.state_vector.shape[0] >= 3
        u = detection.state_vector[2, 0] if has_height else 0.0
        up_variance = 5.0 if has_height else UNKNOWN_HEIGHT_SIGMA_M**2

        # Position states initialized with measurement data; speed/acceleration set to zero
        state_vector = StateVector([e, 0.0, 0.0, n, 0.0, 0.0, u, 0.0, 0.0])

        # Build initial diagonal covariance block matrix
        covar = np.diag(
            [
                10.0,
                self.p_init_val,
                self.p_init_val / 2.0,  # East states [pos, vel, acc]
                10.0,
                self.p_init_val,
                self.p_init_val / 2.0,  # North states
                up_variance,
                self.p_init_val / 2.0,
                self.p_init_val / 4.0,  # Up states
            ]
        )

        prior = GaussianState(state_vector=state_vector, covar=covar, timestamp=detection.timestamp)
        # Seed the track's metadata from the hit that spawned it: Stone Soup only
        # accumulates metadata from Updates, so without this the track is
        # unclassified until its second detection.
        track = Track([prior], init_metadata=dict(detection.metadata))
        self.tracks.add(track)
        # The seeding hit is a real detection, so it anchors staleness.
        self._last_update_time[track.id] = detection.timestamp
        if self.confirm_hits > 1:
            self._tentative[track.id] = [1, detection.timestamp]
        self._bind_object_id(detection, track)

    def _report_track_limit(self) -> None:
        self._refused_tracks += 1
        if time.monotonic() < self._next_track_limit_report:
            return
        self._next_track_limit_report = time.monotonic() + LIMIT_REPORT_SECONDS
        logger.warning(
            "At the %d live-track limit; %d unassociated detection(s) have not started a track "
            "so far. Something is producing detections that do not associate -- clutter, a "
            "sensor whose clock disagrees, or an injected feed.",
            self.max_live_tracks,
            self._refused_tracks,
        )

    def _report_detection_limit(self, event_detections: int) -> None:
        self._shed_detections += event_detections - self.max_event_detections
        if time.monotonic() < self._next_detection_limit_report:
            return
        self._next_detection_limit_report = time.monotonic() + LIMIT_REPORT_SECONDS
        logger.warning(
            "An event carried %d detections, past the %d a single event is associated with; "
            "%d detection(s) have been shed so far. One sensor-instant that wide is a sensor "
            "reporting more than its sweep, or an injected feed. Raise --max-event-detections "
            "with the frame window if the picture really is this dense.",
            event_detections,
            self.max_event_detections,
            self._shed_detections,
        )

    def _report_wide_coast(self, dropped: int) -> None:
        self._dropped_wide_tracks += dropped
        if time.monotonic() < self._next_wide_coast_report:
            return
        self._next_wide_coast_report = time.monotonic() + LIMIT_REPORT_SECONDS
        logger.warning(
            "Dropped %d track(s) whose position uncertainty passed %.0f m, %d so far. They had "
            "coasted long enough to stop saying where their target is, and their gates were "
            "wide enough to take detections from tracks that still do. Sensors are not "
            "revisiting inside --max-coast-seconds (%.0f s).",
            dropped,
            self.max_coast_position_sigma,
            self._dropped_wide_tracks,
            self.max_coast_seconds,
        )

    def _trim_history(self, track):
        """Caps one track's retained history.

        Stone Soup keeps `states` and `metadatas` index-aligned -- each append
        adds one of each, and `track.metadata` is `metadatas[-1]` -- so both must
        be truncated by the same amount or the metadata a track reports stops
        belonging to its current state.
        """
        excess = len(track.states) - self.max_track_history
        if excess > 0:
            del track.states[:excess]
            del track.metadatas[:excess]

    def _prune_stale_tracks(self, current_time):
        """Purges tracks that have coasted too long, in time or in uncertainty.

        Staleness is measured from the last real detection, not from
        track.state.timestamp: coasting appends a prediction each event, so the
        latter always reads ~now and no track would ever expire. That last
        detection is read from explicit per-track bookkeeping rather than
        recovered by scanning history for the newest Update, because history is
        capped -- once a track has coasted past the cap the Update is gone and
        such a scan silently loses its anchor, leaving the track immortal.

        The second rule is the first one in metres, and it is the one that holds
        when the horizon is set too long for the sensors that feed it: a gate is
        measured in the track's own sigmas, so a track coasting with kilometre
        covariance accepts most of the picture at a smaller distance than a
        well-tracked target does, and wins detections away from it.
        """
        active_set = set()
        dropped_wide = 0
        for track in self.tracks:
            last_detection = self._last_update_time.get(track.id, track.states[0].timestamp)
            elapsed = (current_time - last_detection).total_seconds()
            if elapsed > self.max_coast_seconds:
                continue
            pending = self._tentative.get(track.id)
            if (
                pending is not None
                and (current_time - pending[1]).total_seconds() > self.confirm_window_seconds
            ):
                continue  # Never confirmed: a stray report, not an object.
            if _position_sigma(track.state) > self.max_coast_position_sigma:
                dropped_wide += 1
                continue
            active_set.add(track)
        if dropped_wide:
            self._report_wide_coast(dropped_wide)
        self.tracks = active_set
        # Forget pruned tracks, or the bookkeeping outlives them for the whole run.
        active_ids = {track.id for track in active_set}
        self._last_update_time = {
            track_id: seen
            for track_id, seen in self._last_update_time.items()
            if track_id in active_ids
        }
        self._object_tracks = {
            key: track_id for key, track_id in self._object_tracks.items() if track_id in active_ids
        }
        self._tentative = {
            track_id: pending
            for track_id, pending in self._tentative.items()
            if track_id in active_ids
        }
