# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# src/fusion/tracker.py

import logging
import time

import numpy as np
from stonesoup.dataassociator.probability import JPDAwithEHM2
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.predictor.kalman import UnscentedKalmanPredictor
from stonesoup.types.array import StateVector
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.updater.kalman import UnscentedKalmanUpdater

from .models import create_9d_constant_acceleration_model

logger = logging.getLogger(__name__)

# States kept per track. Every event appends one state (and one metadata copy)
# to every live track, so an uncapped run grows for as long as it stays up --
# hundreds of MiB an hour at a few dozen tracks. Nothing downstream reads
# history: the augmentor reads track.state and the serializers read the tactical
# track built from it, so this only has to be deep enough to inspect a track's
# recent past. Kept small deliberately: at ~7.6 KiB a state it is this times
# MAX_LIVE_TRACKS that has to fit the memory limit, and measuring 20 against 200
# on a live feed showed 2 MiB between them.
MAX_TRACK_HISTORY = 20

# Live tracks held at once. Detections that do not associate each start a track,
# so clutter -- weather, birds, a sensor whose clock disagrees, a spoofer -- grows
# the population until per-event JPDA cost passes the event rate, and once fusion
# is behind the backlog keeps it there. Refusing new tracks past this bound sheds
# the clutter and keeps the ones already being followed; the population then
# recovers as coasting tracks are pruned.
MAX_LIVE_TRACKS = 250

# How often to report that the live-track bound is being hit. It is reached by a
# condition that persists, so this is a rate limit rather than a one-shot.
TRACK_LIMIT_REPORT_SECONDS = 60.0


class SapientAsynchronousTracker:
    def __init__(self, q_noise=0.2, p_init_variance=50.0):
        # 1. Instantiate the high-order Constant Acceleration transition model
        self.transition_model = create_9d_constant_acceleration_model(q_process_noise=q_noise)

        # 2. Setup Unscented Kalman Components to manage non-linear kinematics safely
        self.predictor = UnscentedKalmanPredictor(self.transition_model)
        self.updater = UnscentedKalmanUpdater(
            measurement_model=None
        )  # Assigned dynamically via Detections

        # 3. Setup joint probabilistic data association architectures for swarm management
        # clutter_spatial_density=None lets the hypothesiser derive it from each track's
        # validation-region volume; a fixed metre-scale value makes real hits lose to clutter.
        self.hypothesiser = PDAHypothesiser(
            predictor=self.predictor,
            updater=self.updater,
            clutter_spatial_density=None,
            prob_detect=0.9,
        )
        # EHM2 computes exact JPDA marginals without enumerating every joint hypothesis,
        # which is intractable (exponential in track count) for a multi-target swarm.
        self.data_associator = JPDAwithEHM2(self.hypothesiser)

        # Track storage manifest
        self.tracks = set()
        self.p_init_val = p_init_variance
        # Last real detection per track id. Pruning needs it and track history is
        # capped, so it cannot be recovered by scanning states; see
        # _prune_stale_tracks.
        self._last_update_time = {}
        self._refused_tracks = 0
        self._next_track_limit_report = 0.0

    def process_async_event(self, timestamp, detection_group):
        """
        Ingests a completely asynchronous multi-target tracking frame from a single sensor node,
        calculates dynamic time deltas, maps probabilistic associations, and updates state history.
        """
        # If no tracks exist yet, bootstrap the tracking engine using current observations
        if not self.tracks:
            for det in detection_group:
                self._initialize_new_track(det)
            return self.tracks

        # Execute JPDA association matrix solver
        # Each track dynamically calculates its own unique Delta-t prediction internally
        associations = self.data_associator.associate(self.tracks, detection_group, timestamp)

        associated_detections = set()

        for track, hypotheses in associations.items():
            # Pick the most probable association. JPDA.associate inserts the
            # missed-detection hypothesis first, so hypotheses[0] is never the
            # likeliest; using it left every detection unassociated, spawning a
            # new track per hit until JPDA's joint enumeration blew up.
            joint_hypothesis = max(hypotheses, key=lambda hypothesis: hypothesis.probability)

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
        self._prune_stale_tracks(timestamp, max_coastal_seconds=45.0)

        return self.tracks

    def _initialize_new_track(self, detection):
        """Seeds a brand new 9D Constant Acceleration Gaussian state around a 3D Cartesian hit.

        Refused once MAX_LIVE_TRACKS are live: the tracks already being followed
        are worth more than another one built from a hit that associated with
        nothing, and an unbounded population takes per-event cost past the event
        rate.
        """
        if len(self.tracks) >= MAX_LIVE_TRACKS:
            self._report_track_limit()
            return
        e, n, u = (
            detection.state_vector[0, 0],
            detection.state_vector[1, 0],
            detection.state_vector[2, 0],
        )

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
                5.0,
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

    def _report_track_limit(self) -> None:
        self._refused_tracks += 1
        if time.monotonic() < self._next_track_limit_report:
            return
        self._next_track_limit_report = time.monotonic() + TRACK_LIMIT_REPORT_SECONDS
        logger.warning(
            "At the %d live-track limit; %d unassociated detection(s) have not started a track "
            "so far. Something is producing detections that do not associate -- clutter, a "
            "sensor whose clock disagrees, or an injected feed.",
            MAX_LIVE_TRACKS,
            self._refused_tracks,
        )

    def _trim_history(self, track):
        """Caps one track's retained history.

        Stone Soup keeps `states` and `metadatas` index-aligned -- each append
        adds one of each, and `track.metadata` is `metadatas[-1]` -- so both must
        be truncated by the same amount or the metadata a track reports stops
        belonging to its current state.
        """
        excess = len(track.states) - MAX_TRACK_HISTORY
        if excess > 0:
            del track.states[:excess]
            del track.metadatas[:excess]

    def _prune_stale_tracks(self, current_time, max_coastal_seconds):
        """Purges tracks that haven't received physical sensor updates within the timeout window.

        Staleness is measured from the last real detection, not from
        track.state.timestamp: coasting appends a prediction each event, so the
        latter always reads ~now and no track would ever expire. That last
        detection is read from explicit per-track bookkeeping rather than
        recovered by scanning history for the newest Update, because history is
        capped -- once a track has coasted past the cap the Update is gone and
        such a scan silently loses its anchor, leaving the track immortal.
        """
        active_set = set()
        for track in self.tracks:
            last_detection = self._last_update_time.get(track.id, track.states[0].timestamp)
            elapsed = (current_time - last_detection).total_seconds()
            if elapsed <= max_coastal_seconds:
                active_set.add(track)
        self.tracks = active_set
        # Forget pruned tracks, or the bookkeeping outlives them for the whole run.
        active_ids = {track.id for track in active_set}
        self._last_update_time = {
            track_id: seen
            for track_id, seen in self._last_update_time.items()
            if track_id in active_ids
        }
