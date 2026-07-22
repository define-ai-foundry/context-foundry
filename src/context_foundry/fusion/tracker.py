# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# src/fusion/tracker.py

import numpy as np
from stonesoup.dataassociator.probability import JPDAwithEHM2
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.predictor.kalman import UnscentedKalmanPredictor
from stonesoup.types.array import StateVector
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.types.update import Update
from stonesoup.updater.kalman import UnscentedKalmanUpdater

from .models import create_9d_constant_acceleration_model


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
        """Seeds a brand new 9D Constant Acceleration Gaussian state around a 3D Cartesian hit."""
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
        self.tracks.add(Track([prior]))

    def _prune_stale_tracks(self, current_time, max_coastal_seconds):
        """Purges tracks that haven't received physical sensor updates within the timeout window."""
        active_set = set()
        for track in self.tracks:
            # Measure staleness from the last real detection, not track.state.timestamp:
            # coasting appends a prediction each event, so the latter always reads ~now
            # and no track would ever expire.
            last_detection = track.states[0].timestamp
            for state in reversed(track.states):
                if isinstance(state, Update):
                    last_detection = state.timestamp
                    break
            elapsed = (current_time - last_detection).total_seconds()
            if elapsed <= max_coastal_seconds:
                active_set.add(track)
        self.tracks = active_set
