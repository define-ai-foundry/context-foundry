# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# src/fusion/tracker.py

import logging
import time
import uuid
from collections import deque
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import NamedTuple

import numpy as np
from stonesoup.base import Property
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.models.base import LinearModel
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.predictor._utils import predict_lru_cache
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.types.array import CovarianceMatrix, StateVector
from stonesoup.types.hypothesis import SingleDistanceHypothesis
from stonesoup.types.prediction import Prediction
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.updater.kalman import KalmanUpdater, UnscentedKalmanUpdater

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

# How far back a track's swarm count reaches: the largest count any of its detections
# gave within this many seconds. Sensors see a swarm differently -- one counts the swarm,
# another each drone in it, a CoT event nothing at all -- so the latest detection's count
# would flicker between them. Longer than a swarm-counting sensor's revisit (a strategic
# radar's is about 18 s), short enough that a swarm that splits shows it within a minute.
SWARM_COUNT_WINDOW_SECONDS = 60.0

# Track metadata keys the tracker writes from its own per-hit bookkeeping, for the
# augmentor's threat judgement. No detection carries them, so Stone Soup's merge of a
# hit's metadata into the track never overwrites them.
# DECLARATION_KEY holds the (CoT type, fresh until, east, north, radius, made at, velocity
# east, velocity north) of the latest self-report of the uid bound to the track, if one is:
# where the report placed the track and how fast the track was going then. The declaration
# stands while it is fresh and the track within `radius` metres of where that position,
# carried on at that velocity, has got to. SENSOR_HIT_AT_KEY holds the time of the track's
# latest non-CoT hit.
DECLARATION_KEY = "cot_declaration"
SENSOR_HIT_AT_KEY = "sensor_hit_at"

# A self-report's declaration lasts until its event's own `stale`, but at least and at
# most this long after its event time; an event without a readable `stale` gets the least.
DECLARATION_MIN_SECONDS = 60.0
DECLARATION_MAX_SECONDS = 600.0

# A declaration speaks for its track only near where it was made: within this many of the
# report's own horizontal errors (its CoT ce), and never less than DECLARATION_MIN_RADIUS_M.
DECLARATION_RADIUS_ERRORS = 3.0
DECLARATION_MIN_RADIUS_M = 50.0

# The `how` of an emplaced sensor reporting its surveyed position. Its track is held still:
# no velocity, and process noise small enough that its gate stays the size of its error
# for as long as the track lives, so another object passing cannot take it over.
STILL_HOW = "m-i"
STILL_PROCESS_NOISE = 1e-10
STILL_VELOCITY_VARIANCE = 1e-6
STILL_ACCELERATION_VARIANCE = 1e-8

# The CoT `how` of an object reporting its own position: from its GPS (`m-g`), fused from
# its own sensors (`m-f`), surveyed, as an emplaced sensor reports itself (`m-i`), or
# entered by its user without a fix (`h-e`, as ATAK reports itself on losing GPS). How
# values refine by dash (`m-g-l`), so a refinement counts too. A relayed (`m-r`),
# predicted (`m-p`), simulated (`m-s`) or hand-placed (`h-g-i-g-o`) position is another
# party's report of the object and declares nothing.
SELF_REPORT_HOWS = ("m-g", "m-f", "m-i", "h-e")


def is_self_report(how) -> bool:
    """Whether a CoT `how` is that of an object reporting its own position."""
    how = how if isinstance(how, str) else ""
    return any(how == base or how.startswith(base + "-") for base in SELF_REPORT_HOWS)


def declared_until(stale, event_time):
    """Until when a self-report made at `event_time` declares, from its CoT `stale`.

    `stale` is an ISO 8601 string or a datetime; it is clamped to between
    DECLARATION_MIN_SECONDS and DECLARATION_MAX_SECONDS after `event_time`.
    """
    seconds = DECLARATION_MIN_SECONDS
    try:
        if isinstance(stale, str):
            stale = datetime.fromisoformat(stale.strip().replace("Z", "+00:00"))
        if isinstance(stale, datetime):
            seconds = (stale - event_time).total_seconds()
    except (TypeError, ValueError):
        pass  # Unreadable, or naive against aware: the least.
    seconds = min(max(seconds, DECLARATION_MIN_SECONDS), DECLARATION_MAX_SECONDS)
    return event_time + timedelta(seconds=seconds)


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
# keeps its behaviour for callers that do not ask for it; the CLI asks for
# DEFAULT_CONFIRM_HITS.
#
# How many reports, and how long to gather them: 3 reports in 30 s shows a real
# object quickly -- within about 36 s on the simulated Joensuu attack, against 72 s
# for 5 reports in 60 s -- with the same tracks and accuracy on the four simulated
# scenarios. The window has to hold three revisits of the slowest sensor that may
# see an object first: long-range radars revisit every 15-30 s, and with 15 s an
# object only they see could not be confirmed at all -- on the Joensuu attack five
# of six swarms went unshown for 7 to 16 minutes. The cost is noisy sensors: with
# 300 random false reports in two minutes, 3 reports let through 65-74 ghost
# tracks, 5 reports in 60 s only 6-7.
#
# A CoT object reporting its own position (see is_self_report) is no stray report: its
# track is confirmed by its first self-report, whatever the setting.
DEFAULT_CONFIRM_HITS = 3
CONFIRM_WINDOW_SECONDS = 30.0

# Variance (m^2/s^2, east and north) of a new track's velocity, 100 m/s 1-sigma. One
# detection gives a position and no speed, so a track starts at rest, and this is
# how far the next report may be from where it started. About 7 m/s, the old value,
# put a 833 m/s missile's next report -- 4 km on, five seconds later -- far outside
# the gate, so every report started a track of its own. 75-200 m/s followed a fast
# object even from a single radar revisiting every 5 s and left the drone scenarios
# unchanged; 100 m/s is the middle of that range.
DEFAULT_INITIAL_VELOCITY_VARIANCE = 10000.0

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


def _self_report_key(detection):
    """The object key of a CoT self-report with a stable uid, else None."""
    metadata = detection.metadata or {}
    if metadata.get("type") != "CoT" or not is_self_report(metadata.get("how")):
        return None
    return _object_key(detection)


def _is_still_report(detection) -> bool:
    """Whether a detection is a self-report of an emplaced sensor's surveyed position."""
    how = (detection.metadata or {}).get("how")
    return (
        _self_report_key(detection) is not None
        and isinstance(how, str)
        and (how == STILL_HOW or how.startswith(STILL_HOW + "-"))
    )


class _Declaration(NamedTuple):
    """The latest self-report of the uid bound to a track."""

    key: tuple
    cot_type: str | None
    until: datetime
    east: float
    north: float
    radius: float
    still: bool
    # When the report was made, and when the uid's first self-report on this track was.
    at: datetime
    since: datetime


def _hold_still(state) -> None:
    """Sets a 9D state's velocity and acceleration to zero, known to near certainty."""
    for index in (1, 2, 4, 5, 7, 8):
        state.state_vector[index, 0] = 0.0
        state.covar[index, :] = 0.0
        state.covar[:, index] = 0.0
        acceleration = index in (2, 5, 8)
        state.covar[index, index] = (
            STILL_ACCELERATION_VARIANCE if acceleration else STILL_VELOCITY_VARIANCE
        )


def _position_sigma(state):
    """Worst-axis 1-sigma position uncertainty of a 9D state, in metres.

    The worst axis rather than an average: it is the widest reach of the state's
    validation gate that decides what it can pull in, and up is modelled with its
    own, smaller noise.
    """
    covar = state.covar
    return float(np.sqrt(max(covar[0, 0], covar[3, 3], covar[6, 6])))


class _LinearWhereLinearUpdater(UnscentedKalmanUpdater):
    """Unscented where a measurement model is not linear, the Kalman equations where it is.

    Every position report is a linear model of the track's position, and for a linear
    model the unscented transform gives the same measurement prediction through 19 sigma
    points. Association predicts a measurement for every track and detection of an event,
    so this was the second largest cost of fusing. A non-linear model, a bearing-only
    sensor's, still goes through the unscented transform.
    """

    def predict_measurement(
        self, predicted_state, measurement_model=None, measurement_noise=True, **kwargs
    ):
        model = self._check_measurement_model(measurement_model)
        if isinstance(model, LinearModel):
            return KalmanUpdater.predict_measurement(
                self, predicted_state, model, measurement_noise, **kwargs
            )
        return super().predict_measurement(predicted_state, model, measurement_noise, **kwargs)


class _ExcludingHypothesiser(DistanceHypothesiser):
    """A DistanceHypothesiser that leaves out the detections `excludes(track, det)` names.

    A left-out detection gets no hypothesis on that track, so the assignment cannot give
    it that track and it goes to another one or starts its own.
    """

    excludes: Callable = Property(doc="(track, detection) -> whether the track may not take it.")

    def hypothesise(self, track, detections, timestamp, **kwargs):
        detections = {det for det in detections if not self.excludes(track, det)}
        return super().hypothesise(track, detections, timestamp, **kwargs)


class _ConstantAccelerationPredictor(KalmanPredictor):
    """The Kalman predictor for three constant-acceleration axes, in plain numpy.

    Coasting predicts every live track the event does not update, so prediction is most
    of what an event costs, and the general path rebuilt the 9x9 transition matrix and
    noise covariance through each axis model and scipy's block_diag on every call. For
    this model both are fixed formulas of the time step, so they are written out here:
    per axis F = [[1, dt, dt^2/2], [0, 1, dt], [0, 0, 1]] and Q = q [[dt^5/20, dt^4/8,
    dt^3/6], [dt^4/8, dt^3/3, dt^2/2], [dt^3/6, dt^2/2, dt]], with dt the absolute step
    in Q as in Stone Soup's ConstantNthDerivative. A prediction without a time step,
    with a control input or extra arguments takes the general path.
    """

    def __init__(self, transition_model, *args, **kwargs):
        super().__init__(transition_model, *args, **kwargs)
        self._noise = [model.noise_diff_coeff for model in transition_model.model_list]

    @predict_lru_cache()
    def predict(self, prior, timestamp=None, control_input=None, **kwargs):
        if timestamp is None or prior.timestamp is None or control_input is not None or kwargs:
            return super().predict(
                prior, timestamp=timestamp, control_input=control_input, **kwargs
            )
        dt = (timestamp - prior.timestamp).total_seconds()
        step = abs(dt)
        f = np.array([[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
        q = np.array(
            [
                [step**5 / 20.0, step**4 / 8.0, step**3 / 6.0],
                [step**4 / 8.0, step**3 / 3.0, step**2 / 2.0],
                [step**3 / 6.0, step**2 / 2.0, step],
            ]
        )
        transition = np.zeros((9, 9))
        noise = np.zeros((9, 9))
        for axis in range(3):
            block = slice(3 * axis, 3 * axis + 3)
            transition[block, block] = f
            noise[block, block] = self._noise[axis] * q
        mean = transition @ np.asarray(prior.state_vector, dtype=float)
        covar = transition @ np.asarray(prior.covar, dtype=float) @ transition.T + noise
        return Prediction.from_state(
            prior,
            StateVector(mean),
            CovarianceMatrix(covar),
            timestamp=timestamp,
            transition_model=self.transition_model,
            prior=prior,
        )


class _TrackwisePredictor(_ConstantAccelerationPredictor):
    """Predicts a track `is_still(track)` names with `still`, anything else as usual.

    Both are set once built. Only a Track, as association and coasting pass it, is told
    apart; a bare state is predicted as usual.
    """

    still = None
    is_still = None

    def predict(self, prior, *args, **kwargs):
        if isinstance(prior, Track) and self.is_still is not None and self.is_still(prior):
            return self.still.predict(prior, *args, **kwargs)
        return super().predict(prior, *args, **kwargs)


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
        initial_velocity_variance: float = DEFAULT_INITIAL_VELOCITY_VARIANCE,
        max_live_tracks: int = MAX_LIVE_TRACKS,
        max_track_history: int = MAX_TRACK_HISTORY,
        max_coast_seconds: float = MAX_COAST_SECONDS,
        max_event_detections: int = MAX_EVENT_DETECTIONS,
        max_coast_position_sigma: float = MAX_COAST_POSITION_SIGMA_M,
        confirm_hits: int = 1,
        confirm_window_seconds: float = CONFIRM_WINDOW_SECONDS,
        id_namespace: str = "",
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
        # What makes this tracker's track ids its own; see _seed_id.
        self.id_namespace = id_namespace
        # Tentative track id -> [detections gathered, time of its first detection].
        self._tentative = {}

        # 1. Instantiate the high-order Constant Acceleration transition model
        self.transition_model = create_9d_constant_acceleration_model(q_process_noise=q_noise)
        # Each axis's process noise, for the pre-gate, which predicts a position and its
        # variance without building a prediction.
        self._axis_noise = [model.noise_diff_coeff for model in self.transition_model.model_list]

        # 2. Predict with the linear Kalman equations: the transition model is linear,
        # so the unscented transform gives the same mean and covariance through 19
        # sigma points, and it was the largest single cost of association -- every
        # event predicts every live track. The update stays unscented, since a
        # sensor's measurement model need not be linear.
        # A track an emplaced sensor's self-report holds still is predicted with almost no
        # process noise; see STILL_HOW.
        self._still_model = create_9d_constant_acceleration_model(
            q_process_noise=STILL_PROCESS_NOISE
        )
        self._still_noise = [model.noise_diff_coeff for model in self._still_model.model_list]
        self.predictor = _TrackwisePredictor(self.transition_model)
        self.predictor.still = _ConstantAccelerationPredictor(self._still_model)
        self.predictor.is_still = self._is_still
        self.updater = _LinearWhereLinearUpdater(
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
        #
        # Two uids reporting their own positions are two objects, however close, so a
        # track bound to one never takes the other's self-report; see _excluded.
        self.hypothesiser = _ExcludingHypothesiser(
            predictor=self.predictor,
            updater=self.updater,
            measure=Mahalanobis(),
            missed_distance=GATE_MAHALANOBIS_DISTANCE,
            excludes=self._excluded,
        )
        self.data_associator = GNNWith2DAssignment(self.hypothesiser)

        # Track storage manifest
        self.tracks = set()
        self.p_init_val = p_init_variance
        # Horizontal velocity only: a fast object needs room to be moving, but
        # widening the acceleration with it lets a once-seen track's uncertainty
        # grow so fast that it is dropped long before its coast horizon.
        self.initial_velocity_variance = initial_velocity_variance
        # Last real detection per track id. Pruning needs it and track history is
        # capped, so it cannot be recovered by scanning states; see
        # _prune_stale_tracks.
        self._last_update_time = {}
        # (nodeId, objectId) -> id of the track that sensor's object was last fused
        # into, for sensors whose objectId is stable. Pruned with the tracks.
        self._object_tracks = {}
        # Track id -> (timestamp, swarm count) its detections gave within the window.
        self._swarm_counts: dict[str, deque] = {}
        # Track id -> (object key, CoT type, fresh until) of the one self-reporting uid
        # bound to the track and its latest self-report; see _note_threat_evidence.
        self._declarations: dict[str, tuple] = {}
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

        # Drop tracks already past their coast horizon before they can be matched: one
        # silent that long has a covariance wide enough to win a detection of another
        # object anywhere, which would inherit its id, classification and threat. Only
        # that rule: a tentative track still gets this event's detection before it is
        # judged unconfirmed, as before.
        self._drop_coasted_out(timestamp)
        self._release_expired(timestamp)

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
        remaining_detections = detection_group - bound_detections

        # Gate the remaining tracks that can reach a remaining detection, then take the
        # assignment with the lowest total distance. A track that provably reaches none
        # could only be assigned its missed-detection column, so leaving it out changes
        # no assignment; it is what keeps an event's cost to the tracks near it rather
        # than every track in the picture.
        reachable = {
            track for track in remaining_tracks if self._may_gate(track, remaining_detections)
        }
        if reachable:
            associations.update(
                self.data_associator.associate(reachable, remaining_detections, timestamp)
            )
        for track in remaining_tracks - reachable:
            track.append(self.predictor.predict(track, timestamp=timestamp))

        associated_detections = set()

        for track, joint_hypothesis in associations.items():
            # One hypothesis per track: the assignment already chose it, and it is
            # falsy when the track was assigned its own missed-detection column.
            # A detection is assigned to at most one track, so no two tracks are
            # updated from the same hit.
            if not joint_hypothesis:
                # Track fell outside validation gates; coast forward using kinematics prediction
                prediction = self.predictor.predict(track, timestamp=timestamp)
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
                if _is_still_report(det):
                    _hold_still(updated_state)
                track.append(updated_state)
                self._last_update_time[track.id] = timestamp
                self._hold_swarm_count(track, det, timestamp)
                self._note_threat_evidence(track, det, timestamp)
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

    def _may_gate(self, track, detections):
        """Whether a track could fall inside the validation gate of any of the detections.

        A conservative bound, never a false no: the squared Mahalanobis distance is at
        least the squared innovation over the largest eigenvalue of its covariance, and
        that eigenvalue is at most the trace. So a track whose squared distance from a
        detection exceeds the gate squared times that trace (its predicted position
        variance plus the detection's noise, on the measured axes) is outside the gate.
        Worked in plain floats from the track's state, without building a prediction.
        A detection whose model is not a linear position model, or that is older than
        the track's state, is always gated.
        """
        state = track.state
        sv = state.state_vector.ravel().tolist()
        covar = state.covar.tolist()
        axis_noise = self._still_noise if self._is_still(track) else self._axis_noise
        limit = GATE_MAHALANOBIS_DISTANCE**2 * (1.0 + 1e-6)
        for det in detections:
            model = getattr(det, "measurement_model", None)
            mapping = getattr(model, "mapping", None)
            if not isinstance(model, LinearGaussian) or any(m not in (0, 3, 6) for m in mapping):
                return True
            dt = (det.timestamp - state.timestamp).total_seconds()
            if dt < 0:
                return True
            measured = det.state_vector.ravel().tolist()
            noise = model.noise_covar
            miss = spread = 0.0
            for row, index in enumerate(mapping):
                position, variance = self._predicted_axis(sv, covar, index, dt, axis_noise)
                miss += (measured[row] - position) ** 2
                spread += variance + float(noise[row, row])
            if miss <= limit * spread:
                return True
        return False

    @staticmethod
    def _predicted_axis(sv, covar, index, dt, axis_noise):
        """Predicted position and its variance on one axis, `dt` seconds on."""
        h = (1.0, dt, 0.5 * dt * dt)
        position = sv[index] + sv[index + 1] * dt + sv[index + 2] * h[2]
        variance = sum(
            h[i] * h[j] * covar[index + i][index + j] for i in range(3) for j in range(3)
        )
        variance += axis_noise[index // 3] * dt**5 / 20.0
        return position, variance

    def confirmed_tracks(self):
        """The live tracks that have passed confirmation; tentative ones are held back."""
        return {track for track in self.tracks if track.id not in self._tentative}

    def _count_confirmation_hit(self, track):
        """Counts a detection towards a tentative track's confirmation.

        A track that took a self-report, and so is bound to a self-reporting uid, is
        confirmed at once.
        """
        pending = self._tentative.get(track.id)
        if pending is None:
            return
        pending[0] += 1
        if pending[0] >= self.confirm_hits or track.id in self._declarations:
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
            if track is None or track in bound or self._excluded(track, det):
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
        """Remembers which track a stable objectId was fused into.

        Only a self-report moves a self-reporting uid: another party's report of it, such
        as a relay, updates whichever track it went to and leaves the uid bound where it is.
        An object moving to another track takes its declaration with it: the track it left
        is no longer bound to it, so it no longer speaks for that track, and the track may
        take another uid's self-reports.
        """
        key = _object_key(detection)
        if key is None:
            return
        previous = self._object_tracks.get(key)
        if previous is None or previous == track.id:
            self._object_tracks[key] = track.id
            return
        declared = self._declarations.get(previous)
        if declared is not None and declared.key == key:
            if _self_report_key(detection) is None:
                return
            self._undeclare(previous)
        self._object_tracks[key] = track.id

    def _undeclare(self, track_id) -> None:
        """Forget the declaration of the uid bound to a track, if the track is still live."""
        del self._declarations[track_id]
        for live in self.tracks:
            if live.id == track_id:
                live.metadata.pop(DECLARATION_KEY, None)

    def _excluded(self, track, detection) -> bool:
        """Whether the track is bound to a self-reporting uid other than the detection's.

        Only a self-report is kept off such a track: a sensor's detection, or a CoT event
        that is another party's report of an object, may still go to it.
        """
        declared = self._declarations.get(track.id)
        if declared is None:
            return False
        key = _self_report_key(detection)
        return key is not None and key != declared.key

    def _is_still(self, track) -> bool:
        """Whether the track is held still by its uid's latest self-report; see STILL_HOW."""
        declared = self._declarations.get(track.id)
        return declared is not None and declared.still

    def _note_threat_evidence(self, track, detection, timestamp) -> None:
        """Record what a hit tells the threat judgement: a declaration or a sensor's sighting.

        A hit from anything but CoT is a sensor's sighting. A CoT hit is a declaration only
        as a self-report (see is_self_report) from an event with a stable uid, the uid it is
        bound to by this hit; a marker someone placed on the object declares nothing. The
        uid's latest self-report is kept with how long it stands (see declared_until) and
        where it was made, give or take DECLARATION_RADIUS_ERRORS of its own error, with
        the track's velocity after it, and written to the track's metadata under the
        tracker's own key.
        """
        metadata = detection.metadata or {}
        if metadata.get("type") != "CoT":
            track.metadata[SENSOR_HIT_AT_KEY] = timestamp
            return
        key = _self_report_key(detection)
        if key is None:
            return
        cot_type = metadata.get("cot_type")
        until = declared_until(metadata.get("stale"), timestamp)
        noise = detection.measurement_model.noise_covar
        error = float(np.sqrt(max(noise[0, 0], noise[1, 1])))
        radius = max(DECLARATION_RADIUS_ERRORS * error, DECLARATION_MIN_RADIUS_M)
        east, north = float(detection.state_vector[0, 0]), float(detection.state_vector[1, 0])
        previous = self._declarations.get(track.id)
        since = previous.since if previous is not None and previous.key == key else timestamp
        self._declarations[track.id] = _Declaration(
            key,
            cot_type,
            until,
            east,
            north,
            radius,
            _is_still_report(detection),
            timestamp,
            since,
        )
        velocity = track.state.state_vector
        track.metadata[DECLARATION_KEY] = (
            cot_type,
            until,
            east,
            north,
            radius,
            timestamp,
            float(velocity[1, 0]),
            float(velocity[4, 0]),
        )

    def _release_expired(self, timestamp) -> None:
        """Release each uid whose declaration has gone stale from the track it is bound to.

        The track stays, judged by the rule, unless the track of another uid that began
        reporting itself after this uid's last self-report lies inside its gate: then it is
        that object, seen again under a new uid, and is dropped, so what sensors see of it
        goes to the uid that still reports. Another uid reporting all along beside it, a
        vehicle in the same convoy, is another object and leaves the track be.
        """
        expired = [
            track_id
            for track_id, declared in self._declarations.items()
            if timestamp > declared.until
        ]
        if not expired:
            return
        tracks_by_id = {track.id: track for track in self.tracks}
        released = []
        for track_id in expired:
            declared = self._declarations[track_id]
            self._undeclare(track_id)
            if self._object_tracks.get(declared.key) == track_id:
                del self._object_tracks[declared.key]
            if track_id in tracks_by_id:
                released.append((tracks_by_id[track_id], declared.at))
        reporting = [
            (tracks_by_id[track_id], declared.since)
            for track_id, declared in self._declarations.items()
            if track_id in tracks_by_id
        ]
        gone = {
            track
            for track, last in released
            if any(
                since > last and self._gates(track, other, timestamp) for other, since in reporting
            )
        }
        if gone:
            self._forget(gone)

    def _gates(self, track, other, timestamp) -> bool:
        """Whether `other`'s position falls inside `track`'s validation gate at `timestamp`."""
        mine = self.predictor.predict(track, timestamp=timestamp)
        theirs = self.predictor.predict(other, timestamp=timestamp)
        axes = [0, 3, 6]
        miss = np.asarray(mine.state_vector, dtype=float)[axes, 0]
        miss = miss - np.asarray(theirs.state_vector, dtype=float)[axes, 0]
        spread = np.asarray(mine.covar, dtype=float)[np.ix_(axes, axes)]
        spread = spread + np.asarray(theirs.covar, dtype=float)[np.ix_(axes, axes)]
        return float(miss @ np.linalg.solve(spread, miss)) <= GATE_MAHALANOBIS_DISTANCE**2

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
                self.initial_velocity_variance,
                self.p_init_val / 2.0,  # East states [pos, vel, acc]
                10.0,
                self.initial_velocity_variance,
                self.p_init_val / 2.0,  # North states
                up_variance,
                self.p_init_val / 2.0,
                self.p_init_val / 4.0,  # Up states
            ]
        )

        prior = GaussianState(state_vector=state_vector, covar=covar, timestamp=detection.timestamp)
        if _is_still_report(detection):
            _hold_still(prior)
        # Seed the track's metadata from the hit that spawned it: Stone Soup only
        # accumulates metadata from Updates, so without this the track is
        # unclassified until its second detection.
        track = Track([prior], id=self._seed_id(detection), init_metadata=dict(detection.metadata))
        self.tracks.add(track)
        # The seeding hit is a real detection, so it anchors staleness.
        self._last_update_time[track.id] = detection.timestamp
        self._hold_swarm_count(track, detection, detection.timestamp)
        self._note_threat_evidence(track, detection, detection.timestamp)
        if self.confirm_hits > 1 and track.id not in self._declarations:
            self._tentative[track.id] = [1, detection.timestamp]
        self._bind_object_id(detection, track)

    def _hold_swarm_count(self, track, detection, timestamp) -> None:
        """Set the track's swarm count to the largest its detections gave within the window.

        A detection without a count, a CoT event's, adds nothing and leaves the window's
        largest in place. Once the window holds no count the track has none, so the
        augmentor falls back to its default rather than an old count carried forward.
        """
        seen = self._swarm_counts.setdefault(track.id, deque())
        count = (detection.metadata or {}).get("swarm_count")
        if isinstance(count, int) and not isinstance(count, bool):
            seen.append((timestamp, count))
        while seen and (timestamp - seen[0][0]).total_seconds() > SWARM_COUNT_WINDOW_SECONDS:
            seen.popleft()
        if seen:
            track.metadata["swarm_count"] = max(c for _, c in seen)
        else:
            track.metadata.pop("swarm_count", None)

    def _seed_id(self, detection) -> str:
        """A track id fixed by the detection that starts the track, within `id_namespace`.

        Fed the same detections again -- a pipeline stage rebuilding its tracks from a
        re-read window after a restart -- the tracker gives each rebuilt track the id it
        had before, so what it publishes next continues that track rather than starting
        a second one beside it. The id is made of what identifies the report itself --
        its sensor, object id, report id and time, or for a sensor with stable object ids
        its sensor and object id alone -- never its position, which is in a tracking frame
        whose origin a restart moves. The namespace keeps two trackers fed
        the same reports, such as two pipelines' fusion stages, from minting the same ids.
        Two tracks seeded alike while both live are told apart by a count, in the order
        they are seeded, which a replay repeats.
        """
        metadata = detection.metadata or {}
        if metadata.get("stable_object_id") and metadata.get("objectId"):
            # A sensor that keeps one object id for one object names the track by itself,
            # so a track rebuilt from a later report of the same object keeps its id even
            # when the report that first started it lies before the rebuild window.
            parts = (self.id_namespace, str(metadata.get("nodeId")), str(metadata["objectId"]))
        else:
            parts = (
                self.id_namespace,
                str(metadata.get("nodeId")),
                str(metadata.get("objectId")),
                str(metadata.get("reportId")),
                detection.timestamp.isoformat(),
            )
        seed = "|".join(parts)
        live = {track.id for track in self.tracks}
        candidate = str(uuid.uuid5(uuid.NAMESPACE_URL, f"context-foundry-track:{seed}"))
        count = 1
        while candidate in live:
            count += 1
            candidate = str(uuid.uuid5(uuid.NAMESPACE_URL, f"context-foundry-track:{seed}#{count}"))
        return candidate

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

    def _drop_coasted_out(self, current_time) -> None:
        """Drop the tracks silent for longer than max_coast_seconds, and what is kept of them."""
        gone = {
            track
            for track in self.tracks
            if (
                current_time - self._last_update_time.get(track.id, track.states[0].timestamp)
            ).total_seconds()
            > self.max_coast_seconds
        }
        if gone:
            self._forget(gone)

    def _forget(self, gone) -> None:
        """Drop these tracks and everything kept of them."""
        self.tracks -= gone
        ids = {track.id for track in gone}
        books = (
            self._last_update_time,
            self._swarm_counts,
            self._tentative,
            self._declarations,
        )
        for book in books:
            for track_id in ids:
                book.pop(track_id, None)
        self._object_tracks = {k: v for k, v in self._object_tracks.items() if v not in ids}

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
        self._swarm_counts = {
            track_id: seen
            for track_id, seen in self._swarm_counts.items()
            if track_id in active_ids
        }
        self._tentative = {
            track_id: pending
            for track_id, pending in self._tentative.items()
            if track_id in active_ids
        }
        self._declarations = {
            track_id: declared
            for track_id, declared in self._declarations.items()
            if track_id in active_ids
        }
