"""An event's detections assigned to the tracks they can reach, the distances in array calls.

The tracker's associator (Stone Soup's GNNWith2DAssignment over a DistanceHypothesiser)
builds a prediction, a measurement prediction and a hypothesis object for every track and
detection pair, and inverts each pair's innovation covariance on its own; with clutter or
several sites that per-pair Python work is most of an event's cost. Here the same distances
are worked out for every pair at once, the same table is laid out and handed to the same
assignment, and Stone Soup objects are built only for the pairs the assignment takes.

Same answer as the associator, step by step: each track predicted to the detection's
timestamp with its own motion's noise (the tracker's constant-acceleration predictor), the
innovation covariance S = H P H' + R with the detection's own R, the Mahalanobis distance
through the pseudo-inverse of S, a hypothesis only below the missed distance, one table row
per track with a hypothesis, in the order the tracks are iterated, its own missed column at
the missed distance, every other cell infinite, and scipy's linear_sum_assignment. A
detection that is not a linear model of position is left to the associator: `associate`
returns None and the tracker falls back.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.detection import MissedDetection
from stonesoup.types.hypothesis import SingleDistanceHypothesis

from .reach import POSITION_AXES


def _linear_position(detection) -> tuple | None:
    """The detection's measured state indices, if its model is linear in position."""
    model = getattr(detection, "measurement_model", None)
    mapping = getattr(model, "mapping", None)
    if not isinstance(model, LinearGaussian) or mapping is None:
        return None
    mapping = tuple(int(m) for m in mapping)
    if any(m not in POSITION_AXES for m in mapping):
        return None
    return mapping


def _predicted(x, p, dt, noise):
    """Every track's state and covariance `dt` seconds on: the predictor's formulas, batched.

    x (N, 9), p (N, 9, 9), dt (N,), noise (N, 3) -> (N, 9), (N, 9, 9).
    """
    n = len(dt)
    step = np.abs(dt)
    f = np.zeros((n, 3, 3))
    f[:, 0, 0] = f[:, 1, 1] = f[:, 2, 2] = 1.0
    f[:, 0, 1] = f[:, 1, 2] = dt
    f[:, 0, 2] = 0.5 * dt * dt
    q = np.empty((n, 3, 3))
    q[:, 0, 0] = step**5 / 20.0
    q[:, 0, 1] = q[:, 1, 0] = step**4 / 8.0
    q[:, 0, 2] = q[:, 2, 0] = step**3 / 6.0
    q[:, 1, 1] = step**3 / 3.0
    q[:, 1, 2] = q[:, 2, 1] = step**2 / 2.0
    q[:, 2, 2] = step
    transition = np.zeros((n, 9, 9))
    process = np.zeros((n, 9, 9))
    for axis in range(3):
        block = slice(3 * axis, 3 * axis + 3)
        transition[:, block, block] = f
        process[:, block, block] = noise[:, axis, None, None] * q
    # matmul, as the predictor computes it, so the result matches it to the last bit.
    mean = (transition @ x[:, :, None])[:, :, 0]
    covar = transition @ p @ np.transpose(transition, (0, 2, 1)) + process
    return mean, covar


def distances(x, p, stamps, noise, detections, mapping, gate=np.inf) -> np.ndarray:
    """Mahalanobis distance of every track (rows) to every detection (columns).

    The detections share `mapping` and each track is predicted to each detection's own
    timestamp, the step taken from the track's own timestamp as the predictor takes it.
    """
    out = np.empty((len(stamps), len(detections)))
    rows = list(mapping)
    # Pairs this close may gate: the batch sum is within a few bits of the exact one.
    near_squared = (gate * (1.0 + 1e-9)) ** 2
    by_time: dict = {}
    for column, det in enumerate(detections):
        by_time.setdefault(det.timestamp, []).append(column)
    for when, columns in by_time.items():
        dt = np.array([(when - stamp).total_seconds() for stamp in stamps])
        mean, covar = _predicted(x, p, dt, noise)
        z = mean[:, rows]  # (N, k)
        hph = covar[:, rows][:, :, rows]  # (N, k, k)
        measured = np.array(
            [np.asarray(detections[c].state_vector, dtype=float).ravel() for c in columns]
        )  # (D, k)
        noise_r = np.array(
            [np.asarray(detections[c].measurement_model.covar(), dtype=float) for c in columns]
        )  # (D, k, k)
        s = hph[:, None, :, :] + noise_r[None, :, :, :]  # (N, D, k, k)
        delta = z[:, None, :] - measured[None, :, :]  # (N, D, k)
        inverse = np.linalg.pinv(s)
        # A distance must match the measure's to the last bit, so that a tie between
        # identical reports is broken as the associator breaks it. For three measured
        # axes the measure's einsum over a flat list of pairs sums in its own order; for
        # other sizes numpy's einsum sums a single pair differently, so the pairs that
        # may gate are summed again one by one, exactly as the measure sums them.
        k = len(rows)
        flat = delta.reshape(-1, k)
        inverse = inverse.reshape(-1, k, k)
        squared = np.einsum("ni,nij,nj->n", flat, inverse, flat)
        if k != 3:
            for i in np.flatnonzero(squared < near_squared):
                pair = flat[i : i + 1]
                squared[i] = np.einsum("ij,jk,ik->i", pair, inverse[i], pair)[0]
        out[:, columns] = np.sqrt(squared).reshape(delta.shape[:2])
    return out


def associate(tracker, tracks, detections, timestamp, rows_of) -> dict | None:
    """Track -> hypothesis, as the associator returns it, or None to fall back to it.

    `rows_of(track)` gives the track's state vector, covariance, timestamp and its
    motion's noise per axis. The result is ordered as the associator orders it: the
    tracks with no hypothesis first, as its set difference iterates them, then the rest
    in track order. A track left unassigned gets a falsy missed hypothesis.
    """
    tracks = list(tracks)
    detections = list(detections)
    mappings = [_linear_position(det) for det in detections]
    if any(m is None for m in mappings):
        return None
    missed = tracker.hypothesiser.missed_distance
    n, d = len(tracks), len(detections)
    table = np.full((n, d), np.inf)
    if n and d:
        rows = [rows_of(track) for track in tracks]
        x = np.array([r[0] for r in rows])
        p = np.array([r[1] for r in rows])
        stamps = [r[2] for r in rows]
        noise = np.array([r[3] for r in rows])
        by_mapping: dict[tuple, list[int]] = {}
        for column, mapping in enumerate(mappings):
            by_mapping.setdefault(mapping, []).append(column)
        for mapping, columns in by_mapping.items():
            table[:, columns] = distances(
                x, p, stamps, noise, [detections[c] for c in columns], mapping, missed
            )
        # A hypothesis only below the missed distance, as the hypothesiser keeps.
        table[~(table < missed)] = np.inf
        # A detection the tracker keeps off a track gets no hypothesis on it.
        for row, column in zip(*np.nonzero(np.isfinite(table)), strict=True):
            if tracker._excluded(tracks[row], detections[column]):
                table[row, column] = np.inf
    detected = [row for row in range(n) if np.isfinite(table[row]).any()]
    associations = {}
    undetected = dict.fromkeys(tracks).keys() - {tracks[row] for row in detected}
    for track in undetected:
        associations[track] = _missed(timestamp, missed)
    if not detected:
        return associations
    cost = np.full((len(detected), d + len(detected)), np.inf)
    cost[:, :d] = table[detected]
    for j in range(len(detected)):
        cost[j, d + j] = missed
    _, col4row = linear_sum_assignment(cost, False)
    for j, row in enumerate(detected):
        track = tracks[row]
        column = int(col4row[j])
        if column >= d:
            associations[track] = _missed(timestamp, missed)
            continue
        det = detections[column]
        prediction = tracker.predictor.predict(track, timestamp=det.timestamp)
        measurement_prediction = tracker.updater.predict_measurement(
            prediction, det.measurement_model
        )
        distance = tracker.hypothesiser.measure(measurement_prediction, det)
        associations[track] = SingleDistanceHypothesis(
            prediction, det, distance, measurement_prediction
        )
    return associations


def _missed(timestamp, missed):
    """A track's missed-detection hypothesis: falsy, so the track is not updated."""
    return SingleDistanceHypothesis(None, MissedDetection(timestamp=timestamp), missed)
