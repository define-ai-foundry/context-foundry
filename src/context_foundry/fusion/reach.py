"""Which live tracks a group of detections can reach, for every track in one array call.

The tracker gates only the tracks a detection can reach (`SapientAsynchronousTracker.
_may_gate`). Checking that track by track costs a pure-Python prediction of every live
track on every event, which is where most of an event's cost per live track went. Here
each track's numbers are kept as one row, rebuilt only when its state changes, and an
event's rows are predicted and compared with its detections in one vectorised call. The
bound and its arithmetic are `_may_gate`'s; only where the numbers are worked out changes.
"""

from datetime import datetime, timezone

import numpy as np
from stonesoup.models.measurement.linear import LinearGaussian

POSITION_AXES = (0, 3, 6)
_EPOCH_AWARE = datetime(2025, 1, 1, tzinfo=timezone.utc)
_EPOCH_NAIVE = datetime(2025, 1, 1)

# A row: the state's time, its 9 state values, per axis the 6 terms of its position
# variance that a prediction weighs (P00, P01+P10, P02+P20, P11, P12+P21, P22), and per
# axis the motion's noise coefficient.
_TIME = 0
_STATE = 1
_VARIANCE = 10
_NOISE = 28
_WIDTH = 31


def seconds(moment: datetime) -> float:
    """A time as seconds from a fixed epoch near the data, keeping a microsecond apart in order."""
    epoch = _EPOCH_NAIVE if moment.tzinfo is None else _EPOCH_AWARE
    return (moment - epoch).total_seconds()


def track_row(state, axis_noise) -> np.ndarray:
    """The numbers the reach check needs from one track's state."""
    row = np.empty(_WIDTH)
    row[_TIME] = seconds(state.timestamp)
    row[_STATE:_VARIANCE] = np.asarray(state.state_vector, dtype=float).ravel()
    covar = np.asarray(state.covar, dtype=float)
    for axis, index in enumerate(POSITION_AXES):
        block = covar[index : index + 3, index : index + 3]
        base = _VARIANCE + 6 * axis
        row[base] = block[0, 0]
        row[base + 1] = block[0, 1] + block[1, 0]
        row[base + 2] = block[0, 2] + block[2, 0]
        row[base + 3] = block[1, 1]
        row[base + 4] = block[1, 2] + block[2, 1]
        row[base + 5] = block[2, 2]
    row[_NOISE : _NOISE + 3] = axis_noise
    return row


def reachable_rows(rows: np.ndarray, detections, gate_squared: float) -> np.ndarray:
    """For each row, whether any of the detections may fall inside its validation gate.

    The same conservative bound as `_may_gate`: a track is out of a detection's gate when
    its squared distance from the detection exceeds the gate squared times its predicted
    position variance plus the detection's noise, summed over the measured axes. A
    detection that is not a linear position model reaches every track; one older than a
    track's state reaches that track.
    """
    count = len(rows)
    reached = np.zeros(count, dtype=bool)
    if count == 0 or not detections:
        return reached
    groups = {}
    for det in detections:
        model = getattr(det, "measurement_model", None)
        mapping = getattr(model, "mapping", None)
        if not isinstance(model, LinearGaussian) or any(m not in POSITION_AXES for m in mapping):
            reached[:] = True
            return reached
        groups.setdefault(tuple(mapping), []).append(det)
    limit = gate_squared * (1.0 + 1e-6)
    times = rows[:, _TIME][:, None]
    for mapping, dets in groups.items():
        measured = np.array([np.asarray(d.state_vector, dtype=float).ravel() for d in dets])
        noise = np.array([np.diag(np.asarray(d.measurement_model.noise_covar)) for d in dets])
        dt = np.array([seconds(d.timestamp) for d in dets])[None, :] - times
        half = 0.5 * dt * dt
        miss = np.zeros_like(dt)
        spread = np.zeros_like(dt)
        for row_index, index in enumerate(mapping):
            axis = index // 3
            position = (
                rows[:, _STATE + index][:, None]
                + rows[:, _STATE + index + 1][:, None] * dt
                + rows[:, _STATE + index + 2][:, None] * half
            )
            base = _VARIANCE + 6 * axis
            variance = (
                rows[:, base][:, None]
                + rows[:, base + 1][:, None] * dt
                + rows[:, base + 2][:, None] * half
                + rows[:, base + 3][:, None] * dt * dt
                + rows[:, base + 4][:, None] * dt * half
                + rows[:, base + 5][:, None] * half * half
                + rows[:, _NOISE + axis][:, None] * dt**5 / 20.0
            )
            miss += (measured[:, row_index][None, :] - position) ** 2
            spread += variance + noise[:, row_index][None, :]
        reached |= ((dt < 0) | (miss <= limit * spread)).any(axis=1)
    return reached
