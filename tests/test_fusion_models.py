"""Tests for context_foundry.fusion.models."""

import math

import numpy as np
import pytest
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel

from context_foundry.fusion import config
from context_foundry.fusion.models import (
    AcousticAzimuthMeasurementModel,
    create_9d_constant_acceleration_model,
    ecef_to_enu,
    ecef_to_wgs84,
    enu_to_ecef,
    project_geodetic_to_local_enu,
    wgs84_to_ecef,
)

# --- WGS84 <-> ECEF -----------------------------------------------------------


def test_wgs84_to_ecef_and_back_round_trip():
    lat, lon, alt = 62.9, 29.8, 250.0
    x, y, z = wgs84_to_ecef(lat, lon, alt)
    lat2, lon2, alt2 = ecef_to_wgs84(x, y, z)
    assert lat2 == pytest.approx(lat, abs=1e-8)
    assert lon2 == pytest.approx(lon, abs=1e-8)
    assert alt2 == pytest.approx(alt, abs=1e-3)


def test_ecef_to_wgs84_north_pole_case():
    lat, lon, _alt = ecef_to_wgs84(0.0, 0.0, 6356752.0)
    assert lat == 90.0
    assert lon == 0.0


def test_ecef_to_wgs84_south_pole_case():
    lat, lon, _alt = ecef_to_wgs84(0.0, 0.0, -6356752.0)
    assert lat == -90.0
    assert lon == 0.0


# --- ECEF <-> ENU --------------------------------------------------------------


def test_ecef_enu_round_trip():
    lat0, lon0, alt0 = 62.9, 29.8, 100.0
    x, y, z = wgs84_to_ecef(62.905, 29.81, 120.0)
    e, n, u = ecef_to_enu(x, y, z, lat0, lon0, alt0)
    x2, y2, z2 = enu_to_ecef(e, n, u, lat0, lon0, alt0)
    assert x2 == pytest.approx(x, abs=1e-3)
    assert y2 == pytest.approx(y, abs=1e-3)
    assert z2 == pytest.approx(z, abs=1e-3)


def test_ecef_to_enu_origin_gives_zero_vector():
    lat0, lon0, alt0 = 62.9, 29.8, 100.0
    x, y, z = wgs84_to_ecef(lat0, lon0, alt0)
    e, n, u = ecef_to_enu(x, y, z, lat0, lon0, alt0)
    assert e == pytest.approx(0.0, abs=1e-6)
    assert n == pytest.approx(0.0, abs=1e-6)
    assert u == pytest.approx(0.0, abs=1e-6)


# --- project_geodetic_to_local_enu --------------------------------------------


def test_project_geodetic_to_local_enu_raises_without_origin():
    assert config.ENU_ORIGIN_LAT is None
    with pytest.raises(ValueError, match="CRITICAL"):
        project_geodetic_to_local_enu(62.9, 29.8, 100.0)


def test_project_geodetic_to_local_enu_with_origin_set():
    config.load_sensor_network(
        sensor_network_list=[{"id": "anchor", "lat": 62.9, "lon": 29.8, "alt": 100.0}]
    )
    e, n, u = project_geodetic_to_local_enu(62.9, 29.8, 100.0)
    assert e == pytest.approx(0.0, abs=1e-6)
    assert n == pytest.approx(0.0, abs=1e-6)
    assert u == pytest.approx(0.0, abs=1e-6)


# --- create_9d_constant_acceleration_model ------------------------------------


def test_create_9d_constant_acceleration_model_shape():
    model = create_9d_constant_acceleration_model(q_process_noise=0.1)
    assert isinstance(model, CombinedLinearGaussianTransitionModel)
    assert model.ndim_state == 9


# --- AcousticAzimuthMeasurementModel -------------------------------------------


class _ConcreteAcousticModel(AcousticAzimuthMeasurementModel):
    """AcousticAzimuthMeasurementModel is missing pdf/rvs so the ABC can't be
    instantiated directly (see final report: this looks like a source bug).
    Subclassing with trivial stubs lets us exercise its real `function` logic.

    Separately, `noise_covar` is never declared as a stonesoup Property on this
    class or on MeasurementModel itself (only Gaussian-model subclasses like
    LinearGaussian declare it) -- passing noise_covar as a constructor kwarg
    raises TypeError. `function()`'s noise=True branch nonetheless reads
    `self.noise_covar`, so we set it directly as a plain instance attribute
    after construction purely to exercise that branch (see final report).
    """

    def pdf(self, *args, **kwargs):  # pragma: no cover - not exercised
        raise NotImplementedError

    def rvs(self, *args, **kwargs):  # pragma: no cover - not exercised
        raise NotImplementedError


def test_acoustic_model_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        AcousticAzimuthMeasurementModel(ndim_state=9, mapping=(0, 3, 6))


def test_acoustic_model_rejects_noise_covar_constructor_kwarg():
    # noise_covar is not a declared Property anywhere in the MRO (see final report).
    with pytest.raises(TypeError, match="noise_covar"):
        _ConcreteAcousticModel(ndim_state=9, mapping=(0, 3, 6), noise_covar=np.array([[0.01]]))


def _make_acoustic_model():
    return _ConcreteAcousticModel(ndim_state=9, mapping=(0, 3, 6))


def test_acoustic_model_ndim_meas_is_one():
    model = _make_acoustic_model()
    assert model.ndim_meas == 1


def test_acoustic_model_requires_sensor_geodetic_kwarg():
    model = _make_acoustic_model()
    state = _FakeState(np.zeros((9, 1)))
    with pytest.raises(ValueError, match="sensor_geodetic"):
        model.function(state)


def test_acoustic_model_computes_azimuth_without_noise():
    config.load_sensor_network(
        sensor_network_list=[{"id": "sensor", "lat": 62.9, "lon": 29.8, "alt": 0.0}]
    )
    model = _make_acoustic_model()

    # Target due east of the sensor's local origin (which is itself the ENU origin).
    vec = np.zeros((9, 1))
    vec[0, 0] = 100.0  # East
    vec[3, 0] = 0.0  # North
    state = _FakeState(vec)

    result = model.function(
        state, sensor_geodetic={"latitude": 62.9, "longitude": 29.8, "altitude": 0.0}
    )
    azimuth = result[0, 0]
    assert azimuth == pytest.approx(math.pi / 2, abs=1e-6)


def test_acoustic_model_computes_azimuth_with_noise():
    config.load_sensor_network(
        sensor_network_list=[{"id": "sensor", "lat": 62.9, "lon": 29.8, "alt": 0.0}]
    )
    model = _make_acoustic_model()
    model.noise_covar = np.array([[0.001]])

    vec = np.zeros((9, 1))
    vec[0, 0] = 100.0
    vec[3, 0] = 0.0
    state = _FakeState(vec)

    result = model.function(
        state, noise=True, sensor_geodetic={"latitude": 62.9, "longitude": 29.8}
    )
    azimuth = result[0, 0]
    assert 0.0 <= azimuth <= 2 * math.pi


class _FakeState:
    """Minimal duck-typed stand-in for a Stone Soup State: only .state_vector is read."""

    def __init__(self, state_vector):
        self.state_vector = state_vector
