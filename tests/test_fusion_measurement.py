"""Tests for context_foundry.fusion.measurement."""

import math

import numpy as np
import pytest

from context_foundry.fusion.measurement import (
    DEFAULT_MEASUREMENT_MODEL,
    DEFAULT_NOISE_COVAR,
    MIN_SIGMA_M,
    POSITION_MAPPING,
    default_geometric_error,
    measurement_model_for,
    noise_covariance,
)

SENSOR = (0.0, 0.0, 0.0)


def _horizontal_sigma(covar):
    assert covar[0, 0] == pytest.approx(covar[1, 1])
    return math.sqrt(covar[0, 0])


def test_constant_error_is_the_same_at_any_distance():
    error = {"variation_type": "constant", "base_m": 1.5}
    near = noise_covariance(error, SENSOR, (10.0, 0.0, 0.0), 3500.0)
    far = noise_covariance(error, SENSOR, (3000.0, 0.0, 0.0), 3500.0)
    assert _horizontal_sigma(near) == pytest.approx(1.5)
    assert _horizontal_sigma(far) == pytest.approx(1.5)


def test_linear_error_grows_from_base_to_max_range():
    error = {"variation_type": "linear_with_range", "base_m": 5.0, "at_max_range_m": 35.0}
    assert _horizontal_sigma(noise_covariance(error, SENSOR, (0.0, 0.0, 0.0), 1000.0)) == (
        pytest.approx(5.0)
    )
    assert _horizontal_sigma(noise_covariance(error, SENSOR, (300.0, 400.0, 0.0), 1000.0)) == (
        pytest.approx(20.0)
    )
    assert _horizontal_sigma(noise_covariance(error, SENSOR, (0.0, 1000.0, 0.0), 1000.0)) == (
        pytest.approx(35.0)
    )


def test_error_stops_growing_past_the_sensors_range():
    error = {"variation_type": "linear_with_range", "base_m": 5.0, "at_max_range_m": 35.0}
    beyond = noise_covariance(error, SENSOR, (0.0, 5000.0, 0.0), 1000.0)
    assert _horizontal_sigma(beyond) == pytest.approx(35.0)


def test_quadratic_error_grows_with_the_square_of_the_range_fraction():
    error = {"variation_type": "quadratic_with_range", "base_m": 2.0, "at_max_range_m": 17.0}
    half = noise_covariance(error, SENSOR, (2000.0, 0.0, 0.0), 4000.0)
    assert _horizontal_sigma(half) == pytest.approx(2.0 + 15.0 * 0.25)


def test_error_ignores_height_difference_from_the_sensor():
    error = {"variation_type": "linear_with_range", "base_m": 5.0, "at_max_range_m": 35.0}
    covar = noise_covariance(error, SENSOR, (0.0, 0.0, 900.0), 1000.0)
    assert _horizontal_sigma(covar) == pytest.approx(5.0)


def test_bearing_error_is_long_across_the_line_of_sight():
    error = {"variation_type": "bearing", "bearing_deg": 3.5, "range_sigma_m": 1.0}
    # Target due north of the sensor: the line of sight is north, across it is east.
    covar = noise_covariance(error, SENSOR, (0.0, 2000.0, 0.0), 3000.0)
    assert math.sqrt(covar[0, 0]) == pytest.approx(2000.0 * math.tan(math.radians(3.5)))
    assert math.sqrt(covar[1, 1]) == pytest.approx(1.0)
    assert covar[0, 1] == pytest.approx(0.0, abs=1e-9)


def test_bearing_error_rotates_with_the_line_of_sight():
    error = {"variation_type": "bearing", "bearing_deg": 3.5, "range_sigma_m": 1.0}
    covar = noise_covariance(error, SENSOR, (1000.0, 1000.0, 0.0), 3000.0)
    across = math.hypot(1000.0, 1000.0) * math.tan(math.radians(3.5))
    eigenvalues = sorted(np.linalg.eigvalsh(covar[:2, :2]))
    assert math.sqrt(eigenvalues[0]) == pytest.approx(1.0)
    assert math.sqrt(eigenvalues[1]) == pytest.approx(across)


def test_bearing_error_at_the_sensor_itself_is_finite():
    error = {"variation_type": "bearing", "bearing_deg": 3.5}
    covar = noise_covariance(error, SENSOR, SENSOR, 3000.0)
    assert np.all(np.isfinite(covar))
    assert np.linalg.det(covar) > 0


def test_vertical_sigma_defaults_and_can_be_declared():
    error = {"variation_type": "constant", "base_m": 1.5}
    assert math.sqrt(noise_covariance(error, SENSOR, SENSOR, 0.0)[2, 2]) == pytest.approx(10.0)
    declared = {**error, "vertical_m": 1000.0}
    assert math.sqrt(noise_covariance(declared, SENSOR, SENSOR, 0.0)[2, 2]) == pytest.approx(1000.0)


def test_sigma_never_falls_below_the_floor():
    error = {"variation_type": "constant", "base_m": 0.0, "vertical_m": 0.0}
    covar = noise_covariance(error, SENSOR, SENSOR, 0.0)
    assert _horizontal_sigma(covar) == pytest.approx(MIN_SIGMA_M)
    assert math.sqrt(covar[2, 2]) == pytest.approx(MIN_SIGMA_M)


def test_unknown_variation_type_is_refused():
    error = {"variation_type": "logarithmic", "base_m": 5.0, "at_max_range_m": 35.0}
    with pytest.raises(ValueError, match="logarithmic"):
        noise_covariance(error, SENSOR, (10.0, 0.0, 0.0), 1000.0)


def _model_sigma(model):
    return math.sqrt(model.noise_covar[0, 0])


def test_sensor_of_unknown_kind_without_geometric_error_gets_the_cautious_default():
    sensor = {"lat": 0.0, "lon": 0.0, "alt": 0.0, "range_m": 1000.0, "type": "lidar"}
    model = measurement_model_for(sensor, SENSOR, (10.0, 0.0, 0.0))
    assert _model_sigma(model) == pytest.approx(50.0)
    assert math.sqrt(model.noise_covar[2, 2]) == pytest.approx(50.0)


def test_unregistered_sensor_gets_the_cautious_default():
    model = measurement_model_for(None, None, (10.0, 0.0, 0.0))
    assert _model_sigma(model) == pytest.approx(50.0)


@pytest.mark.parametrize(
    "kind, sigma",
    [("RADAR_STRATEGIC", 35.0), ("radar", 35.0), ("THERMAL_CAM", 17.0), ("electro_optical", 17.0)],
)
def test_radar_and_camera_defaults_are_their_cautious_far_end_everywhere(kind, sigma):
    sensor = {"type": kind, "range_m": 1000.0}
    near = measurement_model_for(sensor, SENSOR, (10.0, 0.0, 0.0))
    far = measurement_model_for(sensor, SENSOR, (1000.0, 0.0, 0.0))
    assert _model_sigma(near) == pytest.approx(sigma)
    assert _model_sigma(far) == pytest.approx(sigma)


def test_micro_doppler_default_is_tight_and_constant():
    model = measurement_model_for({"type": "MICRO_DOPPLER"}, SENSOR, (900.0, 0.0, 0.0))
    assert _model_sigma(model) == pytest.approx(5.0)


def test_default_for_a_kind_is_found_by_substring():
    assert default_geometric_error({"type": "RADAR_TACTICAL"})["base_m"] == 35.0


def test_declared_geometric_error_wins_over_the_type_default():
    sensor = {"type": "radar", "geometric_error": {"variation_type": "constant", "base_m": 8.0}}
    model = measurement_model_for(sensor, SENSOR, (10.0, 0.0, 0.0))
    assert _model_sigma(model) == pytest.approx(8.0)


def test_default_model_is_the_historic_fixed_noise():
    assert np.array_equal(DEFAULT_MEASUREMENT_MODEL.covar(), DEFAULT_NOISE_COVAR)
    assert DEFAULT_MEASUREMENT_MODEL.mapping == POSITION_MAPPING


def test_sensor_with_geometric_error_gets_its_own_model():
    sensor = {
        "range_m": 1000.0,
        "geometric_error": {
            "variation_type": "linear_with_range",
            "base_m": 5.0,
            "at_max_range_m": 35.0,
        },
    }
    model = measurement_model_for(sensor, SENSOR, (0.0, 1000.0, 0.0))
    assert model.ndim_state == 9
    assert model.mapping == POSITION_MAPPING
    assert math.sqrt(model.covar()[0, 0]) == pytest.approx(35.0)


def test_position_measurement_with_height_is_three_dimensional():
    from context_foundry.fusion.measurement import POSITION_MAPPING, position_measurement

    vector, model = position_measurement(None, None, (1.0, 2.0, 3.0), has_altitude=True)
    assert vector.ravel().tolist() == [1.0, 2.0, 3.0]
    assert tuple(model.mapping) == POSITION_MAPPING


def test_position_measurement_without_height_uses_east_and_north_only():
    from context_foundry.fusion.measurement import HORIZONTAL_MAPPING, position_measurement

    sensor = {"geometric_error": {"variation_type": "constant", "base_m": 8.0}}
    vector, model = position_measurement(sensor, (0.0, 0.0, 0.0), (1.0, 2.0, 0.0), False)
    assert vector.ravel().tolist() == [1.0, 2.0]
    assert tuple(model.mapping) == HORIZONTAL_MAPPING
    assert np.allclose(model.noise_covar, np.diag([64.0, 64.0]))
