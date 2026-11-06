"""Tests for dev/multisite_benchmark.py.

Dev tooling that measures one engine fusing two distant sites. What it measures is
only as good as the replay it builds, so these pin the transforms: aligning the
sites' clocks, delivering events in true order while a site's clock lags, the
seeded clutter, and attributing tracks, refusals and cost to a site.
"""

import copy
import importlib.util
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from context_foundry.fusion import config
from context_foundry.fusion.sources.json_file import JsonSapientSource

REPO_ROOT = Path(__file__).resolve().parents[1]
START = datetime(2026, 11, 15, 2, 45, tzinfo=timezone.utc)

SENSOR_A = {
    "id": "A-RAD",
    "type": "RADAR",
    "lat": 62.0,
    "lon": 30.0,
    "alt": 100.0,
    "range_m": 20000,
}
SENSOR_B = {
    "id": "B-RAD",
    "type": "RADAR",
    "lat": 67.0,
    "lon": 27.0,
    "alt": 300.0,
    "range_m": 20000,
}


@pytest.fixture(scope="module")
def msb():
    spec = importlib.util.spec_from_file_location(
        "multisite_benchmark", REPO_ROOT / "dev/multisite_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _message(node_id, when, lat, lon, object_id="obj-1"):
    """A SAPIENT detection report shaped like scenario-foundry's output."""
    return {
        "sapientMessage": {
            "timestamp": when.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "nodeId": node_id,
            "detectionReport": {
                "objectId": object_id,
                "state": "ACTIVE",
                "location": {
                    "x": lon,
                    "y": lat,
                    "z": 500.0,
                    "coordinateSystem": "LOCATION_COORDINATE_SYSTEM_LAT_LNG_DEG_M",
                    "datum": "LOCATION_DATUM_WGS84_E",
                },
                "objectInfo": [{"type": "estimatedSwarmCount", "value": "3"}],
                "classification": [{"type": "Air vehicle", "confidence": 0.6, "subClass": []}],
            },
        }
    }


def _stamp(message):
    return message["sapientMessage"]["timestamp"]


# --- aligning the sites -------------------------------------------------------------


def test_alignment_moves_every_site_to_the_earliest_start(msb):
    starts = {"a": START, "b": START + timedelta(minutes=15)}

    assert msb.alignment_shifts(starts) == {"a": 0.0, "b": -900.0}


def test_shift_messages_moves_every_timestamp_and_leaves_the_input_alone(msb):
    original = [
        _message("A-RAD", START, 62.0, 30.0),
        _message("A-RAD", START + timedelta(seconds=1.5), 62.0, 30.0),
    ]
    before = copy.deepcopy(original)

    shifted = msb.shift_messages(original, -900.0)

    assert original == before
    assert [_stamp(m) for m in shifted] == [
        "2026-11-15T02:30:00.000000Z",
        "2026-11-15T02:30:01.500000Z",
    ]


# --- the shared network ---------------------------------------------------------------


def test_union_sensors_keeps_every_site_sensor(msb):
    assert [s["id"] for s in msb.union_sensors([[SENSOR_A], [SENSOR_B]])] == ["A-RAD", "B-RAD"]


def test_union_sensors_refuses_a_node_id_two_sites_share(msb):
    with pytest.raises(ValueError, match="A-RAD"):
        msb.union_sensors([[SENSOR_A], [dict(SENSOR_A, lat=60.0)]])


def test_geodesic_midpoint(msb):
    lat, lon = msb.geodesic_midpoint(0.0, 0.0, 0.0, 90.0)
    assert (lat, lon) == pytest.approx((0.0, 45.0))

    lat, lon = msb.geodesic_midpoint(62.95, 29.8, 67.1333, 26.8833)
    assert (lat, lon) == pytest.approx((65.0487, 28.4561), abs=1e-3)


# --- clutter ----------------------------------------------------------------------------


def _clutter(msb, seed=7, rate=0.5, seconds=100):
    template = _message("A-RAD", START, 62.0, 30.0)
    return msb.clutter_messages(
        [SENSOR_A], template, START, START + timedelta(seconds=seconds), rate, seed
    )


def test_clutter_is_seeded(msb):
    assert _clutter(msb, seed=7) == _clutter(msb, seed=7)
    assert _clutter(msb, seed=7) != _clutter(msb, seed=8)


def test_clutter_fills_the_window_at_the_rate_within_sensor_range(msb):
    messages = _clutter(msb, rate=0.5, seconds=100)

    assert len(messages) == 50
    from_dt = datetime.fromisoformat
    for m in messages:
        sm = m["sapientMessage"]
        when = from_dt(sm["timestamp"].replace("Z", "+00:00"))
        assert START <= when <= START + timedelta(seconds=100)
        assert sm["nodeId"] == "A-RAD"
        loc = sm["detectionReport"]["location"]
        assert msb.fb.haversine_m(SENSOR_A["lat"], SENSOR_A["lon"], loc["y"], loc["x"]) <= 20000
    ids = [m["sapientMessage"]["detectionReport"]["objectId"] for m in messages]
    assert len(set(ids)) == len(ids)


def test_clutter_is_one_object_each_and_does_not_touch_the_template(msb):
    template = _message("A-RAD", START, 62.0, 30.0)
    before = copy.deepcopy(template)

    messages = msb.clutter_messages(
        [SENSOR_A], template, START, START + timedelta(seconds=10), 1.0, 1
    )

    assert template == before
    assert all("objectInfo" not in m["sapientMessage"]["detectionReport"] for m in messages)


def test_clutter_replays_through_the_engine_source(msb, tmp_path):
    config.load_sensor_network(sensor_network_list=[SENSOR_A])
    path = tmp_path / "clutter.json"
    path.write_text(json.dumps(_clutter(msb, rate=0.2, seconds=100)))

    detections = [d for _, dets in JsonSapientSource(path).iter_events() for d in dets]

    assert len(detections) == 20


# --- delivery with a lagging site clock -----------------------------------------------


def _event(node_id, seconds):
    when = START + timedelta(seconds=seconds)
    return when, [SimpleNamespace(metadata={"nodeId": node_id}, timestamp=when)]


def test_delivery_keeps_true_order_and_stamps_the_lagging_site_earlier(msb):
    events = [_event("A-RAD", 0), _event("B-RAD", 1), _event("A-RAD", 2)]
    node_site = {"A-RAD": "a", "B-RAD": "b"}

    delivered = msb.deliver(events, node_site, {"b": 3.0})

    assert [site for _, _, site in delivered] == ["a", "b", "a"]
    assert [stamp for stamp, _, _ in delivered] == [
        START,
        START + timedelta(seconds=-2),
        START + timedelta(seconds=2),
    ]
    assert delivered[1][1][0].timestamp == START + timedelta(seconds=-2)


def test_delivery_without_lag_changes_nothing(msb):
    events = [_event("A-RAD", 0), _event("B-RAD", 1)]

    delivered = msb.deliver(events, {"A-RAD": "a", "B-RAD": "b"}, {})

    assert [(stamp, site) for stamp, _, site in delivered] == [
        (START, "a"),
        (START + timedelta(seconds=1), "b"),
    ]


# --- attribution -------------------------------------------------------------------------


def test_nearest_site_is_the_site_of_the_closest_sensor(msb):
    sensors_enu = [(0.0, 0.0, "a"), (500_000.0, 0.0, "b")]

    assert msb.nearest_site(10_000.0, 0.0, sensors_enu) == "a"
    assert msb.nearest_site(480_000.0, 5_000.0, sensors_enu) == "b"


def test_site_of_track_uses_its_first_position(msb):
    sites = {"a": [SENSOR_A], "b": [SENSOR_B]}
    snapshots = [(START, 67.01, 27.0, 1), (START + timedelta(seconds=5), 62.0, 30.0, 1)]

    assert msb.site_of_snapshots(snapshots, sites) == "b"


def test_counting_tracker_names_the_sensor_it_refused(msb, tmp_path):
    config.load_sensor_network(sensor_network_list=[SENSOR_A, SENSOR_B])
    config.set_reference_origin(62.0, 30.0, 0.0)
    messages = [
        _message("A-RAD", START, 62.05, 30.0, "a-1"),
        _message("B-RAD", START + timedelta(seconds=1), 67.05, 27.0, "b-1"),
    ]
    path = tmp_path / "two.json"
    path.write_text(json.dumps(messages))
    tracker = msb.CountingTracker(max_live_tracks=1)

    for when, dets in JsonSapientSource(path).iter_events():
        tracker.process_async_event(when, set(dets))

    assert tracker.refused_by_node == {"B-RAD": 1}


def test_tracker_is_built_with_the_cli_defaults(msb):
    # The constructor confirms on one hit; the CLI on DEFAULT_CONFIRM_HITS. With the
    # constructor's, alakurtti alone gives 36 tracks where the CLI gives 5.
    from context_foundry.fusion import tracker as tracker_module

    tracker = msb.make_tracker()

    assert isinstance(tracker, msb.CountingTracker)
    assert tracker.confirm_hits == tracker_module.DEFAULT_CONFIRM_HITS
    assert tracker.max_live_tracks == tracker_module.MAX_LIVE_TRACKS
    assert tracker.max_event_detections == tracker_module.MAX_EVENT_DETECTIONS
    assert tracker.max_coast_seconds == tracker_module.MAX_COAST_SECONDS
    assert tracker.max_track_history == tracker_module.MAX_TRACK_HISTORY


# --- cost and frame summaries ----------------------------------------------------------


def test_fit_cost_recovers_own_and_other_site_slopes(msb):
    rng = random.Random(3)
    rows = []
    for _ in range(200):
        own, other = rng.randint(0, 50), rng.randint(0, 50)
        rows.append((1.0 + 0.10 * own + 0.05 * other, own, other))

    fit = msb.fit_cost(rows)

    assert fit["intercept_ms"] == pytest.approx(1.0)
    assert fit["ms_per_own_track"] == pytest.approx(0.10)
    assert fit["ms_per_other_track"] == pytest.approx(0.05)


def test_fit_cost_needs_enough_events(msb):
    assert msb.fit_cost([(1.0, 0, 0)]) is None


def test_error_bins_group_by_distance_from_the_origin(msb):
    samples = [(10.0, 20.0), (50.0, 40.0), (90.0, 30.0), (650.0, 100.0)]

    assert msb.error_bins(samples, width_km=100) == [
        {"from_km": 0, "to_km": 100, "snapshots": 3, "median_error_m": 30.0},
        {"from_km": 600, "to_km": 700, "snapshots": 1, "median_error_m": 100.0},
    ]
