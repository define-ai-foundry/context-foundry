"""Tests for dev/cluster_fusion_benchmark.py.

Dev tooling that splits the fusion work into clusters, one engine per cluster, and
merges their pictures (docs/03-evaluation/cluster-parallel-fusion.md). These pin
the clustering, the routing of messages to clusters, the merge checks for
duplicate tracks and gaps, the stitch across clusters, and the synthetic
boundary case.
"""

import importlib.util
import math
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from context_foundry.fusion.sources.json_file import JsonSapientSource

REPO_ROOT = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 11, 15, 3, 0)
KM_PER_DEG_LAT = 111.195


@pytest.fixture(scope="module")
def cfb():
    spec = importlib.util.spec_from_file_location(
        "cluster_fusion_benchmark", REPO_ROOT / "dev/cluster_fusion_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sensor(node_id, lat, lon, range_m=20000.0):
    return {
        "id": node_id,
        "type": "RADAR",
        "lat": lat,
        "lon": lon,
        "alt": 100.0,
        "range_m": range_m,
    }


def _north(lat, km):
    """The latitude `km` north of `lat`."""
    return lat + km / KM_PER_DEG_LAT


def _located(node_id, when, lat, lon, object_id="obj-1"):
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
                "objectInfo": [{"type": "estimatedSwarmCount", "value": "1"}],
                "classification": [{"type": "Air vehicle", "confidence": 0.6, "subClass": []}],
            },
        }
    }


def _ranged(node_id, when, range_m, azimuth_deg):
    return {
        "sapientMessage": {
            "timestamp": when.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "nodeId": node_id,
            "detectionReport": {
                "objectId": "obj-rb",
                "state": "ACTIVE",
                "rangeBearing": {
                    "elevation": 0.0,
                    "azimuth": azimuth_deg,
                    "range": range_m,
                    "coordinateSystem": "RANGE_BEARING_COORDINATE_SYSTEM_DEGREES_M",
                    "datum": "RANGE_BEARING_DATUM_TRUE",
                },
            },
        }
    }


# --- clustering sensors ---------------------------------------------------------------


def test_single_linkage_joins_chains_and_splits_at_the_threshold(cfb):
    a = _sensor("a", 62.0, 30.0)
    b = _sensor("b", _north(62.0, 40), 30.0)
    c = _sensor("c", _north(62.0, 80), 30.0)  # 80 km from a, 40 km from b
    d = _sensor("d", _north(62.0, 300), 30.0)
    assert cfb.cluster_sensors([d, c, b, a], cfb.link_within(50_000)) == [["a", "b", "c"], ["d"]]
    assert cfb.cluster_sensors([a, b, c, d], cfb.link_within(30_000)) == [
        ["a"],
        ["b"],
        ["c"],
        ["d"],
    ]
    assert cfb.cluster_sensors([a, b, c, d], cfb.link_within(250_000)) == [["a", "b", "c", "d"]]


def test_footprints_overlap_when_the_ranges_reach_past_the_distance(cfb):
    a = _sensor("a", 62.0, 30.0, range_m=20_000)
    near = _sensor("n", _north(62.0, 35), 30.0, range_m=20_000)
    far = _sensor("f", _north(62.0, 45), 30.0, range_m=20_000)
    assert cfb.footprints_overlap(a, near)
    assert not cfb.footprints_overlap(a, far)
    assert cfb.cluster_sensors([a, near, far], cfb.footprints_overlap) == [["a", "f", "n"]]


def test_deployment_clusters_map_every_node_to_its_site(cfb):
    sites = {
        "west": [_sensor("w1", 62, 29), _sensor("w2", 62, 29.5)],
        "east": [_sensor("e1", 62, 31)],
    }
    assert cfb.deployment_clusters(sites) == {"w1": "west", "w2": "west", "e1": "east"}


# --- the grid ------------------------------------------------------------------------


def test_grid_rows_follow_latitude_in_cell_sized_steps(cfb):
    lat_edge = 6 * 100 / KM_PER_DEG_LAT  # row 6 starts 600 km north of the equator
    assert cfb.grid_cell(lat_edge - 0.001, 27.0, 100.0)[0] == 5
    assert cfb.grid_cell(lat_edge + 0.001, 27.0, 100.0)[0] == 6


def test_cell_edge_lon_is_where_the_column_changes_within_a_row(cfb):
    lat = 63.5
    row, col = cfb.grid_cell(lat, 27.0, 100.0)
    edge = cfb.cell_edge_lon(lat, col, 100.0)
    assert edge <= 27.0
    assert cfb.grid_cell(lat, edge + 1e-6, 100.0) == (row, col)
    assert cfb.grid_cell(lat, edge - 1e-6, 100.0) == (row, col - 1)


def test_grid_columns_are_about_a_cell_wide_on_the_ground(cfb):
    lat = 63.5
    _, col = cfb.grid_cell(lat, 27.0, 100.0)
    west = cfb.cell_edge_lon(lat, col, 100.0)
    east = cfb.cell_edge_lon(lat, col + 1, 100.0)
    width_km = (east - west) * KM_PER_DEG_LAT * math.cos(math.radians(lat))
    assert width_km == pytest.approx(100.0, rel=0.02)


# --- routing messages ----------------------------------------------------------------


def test_detection_position_reads_a_location(cfb):
    message = _located("a", T0, 62.5, 29.75)
    assert cfb.detection_position(message, {}) == (62.5, 29.75)


def test_detection_position_resolves_range_and_bearing_from_its_sensor(cfb):
    sensor = _sensor("acu", 62.0, 30.0)
    lat, lon = cfb.detection_position(_ranged("acu", T0, 5000.0, 90.0), {"acu": sensor})
    assert cfb.fb.haversine_m(62.0, 30.0, lat, lon) == pytest.approx(5000.0, rel=1e-6)
    assert lat == pytest.approx(62.0, abs=1e-3)
    assert lon > 30.0


def test_route_by_node_keeps_each_cluster_in_input_order(cfb):
    messages = [
        _located("w1", T0, 62.0, 29.0, "x"),
        _located("e1", T0, 62.0, 31.0, "y"),
        _located("w1", T0 + timedelta(seconds=1), 62.0, 29.0, "z"),
    ]
    routed = cfb.route_by_node(messages, {"w1": "west", "e1": "east"})
    assert list(routed) == ["west", "east"]
    assert [m["sapientMessage"]["detectionReport"]["objectId"] for m in routed["west"]] == [
        "x",
        "z",
    ]
    assert routed["east"] == [messages[1]]


def test_route_by_node_refuses_a_node_in_no_cluster(cfb):
    with pytest.raises(ValueError, match="ghost"):
        cfb.route_by_node([_located("ghost", T0, 62.0, 29.0)], {"w1": "west"})


def test_route_by_cell_sends_each_detection_to_the_cell_it_lies_in(cfb):
    lat = 63.5
    _, col = cfb.grid_cell(lat, 27.0, 100.0)
    edge = cfb.cell_edge_lon(lat, col, 100.0)
    sensor = _sensor("r", lat, edge)
    west = _located("r", T0, lat, edge - 0.01, "w")
    east = _located("r", T0, lat, edge + 0.01, "e")
    ranged = _ranged("r", T0, 3000.0, 270.0)  # 3 km west of the sensor on the edge
    routed = cfb.route_by_cell([west, east, ranged], {"r": sensor}, 100.0)
    assert len(routed) == 2
    by_object = {
        m["sapientMessage"]["detectionReport"]["objectId"]: label
        for label, ms in routed.items()
        for m in ms
    }
    assert by_object["w"] == by_object["obj-rb"] != by_object["e"]


# --- one engine per cluster ----------------------------------------------------------


def test_namespaced_instances_publish_distinct_track_ids(cfb):
    sensor = _sensor("r", 62.0, 30.0)
    messages = [
        _located("r", T0 + timedelta(seconds=2 * i), _north(62.0, 0.2 * i), 30.0) for i in range(6)
    ]
    uids = {}
    for namespace in ("west", "east"):
        fused = cfb.msb.fuse(messages, {"site": [sensor]}, id_namespace=namespace)
        uids[namespace] = set(cfb.fb.read_tracks("\n".join(fused["cot"])))
    assert uids["west"]
    assert uids["east"]
    assert not uids["west"] & uids["east"]


def test_makespan_packs_instances_onto_cores_longest_first(cfb):
    assert cfb.makespan([3.0, 3.0, 2.0, 2.0, 2.0], cores=2) == pytest.approx(7.0)
    assert cfb.makespan([3.0, 1.0], cores=4) == pytest.approx(3.0)
    assert cfb.makespan([3.0, 1.0], cores=1) == pytest.approx(4.0)


# --- merging the instances' pictures ---------------------------------------------------


def _swarm(cfb, wave_id="S", start=T0):
    """One object flying 100 m/s due north from 62N 30E for 10 minutes."""
    north = _north(62.0, 60)
    cfg = {
        "count": 1,
        "launch_delay_sec": 0,
        "speed_kmh": 360,
        "wkt_linestring": f"LINESTRING (30.0 62.0, 30.0 {north})",
    }
    return cfb.fb.Swarm(wave_id, cfg, start)


def _follow(swarm, first_s, last_s, step_s=2):
    """Snapshots of a track exactly on `swarm` from `first_s` to `last_s` after T0."""
    out = []
    for s in range(first_s, last_s + 1, step_s):
        when = T0 + timedelta(seconds=s)
        lat, lon = swarm.position(when)
        out.append((when, lat, lon, 1))
    return out


def test_duplicates_are_tracks_of_one_swarm_alive_together_in_two_instances(cfb):
    swarm = _swarm(cfb)
    tracks = {
        "west": {"t1": _follow(swarm, 0, 100), "t4": _follow(swarm, 0, 100)},
        "east": {"t2": _follow(swarm, 50, 150), "t3": _follow(swarm, 200, 300)},
    }
    assert cfb.duplicates(tracks, [swarm]) == [("t1", "t2", "S"), ("t4", "t2", "S")]


def test_false_tracks_are_never_duplicates(cfb):
    swarm = _swarm(cfb)
    far = [(w, lat + 0.5, lon, n) for w, lat, lon, n in _follow(swarm, 0, 100)]
    tracks = {"west": {"t1": _follow(swarm, 0, 100)}, "east": {"t2": far}}
    assert cfb.duplicates(tracks, [swarm]) == []


def test_gaps_are_uncovered_runs_at_least_the_limit_long(cfb):
    times = [T0 + timedelta(seconds=s) for s in range(101)]
    covered = [(times[0], times[40]), (times[70], times[100])]
    assert cfb.gaps(times, covered, max_gap_s=20) == [(times[41], times[69])]
    assert cfb.gaps(times, covered, max_gap_s=30) == []
    assert cfb.gaps(times, [], max_gap_s=20) == [(times[0], times[100])]


def test_swarm_gaps_count_only_time_in_some_sensors_range(cfb):
    swarm = _swarm(cfb)
    # The swarm flies north from the sensor at 100 m/s: in range for its first 200 s.
    sensor = _sensor("r", 62.0, 30.0, range_m=20_000)
    track = _follow(swarm, 0, 60)
    found = cfb.swarm_gaps(swarm, [track], [sensor], max_gap_s=30)
    assert len(found) == 1
    start, end = found[0]
    assert (start - T0).total_seconds() == pytest.approx(61, abs=1)
    assert (end - T0).total_seconds() == pytest.approx(200, abs=2)


# --- stitching across clusters ----------------------------------------------------------


def test_stitch_joins_a_track_ending_in_one_instance_to_one_starting_in_another(cfb):
    swarm = _swarm(cfb)
    tracks = {"west": {"a": _follow(swarm, 0, 100)}, "east": {"b": _follow(swarm, 104, 200)}}
    merged, links = cfb.stitch(tracks, max_dt_s=10, max_dist_m=1000)
    assert links == [("a", "b")]
    assert list(merged) == ["a"]
    assert merged["a"] == tracks["west"]["a"] + tracks["east"]["b"]


def test_stitch_declines_pairs_too_late_too_far_or_in_one_instance(cfb):
    swarm = _swarm(cfb)
    a = _follow(swarm, 0, 100)
    late = _follow(swarm, 130, 200)
    far = [(w, lat + 0.1, lon, n) for w, lat, lon, n in _follow(swarm, 104, 200)]
    same = _follow(swarm, 104, 200)
    for tracks in (
        {"west": {"a": a}, "east": {"b": late}},
        {"west": {"a": a}, "east": {"b": far}},
        {"west": {"a": a, "b": same}},
    ):
        merged, links = cfb.stitch(tracks, max_dt_s=10, max_dist_m=1000)
        assert links == []
        assert set(merged) == {"a", "b"}


def test_stitch_joins_overlapping_tracks_that_stay_together(cfb):
    swarm = _swarm(cfb)
    a = _follow(swarm, 0, 120)
    b = _follow(swarm, 60, 200)
    merged, links = cfb.stitch({"west": {"a": a}, "east": {"b": b}}, max_dt_s=10, max_dist_m=1000)
    assert links == [("a", "b")]
    # b continues a from a's last snapshot on, so no instant is claimed twice.
    assert merged["a"] == a + [s for s in b if s[0] > a[-1][0]]


def test_stitch_keeps_overlapping_tracks_apart_when_they_are_apart(cfb):
    swarm = _swarm(cfb)
    a = _follow(swarm, 0, 120)
    beside = [(w, lat, lon + 0.1, n) for w, lat, lon, n in _follow(swarm, 60, 200)]  # ~5 km east
    merged, links = cfb.stitch(
        {"west": {"a": a}, "east": {"b": beside}}, max_dt_s=10, max_dist_m=1000
    )
    assert links == []
    assert set(merged) == {"a", "b"}


def test_stitch_chains_more_than_two_instances(cfb):
    swarm = _swarm(cfb)
    tracks = {
        "one": {"a": _follow(swarm, 0, 100)},
        "two": {"b": _follow(swarm, 104, 200)},
        "three": {"c": _follow(swarm, 204, 300)},
    }
    merged, links = cfb.stitch(tracks, max_dt_s=10, max_dist_m=1000)
    assert links == [("a", "b"), ("b", "c")]
    assert list(merged) == ["a"]
    assert merged["a"] == tracks["one"]["a"] + tracks["two"]["b"] + tracks["three"]["c"]


# --- the synthetic boundary case ------------------------------------------------------------


@pytest.fixture(scope="module")
def case(cfb):
    return cfb.boundary_case(seed=42)


def test_boundary_case_is_seeded(cfb, case):
    assert cfb.boundary_case(seed=42)["messages"] == case["messages"]
    assert cfb.boundary_case(seed=7)["messages"] != case["messages"]


def test_boundary_case_groups_overlap_across_the_boundary(cfb, case):
    west, east = case["sites"]["west"], case["sites"]["east"]
    assert any(cfb.footprints_overlap(w, e) for w in west for e in east)
    assert cfb.cluster_sensors(west + east, cfb.footprints_overlap) == [
        sorted(s["id"] for s in west + east)
    ]


def test_boundary_case_detections_lie_within_their_sensors_range_near_the_truth(cfb, case):
    sensors = {s["id"]: s for group in case["sites"].values() for s in group}
    swarms = {w.wave_id: w for w in cfb.case_swarms(case)}
    for message in case["messages"]:
        body = message["sapientMessage"]
        lat, lon = cfb.detection_position(message, sensors)
        sensor = sensors[body["nodeId"]]
        assert cfb.fb.haversine_m(sensor["lat"], sensor["lon"], lat, lon) <= sensor["range_m"]
        when = cfb.fb.parse_time(body["timestamp"])
        wave = body["detectionReport"]["objectId"].split(":")[1]
        truth = swarms[wave].position(when)
        assert cfb.fb.haversine_m(lat, lon, *truth) < 500


def test_boundary_case_crossing_target_is_seen_by_both_groups(case):
    seen = {
        group
        for group, sensors in case["sites"].items()
        for m in case["messages"]
        if m["sapientMessage"]["nodeId"] in {s["id"] for s in sensors}
        and m["sapientMessage"]["detectionReport"]["objectId"].endswith(":T1_CROSSING")
    }
    assert seen == {"west", "east"}


def test_boundary_case_loiter_target_flies_on_a_cell_edge(cfb, case):
    swarm = next(w for w in cfb.case_swarms(case) if w.wave_id == "T2_OVERLAP")
    lat, lon = swarm.position(swarm.launch + timedelta(seconds=60))
    _, col = cfb.grid_cell(lat, lon + 0.05, case["cell_km"])
    assert lon == pytest.approx(cfb.cell_edge_lon(lat, col, case["cell_km"]), abs=1e-6)


def test_boundary_case_replays_through_the_engine_source(cfb, case, tmp_path):
    from context_foundry.fusion import config

    sensors = [s for group in case["sites"].values() for s in group]
    config.load_sensor_network(sensor_network_list=sensors, primary_anchor_node=None)
    config.set_reference_origin(None, None, None)
    path = tmp_path / "b1.json"
    path.write_text(cfb.json.dumps(case["messages"]), encoding="utf-8")
    events = list(JsonSapientSource(path).iter_events())
    assert sum(len(d) for _, d in events) == len(case["messages"])
