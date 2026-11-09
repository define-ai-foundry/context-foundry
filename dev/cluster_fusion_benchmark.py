#!/usr/bin/env python3
# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# Measures what splitting fusion into clusters gains, with one engine per cluster
# (docs/03-evaluation/cluster-parallel-fusion.md). Messages are routed to clusters
# by their sensor's deployment (method A) or by the detection's own position on a
# fixed grid (method C); each cluster is fused by its own engine in its own
# process, with its own frame, late gate, caps and track-id namespace, and the
# instances' pictures are merged, checked for duplicate tracks and gaps, and
# scored per site against one engine fusing everything and each site alone.
# Method B, single-linkage over sensor positions, is reported from sensor
# geometry. A synthetic boundary case (B1) puts two radar groups with
# overlapping footprints either side of a grid edge. Dev tooling only.

import argparse
import importlib.util
import json
import math
import random
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "multisite_benchmark", Path(__file__).with_name("multisite_benchmark.py")
)
msb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(msb)
fb = msb.fb

REPO = fb.REPO
FAR_PAIR = msb.FAR_PAIR
KM_PER_DEG = fb.EARTH_RADIUS_M * math.pi / 180.0 / 1000.0
DEFAULT_CELL_KM = 100.0
# Stitch limits: a coast window apart in time, and the scoring's own false-track distance.
STITCH_DT_S = msb.MAX_COAST_SECONDS
STITCH_DIST_M = fb.FALSE_TRACK_DISTANCE_M
LINKAGE_KM = (50, 100, 150, 250, 300)
ALL_SITES = ("joensuu", "alakurtti", "hamina_kotka", "porvoo_oil_refinery")


# --- clustering sensors ------------------------------------------------------------------


def _distance_m(a, b):
    return fb.haversine_m(a["lat"], a["lon"], b["lat"], b["lon"])


def link_within(threshold_m):
    """Link two sensors closer than `threshold_m`."""
    return lambda a, b: _distance_m(a, b) < threshold_m


def footprints_overlap(a, b):
    """Link two sensors whose coverage circles overlap."""
    return _distance_m(a, b) < float(a["range_m"]) + float(b["range_m"])


def cluster_sensors(sensors, link):
    """Single-linkage clusters of sensor ids under `link`, each sorted, in order of first id."""
    ids = [s["id"] for s in sensors]
    parent = list(range(len(sensors)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(sensors)):
        for j in range(i + 1, len(sensors)):
            if link(sensors[i], sensors[j]):
                parent[root(i)] = root(j)
    groups = defaultdict(list)
    for i, node in enumerate(ids):
        groups[root(i)].append(node)
    return sorted(sorted(g) for g in groups.values())


def deployment_clusters(sites):
    """Method A: node id -> the site (sensor-network config) it belongs to."""
    return {s["id"]: name for name, sensors in sites.items() for s in sensors}


# --- the grid ----------------------------------------------------------------------------


def _row_centre_lat(row, cell_km):
    return (row + 0.5) * cell_km / KM_PER_DEG


def grid_cell(lat, lon, cell_km):
    """(row, col) of a fixed grid of cells about `cell_km` square; columns follow each row."""
    row = math.floor(lat * KM_PER_DEG / cell_km)
    scale = KM_PER_DEG * math.cos(math.radians(_row_centre_lat(row, cell_km)))
    return row, math.floor(lon * scale / cell_km)


def cell_edge_lon(lat, col, cell_km):
    """The west edge of column `col`, in the row that holds `lat`."""
    row = math.floor(lat * KM_PER_DEG / cell_km)
    scale = KM_PER_DEG * math.cos(math.radians(_row_centre_lat(row, cell_km)))
    return col * cell_km / scale


# --- routing messages --------------------------------------------------------------------


def detection_position(message, sensors_by_id):
    """(lat, lon) of a detection report: its location, or its range and bearing from its sensor."""
    body = message["sapientMessage"]
    report = body["detectionReport"]
    if "location" in report:
        return report["location"]["y"], report["location"]["x"]
    rb = report["rangeBearing"]
    sensor = sensors_by_id[body["nodeId"]]
    ground = rb["range"] * math.cos(math.radians(rb.get("elevation", 0.0)))
    return msb._destination(sensor["lat"], sensor["lon"], ground, math.radians(rb["azimuth"]))


def route_by_node(messages, node_cluster):
    """Methods A and B: cluster -> its messages, in input order."""
    routed = {}
    for message in messages:
        node = message["sapientMessage"]["nodeId"]
        if node not in node_cluster:
            raise ValueError(f"sensor {node} is in no cluster")
        routed.setdefault(node_cluster[node], []).append(message)
    return routed


def route_by_cell(messages, sensors_by_id, cell_km):
    """Method C: grid cell label -> the messages whose detection lies in it, in input order."""
    routed = {}
    for message in messages:
        row, col = grid_cell(*detection_position(message, sensors_by_id), cell_km)
        routed.setdefault(f"r{row}c{col}", []).append(message)
    return routed


def makespan(seconds, cores):
    """Wall-clock of jobs of `seconds` on `cores`, longest first onto the least-loaded core."""
    loads = [0.0] * cores
    for job in sorted(seconds, reverse=True):
        loads[loads.index(min(loads))] += job
    return max(loads)


# --- merging the instances' pictures -------------------------------------------------------


def _followed(instance_tracks, swarms):
    """(instance, uid, wave, snapshots) for every track that follows a swarm."""
    out = []
    for instance, tracks in instance_tracks.items():
        for uid, snapshots in tracks.items():
            wave, median = fb.match_track(snapshots, swarms)
            if wave is not None and median <= fb.FALSE_TRACK_DISTANCE_M:
                out.append((instance, uid, wave, snapshots))
    return out


def duplicates(instance_tracks, swarms):
    """(uid, uid, wave) for tracks in different instances following one swarm at the same time."""
    followed = _followed(instance_tracks, swarms)
    out = []
    for i, (ia, ua, wa, sa) in enumerate(followed):
        for ib, ub, wb, sb in followed[i + 1 :]:
            if ia != ib and wa == wb and sa[0][0] <= sb[-1][0] and sb[0][0] <= sa[-1][0]:
                out.append((ua, ub, wa))
    return out


def gaps(times, intervals, max_gap_s):
    """(first, last) of each run of `times` outside every interval, `max_gap_s` or longer."""
    found = []
    run = []
    for when in [*times, None]:
        covered = when is not None and any(s <= when <= e for s, e in intervals)
        if when is not None and not covered:
            run.append(when)
            continue
        if run and (run[-1] - run[0]).total_seconds() >= max_gap_s:
            found.append((run[0], run[-1]))
        run = []
    return found


def swarm_gaps(swarm, tracks, sensors, max_gap_s, step_s=1.0):
    """Gaps in `tracks`' cover of `swarm` while it flies within some sensor's range."""
    times = []
    when = swarm.launch
    while (pos := swarm.position(when)) is not None:
        if any(fb.haversine_m(*pos, s["lat"], s["lon"]) <= s["range_m"] for s in sensors):
            times.append(when)
        when += timedelta(seconds=step_s)
    return gaps(times, [(t[0][0], t[-1][0]) for t in tracks], max_gap_s)


def _position_at(snapshots, when):
    return min(snapshots, key=lambda s: abs((s[0] - when).total_seconds()))[1:3]


def _apart_m(a, b):
    """How far `b` starts from `a`: at a's end if b starts after it, else median over the overlap."""
    if b[0][0] > a[-1][0]:
        return fb.haversine_m(*a[-1][1:3], *b[0][1:3])
    shared = [s for s in b if s[0] <= a[-1][0]]
    dists = sorted(fb.haversine_m(*_position_at(a, s[0]), *s[1:3]) for s in shared)
    return dists[len(dists) // 2]


def stitch(instance_tracks, max_dt_s, max_dist_m):
    """Join tracks across instances that continue each other.

    Track b continues track a, in another instance, when b starts after a and no
    more than `max_dt_s` after a ends, and b is within `max_dist_m` of a: of a's
    last position if it starts after a ends, else over the time both exist. Each
    track continues at most one and is continued by at most one, closest first.
    Returns (uid -> snapshots, with each chain under its first uid; links).
    """
    entries = [
        (instance, uid, snapshots)
        for instance, tracks in instance_tracks.items()
        for uid, snapshots in tracks.items()
    ]
    order = {uid: k for k, (_, uid, _) in enumerate(entries)}
    candidates = []
    for ia, ua, sa in entries:
        for ib, ub, sb in entries:
            if ia == ib or (sb[0][0], order[ub]) <= (sa[0][0], order[ua]):
                continue
            if (sb[0][0] - sa[-1][0]).total_seconds() > max_dt_s:
                continue
            apart = _apart_m(sa, sb)
            if apart <= max_dist_m:
                candidates.append((apart, order[ua], ua, ub))
    successor, predecessor = {}, {}
    for _, _, ua, ub in sorted(candidates):
        if ua not in successor and ub not in predecessor:
            successor[ua] = ub
            predecessor[ub] = ua
    by_uid = {uid: snapshots for _, uid, snapshots in entries}
    merged = {}
    for _, uid, snapshots in entries:
        if uid in predecessor:
            continue
        chain = list(snapshots)
        nxt = successor.get(uid)
        while nxt is not None:
            chain += [s for s in by_uid[nxt] if s[0] > chain[-1][0]]
            nxt = successor.get(nxt)
        merged[uid] = chain
    links = sorted(successor.items(), key=lambda link: (by_uid[link[0]][0][0], order[link[0]]))
    return merged, links


# --- the synthetic boundary case ------------------------------------------------------------

B1_LAT = 63.5
B1_START = "2026-11-15T03:00:00.000000Z"
B1_RANGE_M = 40_000.0
B1_PERIOD_S = 2.0
B1_NOISE_M = 30.0
# Radars along one parallel, km east of the grid edge; every footprint overlaps the next.
B1_RADARS = {"west": (-60.0, -30.0), "east": (30.0, 60.0)}


def _east_lon(lat, lon, km):
    return lon + km / (KM_PER_DEG * math.cos(math.radians(lat)))


def boundary_case(seed, cell_km=DEFAULT_CELL_KM):
    """B1, synthetic: two radar groups whose footprints overlap across a grid cell edge.

    T1_CROSSING flies from the west group's area to the east group's; T2_OVERLAP
    flies north along the cell edge, in both groups' range. Every radar reports
    every target in its range each period, with seeded Gaussian position noise.
    Object ids are "<radar>:<wave>".
    """
    rng = random.Random(seed)
    _, col = grid_cell(B1_LAT, 27.0, cell_km)
    edge = cell_edge_lon(B1_LAT, col, cell_km)
    sites = {
        group: [
            {
                "id": f"B1-{group.upper()}-RAD-{k + 1}",
                "type": "RADAR",
                "lat": B1_LAT,
                "lon": _east_lon(B1_LAT, edge, km),
                "alt": 100.0,
                "range_m": B1_RANGE_M,
            }
            for k, km in enumerate(kms)
        ]
        for group, kms in B1_RADARS.items()
    }
    lat_t1 = B1_LAT + 5 / KM_PER_DEG
    swarms_cfg = {
        "T1_CROSSING": {
            "count": 1,
            "launch_delay_sec": 0,
            "speed_kmh": 360,
            "wkt_linestring": (
                f"LINESTRING ({_east_lon(lat_t1, edge, -90)} {lat_t1}, "
                f"{_east_lon(lat_t1, edge, 90)} {lat_t1})"
            ),
        },
        "T2_OVERLAP": {
            "count": 1,
            "launch_delay_sec": 0,
            "speed_kmh": 60,
            "wkt_linestring": (
                f"LINESTRING ({edge} {B1_LAT - 15 / KM_PER_DEG}, {edge} {B1_LAT + 15 / KM_PER_DEG})"
            ),
        },
    }
    case = {"sites": sites, "swarms_cfg": swarms_cfg, "start": B1_START, "cell_km": cell_km}
    swarms = case_swarms(case)
    end = max(s.launch + timedelta(seconds=sum(s.legs) / s.mps) for s in swarms)
    start = fb.parse_time(B1_START)
    radars = [r for group in sites.values() for r in group]
    messages = []
    for k, radar in enumerate(radars):
        when = start + timedelta(seconds=0.5 * k)
        while when <= end:
            for swarm in swarms:
                truth = swarm.position(when)
                if truth is None:
                    continue
                north = rng.gauss(0.0, B1_NOISE_M) / 1000.0
                east = rng.gauss(0.0, B1_NOISE_M) / 1000.0
                lat = truth[0] + north / KM_PER_DEG
                lon = _east_lon(truth[0], truth[1], east)
                margin = 10 * B1_NOISE_M
                if fb.haversine_m(radar["lat"], radar["lon"], *truth) > B1_RANGE_M - margin:
                    continue
                messages.append(_b1_message(radar["id"], when, lat, lon, swarm.wave_id))
            when += timedelta(seconds=B1_PERIOD_S)
    messages.sort(key=lambda m: m["sapientMessage"]["timestamp"])
    return {**case, "messages": messages}


def _b1_message(node_id, when, lat, lon, wave):
    return {
        "sapientMessage": {
            "timestamp": when.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "nodeId": node_id,
            "detectionReport": {
                "objectId": f"{node_id}:{wave}",
                "state": "ACTIVE",
                "location": {
                    "x": lon,
                    "y": lat,
                    "z": 500.0,
                    "coordinateSystem": "LOCATION_COORDINATE_SYSTEM_LAT_LNG_DEG_M",
                    "datum": "LOCATION_DATUM_WGS84_E",
                },
                "objectInfo": [{"type": "estimatedSwarmCount", "value": "1"}],
                "classification": [{"type": "Air vehicle", "confidence": 0.7, "subClass": []}],
            },
        }
    }


def case_swarms(case):
    """B1's true paths as fusion_benchmark Swarms."""
    start = fb.parse_time(case["start"])
    return [fb.Swarm(wave, cfg, start) for wave, cfg in case["swarms_cfg"].items()]


# --- one instance, in this process -----------------------------------------------------------


def run_instance(spec):
    """Fuse one cluster's messages with one engine; the spec names a messages file."""
    messages = json.loads(Path(spec["messages"]).read_text())
    started = time.perf_counter()
    fused = msb.fuse(
        messages,
        spec["sites"],
        anchor=spec.get("anchor"),
        lags=spec.get("lags", {}),
        id_namespace=spec["namespace"],
    )
    fused["process_wall_seconds"] = time.perf_counter() - started
    return fused


def _launch(spec, workdir, name):
    path = Path(workdir) / f"{name}.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    # The command is this script on a spec it wrote itself.
    return subprocess.Popen(  # noqa: S603
        [sys.executable, __file__, "instance", "--spec", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _collect(proc):
    out, err = proc.communicate()
    if proc.returncode:
        raise RuntimeError(f"instance failed: {err[-2000:]}")
    return json.loads(out)


def run_instances(clusters, parallel):
    """Each cluster's instance in its own process: all at once, or one after another.

    `clusters` maps a cluster name to (messages, sites, anchor, lags). Returns the
    instances' results by cluster and the wall-clock of the whole run.
    """
    results = {}
    with tempfile.TemporaryDirectory() as workdir:
        specs = {}
        for name, (messages, sites, anchor, lags) in clusters.items():
            path = Path(workdir) / f"{name}-messages.json"
            path.write_text(json.dumps(messages), encoding="utf-8")
            specs[name] = {
                "messages": str(path),
                "sites": sites,
                "anchor": anchor,
                "lags": lags,
                "namespace": name,
            }
        began = time.perf_counter()
        if parallel:
            procs = {name: _launch(spec, workdir, name) for name, spec in specs.items()}
            results = {name: _collect(proc) for name, proc in procs.items()}
        else:
            results = {name: _collect(_launch(spec, workdir, name)) for name, spec in specs.items()}
        wall = time.perf_counter() - began
    return results, wall


# --- one clustered condition -------------------------------------------------------------------


def plan_clusters(built, method, cell_km):
    """cluster -> (messages, sites, anchor, lags) for a condition under a method.

    "single" is one engine for everything (today's pipeline), "A" one per
    deployment, "C" one per grid cell, each with the whole sensor registry.
    """
    sites, lags = built["sites"], built["lags"]
    if method == "single":
        return {"all": (built["messages"], sites, built["anchor"], lags)}
    if method == "A":
        routed = route_by_node(built["messages"], deployment_clusters(sites))
        return {
            n: (ms, {n: sites[n]}, built["anchors"][n], {k: v for k, v in lags.items() if k == n})
            for n, ms in routed.items()
        }
    if method == "C":
        by_id = {s["id"]: s for ss in sites.values() for s in ss}
        routed = route_by_cell(built["messages"], by_id, cell_km)
        return {cell: (ms, sites, built["anchor"], lags) for cell, ms in routed.items()}
    raise ValueError(f"unknown method {method}")


def merge_and_score(instances, sites, swarms):
    """Score the merged picture per site, with and without the stitch, and check it."""
    instance_tracks = {name: fb.read_tracks("\n".join(r["cot"])) for name, r in instances.items()}
    all_swarms = [w for n in sites for w in swarms[n]]
    all_sensors = [s for ss in sites.values() for s in ss]
    cot = [c for r in instances.values() for c in r["cot"]]
    scores, _ = msb.score_by_site(cot, sites, swarms, (0.0, 0.0, 0.0))
    stitched, links = stitch(instance_tracks, STITCH_DT_S, STITCH_DIST_M)
    stitched_scores = {}
    by_site = defaultdict(dict)
    for uid, snapshots in stitched.items():
        by_site[msb.site_of_snapshots(snapshots, sites)][uid] = snapshots
    for n in sites:
        stitched_scores[n] = fb.score_tracks(n, by_site.get(n, {}), swarms[n])

    def gap_report(tracks):
        found = {}
        followed = _followed({"merged": tracks}, all_swarms)
        for swarm in all_swarms:
            mine = [s for _, _, wave, s in followed if wave == swarm.wave_id]
            runs = swarm_gaps(swarm, mine, all_sensors, msb.MAX_COAST_SECONDS)
            found[swarm.wave_id] = [(b - a).total_seconds() for a, b in runs]
        return found

    merged_raw = {uid: s for tracks in instance_tracks.values() for uid, s in tracks.items()}
    raw = duplicates(instance_tracks, all_swarms)
    # A duplicate the stitch left standing is one whose two tracks head different chains.
    head = {uid: uid for uid in merged_raw}
    for first, nxt in links:
        head[nxt] = head[first]
    return {
        "scores": scores,
        "duplicates": [list(d) for d in raw],
        "gaps_s": gap_report(merged_raw),
        "stitched": {
            "links": [list(link) for link in links],
            "scores": stitched_scores,
            "duplicates": [list(d) for d in raw if head[d[0]] != head[d[1]]],
            "gaps_s": gap_report(stitched),
        },
    }


def site_totals(instances, names):
    """Per-site counts over all instances, with the median over every event's time."""
    out = {}
    for n in names:
        parts = [r["sites"][n] for r in instances.values() if n in r["sites"]]
        ms = [m for r in instances.values() for m in r["ms_by_site"].get(n, [])]
        out[n] = {
            "events": sum(p["events"] for p in parts),
            "detections": sum(p["detections"] for p in parts),
            "dropped": sum(p["dropped"] for p in parts),
            "refused_tracks": sum(p["refused_tracks"] for p in parts),
            "peak_live_tracks": max((p["peak_live_tracks"] for p in parts), default=0),
            "median_ms_per_event": statistics.median(ms) if ms else None,
        }
    return out


def throughput(instances, wall):
    """Detections per second: per instance, and for all, sequentially and on cores."""
    seconds = {name: r["process_seconds"] for name, r in instances.items()}
    offered = {name: r["detections_offered"] for name, r in instances.items()}
    total = sum(offered.values())
    per = {name: offered[name] / seconds[name] if seconds[name] else None for name in instances}
    return {
        "detections": total,
        "per_instance": per,
        "process_seconds": seconds,
        # One core doing every cluster's work: the engine-level partitioning estimate.
        "one_core": total / sum(seconds.values()),
        "two_cores": total / makespan(list(seconds.values()), 2),
        "core_per_cluster": total / max(seconds.values()),
        "wall_seconds": wall,
    }


def run_clustered(built, swarms, method, parallel, cell_km=DEFAULT_CELL_KM):
    began = time.perf_counter()
    clusters = plan_clusters(built, method, cell_km)
    routing_s = time.perf_counter() - began
    instances, wall = run_instances(clusters, parallel)
    merged = merge_and_score(instances, built["sites"], swarms)
    return {
        "method": method,
        "parallel": parallel,
        "clusters": {name: len(c[0]) for name, c in clusters.items()},
        "routing_us_per_message": routing_s * 1e6 / max(1, len(built["messages"])),
        "sites": site_totals(instances, built["names"]),
        "shed_detections": sum(r["shed_detections"] for r in instances.values()),
        "cap_full": {name: r["cap_full"] for name, r in instances.items()},
        "throughput": throughput(instances, wall),
        **merged,
    }


def boundary_built(seed, cell_km):
    case = boundary_case(seed, cell_km)
    names = list(case["sites"])
    swarms = case_swarms(case)
    return (
        {
            "names": names,
            "sites": case["sites"],
            "anchor": None,
            "anchors": dict.fromkeys(names),
            "lags": {},
            "messages": case["messages"],
        },
        # Each group is scored against both targets: either may cross into it.
        dict.fromkeys(names, swarms),
    )


# --- geometry: method B -------------------------------------------------------------------------


def geometry(sensors_dir):
    """Method B over every site's sensor config: clusters per threshold, and its own cost."""
    sites = {n: json.loads((sensors_dir / f"{n}.json").read_text())["sensors"] for n in ALL_SITES}
    sensors = [s for n in ALL_SITES for s in sites[n]]
    owner = deployment_clusters(sites)
    rows = []
    for km in LINKAGE_KM:
        clusters = cluster_sensors(sensors, link_within(km * 1000))
        rows.append(
            {
                "link": f"< {km} km",
                "clusters": len(clusters),
                "split_sites": sorted(
                    n for n in ALL_SITES if sum(any(owner[i] == n for i in c) for c in clusters) > 1
                ),
            }
        )
    rows.append(
        {
            "link": "footprints overlap",
            "clusters": len(cluster_sensors(sensors, footprints_overlap)),
            "split_sites": [],
        }
    )
    began = time.perf_counter()
    repeats = 100
    for _ in range(repeats):
        cluster_sensors(sensors, link_within(100_000))
    recluster_ms = (time.perf_counter() - began) * 1000 / repeats
    span = {
        n: max((_distance_m(a, b) for a in sites[n] for b in sites[n]), default=0) / 1000
        for n in ALL_SITES
    }
    return {
        "sensors": len(sensors),
        "thresholds": rows,
        "recluster_ms": recluster_ms,
        "site_span_km": span,
    }


# --- experiments ------------------------------------------------------------------------------


def _scenario_conditions(args):
    base = {"scenario_foundry": str(args.scenario_foundry), "sensors_dir": str(args.sensors_dir)}
    a, b = FAR_PAIR
    out = [("C1", "combined", {**base, "sites": FAR_PAIR}, ("single", "A", "C"))]
    for lag in args.lags:
        out.append(
            (
                "C1",
                f"{b} {lag:g} s behind",
                {**base, "sites": FAR_PAIR, "lags": {b: lag}},
                ("single", "A"),
            )
        )
    for rate in args.clutter_rates:
        methods = ("single", "A", "C") if rate == max(args.clutter_rates) else ("single", "A")
        out.append(
            (
                "C1",
                f"clutter {rate:g}/s at {a}",
                {
                    **base,
                    "sites": FAR_PAIR,
                    "clutter": {"site": a, "rate": rate, "seed": args.seed},
                },
                methods,
            )
        )
    for delay in args.p0_delays:
        out.append(
            (
                "P0",
                f"clutter {max(args.clutter_rates):g}/s at {a}, {b} {delay:g} s later",
                {
                    **base,
                    "sites": FAR_PAIR,
                    "delays": {b: delay},
                    "clutter": {"site": a, "rate": max(args.clutter_rates), "seed": args.seed},
                },
                ("single", "A"),
            )
        )
    return out


def _alone(args):
    base = {"scenario_foundry": str(args.scenario_foundry), "sensors_dir": str(args.sensors_dir)}
    out = [(f"{n} alone", {**base, "sites": [n]}) for n in FAR_PAIR]
    return out


def _repeat(fn, repeats):
    runs = [fn() for _ in range(repeats)]
    keep = ("scores", "duplicates", "gaps_s", "stitched")
    first = runs[0]
    deterministic = all(
        {k: r[k] for k in keep} == {k: first[k] for k in keep}
        and {
            n: {x: v for x, v in s.items() if x != "median_ms_per_event"}
            for n, s in r["sites"].items()
        }
        == {
            n: {x: v for x, v in s.items() if x != "median_ms_per_event"}
            for n, s in first["sites"].items()
        }
        for r in runs
    )
    summary = dict(first)
    summary["repeats"] = len(runs)
    summary["deterministic"] = deterministic
    for n in summary["sites"]:
        values = [r["sites"][n]["median_ms_per_event"] for r in runs]
        values = [v for v in values if v is not None]
        summary["sites"][n] = {
            **first["sites"][n],
            "median_ms_per_event": statistics.median(values) if values else None,
        }
    tp = {}
    for key in ("one_core", "two_cores", "core_per_cluster", "wall_seconds"):
        tp[key] = statistics.median(r["throughput"][key] for r in runs)
    tp["detections"] = first["throughput"]["detections"]
    tp["per_instance"] = {
        name: statistics.median(r["throughput"]["per_instance"][name] for r in runs)
        for name in first["throughput"]["per_instance"]
    }
    tp["process_seconds"] = {
        name: statistics.median(r["throughput"]["process_seconds"][name] for r in runs)
        for name in first["throughput"]["process_seconds"]
    }
    summary["throughput"] = tp
    summary["routing_us_per_message"] = statistics.median(r["routing_us_per_message"] for r in runs)
    return summary


def experiments(args):
    results = []
    only = set(args.only)
    if "C1" in only or "P0" in only:
        for experiment, label, spec, methods in _scenario_conditions(args):
            if experiment not in only:
                continue
            built = msb.build_messages(spec)
            swarms = msb.condition_swarms(spec, built)
            for method in methods:
                modes = (False,) if method == "single" else (False, True)
                for parallel in modes:
                    print(f"{experiment} {label} {method} parallel={parallel}", file=sys.stderr)
                    run = _repeat(
                        lambda b=built, s=swarms, m=method, p=parallel: run_clustered(
                            b, s, m, p, args.cell_km
                        ),
                        args.repeats,
                    )
                    results.append({"experiment": experiment, "label": label, **run})
        for label, spec in _alone(args):
            if "C1" not in only:
                break
            built = msb.build_messages(spec)
            swarms = msb.condition_swarms(spec, built)
            run = _repeat(
                lambda b=built, s=swarms: run_clustered(b, s, "single", False), args.repeats
            )
            results.append({"experiment": "C1", "label": label, **run})
    if "C3" in only:
        built, swarms = boundary_built(args.seed, args.cell_km)
        for method in ("single", "A", "C"):
            print(f"C3 B1 {method}", file=sys.stderr)
            run = _repeat(
                lambda m=method: run_clustered(built, swarms, m, False, args.cell_km), args.repeats
            )
            results.append({"experiment": "C3", "label": "B1 (synthetic)", **run})
    out = {
        "context_foundry": msb._commit(REPO),
        "scenario_foundry": msb._commit(args.scenario_foundry),
        "machine": msb._machine(),
        "seed": args.seed,
        "cell_km": args.cell_km,
        "results": results,
    }
    if "C2" in only:
        out["geometry"] = geometry(args.sensors_dir)
    return out


def print_results(out):
    for r in out["results"]:
        tp = r["throughput"]
        mode = "parallel" if r["parallel"] else "sequential"
        print(
            f"\n== {r['experiment']} {r['label']} | {r['method']} {mode} | "
            f"clusters {len(r['clusters'])} | repeats {r['repeats']} det {r['deterministic']}"
        )
        for n, s in r["sites"].items():
            sc, st = r["scores"][n], r["stitched"]["scores"][n]
            ms = f"{s['median_ms_per_event']:.2f}" if s["median_ms_per_event"] is not None else "-"
            print(
                f"  {n:22s} tracks {sc['tracks']:4d} false {sc['false_tracks']:4d} "
                f"err {sc['median_error_m']} drop {s['dropped']:4d} refused {s['refused_tracks']:4d} "
                f"ms/ev {ms:>6} | stitched tracks {st['tracks']} false {st['false_tracks']}"
            )
        gaps_raw = {w: g for w, g in r["gaps_s"].items() if g}
        gaps_st = {w: g for w, g in r["stitched"]["gaps_s"].items() if g}
        print(
            f"  duplicates {len(r['duplicates'])}, links {len(r['stitched']['links'])}, "
            f"gaps {gaps_raw}, gaps stitched {gaps_st}"
        )
        per = ", ".join(f"{k} {v:.0f}" for k, v in tp["per_instance"].items())
        print(
            f"  det/s per instance: {per}; all: one core {tp['one_core']:.0f}, "
            f"two cores {tp['two_cores']:.0f}, core per cluster {tp['core_per_cluster']:.0f}; "
            f"routing {r['routing_us_per_message']:.1f} us/msg"
        )
    if "geometry" in out:
        print("\n== C2 geometry")
        print(json.dumps(out["geometry"], indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("instance", help="fuse one cluster in this process and print its JSON")
    one.add_argument("--spec", type=Path, required=True)
    exp = sub.add_parser("experiments", help="run the experiments")
    exp.add_argument("--scenario-foundry", type=Path, default=fb.DEFAULT_SCENARIO_FOUNDRY)
    exp.add_argument("--sensors-dir", type=Path, default=REPO / "config" / "sensors")
    exp.add_argument("--only", nargs="+", default=["P0", "C1", "C2", "C3"])
    exp.add_argument("--repeats", type=int, default=3)
    exp.add_argument("--lags", nargs="+", type=float, default=[3, 5])
    exp.add_argument("--clutter-rates", nargs="+", type=float, default=[1, 2, 5])
    exp.add_argument("--p0-delays", nargs="+", type=float, default=[])
    exp.add_argument("--cell-km", type=float, default=DEFAULT_CELL_KM)
    exp.add_argument("--seed", type=int, default=42, help="clutter and B1 seed")
    exp.add_argument("--save", type=Path, help="write all results as JSON")
    args = parser.parse_args()

    if args.command == "instance":
        print(json.dumps(run_instance(json.loads(args.spec.read_text())), default=str))
        return
    out = experiments(args)
    print_results(out)
    if args.save:
        args.save.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
