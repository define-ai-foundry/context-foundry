#!/usr/bin/env python3
# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# Measures one fusion engine fusing two unrelated, distant sites at once, against
# each site fused alone (docs/03-evaluation/multisite-fusion.md). The sites'
# scenario-foundry logs are aligned to one start, merged, and fed to the engine
# in-process in true time order, so a site whose clock lags reaches the engine
# the way it would live: on time, stamped early. A replay file through the CLI
# cannot show that, because its source sorts every event by its stamp. Each
# condition runs in its own process, as the sensor registry and the frame origin
# are module-global. Tracks are scored per site with fusion_benchmark's scoring.
# Dev tooling only.

import argparse
import copy
import importlib.util
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from context_foundry.fusion import config
from context_foundry.fusion.augmentor import TacticalContextAugmentor
from context_foundry.fusion.engine import FusionEngine
from context_foundry.fusion.publish import (
    DEFAULT_COAST_SECONDS,
    DEFAULT_WINDOW_SECONDS,
    PublishPolicy,
)
from context_foundry.fusion.serializers import DEFAULT_STALE_SECONDS, CotSerializer
from context_foundry.fusion.sources.json_file import JsonSapientSource
from context_foundry.fusion.tracker import (
    DEFAULT_CONFIRM_HITS,
    MAX_COAST_SECONDS,
    MAX_EVENT_DETECTIONS,
    MAX_LIVE_TRACKS,
    MAX_TRACK_HISTORY,
    SapientAsynchronousTracker,
)

_spec = importlib.util.spec_from_file_location(
    "fusion_benchmark", Path(__file__).with_name("fusion_benchmark.py")
)
fb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fb)

REPO = fb.REPO
FAR_PAIR = ["joensuu", "alakurtti"]
FRAME_BIN_KM = 100


# --- timestamps -----------------------------------------------------------------------


def _parse_stamp(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _format_stamp(moment):
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def alignment_shifts(starts):
    """Seconds to move each site's clock by so that every site starts with the earliest."""
    earliest = min(starts.values())
    return {name: (earliest - start).total_seconds() for name, start in starts.items()}


def shift_messages(messages, seconds):
    """Copies of SAPIENT messages with every timestamp moved by `seconds`."""
    shifted = copy.deepcopy(messages)
    for message in shifted:
        sm = message["sapientMessage"]
        sm["timestamp"] = _format_stamp(_parse_stamp(sm["timestamp"]) + timedelta(seconds=seconds))
    return shifted


# --- the shared network ---------------------------------------------------------------


def union_sensors(sensor_lists):
    """One sensor network from several sites'; a node id two sites share is an error."""
    seen = set()
    union = []
    for sensors in sensor_lists:
        for sensor in sensors:
            if sensor["id"] in seen:
                raise ValueError(f"node id {sensor['id']} is in more than one site")
            seen.add(sensor["id"])
            union.append(sensor)
    return union


def geodesic_midpoint(lat1, lon1, lat2, lon2):
    """The point halfway along the great circle between two points, in degrees."""
    p1, l1, p2, l2 = map(math.radians, (lat1, lon1, lat2, lon2))
    bx = math.cos(p2) * math.cos(l2 - l1)
    by = math.cos(p2) * math.sin(l2 - l1)
    lat = math.atan2(math.sin(p1) + math.sin(p2), math.hypot(math.cos(p1) + bx, by))
    lon = l1 + math.atan2(by, math.cos(p1) + bx)
    return math.degrees(lat), math.degrees(lon)


def _destination(lat, lon, distance_m, bearing):
    """The point `distance_m` from (lat, lon) along `bearing` (radians), on fb's sphere."""
    d = distance_m / fb.EARTH_RADIUS_M
    p1, l1 = math.radians(lat), math.radians(lon)
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(bearing))
    l2 = l1 + math.atan2(
        math.sin(bearing) * math.sin(d) * math.cos(p1), math.cos(d) - math.sin(p1) * math.sin(p2)
    )
    return math.degrees(p2), math.degrees(l2)


# --- clutter ----------------------------------------------------------------------------


def clutter_messages(sensors, template, start, end, rate_per_s, seed):
    """Seeded false detections from `sensors`, `rate_per_s` on average over [start, end].

    Each is one object of its own (no swarm count) at a random point inside a
    random sensor's range, so it associates with nothing real and seeds a
    tentative track: load on the shared track cap, from one site only.
    """
    rng = random.Random(seed)
    span = (end - start).total_seconds()
    placed = [s for s in sensors if "lat" in s and "lon" in s]
    messages = []
    for i in range(round(rate_per_s * span)):
        when = start + timedelta(seconds=rng.uniform(0.0, span))
        sensor = rng.choice(placed)
        reach = float(sensor.get("range_m", 10000.0)) * math.sqrt(rng.random())
        lat, lon = _destination(sensor["lat"], sensor["lon"], reach, rng.uniform(0, 2 * math.pi))
        message = copy.deepcopy(template)
        sm = message["sapientMessage"]
        sm["timestamp"] = _format_stamp(when)
        sm["nodeId"] = sensor["id"]
        report = sm["detectionReport"]
        report["objectId"] = f"CLUTTER-{seed}-{i}"
        report["location"].update(x=lon, y=lat, z=rng.uniform(100.0, 1500.0))
        report.pop("objectInfo", None)
        messages.append(message)
    messages.sort(key=lambda m: m["sapientMessage"]["timestamp"])
    return messages


# --- delivery and attribution -------------------------------------------------------------


def deliver(events, node_site, lags):
    """(stamp, detections, site) in true order, a lagging site's events stamped early.

    `events` are (true time, detections) in true time order; `lags` gives seconds
    per site. A lagged site's detections are restamped in place.
    """
    delivered = []
    for when, detections in events:
        site = node_site.get(detections[0].metadata.get("nodeId")) if detections else None
        lag = lags.get(site, 0.0)
        stamp = when - timedelta(seconds=lag) if lag else when
        if lag:
            for detection in detections:
                detection.timestamp = stamp
        delivered.append((stamp, detections, site))
    return delivered


def nearest_site(east, north, sensors_enu):
    """The site of the sensor closest to (east, north); `sensors_enu` is (e, n, site)."""
    return min(sensors_enu, key=lambda s: (s[0] - east) ** 2 + (s[1] - north) ** 2)[2]


def site_of_snapshots(snapshots, sites):
    """The site of the sensor closest to a track's first position; `sites` maps to sensors."""
    _, lat, lon, _ = snapshots[0]
    return min(
        (
            (fb.haversine_m(lat, lon, s["lat"], s["lon"]), name)
            for name, ss in sites.items()
            for s in ss
        ),
    )[1]


class CountingTracker(SapientAsynchronousTracker):
    """The engine's tracker, also counting which sensor each refused new track came from."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.refused_by_node = Counter()

    def _initialize_new_track(self, detection):
        if len(self.tracks) >= self.max_live_tracks:
            self.refused_by_node[detection.metadata.get("nodeId")] += 1
        return super()._initialize_new_track(detection)


def make_tracker():
    """A CountingTracker with the CLI's defaults, which are not all the constructor's."""
    return CountingTracker(
        max_live_tracks=MAX_LIVE_TRACKS,
        max_track_history=MAX_TRACK_HISTORY,
        max_coast_seconds=MAX_COAST_SECONDS,
        max_event_detections=MAX_EVENT_DETECTIONS,
        confirm_hits=DEFAULT_CONFIRM_HITS,
    )


# --- summaries ----------------------------------------------------------------------------


def fit_cost(rows):
    """Least-squares ms per event = a + b * own-site live tracks + c * other-site live tracks."""
    if len(rows) < 3:
        return None
    ms = np.array([r[0] for r in rows], dtype=float)
    design = np.array([[1.0, r[1], r[2]] for r in rows], dtype=float)
    (a, b, c), *_ = np.linalg.lstsq(design, ms, rcond=None)
    return {"intercept_ms": float(a), "ms_per_own_track": float(b), "ms_per_other_track": float(c)}


def error_bins(samples, width_km):
    """Median error of (distance from the frame origin in km, error in m) samples, per bin."""
    bins = defaultdict(list)
    for km, err in samples:
        bins[int(km // width_km)].append(err)
    return [
        {
            "from_km": b * width_km,
            "to_km": (b + 1) * width_km,
            "snapshots": len(errs),
            "median_error_m": statistics.median(errs),
        }
        for b, errs in sorted(bins.items())
    ]


# --- one condition, in this process -------------------------------------------------------


def _scenario_start(name, scenario_foundry):
    tactical = json.loads(
        (scenario_foundry / "data/tactical_scenarios" / f"{name}_tactical.json").read_text()
    )
    return _parse_stamp(tactical["scenario_meta"]["start_time_iso"])


def _origin(spec_origin, sensors):
    if spec_origin is None:
        return None
    if "node" in spec_origin:
        sensor = next(s for s in sensors if s["id"] == spec_origin["node"])
        return sensor["lat"], sensor["lon"], float(sensor.get("alt", 0.0))
    return spec_origin["lat"], spec_origin["lon"], spec_origin.get("alt", 0.0)


def run_condition(spec):
    """Fuse one condition and return its measurements. Changes module-global engine state."""
    sf = Path(spec["scenario_foundry"])
    names = spec["sites"]
    lags = {k: float(v) for k, v in spec.get("lags", {}).items()}
    networks = {n: json.loads((Path(spec["sensors_dir"]) / f"{n}.json").read_text()) for n in names}
    sites = {n: networks[n]["sensors"] for n in names}
    shifts = alignment_shifts({n: _scenario_start(n, sf) for n in names})

    messages = []
    for n in names:
        raw = json.loads((sf / "data/generated_output" / f"{n}_messages.json").read_text())
        messages += shift_messages(raw, shifts[n])
    clutter = spec.get("clutter")
    if clutter and clutter["rate"] > 0:
        stamps = sorted(_parse_stamp(m["sapientMessage"]["timestamp"]) for m in messages)
        own = {s["id"] for s in sites[clutter["site"]]}
        template = next(m for m in messages if m["sapientMessage"]["nodeId"] in own)
        messages += clutter_messages(
            sites[clutter["site"]],
            template,
            stamps[0],
            stamps[-1],
            clutter["rate"],
            clutter["seed"],
        )

    sensors = union_sensors([sites[n] for n in names])
    config.load_sensor_network(
        sensor_network_list=sensors,
        primary_anchor_node=networks[names[0]].get("primary_anchor_node"),
    )
    # As the pipeline stage does: no frame until the first point projected sets it,
    # unless the condition places the origin.
    config.set_reference_origin(None, None, None)
    origin = _origin(spec.get("origin"), sensors)
    if origin is not None:
        config.set_reference_origin(*origin)

    with tempfile.TemporaryDirectory() as workdir:
        path = Path(workdir) / "replay.json"
        path.write_text(json.dumps(messages), encoding="utf-8")
        events = list(JsonSapientSource(path).iter_events())
    frame_origin = (config._origin_lat, config._origin_lon, config._origin_alt)

    node_site = {s["id"]: n for n in names for s in sites[n]}
    sensors_enu = []
    for n in names:
        for s in sites[n]:
            e, north, _ = config.wgs84_to_enu(s["lat"], s["lon"], float(s.get("alt", 0.0)))
            sensors_enu.append((e, north, n))

    tracker = make_tracker()
    engine = FusionEngine(
        tracker,
        PublishPolicy("window", DEFAULT_COAST_SECONDS, DEFAULT_WINDOW_SECONDS),
        TacticalContextAugmentor(),
    )
    serializer = CotSerializer(stale_seconds=DEFAULT_STALE_SECONDS)

    cot = []
    rows = []
    per_site = {n: {"events": 0, "detections": 0, "dropped": 0, "ms": []} for n in names}
    peak_live = dict.fromkeys(names, 0)
    worst_skew = {}
    began = time.perf_counter()
    for stamp, detections, site in deliver(events, node_site, lags):
        live = Counter(
            nearest_site(t.state_vector[0, 0], t.state_vector[3, 0], sensors_enu)
            for t in tracker.tracks
        )
        for n in names:
            peak_live[n] = max(peak_live[n], live[n])
        t0 = time.perf_counter()
        result = engine.process(stamp, detections)
        ms = (time.perf_counter() - t0) * 1000.0
        stats = per_site[site]
        if result.dropped:
            stats["dropped"] += 1
            worst_skew[site] = max(worst_skew.get(site, 0.0), float(result.skew.split("s ")[0]))
            continue
        stats["events"] += 1
        stats["detections"] += len(detections)
        stats["ms"].append(ms)
        rows.append((ms, live[site], sum(live.values()) - live[site]))
        cot.extend(serializer.serialize(tactical) for _, tactical in result.published)
    loop_seconds = time.perf_counter() - began

    tracks = fb.read_tracks("\n".join(cot))
    by_site = defaultdict(dict)
    for uid, snapshots in tracks.items():
        by_site[site_of_snapshots(snapshots, sites)][uid] = snapshots
    scores = {}
    samples = []
    for n in names:
        swarms = fb.load_swarms(n, sf, shifts[n] - lags.get(n, 0.0))
        scores[n] = fb.score_tracks(n, by_site.get(n, {}), swarms)
        for snapshots in by_site.get(n, {}).values():
            for when, lat, lon, _ in snapshots:
                match = fb.nearest_swarm(swarms, when, lat, lon)
                if match is not None and match[1] <= fb.FALSE_TRACK_DISTANCE_M:
                    km = fb.haversine_m(frame_origin[0], frame_origin[1], lat, lon) / 1000.0
                    samples.append((km, match[1]))

    refused_by_site = Counter()
    for node, count in tracker.refused_by_node.items():
        refused_by_site[node_site.get(node, "unknown")] += count
    all_ms = [ms for s in per_site.values() for ms in s["ms"]]
    all_dets = sum(s["detections"] for s in per_site.values())
    return {
        "spec": spec,
        "frame_origin": frame_origin,
        "scores": scores,
        "sites": {
            n: {
                "events": per_site[n]["events"],
                "detections": per_site[n]["detections"],
                "dropped": per_site[n]["dropped"],
                "worst_skew_s": worst_skew.get(n),
                "median_ms_per_event": statistics.median(per_site[n]["ms"])
                if per_site[n]["ms"]
                else None,
                "peak_live_tracks": peak_live[n],
                "refused_tracks": refused_by_site.get(n, 0),
            }
            for n in names
        },
        "refused_tracks": tracker._refused_tracks,
        "shed_detections": tracker._shed_detections,
        "ms_per_detection": sum(all_ms) / all_dets if all_dets else None,
        "loop_seconds": loop_seconds,
        "cost_fit": fit_cost(rows),
        "error_by_frame_distance": error_bins(samples, FRAME_BIN_KM),
    }


# --- experiments -------------------------------------------------------------------------


def conditions(args):
    """(experiment, label, spec) for every condition asked for."""
    base = {"scenario_foundry": str(args.scenario_foundry), "sensors_dir": str(args.sensors_dir)}
    a, b = FAR_PAIR
    nets = {n: json.loads((args.sensors_dir / f"{n}.json").read_text()) for n in FAR_PAIR}
    anchors = {
        n: next(s for s in nets[n]["sensors"] if s["id"] == nets[n]["primary_anchor_node"])
        for n in FAR_PAIR
    }
    mid_lat, mid_lon = geodesic_midpoint(
        anchors[a]["lat"], anchors[a]["lon"], anchors[b]["lat"], anchors[b]["lon"]
    )
    mid_alt = (anchors[a].get("alt", 0.0) + anchors[b].get("alt", 0.0)) / 2

    out = []
    if "E1" in args.only:
        out += [
            ("E1", f"{a} alone", {**base, "sites": [a]}),
            ("E1", f"{b} alone", {**base, "sites": [b]}),
            ("E1", "combined", {**base, "sites": FAR_PAIR}),
        ]
    if "E2" in args.only:
        for n in FAR_PAIR:
            out.append(
                (
                    "E2",
                    f"origin {anchors[n]['id']}",
                    {**base, "sites": FAR_PAIR, "origin": {"node": anchors[n]["id"]}},
                )
            )
        out.append(
            (
                "E2",
                f"origin midpoint {mid_lat:.3f}N {mid_lon:.3f}E",
                {
                    **base,
                    "sites": FAR_PAIR,
                    "origin": {"lat": mid_lat, "lon": mid_lon, "alt": mid_alt},
                },
            )
        )
    if "E3" in args.only:
        for lag in args.lags:
            out.append(
                ("E3", f"{b} {lag:g} s behind", {**base, "sites": FAR_PAIR, "lags": {b: lag}})
            )
    if "E4" in args.only:
        for rate in args.clutter_rates:
            out.append(
                (
                    "E4",
                    f"clutter {rate:g}/s at {a}",
                    {
                        **base,
                        "sites": FAR_PAIR,
                        "clutter": {"site": a, "rate": rate, "seed": args.seed},
                    },
                )
            )
    return out


def run_in_subprocess(spec):
    with tempfile.TemporaryDirectory() as workdir:
        path = Path(workdir) / "spec.json"
        path.write_text(json.dumps(spec), encoding="utf-8")
        # The command is this script on a spec it wrote itself.
        done = subprocess.run(  # noqa: S603
            [sys.executable, __file__, "run", "--spec", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    return json.loads(done.stdout)


def summarize(runs):
    """One condition's repeats: accuracy and losses from the first, cost as medians."""
    first = runs[0]
    keep = ("scores", "refused_tracks", "shed_detections")
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
    summary = {
        k: v for k, v in first.items() if k not in ("loop_seconds", "ms_per_detection", "cost_fit")
    }
    summary["repeats"] = len(runs)
    summary["deterministic"] = deterministic
    summary["ms_per_detection"] = statistics.median(r["ms_per_detection"] for r in runs)
    summary["loop_seconds"] = statistics.median(r["loop_seconds"] for r in runs)
    for n in summary["sites"]:
        values = [
            r["sites"][n]["median_ms_per_event"]
            for r in runs
            if r["sites"][n]["median_ms_per_event"] is not None
        ]
        summary["sites"][n]["median_ms_per_event"] = statistics.median(values) if values else None
    fits = [r["cost_fit"] for r in runs if r["cost_fit"]]
    summary["cost_fit"] = (
        {k: statistics.median(f[k] for f in fits) for k in fits[0]} if fits else None
    )
    return summary


def _git(path, *args):
    command = ["git", "-C", str(path), *args]
    # git on the checkouts this run read from.
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()  # noqa: S603


def _commit(path):
    """HEAD of a checkout, marked +dirty when tracked files differ from it."""
    dirty = _git(path, "status", "--porcelain", "--untracked-files=no")
    return _git(path, "rev-parse", "HEAD") + ("+dirty" if dirty else "")


def _machine():
    model = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    return {
        "cpu": model,
        "cpus": os.cpu_count(),
        "python": platform.python_version(),
        "system": f"{platform.system()} {platform.machine()}",
    }


def print_results(results):
    for experiment in sorted({r["experiment"] for r in results}):
        print(f"\n== {experiment}")
        print(
            f"{'condition':34s} {'site':10s} {'tracks':>6s} {'false':>5s} {'err m':>6s} "
            f"{'events':>6s} {'drop':>5s} {'refused':>7s} {'peak':>5s} {'ms/ev':>6s}  tracks per swarm"
        )
        for r in (r for r in results if r["experiment"] == experiment):
            for n, s in r["sites"].items():
                sc = r["scores"][n]
                split = " ".join(f"{w}={c}" for w, c in sc["tracks_per_swarm"].items())
                ms = (
                    f"{s['median_ms_per_event']:.2f}"
                    if s["median_ms_per_event"] is not None
                    else "-"
                )
                err = sc["median_error_m"] if sc["median_error_m"] is not None else "-"
                print(
                    f"{r['label']:34s} {n:10s} {sc['tracks']:6d} {sc['false_tracks']:5d} {err:>6} "
                    f"{s['events']:6d} {s['dropped']:5d} {s['refused_tracks']:7d} "
                    f"{s['peak_live_tracks']:5d} {ms:>6}  {split}"
                )
            fit = r["cost_fit"]
            fit_text = (
                f"fit {fit['intercept_ms']:.2f} ms + {fit['ms_per_own_track']:.3f}/own + "
                f"{fit['ms_per_other_track']:.3f}/other track"
                if fit
                else "fit -"
            )
            print(
                f"{'':34s} ms/detection {r['ms_per_detection']:.2f}, shed {r['shed_detections']}, "
                f"{fit_text}, repeats {r['repeats']}, deterministic {r['deterministic']}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("run", help="fuse one condition in this process and print its JSON")
    one.add_argument("--spec", type=Path, required=True)
    exp = sub.add_parser(
        "experiments", help="run the experiments, each condition in its own process"
    )
    exp.add_argument("--scenario-foundry", type=Path, default=fb.DEFAULT_SCENARIO_FOUNDRY)
    exp.add_argument("--sensors-dir", type=Path, default=REPO / "config" / "sensors")
    exp.add_argument("--only", nargs="+", default=["E1", "E2", "E3", "E4"])
    exp.add_argument("--repeats", type=int, default=3)
    exp.add_argument("--lags", nargs="+", type=float, default=[1, 2, 3, 5])
    exp.add_argument("--clutter-rates", nargs="+", type=float, default=[0.5, 1, 2, 5])
    exp.add_argument("--seed", type=int, default=42, help="clutter seed")
    exp.add_argument("--save", type=Path, help="write all results as JSON")
    args = parser.parse_args()

    if args.command == "run":
        print(json.dumps(run_condition(json.loads(args.spec.read_text()))))
        return

    results = []
    for experiment, label, spec in conditions(args):
        print(f"{experiment} {label} ...", file=sys.stderr)
        runs = [run_in_subprocess(spec) for _ in range(args.repeats)]
        results.append({"experiment": experiment, "label": label, **summarize(runs)})
    print_results(results)
    if args.save:
        record = {
            "context_foundry": _commit(REPO),
            "scenario_foundry": _commit(args.scenario_foundry),
            "machine": _machine(),
            "clutter_seed": args.seed,
            "results": results,
        }
        args.save.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"\nsaved {args.save}", file=sys.stderr)


if __name__ == "__main__":
    main()
