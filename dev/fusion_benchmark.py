#!/usr/bin/env python3
# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# Scores the fusion engine against scenario-foundry's ground truth: how close the
# number of tracks it produces comes to the number of swarms actually flying.
# For each scenario it replays the simulator's SAPIENT log through the engine at
# full speed under the scenario's own clock, reads the fused CoT back, and
# compares every track snapshot with where each swarm truly was at that instant,
# computed from the tactical scenario's flight paths. Dev tooling only.

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SCENARIO_FOUNDRY = REPO.parent / "scenario-foundry"
DEFAULT_SCENARIOS = ["joensuu", "alakurtti", "hamina_kotka", "porvoo_oil_refinery"]

EARTH_RADIUS_M = 6371000.0
# A track whose median distance from its nearest swarm exceeds this follows no
# swarm at all, and is counted as false.
FALSE_TRACK_DISTANCE_M = 2000.0

EVENT_RE = re.compile(
    r'<event[^>]*\buid="([^"]+)"[^>]*\btime="([^"]+)"[^>]*>'
    r'<point lat="([^"]+)" lon="([^"]+)".*?callsign="([^"]*)"'
)
SWARM_RE = re.compile(r"SWM\((\d+)\)")


def parse_time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def parse_wkt(wkt):
    inner = re.search(r"LINESTRING\s*\((.*)\)", wkt, re.IGNORECASE).group(1)
    return [tuple(float(v) for v in pair.split())[::-1] for pair in inner.split(",")]


class Swarm:
    """One threat wave's true path, positioned the way scenario-foundry positions it."""

    def __init__(self, wave_id, cfg, start):
        self.wave_id = wave_id
        self.count = cfg["count"]
        self.launch = start + timedelta(seconds=cfg["launch_delay_sec"])
        self.mps = cfg["speed_kmh"] * 1000 / 3600
        self.points = parse_wkt(cfg["wkt_linestring"])
        self.legs = [
            haversine_m(*self.points[i], *self.points[i + 1]) for i in range(len(self.points) - 1)
        ]

    def position(self, when):
        """(lat, lon) at `when`, or None before launch and after impact."""
        if when < self.launch:
            return None
        travelled = self.mps * (when - self.launch).total_seconds()
        for (lat1, lon1), (lat2, lon2), leg in zip(
            self.points, self.points[1:], self.legs, strict=False
        ):
            if travelled <= leg:
                ratio = travelled / leg if leg else 0.0
                return lat1 + ratio * (lat2 - lat1), lon1 + ratio * (lon2 - lon1)
            travelled -= leg
        return None


def run_fusion(messages, sensors, workdir):
    """Fuse a SAPIENT log, returning the CoT the engine wrote."""
    cmd = [
        str(REPO / ".venv" / "bin" / "context-foundry-fusion"),
        "--replay-file",
        str(messages),
        "--config",
        str(sensors),
        "--log-to-file",
        "--realtime-factor",
        "0",
        "--use-scenario-timestamps",
    ]
    # The command is this repository's own engine on paths the caller chose.
    subprocess.run(cmd, cwd=workdir, check=True, capture_output=True, text=True)  # noqa: S603
    # The file sink appends, so each run gets a fresh working directory.
    return (Path(workdir) / "fused_tracks_debug.xml").read_text(encoding="utf-8")


def read_tracks(cot):
    """Track id -> time-ordered snapshots of (time, lat, lon, swarm count)."""
    tracks = defaultdict(list)
    for uid, time, lat, lon, callsign in EVENT_RE.findall(cot):
        swarm = SWARM_RE.search(callsign)
        tracks[uid].append(
            (parse_time(time), float(lat), float(lon), int(swarm.group(1)) if swarm else 1)
        )
    for snapshots in tracks.values():
        snapshots.sort()
    return tracks


def nearest_swarm(swarms, when, lat, lon):
    best = None
    for swarm in swarms:
        pos = swarm.position(when)
        if pos is not None:
            dist = haversine_m(lat, lon, *pos)
            if best is None or dist < best[1]:
                best = (swarm.wave_id, dist)
    return best


def peak_concurrent(tracks):
    """Most tracks alive at once, a track being alive from its first to its last snapshot."""
    edges = []
    for snapshots in tracks.values():
        edges += [(snapshots[0][0], 1), (snapshots[-1][0], -1)]
    alive = peak = 0
    for _, step in sorted(edges, key=lambda e: (e[0], -e[1])):
        alive += step
        peak = max(peak, alive)
    return peak


def peak_estimated_objects(tracks):
    """Most objects the picture claims at once: the alive tracks' latest swarm counts, summed."""
    times = sorted({s[0] for snapshots in tracks.values() for s in snapshots})
    peak = 0
    for when in times[:: max(1, len(times) // 400)]:
        total = 0
        for snapshots in tracks.values():
            if snapshots[0][0] <= when <= snapshots[-1][0]:
                total += next(s[3] for s in reversed(snapshots) if s[0] <= when)
        peak = max(peak, total)
    return peak


def score(name, scenario_foundry, sensors_dir):
    tactical = json.loads(
        (scenario_foundry / "data/tactical_scenarios" / f"{name}_tactical.json").read_text()
    )
    messages = scenario_foundry / "data/generated_output" / f"{name}_messages.json"
    start = parse_time(tactical["scenario_meta"]["start_time_iso"])
    swarms = [Swarm(wid, cfg, start) for wid, cfg in tactical["threat_profiles"].items()]

    with tempfile.TemporaryDirectory() as workdir:
        tracks = read_tracks(run_fusion(messages, sensors_dir / f"{name}.json", workdir))

    per_swarm = Counter()
    false_tracks = 0
    errors = []
    for snapshots in tracks.values():
        matches = [nearest_swarm(swarms, t, lat, lon) for t, lat, lon, _ in snapshots]
        matches = [m for m in matches if m is not None]
        if not matches:
            false_tracks += 1
            continue
        dists = sorted(d for _, d in matches)
        median = dists[len(dists) // 2]
        if median > FALSE_TRACK_DISTANCE_M:
            false_tracks += 1
            continue
        per_swarm[Counter(w for w, _ in matches).most_common(1)[0][0]] += 1
        errors.append(median)

    return {
        "scenario": name,
        "swarms": len(swarms),
        "objects": sum(s.count for s in swarms),
        "tracks": len(tracks),
        "peak_concurrent": peak_concurrent(tracks),
        "false_tracks": false_tracks,
        "tracks_per_swarm": {s.wave_id: per_swarm.get(s.wave_id, 0) for s in swarms},
        "peak_estimated_objects": peak_estimated_objects(tracks),
        "median_error_m": round(sorted(errors)[len(errors) // 2]) if errors else None,
    }


def regenerate(name, scenario_foundry, seed):
    python = scenario_foundry / ".venv" / "bin" / "python"
    cmd = [str(python), "-m", "src.scenario_foundry.generate_scenario", "--scenario", name]
    cmd += ["--seed", str(seed)]
    subprocess.run(cmd, cwd=scenario_foundry, check=True, capture_output=True)  # noqa: S603


def print_table(results):
    print(
        f"{'scenario':22s} {'swarms':>6s} {'tracks':>6s} {'peak':>5s} {'false':>5s} "
        f"{'objects':>7s} {'est.peak':>8s} {'err m':>6s}  tracks per swarm"
    )
    for r in results:
        split = " ".join(f"{w}={n}" for w, n in r["tracks_per_swarm"].items())
        print(
            f"{r['scenario']:22s} {r['swarms']:6d} {r['tracks']:6d} {r['peak_concurrent']:5d} "
            f"{r['false_tracks']:5d} {r['objects']:7d} {r['peak_estimated_objects']:8d} "
            f"{r['median_error_m'] if r['median_error_m'] is not None else '-':>6}  {split}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Score fusion's track count against scenario-foundry ground truth."
    )
    parser.add_argument("scenarios", nargs="*", default=DEFAULT_SCENARIOS)
    parser.add_argument("--scenario-foundry", type=Path, default=DEFAULT_SCENARIO_FOUNDRY)
    parser.add_argument("--sensors-dir", type=Path, default=REPO / "config" / "sensors")
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="re-run scenario-foundry for each scenario first (with --seed)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=Path, help="write the results as JSON for later comparison")
    args = parser.parse_args()

    results = []
    for name in args.scenarios:
        if args.regenerate:
            regenerate(name, args.scenario_foundry, args.seed)
        print(f"scoring {name} ...", file=sys.stderr)
        results.append(score(name, args.scenario_foundry, args.sensors_dir))

    print_table(results)
    if args.save:
        args.save.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\nsaved {args.save}", file=sys.stderr)


if __name__ == "__main__":
    main()
