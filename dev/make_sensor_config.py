#!/usr/bin/env python3
# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# Builds a fusion sensor manifest (config/sensors/<name>.json) from a
# scenario-foundry scenario file, so a simulated scenario can be fused without a
# hand-written manifest. Fusion reads a sensor's position only, and the scenario
# carries no sensor altitude, so each sensor's altitude is its ground elevation
# from the scenario's terrain anchors -- the same inverse-distance-squared
# weighting scenario-foundry falls back to when no terrain tile is cached -- plus
# any height the scenario gives the sensor itself. Dev tooling only.

import argparse
import json
import math
from pathlib import Path

EARTH_RADIUS_M = 6371000.0
# scenario-foundry's elevation when a scenario has no terrain anchors.
DEFAULT_TERRAIN_ELEVATION_M = 80.0


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def ground_elevation(lat, lon, anchors):
    """Ground elevation at a point, weighted by inverse squared distance to the anchors."""
    if not anchors:
        return DEFAULT_TERRAIN_ELEVATION_M
    total_weight, weighted = 0.0, 0.0
    for anchor in anchors:
        weight = 1.0 / max(haversine_m(lat, lon, anchor["lat"], anchor["lon"]), 1.0) ** 2
        total_weight += weight
        weighted += anchor["elevation_msl"] * weight
    return weighted / total_weight


# Each sensor type's geometric_error, in the shape a Registration message's
# DetectionDefinition.geometric_error carries it, set to the position noise
# scenario-foundry adds to that type's detections (compute_noisy_position and the
# acoustic branch of DetectionReportBuilder). It is what a sensor registering
# honestly would declare, and what lets fusion weigh each sensor's hits by what
# they are actually worth instead of by one figure for every sensor.
SIMULATOR_GEOMETRIC_ERROR = {
    "RADAR_STRATEGIC": {
        "variation_type": "linear_with_range",
        "base_m": 5.0,
        "at_max_range_m": 35.0,
    },
    "RADAR_TACTICAL": {
        "variation_type": "linear_with_range",
        "base_m": 5.0,
        "at_max_range_m": 35.0,
    },
    "THERMAL_CAM": {
        "variation_type": "quadratic_with_range",
        "base_m": 2.0,
        "at_max_range_m": 17.0,
    },
    "VISUAL_CAM": {"variation_type": "quadratic_with_range", "base_m": 2.0, "at_max_range_m": 17.0},
    "MICRO_DOPPLER": {"variation_type": "constant", "base_m": 1.5},
    # Range is reported exactly and bearing to 3.5 degrees; with no elevation in the
    # report the target's height is unknown, which a wide vertical sigma says.
    "ACOUSTIC": {
        "variation_type": "bearing",
        "bearing_deg": 3.5,
        "range_sigma_m": 1.0,
        "vertical_m": 1000.0,
    },
}


def primary_anchor(sensors):
    """The first strategic radar, else the first sensor: the network's ENU origin."""
    return next((s["id"] for s in sensors if s["type"] == "RADAR_STRATEGIC"), sensors[0]["id"])


def build_manifest(scenario, geometric_error=False, tracking_type=None):
    anchors = scenario.get("terrain_elevation_anchors", [])
    sensors = [
        {
            "id": s["id"],
            "type": s["type"],
            "lat": s["lat"],
            "lon": s["lon"],
            "alt": round(ground_elevation(s["lat"], s["lon"], anchors) + s.get("alt", 0.0), 1),
            "range_m": s["range_m"],
            "update_rate_sec": s["update_rate_sec"],
        }
        for s in scenario["sensor_network"]
    ]
    if geometric_error:
        for sensor in sensors:
            sensor["geometric_error"] = {
                "type": "standard_deviation",
                "units": "m",
                **SIMULATOR_GEOMETRIC_ERROR[sensor["type"]],
            }
    if tracking_type:
        for sensor in sensors:
            sensor["tracking_type"] = tracking_type
    return {
        "version": "1.1",
        "name": scenario["scenario_meta"]["name"],
        "description": "Generated from a scenario-foundry scenario by dev/make_sensor_config.py; "
        "altitudes are ground elevation from the scenario's terrain anchors.",
        "primary_anchor_node": primary_anchor(sensors),
        "coordinate_system": "WGS84",
        "sensors": sensors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path, help="scenario-foundry scenario JSON")
    parser.add_argument(
        "--output",
        type=Path,
        help="manifest path (default: config/sensors/<scenario name>.json)",
    )
    parser.add_argument(
        "--geometric-error",
        action="store_true",
        help="declare each sensor's geometric_error, set to the noise scenario-foundry adds",
    )
    parser.add_argument(
        "--tracking-type",
        choices=[
            "TRACKING_TYPE_NONE",
            "TRACKING_TYPE_TRACKLET",
            "TRACKING_TYPE_TRACK",
            "TRACKING_TYPE_TRACK_WITH_RE_ID",
        ],
        help="declare every sensor's Registration tracking_type. scenario-foundry keeps one "
        "objectId per swarm per sensor for the whole run, which is TRACKING_TYPE_TRACK",
    )
    args = parser.parse_args()

    scenario = json.loads(args.scenario.read_text(encoding="utf-8"))
    output = args.output or Path("config/sensors") / args.scenario.name
    manifest = build_manifest(
        scenario, geometric_error=args.geometric_error, tracking_type=args.tracking_type
    )
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
