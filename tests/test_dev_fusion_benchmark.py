"""Tests for the scoring in dev/fusion_benchmark.py.

Dev tooling, but other benchmarks score with it, so the split between running the
engine and scoring its tracks has to keep giving the numbers the benchmark prints.
"""

import importlib.util
from datetime import datetime, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
START = datetime(2026, 11, 15, 2, 45)


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location(
        "fusion_benchmark", REPO_ROOT / "dev/fusion_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _swarm(bench, wave_id="W1", count=4):
    # Due north from 62.0N 30.0E at 360 km/h: 100 m/s.
    cfg = {
        "count": count,
        "launch_delay_sec": 0,
        "speed_kmh": 360,
        "wkt_linestring": "LINESTRING (30.0 62.0, 30.0 62.5)",
    }
    return bench.Swarm(wave_id, cfg, START)


def _following(swarm, offset_deg=0.0, seconds=(10, 20, 30), count=4):
    snaps = []
    for s in seconds:
        when = START + timedelta(seconds=s)
        lat, lon = swarm.position(when)
        snaps.append((when, lat + offset_deg, lon, count))
    return snaps


def test_score_tracks_counts_followers_and_false_tracks(bench):
    swarm = _swarm(bench)
    tracks = {
        "t-follow": _following(swarm),
        # 0.1 degree of latitude is about 11 km: follows no swarm.
        "t-false": _following(swarm, offset_deg=0.1, count=1),
    }

    result = bench.score_tracks("site", tracks, [swarm])

    assert result["scenario"] == "site"
    assert result["swarms"] == 1
    assert result["objects"] == 4
    assert result["tracks"] == 2
    assert result["false_tracks"] == 1
    assert result["tracks_per_swarm"] == {"W1": 1}
    assert result["median_error_m"] == 0


def test_score_tracks_with_no_tracks_has_no_error(bench):
    result = bench.score_tracks("site", {}, [_swarm(bench)])

    assert result["tracks"] == 0
    assert result["tracks_per_swarm"] == {"W1": 0}
    assert result["median_error_m"] is None


def test_match_track_gives_the_followed_swarm_and_median_distance(bench):
    swarm = _swarm(bench)

    wave_id, median = bench.match_track(_following(swarm), [swarm])

    assert wave_id == "W1"
    assert median == pytest.approx(0.0, abs=1e-6)


def test_match_track_before_any_launch_matches_nothing(bench):
    swarm = _swarm(bench)
    early = [(START - timedelta(seconds=5), 62.0, 30.0, 1)]

    assert bench.match_track(early, [swarm]) == (None, None)
