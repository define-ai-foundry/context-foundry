"""Tests for context_foundry.cli.

fusion_main() wires up real UDP-socket-backed sources/sinks and runs an
unconditional `while True:` loop, so every test replaces the source classes,
the tracker, and the augmentor with lightweight fakes (monkeypatched into the
`context_foundry.cli` module namespace, exactly where fusion_main looks them
up) and never touches a real socket. Sources that don't naturally terminate
(the --enable-sapient/--enable-cot live-stream case) are given a fake
iter_events() that raises a sentinel exception on its second call, which
deterministically escapes fusion_main's infinite loop.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from context_foundry import cli
from context_foundry.fusion.schemas import TacticalTrack

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _Stop(BaseException):
    """Sentinel used to escape fusion_main's infinite while-loop from a test."""


class _EventsOnceSource:
    """Fake source: yields `events` on the first iter_events() call, nothing after."""

    def __init__(self, events):
        self._events = events
        self.calls = 0

    def iter_events(self):
        self.calls += 1
        if self.calls == 1:
            yield from self._events


class _EventsThenStopSource:
    """Fake source: yields `events` once, then raises _Stop on the next call."""

    def __init__(self, events):
        self._events = events
        self.calls = 0

    def iter_events(self):
        self.calls += 1
        if self.calls == 1:
            yield from self._events
        else:
            raise _Stop


class _FakeTracker:
    def __init__(self, tracks_by_call):
        self._tracks_by_call = tracks_by_call
        self.calls = 0

    def process_async_event(self, timestamp, detection_group):
        tracks = self._tracks_by_call[min(self.calls, len(self._tracks_by_call) - 1)]
        self.calls += 1
        return tracks


class _FakeAugmentor:
    def extract_tactical_track(self, track):
        return track.tactical


def _fake_track(track_id, timestamp, tactical_kwargs=None):
    kwargs = {
        "track_id": track_id,
        "timestamp": timestamp,
        "latitude": 62.9,
        "longitude": 29.8,
        "altitude": 100.0,
        "speed_mps": 5.0,
        "heading_deg": 90.0,
        "classification": "UAS",
        "swarm_count": 1,
        "threat_level": "suspect",
    }
    if tactical_kwargs:
        kwargs.update(tactical_kwargs)
    tactical = TacticalTrack(**kwargs)
    return SimpleNamespace(state=SimpleNamespace(timestamp=timestamp), tactical=tactical)


@pytest.fixture(autouse=True)
def _no_real_sensor_config_load(monkeypatch):
    """Every cli test controls the sensor registry explicitly via fakes; make
    config.load_sensor_network a no-op unless a test overrides it again."""
    monkeypatch.setattr(cli.config, "load_sensor_network", lambda **kwargs: None)


def test_missing_required_config_arg_exits(monkeypatch):
    monkeypatch.setattr("sys.argv", ["fusion", "--replay-file", "x.json"])
    with pytest.raises(SystemExit):
        cli.fusion_main()


def test_sensor_config_load_failure_returns_without_raising(monkeypatch, caplog):
    def _raise(**kwargs):
        raise RuntimeError("bad config")

    monkeypatch.setattr(cli.config, "load_sensor_network", _raise)
    monkeypatch.setattr(
        "sys.argv", ["fusion", "--config", "sensors.json", "--replay-file", "x.json"]
    )

    with caplog.at_level("ERROR"):
        result = cli.fusion_main()

    assert result is None
    assert any("Failed to load sensor config" in r.message for r in caplog.records)


def test_no_sources_enabled_returns_without_raising(monkeypatch, caplog):
    monkeypatch.setattr("sys.argv", ["fusion", "--config", "sensors.json"])

    with caplog.at_level("ERROR"):
        result = cli.fusion_main()

    assert result is None
    assert any("No sources enabled" in r.message for r in caplog.records)


def test_replay_file_log_to_file_broadcasts_only_matching_timestamp(monkeypatch, caplog, tmp_path):
    monkeypatch.chdir(tmp_path)
    replay_path = tmp_path / "scenario.json"
    replay_path.write_text("[]", encoding="utf-8")

    fresh = _fake_track("fresh-track-0001", T0)
    stale = _fake_track(
        "stale-track-0002", T0 - timedelta(seconds=100), {"threat_level": "hostile"}
    )

    monkeypatch.setattr(
        cli, "JsonSapientSource", lambda path: _EventsOnceSource([(T0, [object()])])
    )
    monkeypatch.setattr(cli, "SapientAsynchronousTracker", lambda: _FakeTracker([[fresh, stale]]))
    monkeypatch.setattr(cli, "TacticalContextAugmentor", _FakeAugmentor)

    monkeypatch.setattr(
        "sys.argv",
        ["fusion", "--config", "sensors.json", "--replay-file", str(replay_path), "--log-to-file"],
    )

    with caplog.at_level("INFO"):
        cli.fusion_main()

    assert any("Replay file processing complete" in r.message for r in caplog.records)
    assert any("Broadcast Update for Track 0001" in r.message for r in caplog.records)
    assert not any("Broadcast Update for Track 0002" in r.message for r in caplog.records)

    output = (tmp_path / "fused_tracks_debug.xml").read_text(encoding="utf-8")
    assert "TRK-fresh-track-0001" in output
    assert "TRK-stale-track-0002" not in output


def test_replay_file_udp_broadcast_success_then_network_error(monkeypatch, caplog, tmp_path):
    replay_path = tmp_path / "scenario.json"
    replay_path.write_text("[]", encoding="utf-8")

    track1 = _fake_track("track-aaaa", T0)
    track2 = _fake_track("track-bbbb", T0)

    monkeypatch.setattr(
        cli, "JsonSapientSource", lambda path: _EventsOnceSource([(T0, [object(), object()])])
    )
    monkeypatch.setattr(cli, "SapientAsynchronousTracker", lambda: _FakeTracker([[track1, track2]]))
    monkeypatch.setattr(cli, "TacticalContextAugmentor", _FakeAugmentor)

    call_state = {"n": 0}

    def sendto(payload, addr):
        call_state["n"] += 1
        if call_state["n"] == 2:
            raise OSError("network unreachable")

    fake_socket = SimpleNamespace(setsockopt=lambda *a, **kw: None, sendto=sendto)
    monkeypatch.setattr(cli.socket, "socket", lambda *a, **kw: fake_socket)

    monkeypatch.setattr(
        "sys.argv",
        [
            "fusion",
            "--config",
            "sensors.json",
            "--replay-file",
            str(replay_path),
            "--tak-ip",
            "239.9.9.9",
            "--tak-port",
            "7000",
        ],
    )

    with caplog.at_level("INFO"):
        cli.fusion_main()

    assert call_state["n"] == 2
    assert any("Network error" in r.message for r in caplog.records)
    assert any(
        "TAK Multicast Broadcast active on 239.9.9.9:7000" in r.message for r in caplog.records
    )


def test_enable_sapient_and_cot_sources_are_constructed_with_expected_ports(monkeypatch):
    created = {}

    def fake_sapient_stream(port):
        created["sapient_port"] = port
        return _EventsThenStopSource([(T0, [object()])])

    def fake_cot_stream(port):
        created["cot_port"] = port
        return _EventsOnceSource([])

    monkeypatch.setattr(cli, "NetworkSapientStream", fake_sapient_stream)
    monkeypatch.setattr(cli, "CotNetworkStream", fake_cot_stream)
    monkeypatch.setattr(cli, "SapientAsynchronousTracker", lambda: _FakeTracker([[]]))
    monkeypatch.setattr(cli, "TacticalContextAugmentor", _FakeAugmentor)
    monkeypatch.setattr(
        cli.socket, "socket", lambda *a, **kw: SimpleNamespace(setsockopt=lambda *a, **kw: None)
    )

    monkeypatch.setattr(
        "sys.argv", ["fusion", "--config", "sensors.json", "--enable-sapient", "--enable-cot"]
    )

    with pytest.raises(_Stop):
        cli.fusion_main()

    assert created == {"sapient_port": 5000, "cot_port": 6969}
