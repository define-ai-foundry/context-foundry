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


class _ResettableEventsSource:
    """Fake replay source: yields `events` on the first iter_events() after each
    reset(), nothing after. Records reset() invocations for loop assertions."""

    def __init__(self, events):
        self._events = events
        self.calls = 0
        self.resets = 0

    def iter_events(self):
        self.calls += 1
        if self.calls == 1:
            yield from self._events

    def reset(self):
        self.resets += 1
        self.calls = 0


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


def test_no_sink_configured_errors_with_guidance(monkeypatch, caplog, tmp_path):
    replay_path = tmp_path / "scenario.json"
    replay_path.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(cli, "JsonSapientSource", lambda path: _EventsOnceSource([]))
    monkeypatch.setattr(cli, "SapientAsynchronousTracker", lambda: _FakeTracker([[]]))
    monkeypatch.setattr(cli, "TacticalContextAugmentor", _FakeAugmentor)

    monkeypatch.setattr(
        "sys.argv",
        ["fusion", "--config", "sensors.json", "--replay-file", str(replay_path)],
    )

    with caplog.at_level("ERROR"):
        result = cli.fusion_main()

    assert result is None
    # The error names both sinks so the operator knows how to proceed.
    assert any(
        "No output sink configured" in r.message
        and "--tak-tls-host" in r.message
        and "--log-to-file" in r.message
        for r in caplog.records
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
        "sys.argv",
        ["fusion", "--config", "sensors.json", "--enable-sapient", "--enable-cot", "--log-to-file"],
    )

    with pytest.raises(_Stop):
        cli.fusion_main()

    assert created == {"sapient_port": 5000, "cot_port": 6969}


def test_tak_tls_host_constructs_sink_streams_payload_and_closes(monkeypatch, caplog, tmp_path):
    replay_path = tmp_path / "scenario.json"
    replay_path.write_text("[]", encoding="utf-8")

    track = _fake_track("track-tls0", T0)
    monkeypatch.setattr(
        cli, "JsonSapientSource", lambda path: _EventsOnceSource([(T0, [object()])])
    )
    monkeypatch.setattr(cli, "SapientAsynchronousTracker", lambda: _FakeTracker([[track]]))
    monkeypatch.setattr(cli, "TacticalContextAugmentor", _FakeAugmentor)

    created = {}

    class _FakeTakTlsSink:
        def __init__(self, host, port, cert=None, key=None, ca=None):
            created.update(host=host, port=port, cert=cert, key=key, ca=ca, obj=self)
            self.sent = []
            self.closed = False

        def send(self, payload):
            self.sent.append(payload)

        def close(self):
            self.closed = True

    monkeypatch.setattr(cli, "TakTlsSink", _FakeTakTlsSink)

    monkeypatch.setattr(
        "sys.argv",
        [
            "fusion",
            "--config",
            "sensors.json",
            "--replay-file",
            str(replay_path),
            "--tak-tls-host",
            "takhost",
            "--tak-tls-cert",
            "client.pem",
            "--tak-tls-key",
            "client.key",
            "--tak-tls-ca",
            "ca.pem",
        ],
    )

    with caplog.at_level("INFO"):
        cli.fusion_main()

    assert created["host"] == "takhost"
    assert created["port"] == 8089
    assert (created["cert"], created["key"], created["ca"]) == (
        "client.pem",
        "client.key",
        "ca.pem",
    )
    sink = created["obj"]
    assert any("TRK-track-tls0" in payload for payload in sink.sent)
    assert sink.closed is True
    assert any(
        "Streaming CoT to TAK Server takhost:8089 over TLS" in r.message for r in caplog.records
    )


def test_tak_ws_host_constructs_sink_streams_payload_and_closes(monkeypatch, caplog, tmp_path):
    replay_path = tmp_path / "scenario.json"
    replay_path.write_text("[]", encoding="utf-8")

    track = _fake_track("track-ws00", T0)
    monkeypatch.setattr(
        cli, "JsonSapientSource", lambda path: _EventsOnceSource([(T0, [object()])])
    )
    monkeypatch.setattr(cli, "SapientAsynchronousTracker", lambda: _FakeTracker([[track]]))
    monkeypatch.setattr(cli, "TacticalContextAugmentor", _FakeAugmentor)

    created = {}

    class _FakeTakWsSink:
        def __init__(
            self,
            host,
            port,
            *,
            token_url=None,
            client_id=None,
            client_secret=None,
            static_token=None,
            verify_tls=False,
        ):
            created.update(
                host=host,
                port=port,
                token_url=token_url,
                client_id=client_id,
                client_secret=client_secret,
                static_token=static_token,
                verify_tls=verify_tls,
                obj=self,
            )
            self.sent = []
            self.closed = False

        def send(self, payload):
            self.sent.append(payload)

        def close(self):
            self.closed = True

    monkeypatch.setattr(cli, "TakWsSink", _FakeTakWsSink)

    monkeypatch.setattr(
        "sys.argv",
        [
            "fusion",
            "--config",
            "sensors.json",
            "--replay-file",
            str(replay_path),
            "--tak-ws-host",
            "takhost",
            "--keycloak-token-url",
            "https://iam/token",
            "--oidc-client-id",
            "cf-a",
            "--oidc-client-secret",
            "secret",
        ],
    )

    with caplog.at_level("INFO"):
        cli.fusion_main()

    assert created["host"] == "takhost"
    assert created["port"] == 8446
    assert created["token_url"] == "https://iam/token"
    assert created["client_id"] == "cf-a"
    assert created["client_secret"] == "secret"
    assert created["verify_tls"] is False
    sink = created["obj"]
    assert any("TRK-track-ws00" in payload for payload in sink.sent)
    assert sink.closed is True
    assert any("Streaming CoT to TAK Server takhost:8446" in r.message for r in caplog.records)


def _run_looping_replay(monkeypatch, tmp_path, extra_argv):
    """Drive fusion_main in --loop mode with a finite replay + file sink.

    Returns (source, sleep_seconds, tracker_count). cli.time.sleep is patched to
    record its argument and raise _Stop so the second (drained) pass escapes.
    """
    monkeypatch.chdir(tmp_path)
    replay_path = tmp_path / "scenario.json"
    replay_path.write_text("[]", encoding="utf-8")

    source = _ResettableEventsSource([(T0, [object()])])
    track = _fake_track("loop-track-01", T0)
    tracker_count = {"n": 0}

    def _make_tracker():
        tracker_count["n"] += 1
        return _FakeTracker([[track]])

    monkeypatch.setattr(cli, "JsonSapientSource", lambda path: source)
    monkeypatch.setattr(cli, "SapientAsynchronousTracker", _make_tracker)
    monkeypatch.setattr(cli, "TacticalContextAugmentor", _FakeAugmentor)

    recorded = {}

    def _sleep_stop(seconds):
        recorded["sleep"] = seconds
        raise _Stop

    monkeypatch.setattr(cli.time, "sleep", _sleep_stop)

    argv = [
        "fusion",
        "--config",
        "sensors.json",
        "--replay-file",
        str(replay_path),
        "--log-to-file",
    ]
    argv += extra_argv
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(_Stop):
        cli.fusion_main()

    return source, recorded["sleep"], tracker_count["n"]


def test_loop_reruns_replay_resets_source_and_rebuilds_tracker(monkeypatch, tmp_path):
    source, sleep_seconds, tracker_count = _run_looping_replay(monkeypatch, tmp_path, ["--loop"])

    assert sleep_seconds == 60.0  # default loop delay
    assert source.resets == 1  # source rewound before the next pass
    assert tracker_count >= 2  # fresh tracker per loop


def test_loop_continues_to_a_second_iteration(monkeypatch, tmp_path):
    """The loop `continue`s past sleep back into a fresh pass; escape on the
    second sleep so a full re-iteration (reset -> re-yield -> drain) is exercised."""
    monkeypatch.chdir(tmp_path)
    replay_path = tmp_path / "scenario.json"
    replay_path.write_text("[]", encoding="utf-8")

    source = _ResettableEventsSource([(T0, [object()])])
    track = _fake_track("loop-track-02", T0)
    monkeypatch.setattr(cli, "JsonSapientSource", lambda path: source)
    monkeypatch.setattr(cli, "SapientAsynchronousTracker", lambda: _FakeTracker([[track]]))
    monkeypatch.setattr(cli, "TacticalContextAugmentor", _FakeAugmentor)

    sleeps = []

    def _sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) >= 2:
            raise _Stop

    monkeypatch.setattr(cli.time, "sleep", _sleep)
    monkeypatch.setattr(
        "sys.argv",
        [
            "fusion",
            "--config",
            "sensors.json",
            "--replay-file",
            str(replay_path),
            "--log-to-file",
            "--loop",
        ],
    )

    with pytest.raises(_Stop):
        cli.fusion_main()

    assert len(sleeps) == 2  # looped once past `continue`, escaped on the second drain
    assert source.resets == 2


def test_loop_delay_is_honored(monkeypatch, tmp_path):
    _, sleep_seconds, _ = _run_looping_replay(
        monkeypatch, tmp_path, ["--loop", "--loop-delay", "5"]
    )
    assert sleep_seconds == 5.0


def test_negative_loop_delay_clamps_to_zero(monkeypatch, tmp_path):
    _, sleep_seconds, _ = _run_looping_replay(
        monkeypatch, tmp_path, ["--loop", "--loop-delay", "-3"]
    )
    assert sleep_seconds == 0.0


def test_loop_without_replay_file_warns_and_is_noop(monkeypatch, caplog):
    monkeypatch.setattr(
        cli, "NetworkSapientStream", lambda port: _EventsThenStopSource([(T0, [object()])])
    )
    monkeypatch.setattr(cli, "SapientAsynchronousTracker", lambda: _FakeTracker([[]]))
    monkeypatch.setattr(cli, "TacticalContextAugmentor", _FakeAugmentor)
    monkeypatch.setattr(
        "sys.argv",
        ["fusion", "--config", "sensors.json", "--enable-sapient", "--loop", "--log-to-file"],
    )

    with caplog.at_level("WARNING"), pytest.raises(_Stop):
        cli.fusion_main()

    assert any("--loop has no effect without --replay-file" in r.message for r in caplog.records)


def test_no_sink_message_names_the_ws_sink(monkeypatch, caplog, tmp_path):
    replay_path = tmp_path / "scenario.json"
    replay_path.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(cli, "JsonSapientSource", lambda path: _EventsOnceSource([]))
    monkeypatch.setattr(cli, "SapientAsynchronousTracker", lambda: _FakeTracker([[]]))
    monkeypatch.setattr(cli, "TacticalContextAugmentor", _FakeAugmentor)
    monkeypatch.setattr(
        "sys.argv", ["fusion", "--config", "sensors.json", "--replay-file", str(replay_path)]
    )

    with caplog.at_level("ERROR"):
        cli.fusion_main()

    assert any("--tak-ws-host" in r.message for r in caplog.records)


def test_file_and_tak_tls_sinks_are_both_active(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    replay_path = tmp_path / "scenario.json"
    replay_path.write_text("[]", encoding="utf-8")

    track = _fake_track("track-both0", T0)
    monkeypatch.setattr(
        cli, "JsonSapientSource", lambda path: _EventsOnceSource([(T0, [object()])])
    )
    monkeypatch.setattr(cli, "SapientAsynchronousTracker", lambda: _FakeTracker([[track]]))
    monkeypatch.setattr(cli, "TacticalContextAugmentor", _FakeAugmentor)

    tls_sent = []

    class _FakeTakTlsSink:
        def __init__(self, host, port, cert=None, key=None, ca=None):
            pass

        def send(self, payload):
            tls_sent.append(payload)

        def close(self):
            pass

    monkeypatch.setattr(cli, "TakTlsSink", _FakeTakTlsSink)

    monkeypatch.setattr(
        "sys.argv",
        [
            "fusion",
            "--config",
            "sensors.json",
            "--replay-file",
            str(replay_path),
            "--log-to-file",
            "--tak-tls-host",
            "takhost",
        ],
    )

    cli.fusion_main()

    # One run, both sinks: the file has the event AND the TLS sink received it.
    file_out = (tmp_path / "fused_tracks_debug.xml").read_text(encoding="utf-8")
    assert "TRK-track-both0" in file_out
    assert any("TRK-track-both0" in payload for payload in tls_sent)
