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
from context_foundry.fusion.sources.offset import OffsetReplaySource

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _Detection:
    """Fake Stone Soup Detection. Only .timestamp is touched by the code under
    test -- OffsetReplaySource rewrites it, so it must be assignable."""

    def __init__(self, timestamp=T0):
        self.timestamp = timestamp


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
        cli, "JsonSapientSource", lambda path: _EventsOnceSource([(T0, [_Detection()])])
    )
    monkeypatch.setattr(cli, "SapientAsynchronousTracker", lambda: _FakeTracker([[fresh, stale]]))
    monkeypatch.setattr(cli, "TacticalContextAugmentor", _FakeAugmentor)

    monkeypatch.setattr(
        "sys.argv",
        [
            "fusion",
            "--config",
            "sensors.json",
            "--replay-file",
            str(replay_path),
            "--use-scenario-timestamps",
            "--log-to-file",
        ],
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


def test_enable_sapient_and_cot_reads_both_sources_concurrently(monkeypatch, caplog):
    """Both live sources must be drained. Consuming them in sequence starved the
    CoT socket behind the SAPIENT source's blocking read, so it was never served.
    """
    created = {}
    sapient = _EventsOnceSource([(T0, [_Detection()])])
    cot = _EventsOnceSource([(T0, [_Detection()])])

    def fake_sapient_stream(port):
        created["sapient_port"] = port
        return sapient

    def fake_cot_stream(port):
        created["cot_port"] = port
        return cot

    monkeypatch.setattr(cli, "NetworkSapientStream", fake_sapient_stream)
    monkeypatch.setattr(cli, "CotNetworkStream", fake_cot_stream)
    monkeypatch.setattr(cli, "SapientAsynchronousTracker", lambda: _FakeTracker([[], []]))
    monkeypatch.setattr(cli, "TacticalContextAugmentor", _FakeAugmentor)

    monkeypatch.setattr(
        "sys.argv",
        ["fusion", "--config", "sensors.json", "--enable-sapient", "--enable-cot", "--log-to-file"],
    )

    # Both fakes run dry, so the loop exits instead of re-polling dead sources.
    with caplog.at_level("ERROR"):
        cli.fusion_main()

    assert created == {"sapient_port": 5000, "cot_port": 6969}
    assert sapient.calls >= 1
    assert cot.calls >= 1
    assert any("Every live source ended" in r.message for r in caplog.records)


def test_tak_tls_host_constructs_sink_streams_payload_and_closes(monkeypatch, caplog, tmp_path):
    replay_path = tmp_path / "scenario.json"
    replay_path.write_text("[]", encoding="utf-8")

    track = _fake_track("track-tls0", T0)
    monkeypatch.setattr(
        cli, "JsonSapientSource", lambda path: _EventsOnceSource([(T0, [_Detection()])])
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
            "--use-scenario-timestamps",
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
        cli, "JsonSapientSource", lambda path: _EventsOnceSource([(T0, [_Detection()])])
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
            "--use-scenario-timestamps",
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

    source = _ResettableEventsSource([(T0, [_Detection()])])
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

    source = _ResettableEventsSource([(T0, [_Detection()])])
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
        cli, "NetworkSapientStream", lambda port: _EventsThenStopSource([(T0, [_Detection()])])
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


class _FakeRealtimeReplaySource:
    """Fake wrapper standing in for the real RealtimeReplaySource: records the
    (inner, factor) it was constructed with and passes iter_events()/reset()
    straight through to the inner fake, so fusion_main still runs to
    completion when this replaces cli.RealtimeReplaySource."""

    def __init__(self, inner, factor):
        self.inner = inner
        self.factor = factor

    def iter_events(self):
        yield from self.inner.iter_events()

    def reset(self):
        self.inner.reset()


def _patch_realtime_wrapper(monkeypatch):
    """Patch cli.RealtimeReplaySource with the fake; returns the list of
    wrappers it constructs (empty = pacing was never wired up)."""
    built = []

    def _build(inner, factor):
        wrapper = _FakeRealtimeReplaySource(inner, factor)
        built.append(wrapper)
        return wrapper

    monkeypatch.setattr(cli, "RealtimeReplaySource", _build)
    return built


def test_default_wraps_replay_source_in_realtime_replay_source_at_factor_one(monkeypatch, tmp_path):
    replay_path = tmp_path / "scenario.json"
    replay_path.write_text("[]", encoding="utf-8")

    inner = _EventsOnceSource([(T0, [_Detection()])])
    monkeypatch.setattr(cli, "JsonSapientSource", lambda path: inner)
    built = _patch_realtime_wrapper(monkeypatch)
    monkeypatch.setattr(cli, "SapientAsynchronousTracker", lambda: _FakeTracker([[]]))
    monkeypatch.setattr(cli, "TacticalContextAugmentor", _FakeAugmentor)

    monkeypatch.setattr(
        "sys.argv",
        ["fusion", "--config", "sensors.json", "--replay-file", str(replay_path), "--log-to-file"],
    )

    cli.fusion_main()

    assert len(built) == 1
    assert built[0].factor == 1.0
    # Offset inside pacer: the pacer sleeps against the intervals between
    # events, which the offset leaves untouched. Reverse the two and the pacer
    # would be measuring against timestamps that no longer match the file.
    assert isinstance(built[0].inner, OffsetReplaySource)
    assert built[0].inner.inner is inner


def test_realtime_factor_flag_sets_factor_on_the_wrapper(monkeypatch, tmp_path):
    replay_path = tmp_path / "scenario.json"
    replay_path.write_text("[]", encoding="utf-8")

    inner = _EventsOnceSource([(T0, [_Detection()])])
    monkeypatch.setattr(cli, "JsonSapientSource", lambda path: inner)
    built = _patch_realtime_wrapper(monkeypatch)
    monkeypatch.setattr(cli, "SapientAsynchronousTracker", lambda: _FakeTracker([[]]))
    monkeypatch.setattr(cli, "TacticalContextAugmentor", _FakeAugmentor)

    monkeypatch.setattr(
        "sys.argv",
        [
            "fusion",
            "--config",
            "sensors.json",
            "--replay-file",
            str(replay_path),
            "--log-to-file",
            "--realtime-factor",
            "5",
        ],
    )

    cli.fusion_main()

    assert len(built) == 1
    assert built[0].factor == 5.0


def test_realtime_factor_zero_does_not_wrap_and_logs_unpaced(monkeypatch, caplog, tmp_path):
    replay_path = tmp_path / "scenario.json"
    replay_path.write_text("[]", encoding="utf-8")

    inner = _EventsOnceSource([(T0, [_Detection()])])
    monkeypatch.setattr(cli, "JsonSapientSource", lambda path: inner)
    built = _patch_realtime_wrapper(monkeypatch)
    monkeypatch.setattr(cli, "SapientAsynchronousTracker", lambda: _FakeTracker([[]]))
    monkeypatch.setattr(cli, "TacticalContextAugmentor", _FakeAugmentor)

    monkeypatch.setattr(
        "sys.argv",
        [
            "fusion",
            "--config",
            "sensors.json",
            "--replay-file",
            str(replay_path),
            "--log-to-file",
            "--realtime-factor",
            "0",
        ],
    )

    with caplog.at_level("INFO"):
        cli.fusion_main()

    assert built == []  # never constructed
    assert any("unpaced" in r.message and str(replay_path) in r.message for r in caplog.records)


def test_negative_realtime_factor_exits(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv", ["fusion", "--config", "sensors.json", "--realtime-factor", "-1"]
    )

    with pytest.raises(SystemExit):
        cli.fusion_main()

    assert "--realtime-factor must be >= 0" in capsys.readouterr().err


def test_realtime_factor_without_replay_file_warns(monkeypatch, caplog):
    monkeypatch.setattr(
        "sys.argv", ["fusion", "--config", "sensors.json", "--realtime-factor", "5"]
    )

    with caplog.at_level("WARNING"):
        result = cli.fusion_main()

    assert result is None  # falls through to "No sources enabled" and returns
    assert any("--realtime-factor has no effect" in r.message for r in caplog.records)


def test_file_and_tak_tls_sinks_are_both_active(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    replay_path = tmp_path / "scenario.json"
    replay_path.write_text("[]", encoding="utf-8")

    track = _fake_track("track-both0", T0)
    monkeypatch.setattr(
        cli, "JsonSapientSource", lambda path: _EventsOnceSource([(T0, [_Detection()])])
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
            "--use-scenario-timestamps",
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


class _EchoTracker:
    """Fake tracker that stamps its track with the timestamp it was handed, so
    the broadcast gate matches whatever the source actually emitted."""

    def __init__(self, track_id):
        self.track_id = track_id
        self.seen = []

    def process_async_event(self, timestamp, detection_group):
        self.seen.append(timestamp)
        return [_fake_track(self.track_id, timestamp)]


def test_default_offsets_replay_timestamps_and_still_broadcasts(monkeypatch, caplog, tmp_path):
    """End-to-end guard on the default path.

    The scenario is dated months from now; the emitted CoT must carry the
    present instead. This also covers the silent-failure mode: if the offset
    moved the event timestamp without moving the detections', the tracker's
    track would not match and nothing would be broadcast at all.
    """
    monkeypatch.chdir(tmp_path)
    replay_path = tmp_path / "scenario.json"
    replay_path.write_text("[]", encoding="utf-8")

    scenario_time = datetime(2026, 11, 15, 2, 45, tzinfo=timezone.utc)
    tracker = _EchoTracker("offset-track-0001")

    monkeypatch.setattr(
        cli,
        "JsonSapientSource",
        lambda path: _EventsOnceSource([(scenario_time, [_Detection(scenario_time)])]),
    )
    monkeypatch.setattr(cli, "SapientAsynchronousTracker", lambda: tracker)
    monkeypatch.setattr(cli, "TacticalContextAugmentor", _FakeAugmentor)

    monkeypatch.setattr(
        "sys.argv",
        ["fusion", "--config", "sensors.json", "--replay-file", str(replay_path), "--log-to-file"],
    )

    with caplog.at_level("INFO"):
        cli.fusion_main()

    assert any("Broadcast Update for Track 0001" in r.message for r in caplog.records)

    emitted = tracker.seen[0]
    assert emitted != scenario_time
    assert abs((emitted - datetime.now(timezone.utc)).total_seconds()) < 60

    # And the serialised CoT carries the shifted time, not the file's.
    output = (tmp_path / "fused_tracks_debug.xml").read_text(encoding="utf-8")
    assert "2026-11-15" not in output


def test_use_scenario_timestamps_keeps_the_file_clock_and_skips_the_wrapper(
    monkeypatch, caplog, tmp_path
):
    replay_path = tmp_path / "scenario.json"
    replay_path.write_text("[]", encoding="utf-8")

    scenario_time = datetime(2026, 11, 15, 2, 45, tzinfo=timezone.utc)
    tracker = _EchoTracker("scenario-track-0001")

    monkeypatch.setattr(
        cli,
        "JsonSapientSource",
        lambda path: _EventsOnceSource([(scenario_time, [_Detection(scenario_time)])]),
    )
    monkeypatch.setattr(cli, "SapientAsynchronousTracker", lambda: tracker)
    monkeypatch.setattr(cli, "TacticalContextAugmentor", _FakeAugmentor)
    built = _patch_realtime_wrapper(monkeypatch)

    monkeypatch.setattr(
        "sys.argv",
        [
            "fusion",
            "--config",
            "sensors.json",
            "--replay-file",
            str(replay_path),
            "--use-scenario-timestamps",
            "--log-to-file",
        ],
    )

    with caplog.at_level("INFO"):
        cli.fusion_main()

    assert tracker.seen == [scenario_time]
    assert not isinstance(built[0].inner, OffsetReplaySource)
    assert any("with the scenario's own timestamps" in r.message for r in caplog.records)


def test_use_scenario_timestamps_without_replay_file_warns(monkeypatch, caplog):
    monkeypatch.setattr(
        "sys.argv", ["fusion", "--config", "sensors.json", "--use-scenario-timestamps"]
    )

    with caplog.at_level("WARNING"):
        cli.fusion_main()

    assert any("--use-scenario-timestamps has no effect" in r.message for r in caplog.records)


def test_cot_stale_seconds_flag_reaches_the_serializer(monkeypatch, tmp_path):
    replay_path = tmp_path / "scenario.json"
    replay_path.write_text("[]", encoding="utf-8")

    built = []
    monkeypatch.setattr(
        cli,
        "CotSerializer",
        lambda stale_seconds: built.append(stale_seconds) or _FakeAugmentor(),
    )
    monkeypatch.setattr(cli, "JsonSapientSource", lambda path: _EventsOnceSource([]))
    monkeypatch.setattr(cli, "SapientAsynchronousTracker", lambda: _FakeTracker([[]]))
    monkeypatch.setattr(cli, "TacticalContextAugmentor", _FakeAugmentor)

    monkeypatch.setattr(
        "sys.argv",
        [
            "fusion",
            "--config",
            "sensors.json",
            "--replay-file",
            str(replay_path),
            "--log-to-file",
            "--cot-stale-seconds",
            "90",
        ],
    )

    cli.fusion_main()

    assert built == [90.0]


def test_cot_stale_seconds_defaults_to_the_serializer_default(monkeypatch, tmp_path):
    replay_path = tmp_path / "scenario.json"
    replay_path.write_text("[]", encoding="utf-8")

    built = []
    monkeypatch.setattr(
        cli,
        "CotSerializer",
        lambda stale_seconds: built.append(stale_seconds) or _FakeAugmentor(),
    )
    monkeypatch.setattr(cli, "JsonSapientSource", lambda path: _EventsOnceSource([]))
    monkeypatch.setattr(cli, "SapientAsynchronousTracker", lambda: _FakeTracker([[]]))
    monkeypatch.setattr(cli, "TacticalContextAugmentor", _FakeAugmentor)

    monkeypatch.setattr(
        "sys.argv",
        ["fusion", "--config", "sensors.json", "--replay-file", str(replay_path), "--log-to-file"],
    )

    cli.fusion_main()

    assert built == [cli.DEFAULT_STALE_SECONDS]


def test_sigterm_handler_is_installed_and_stops_the_run(monkeypatch):
    """A live producer only ever ends by signal, and as PID 1 it gets no default
    SIGTERM action -- without an explicit handler it runs until SIGKILL.
    """
    handler = None

    def capture(signum, hnd):
        nonlocal handler
        assert signum == cli.signal.SIGTERM
        handler = hnd

    monkeypatch.setattr(cli.signal, "signal", capture)

    class _SignalOnFirstEvent:
        """Fires the installed SIGTERM handler mid-run, as the runtime would."""

        def iter_events(self):
            handler(cli.signal.SIGTERM, None)
            yield  # pragma: no cover  -- the handler raises first

    monkeypatch.setattr(cli, "NetworkSapientStream", lambda port: _SignalOnFirstEvent())
    monkeypatch.setattr(cli, "SapientAsynchronousTracker", lambda: _FakeTracker([[]]))
    monkeypatch.setattr(cli, "TacticalContextAugmentor", _FakeAugmentor)

    closed = []

    class _RecordingSink:
        def send(self, payload):  # pragma: no cover  -- no event is ever emitted
            pass

        def close(self):
            closed.append(True)

    monkeypatch.setattr(cli, "FileCotSink", lambda path: _RecordingSink())
    monkeypatch.setattr(
        "sys.argv", ["fusion", "--config", "sensors.json", "--enable-sapient", "--log-to-file"]
    )

    with pytest.raises(SystemExit):
        cli.fusion_main()

    # Sinks are flushed and closed on the way out, which a SIGKILL would skip.
    assert closed == [True]
