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

import threading
import time
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

    def __init__(self, timestamp=T0, node_id=None):
        self.timestamp = timestamp
        self.metadata = {"nodeId": node_id} if node_id else {}


class _Stop(BaseException):
    """Sentinel used to escape fusion_main's infinite while-loop from a test."""


class _FakeClock:
    """Stands in for the wall clock the out-of-step gate reads.

    The gate reads it once per event, so advancing a fixed step per read makes a
    live run deterministic: event n is evaluated against a present of
    start + n * step.
    """

    def __init__(self, start, step_seconds=1.0):
        self.start = start
        self.step = timedelta(seconds=step_seconds)
        self.reads = 0

    def now(self, tz=None):
        reading = self.start + self.reads * self.step
        self.reads += 1
        return reading


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


def test_sensor_config_load_failure_exits_nonzero(monkeypatch, caplog):
    """A startup misconfiguration must not exit 0: the runtime reports that as
    'Completed', indistinguishable from a clean finish, and nothing alerting on
    a non-zero exit fires."""

    def _raise(**kwargs):
        raise RuntimeError("bad config")

    monkeypatch.setattr(cli.config, "load_sensor_network", _raise)
    monkeypatch.setattr(
        "sys.argv", ["fusion", "--config", "sensors.json", "--replay-file", "x.json"]
    )

    with caplog.at_level("ERROR"), pytest.raises(SystemExit) as exit_info:
        cli.fusion_main()

    assert exit_info.value.code == 1
    assert any("Failed to load sensor config" in r.message for r in caplog.records)


def test_no_sources_enabled_exits_nonzero(monkeypatch, caplog):
    monkeypatch.setattr("sys.argv", ["fusion", "--config", "sensors.json"])

    with caplog.at_level("ERROR"), pytest.raises(SystemExit) as exit_info:
        cli.fusion_main()

    assert exit_info.value.code == 1
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

    with caplog.at_level("DEBUG"):
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

    with caplog.at_level("ERROR"), pytest.raises(SystemExit) as exit_info:
        cli.fusion_main()

    assert exit_info.value.code == 1
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

    def fake_sapient_stream(port, **_):
        created["sapient_port"] = port
        return sapient

    def fake_cot_stream(port, **_):
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
    with caplog.at_level("ERROR"), pytest.raises(SystemExit) as exit_info:
        cli.fusion_main()

    # Non-zero: an ingress that has stopped reading must not look healthy.
    assert exit_info.value.code == 1

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
            ca=None,
            token_timeout=None,
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
        cli, "NetworkSapientStream", lambda port, **_: _EventsThenStopSource([(T0, [_Detection()])])
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

    with caplog.at_level("ERROR"), pytest.raises(SystemExit):
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

    with caplog.at_level("WARNING"), pytest.raises(SystemExit) as exit_info:
        cli.fusion_main()

    assert exit_info.value.code == 1  # falls through to "No sources enabled"
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

    with caplog.at_level("DEBUG"):
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

    with caplog.at_level("WARNING"), pytest.raises(SystemExit):
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

    monkeypatch.setattr(cli, "NetworkSapientStream", lambda port, **_: _SignalOnFirstEvent())
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


def test_replay_file_with_a_live_source_is_refused(monkeypatch, capsys):
    """The two run on clocks that do not line up, so the tracker would discard
    whichever timeline is behind -- silently, and for the whole run."""
    monkeypatch.setattr(
        "sys.argv",
        ["fusion", "--config", "sensors.json", "--replay-file", "x.json", "--enable-sapient"],
    )

    with pytest.raises(SystemExit):
        cli.fusion_main()

    assert "cannot be combined with" in capsys.readouterr().err


def test_events_older_than_the_last_processed_are_dropped(monkeypatch, caplog):
    """Stone Soup cannot predict backwards: a late event rewinds every track's
    timestamp and re-broadcasts the lot with a CoT time that moves into the past.
    """
    late = T0 - timedelta(seconds=5)
    tracker = _EchoTracker("late-track-0001")

    monkeypatch.setattr(
        cli,
        "NetworkSapientStream",
        lambda port, **_: _EventsOnceSource(
            [
                (T0, [_Detection(T0)]),
                (late, [_Detection(late, node_id="node-late")]),
                (T0, [_Detection(T0)]),
            ]
        ),
    )
    monkeypatch.setattr(cli, "SapientAsynchronousTracker", lambda: tracker)
    monkeypatch.setattr(cli, "TacticalContextAugmentor", _FakeAugmentor)
    monkeypatch.setattr(
        "sys.argv", ["fusion", "--config", "sensors.json", "--enable-sapient", "--log-to-file"]
    )

    with caplog.at_level("WARNING"), pytest.raises(SystemExit):
        cli.fusion_main()

    assert tracker.seen == [T0, T0]
    warnings = [r for r in caplog.records if "5.0s behind the newest event" in r.message]
    assert len(warnings) == 1
    # Names the sensor whose clock is out of step, not just the fact of a drop.
    assert "node-late" in warnings[0].message


def test_late_events_are_reported_periodically_not_once(monkeypatch, caplog):
    """A skewed sensor clock drops every packet it sends; one line for the whole
    run leaves an operator with a permanently degraded picture and no signal."""
    late = T0 - timedelta(seconds=5)
    events = [(T0, [_Detection(T0)])] + [(late, [_Detection(late)]) for _ in range(4)]

    monkeypatch.setattr(cli, "NetworkSapientStream", lambda port, **_: _EventsOnceSource(events))
    monkeypatch.setattr(cli, "SapientAsynchronousTracker", lambda: _EchoTracker("t-0001"))
    monkeypatch.setattr(cli, "TacticalContextAugmentor", _FakeAugmentor)
    monkeypatch.setattr(cli, "LATE_EVENT_REPORT_SECONDS", 0.0)  # report every drop
    monkeypatch.setattr(
        "sys.argv", ["fusion", "--config", "sensors.json", "--enable-sapient", "--log-to-file"]
    )

    with caplog.at_level("WARNING"), pytest.raises(SystemExit):
        cli.fusion_main()

    reports = [r for r in caplog.records if "behind the newest event" in r.message]
    assert len(reports) == 4
    # Each report carries the running total, so the scale of the loss is visible.
    assert "Dropped 4 event(s)" in reports[-1].message
    # And the total is flushed on the way out, for a run that ends mid-window.
    assert any("in total for timestamps out of step" in r.message for r in caplog.records)


def test_a_future_stamped_live_event_does_not_become_the_watermark(monkeypatch, caplog):
    """One sensor with a fast clock, an NTP step, or a spoofed datagram would
    otherwise set a mark every healthy event afterwards falls behind, silencing
    the engine for the rest of the run."""
    now = datetime.now(timezone.utc)
    far_future = now + timedelta(hours=6)
    good = [
        (now + timedelta(seconds=i), [_Detection(now + timedelta(seconds=i))]) for i in range(3)
    ]
    events = [good[0], (far_future, [_Detection(far_future, node_id="node-fast")]), *good[1:]]

    tracker = _EchoTracker("future-track-0001")
    monkeypatch.setattr(cli, "NetworkSapientStream", lambda port, **_: _EventsOnceSource(events))
    monkeypatch.setattr(cli, "SapientAsynchronousTracker", lambda: tracker)
    monkeypatch.setattr(cli, "TacticalContextAugmentor", _FakeAugmentor)
    monkeypatch.setattr(
        "sys.argv", ["fusion", "--config", "sensors.json", "--enable-sapient", "--log-to-file"]
    )

    with caplog.at_level("WARNING"), pytest.raises(SystemExit):
        cli.fusion_main()

    # The future event is refused; every healthy one still gets through.
    assert tracker.seen == [e[0] for e in good]
    reports = [r for r in caplog.records if "ahead of the present" in r.message]
    assert len(reports) == 1
    assert "node-fast" in reports[0].message


def test_a_slightly_fast_sensor_does_not_drag_the_watermark_past_the_present(
    monkeypatch, caplog, tmp_path
):
    """A sensor a few seconds fast is inside the future horizon, so its event is
    fused -- but if it also set the watermark, every sensor reporting the actual
    present would fall behind it and be dropped, one bad clock erasing the
    picture the healthy sensors built.
    """
    monkeypatch.chdir(tmp_path)
    clock = _FakeClock(T0)
    monkeypatch.setattr(cli, "datetime", clock)

    fast = T0 + timedelta(seconds=5)  # read 4s ahead of the present: inside the horizon
    healthy = [T0, T0 + timedelta(seconds=2), T0 + timedelta(seconds=3)]
    events = [
        (healthy[0], [_Detection(healthy[0], node_id="node-a")]),
        (fast, [_Detection(fast, node_id="node-fast")]),
        (healthy[1], [_Detection(healthy[1], node_id="node-a")]),
        (healthy[2], [_Detection(healthy[2], node_id="node-b")]),
    ]

    tracker = _EchoTracker("fast-track-0001")
    monkeypatch.setattr(cli, "NetworkSapientStream", lambda port, **_: _EventsOnceSource(events))
    monkeypatch.setattr(cli, "SapientAsynchronousTracker", lambda: tracker)
    monkeypatch.setattr(cli, "TacticalContextAugmentor", _FakeAugmentor)
    monkeypatch.setattr(
        "sys.argv", ["fusion", "--config", "sensors.json", "--enable-sapient", "--log-to-file"]
    )

    with caplog.at_level("WARNING"), pytest.raises(SystemExit):
        cli.fusion_main()

    # Every event is fused, each with its own timestamp; nothing is dropped.
    assert tracker.seen == [healthy[0], fast, healthy[1], healthy[2]]
    assert not [r for r in caplog.records if "out of step" in r.message]


def test_the_drop_warning_names_the_sensor_that_set_the_watermark(monkeypatch, caplog, tmp_path):
    """Naming only the dropped event's sensor blames the victim: in a mixed-clock
    network the sensor that set the mark is the one to go and look at."""
    monkeypatch.chdir(tmp_path)
    clock = _FakeClock(T0)
    monkeypatch.setattr(cli, "datetime", clock)

    fast = T0 + timedelta(seconds=5)
    # Beyond the lateness tolerance, so these are a disagreeing clock rather than
    # the ordinary arrival delay the gate has to let through.
    late = T0 - timedelta(seconds=30)
    later = T0 - timedelta(seconds=29)
    events = [
        (T0, [_Detection(T0, node_id="node-a")]),
        (fast, [_Detection(fast, node_id="node-fast")]),
        (late, [_Detection(late, node_id="node-late")]),
        (later, [_Detection(later, node_id="node-late")]),
    ]

    monkeypatch.setattr(cli, "NetworkSapientStream", lambda port, **_: _EventsOnceSource(events))
    monkeypatch.setattr(cli, "SapientAsynchronousTracker", lambda: _EchoTracker("mark-track-0001"))
    monkeypatch.setattr(cli, "TacticalContextAugmentor", _FakeAugmentor)
    monkeypatch.setattr(
        "sys.argv", ["fusion", "--config", "sensors.json", "--enable-sapient", "--log-to-file"]
    )

    with caplog.at_level("WARNING"), pytest.raises(SystemExit):
        cli.fusion_main()

    # Two drops, one report: the warning is rate-limited, not one line per packet.
    (report,) = [r for r in caplog.records if "behind the newest event" in r.message]
    assert "node-late" in report.message  # the event that was dropped
    assert "node-fast" in report.message  # the event that set the mark it fell behind


def test_the_drop_warning_reports_an_unset_watermark(monkeypatch, caplog, tmp_path):
    """The first event of a run can itself be the one dropped, leaving no sensor
    to name as having set the mark."""
    monkeypatch.chdir(tmp_path)
    clock = _FakeClock(T0)
    monkeypatch.setattr(cli, "datetime", clock)

    far = T0 + timedelta(minutes=5)
    events = [(far, [_Detection(far, node_id="node-fast")])]

    monkeypatch.setattr(cli, "NetworkSapientStream", lambda port, **_: _EventsOnceSource(events))
    monkeypatch.setattr(cli, "SapientAsynchronousTracker", lambda: _EchoTracker("none-track-0001"))
    monkeypatch.setattr(cli, "TacticalContextAugmentor", _FakeAugmentor)
    monkeypatch.setattr(
        "sys.argv", ["fusion", "--config", "sensors.json", "--enable-sapient", "--log-to-file"]
    )

    with caplog.at_level("WARNING"), pytest.raises(SystemExit):
        cli.fusion_main()

    (report,) = [r for r in caplog.records if "ahead of the present" in r.message]
    assert "no event accepted yet" in report.message


def test_a_replay_running_ahead_of_the_wall_clock_is_not_gated(monkeypatch, caplog, tmp_path):
    """A replay legitimately emits timestamps minutes into the future -- the
    scenario is re-stamped to start now and then runs on its own clock."""
    replay_path = tmp_path / "scenario.json"
    replay_path.write_text("[]", encoding="utf-8")

    ahead = datetime.now(timezone.utc) + timedelta(minutes=30)
    tracker = _EchoTracker("ahead-track-0001")
    monkeypatch.setattr(
        cli, "JsonSapientSource", lambda path: _EventsOnceSource([(ahead, [_Detection(ahead)])])
    )
    monkeypatch.setattr(cli, "SapientAsynchronousTracker", lambda: tracker)
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

    cli.fusion_main()

    assert tracker.seen == [ahead]
    assert not [r for r in caplog.records if "out of step" in r.message]


def _run_replay_events(monkeypatch, tmp_path, events, tracker, extra_argv=()):
    """Drive one unpaced replay pass over `events` with a file sink."""
    monkeypatch.chdir(tmp_path)
    replay_path = tmp_path / "scenario.json"
    replay_path.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(cli, "JsonSapientSource", lambda path: _EventsOnceSource(events))
    monkeypatch.setattr(cli, "SapientAsynchronousTracker", lambda: tracker)
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
            "--realtime-factor",
            "0",
            "--log-to-file",
            *extra_argv,
        ],
    )
    cli.fusion_main()


def test_one_info_line_per_event_replaces_the_per_track_broadcast_lines(
    monkeypatch, caplog, tmp_path
):
    """A line per track per event is ~20 a second on a busy picture, which buries
    the once-a-minute drop warnings in it."""
    tracks = [_fake_track("summary-track-0001", T0), _fake_track("summary-track-0002", T0)]
    events = [(T0, [_Detection(T0), _Detection(T0)])]

    with caplog.at_level("DEBUG"):
        _run_replay_events(monkeypatch, tmp_path, events, _FakeTracker([tracks]))

    per_track = [r for r in caplog.records if "Broadcast Update for Track" in r.message]
    assert len(per_track) == 2
    assert {r.levelname for r in per_track} == {"DEBUG"}

    (per_event,) = [r for r in caplog.records if "Fused event at" in r.message]
    assert per_event.levelname == "INFO"
    assert "2 detection(s) in, 2 track(s) broadcast" in per_event.message


def test_the_counters_track_what_the_engine_actually_did(monkeypatch, caplog, tmp_path):
    """An engine that is up but fusing nothing logs the same startup lines as a
    healthy one; the counters are what separate them, so they have to be right."""
    events = [
        (T0 + timedelta(seconds=i), [_Detection(T0 + timedelta(seconds=i))]) for i in range(3)
    ]

    with caplog.at_level("INFO"):
        _run_replay_events(monkeypatch, tmp_path, events, _EchoTracker("periodic-track-0001"))

    (final,) = [r for r in caplog.records if "Fusion totals at shutdown" in r.message]
    assert "3 event(s) fused" in final.message
    assert "3 detection(s) ingested" in final.message
    assert "3 CoT message(s) sent" in final.message
    assert "0 event(s) dropped out of step" in final.message


def test_summary_is_rate_limited_and_flushed_on_shutdown(monkeypatch, caplog, tmp_path):
    events = [
        (T0 + timedelta(seconds=i), [_Detection(T0 + timedelta(seconds=i))]) for i in range(3)
    ]

    with caplog.at_level("INFO"):
        _run_replay_events(monkeypatch, tmp_path, events, _EchoTracker("shutdown-track-0001"))

    # Default interval: nothing periodic in a run this short.
    assert not [r for r in caplog.records if "Fusion summary" in r.message]

    (final,) = [r for r in caplog.records if "Fusion totals at shutdown" in r.message]
    assert final.levelname == "INFO"
    assert "3 event(s) fused" in final.message
    assert "3 detection(s) ingested" in final.message
    assert "3 CoT message(s) sent" in final.message
    assert "0 event(s) dropped out of step" in final.message


def test_shutdown_summary_counts_dropped_events(monkeypatch, caplog, tmp_path):
    """Arriving-and-dropped has to read differently from nothing-arriving."""
    late = T0 - timedelta(seconds=5)
    events = [(T0, [_Detection(T0)]), (late, [_Detection(late, node_id="node-late")])]

    with caplog.at_level("INFO"):
        _run_replay_events(monkeypatch, tmp_path, events, _EchoTracker("dropped-track-0001"))

    (final,) = [r for r in caplog.records if "Fusion totals at shutdown" in r.message]
    assert "1 event(s) fused" in final.message
    assert "1 event(s) dropped out of step" in final.message


def test_the_summary_is_reported_even_when_nothing_arrives(monkeypatch, caplog):
    """An ingress that has gone quiet is the state most worth reporting, and a
    loop-driven report is silent exactly then."""
    monkeypatch.setattr(cli, "SUMMARY_REPORT_SECONDS", 0.02)
    counters = {"events": 0, "detections": 0, "cot": 0, "dropped": 0}

    with caplog.at_level("INFO"):
        stop = cli._start_summary_reporter(counters)
        time.sleep(0.12)
        stop.set()

    summaries = [r for r in caplog.records if "Fusion summary" in r.message]
    assert len(summaries) >= 2
    assert "0 event(s) fused" in summaries[0].message


def test_the_summary_reporter_stops_when_told(monkeypatch, caplog):
    monkeypatch.setattr(cli, "SUMMARY_REPORT_SECONDS", 0.02)
    stop = cli._start_summary_reporter({"events": 0, "detections": 0, "cot": 0, "dropped": 0})
    stop.set()
    time.sleep(0.08)

    with caplog.at_level("INFO"):
        time.sleep(0.08)

    assert not [r for r in caplog.records if "Fusion summary" in r.message]
    assert not [t for t in threading.enumerate() if t.name == "summary-reporter"]


def test_the_frame_window_reaches_both_live_sources(monkeypatch):
    """A sweep spread over more than the window splits into single-detection
    events, so an operator whose sensors are jittery has to be able to raise it."""
    windows = {}
    monkeypatch.setattr(
        cli,
        "NetworkSapientStream",
        lambda port, frame_window_seconds: (
            windows.setdefault("sapient", frame_window_seconds) or _EventsOnceSource([])
        ),
    )
    monkeypatch.setattr(
        cli,
        "CotNetworkStream",
        lambda port, frame_window_seconds: (
            windows.setdefault("cot", frame_window_seconds) or _EventsOnceSource([])
        ),
    )
    monkeypatch.setattr(cli, "SapientAsynchronousTracker", lambda: _FakeTracker([[]]))
    monkeypatch.setattr(cli, "TacticalContextAugmentor", _FakeAugmentor)
    monkeypatch.setattr(
        "sys.argv",
        [
            "fusion",
            "--config",
            "sensors.json",
            "--enable-sapient",
            "--enable-cot",
            "--log-to-file",
            "--frame-window-seconds",
            "0.4",
        ],
    )

    with pytest.raises(SystemExit):
        cli.fusion_main()

    assert windows == {"sapient": 0.4, "cot": 0.4}


def test_the_tak_ws_trust_and_token_timeout_flags_reach_the_sink(monkeypatch, tmp_path):
    """verify_tls cannot verify a TAK server dialled by a name its cert does not
    carry, so the CA is the only usable prod setting -- it has to arrive."""
    replay_path = tmp_path / "scenario.json"
    replay_path.write_text("[]", encoding="utf-8")
    built = {}

    class _FakeTakWsSink:
        def __init__(self, host, port, **kwargs):
            built.update(kwargs)

        def send(self, payload):  # pragma: no cover  -- no event is emitted
            pass

        def close(self):
            pass

    monkeypatch.setattr(cli, "TakWsSink", _FakeTakWsSink)
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
            "--tak-ws-host",
            "tak-server.tak.svc.cluster.local",
            "--tak-ws-ca",
            "/etc/context-foundry-tak-ca/ca.crt",
            "--tak-ws-token-timeout",
            "3",
        ],
    )

    cli.fusion_main()

    assert built["ca"] == "/etc/context-foundry-tak-ca/ca.crt"
    assert built["token_timeout"] == 3.0


def test_an_event_within_the_lateness_tolerance_is_still_fused(monkeypatch, caplog, tmp_path):
    """A sensor stamps before it transmits, so every event arrives a little in the
    past while the mark is pinned to the present. With no tolerance a sender whose
    clock runs ahead pulls the mark up to now and every healthy sensor behind it
    is refused -- measured as a total blackout from one datagram per second."""
    monkeypatch.chdir(tmp_path)
    clock = _FakeClock(T0)
    monkeypatch.setattr(cli, "datetime", clock)

    ahead = T0 + timedelta(seconds=9)  # inside the future horizon, so it sets the mark
    healthy = T0 - timedelta(milliseconds=300)  # ordinary flight + assembly delay
    tracker = _EchoTracker("tolerance-track-0001")
    events = [
        (ahead, [_Detection(ahead, node_id="node-ahead")]),
        (healthy, [_Detection(healthy, node_id="node-healthy")]),
    ]

    monkeypatch.setattr(cli, "NetworkSapientStream", lambda port, **_: _EventsOnceSource(events))
    monkeypatch.setattr(cli, "SapientAsynchronousTracker", lambda: tracker)
    monkeypatch.setattr(cli, "TacticalContextAugmentor", _FakeAugmentor)
    monkeypatch.setattr(
        "sys.argv", ["fusion", "--config", "sensors.json", "--enable-sapient", "--log-to-file"]
    )

    with caplog.at_level("WARNING"), pytest.raises(SystemExit):
        cli.fusion_main()

    assert tracker.seen == [ahead, healthy]
    assert not [r for r in caplog.records if "behind the newest event" in r.message]
