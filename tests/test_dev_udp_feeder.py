"""Tests for dev/udp_feeder.py.

Dev tooling, but it is the only thing that drives the live ingress paths, and
the one property that matters is easy to break silently: a sensor stamps the
present. Stamping the scenario's own spacing instead puts a compressed feed
minutes into the future, where the engine's out-of-step gate discards all of it
and the demo delivers nothing.
"""

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def feeder():
    spec = importlib.util.spec_from_file_location("udp_feeder", REPO_ROOT / "dev/udp_feeder.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scenario(tmp_path, count=3, spacing_seconds=600):
    """A scenario whose events are far enough apart to run into the future."""
    start = datetime(2026, 11, 15, 2, 45, tzinfo=timezone.utc)
    messages = [
        {
            "sapientMessage": {
                "timestamp": (start + timedelta(seconds=i * spacing_seconds))
                .isoformat()
                .replace("+00:00", "Z"),
                "nodeId": "FI-MIL-RAD-KOLI-01",
                "detectionReport": {
                    "objectId": f"obj-{i}",
                    "location": {
                        "x": 30.6,
                        "y": 62.1,
                        "z": 1500.0,
                        "coordinateSystem": "LOCATION_COORDINATE_SYSTEM_LAT_LNG_DEG_M",
                        "datum": "LOCATION_DATUM_WGS84_E",
                    },
                },
            }
        }
        for i in range(count)
    ]
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(messages), encoding="utf-8")
    return path


def _sent_timestamps(feeder, monkeypatch, argv):
    sent = []
    monkeypatch.setattr(feeder.socket, "socket", lambda *a, **kw: _CapturingSocket(sent))
    monkeypatch.setattr("sys.argv", argv)
    feeder.main()
    return sent


class _CapturingSocket:
    def __init__(self, sent):
        self._sent = sent

    def sendto(self, datagram, _addr):
        self._sent.append(datagram)


def test_datagrams_are_stamped_with_the_present(feeder, monkeypatch, tmp_path, capsys):
    """Not with the scenario's own spacing, which a compressed feed would push
    minutes ahead of the wall clock -- where no sensor ever reports."""
    from sapient_msg.bsi_flex_335_v2_0.sapient_message_pb2 import SapientMessage

    path = _scenario(tmp_path, count=3, spacing_seconds=600)
    before = datetime.now(timezone.utc)
    sent = _sent_timestamps(
        feeder,
        monkeypatch,
        ["feeder", "--scenario", str(path), "--realtime-factor", "0"],
    )
    after = datetime.now(timezone.utc)
    capsys.readouterr()

    assert len(sent) == 3
    for datagram in sent:
        msg = SapientMessage()
        msg.ParseFromString(datagram)
        stamped = msg.timestamp.ToDatetime().replace(tzinfo=timezone.utc)
        assert before <= stamped <= after


class _TickingClock(datetime):
    """Stands in for `datetime`, handing out a distinct instant per now() call."""

    _calls = 0

    @classmethod
    def now(cls, _tz=None):
        cls._calls += 1
        return datetime(2026, 8, 3, 12, 0, cls._calls, tzinfo=timezone.utc)


def _sweep_scenario(tmp_path, sweep_sizes, spacing_seconds=600):
    """A scenario with one message per detection but shared (timestamp, nodeId)
    within each sweep, as a real SAPIENT node reports a multi-object sweep."""
    start = datetime(2026, 11, 15, 2, 45, tzinfo=timezone.utc)
    messages = []
    for sweep_index, size in enumerate(sweep_sizes):
        sweep_time = (
            (start + timedelta(seconds=sweep_index * spacing_seconds))
            .isoformat()
            .replace("+00:00", "Z")
        )
        for obj_index in range(size):
            messages.append(
                {
                    "sapientMessage": {
                        "timestamp": sweep_time,
                        "nodeId": "FI-MIL-RAD-KOLI-01",
                        "detectionReport": {
                            "objectId": f"sweep{sweep_index}-obj{obj_index}",
                            "location": {
                                "x": 30.6,
                                "y": 62.1,
                                "z": 1500.0,
                                "coordinateSystem": "LOCATION_COORDINATE_SYSTEM_LAT_LNG_DEG_M",
                                "datum": "LOCATION_DATUM_WGS84_E",
                            },
                        },
                    }
                }
            )
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(messages), encoding="utf-8")
    return path


def test_a_sweep_shares_one_send_time_stamp(feeder, monkeypatch, tmp_path, capsys):
    """A sweep is several detections reported at the same scenario instant by
    the same node; the feeder must send them with one identical timestamp, and
    a later sweep must get a different one."""
    from sapient_msg.bsi_flex_335_v2_0.sapient_message_pb2 import SapientMessage

    path = _sweep_scenario(tmp_path, sweep_sizes=[4, 3])
    # A stub clock, so "the two sweeps differ" cannot hinge on whether two
    # unpaced sends happen to straddle a microsecond boundary.
    monkeypatch.setattr(feeder, "datetime", _TickingClock)
    sent = _sent_timestamps(
        feeder,
        monkeypatch,
        ["feeder", "--scenario", str(path), "--realtime-factor", "0"],
    )
    capsys.readouterr()

    assert len(sent) == 7
    stamps = []
    for datagram in sent:
        msg = SapientMessage()
        msg.ParseFromString(datagram)
        stamps.append(msg.timestamp.ToDatetime().replace(tzinfo=timezone.utc))

    first_sweep_stamps = stamps[:4]
    second_sweep_stamps = stamps[4:]
    assert len(set(first_sweep_stamps)) == 1
    assert len(set(second_sweep_stamps)) == 1
    assert first_sweep_stamps[0] != second_sweep_stamps[0]


def test_use_scenario_timestamps_sends_the_file_clock(feeder, monkeypatch, tmp_path, capsys):
    from sapient_msg.bsi_flex_335_v2_0.sapient_message_pb2 import SapientMessage

    path = _scenario(tmp_path, count=1)
    sent = _sent_timestamps(
        feeder,
        monkeypatch,
        [
            "feeder",
            "--scenario",
            str(path),
            "--realtime-factor",
            "0",
            "--use-scenario-timestamps",
        ],
    )
    capsys.readouterr()

    msg = SapientMessage()
    msg.ParseFromString(sent[0])
    assert msg.timestamp.ToDatetime().year == 2026
    assert msg.timestamp.ToDatetime().month == 11


def test_cot_mode_emits_schema_valid_events(feeder, monkeypatch, tmp_path, capsys):
    from context_foundry.fusion.validators.cot import CotValidator

    path = _scenario(tmp_path, count=2)
    sent = _sent_timestamps(
        feeder,
        monkeypatch,
        ["feeder", "--scenario", str(path), "--protocol", "cot", "--realtime-factor", "0"],
    )
    capsys.readouterr()

    validator = CotValidator(
        xsd_path=str(REPO_ROOT / "protos/cot/CoT Base-Event Schema  (PUBLIC RELEASE).xsd")
    )
    for datagram in sent:
        valid, reason = validator.validate(datagram.decode("utf-8"))
        assert valid, reason


def test_an_empty_scenario_is_refused(feeder, monkeypatch, tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["feeder", "--scenario", str(path)])

    with pytest.raises(SystemExit, match="no detection reports"):
        feeder.main()
