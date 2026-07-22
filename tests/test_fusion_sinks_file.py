"""Tests for FileCotSink: appends one newline-terminated CoT document per send."""

from context_foundry.fusion.sinks.file import FileCotSink


def test_send_appends_newline_terminated_payloads(tmp_path):
    path = tmp_path / "out.xml"
    sink = FileCotSink(str(path))

    sink.send("<event uid='a'/>")
    sink.send("<event uid='b'/>")

    assert path.read_text(encoding="utf-8") == "<event uid='a'/>\n<event uid='b'/>\n"
