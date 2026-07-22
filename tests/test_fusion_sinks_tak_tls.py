"""Tests for TakTlsSink.

The sink opens a real TCP+TLS connection to a TAK Server. We replace both
socket.socket and ssl.SSLContext so no real socket or handshake occurs; the
wrapped SSL socket is a MagicMock recording connect()/sendall()/close().
"""

import ssl
from unittest.mock import MagicMock

from context_foundry.fusion.sinks.tak_tls import TakTlsSink


def _patch(monkeypatch):
    """Install fake socket + ssl context; return (context, ssl_sock) mocks."""
    context = MagicMock()
    ssl_sock = MagicMock()
    context.wrap_socket.return_value = ssl_sock
    monkeypatch.setattr(
        "context_foundry.fusion.sinks.tak_tls.socket.socket", lambda *a, **kw: MagicMock()
    )
    monkeypatch.setattr(
        "context_foundry.fusion.sinks.tak_tls.ssl.SSLContext", lambda proto: context
    )
    return context, ssl_sock


def test_first_send_connects_then_writes_newline_framed_payload(monkeypatch, caplog):
    _, ssl_sock = _patch(monkeypatch)
    sink = TakTlsSink("tak-server", 8089)

    with caplog.at_level("INFO"):
        sink.send("<event/>")

    ssl_sock.connect.assert_called_once_with(("tak-server", 8089))
    ssl_sock.sendall.assert_called_once_with(b"<event/>\n")
    assert any(
        "CoT-over-TLS stream connected to tak-server:8089" in r.message for r in caplog.records
    )


def test_connection_is_reused_across_sends(monkeypatch):
    _, ssl_sock = _patch(monkeypatch)
    sink = TakTlsSink("tak-server")

    sink.send("<event uid='a'/>")
    sink.send("<event uid='b'/>")

    assert ssl_sock.connect.call_count == 1
    assert ssl_sock.sendall.call_count == 2


def test_no_ca_disables_verification(monkeypatch):
    context, _ = _patch(monkeypatch)
    sink = TakTlsSink("tak-server")

    sink.send("<event/>")

    assert context.check_hostname is False
    assert context.verify_mode == ssl.CERT_NONE
    context.load_verify_locations.assert_not_called()


def test_ca_and_client_cert_are_loaded(monkeypatch):
    context, _ = _patch(monkeypatch)
    sink = TakTlsSink("tak-server", cert="client.pem", key="client.key", ca="ca.pem")

    sink.send("<event/>")

    context.load_verify_locations.assert_called_once_with("ca.pem")
    context.load_cert_chain.assert_called_once_with(certfile="client.pem", keyfile="client.key")
    assert context.check_hostname is False


def test_send_error_logs_drops_socket_and_reconnects(monkeypatch, caplog):
    _, ssl_sock = _patch(monkeypatch)
    ssl_sock.sendall.side_effect = [ssl.SSLError("broken pipe"), None]
    sink = TakTlsSink("tak-server")

    with caplog.at_level("ERROR"):
        sink.send("<event uid='a'/>")  # fails, closes the socket

    assert any("TAK TLS stream error" in r.message for r in caplog.records)
    ssl_sock.close.assert_called_once()
    assert sink.sock is None

    sink.send("<event uid='b'/>")  # reconnects
    assert ssl_sock.connect.call_count == 2


def test_close_without_connection_is_noop(monkeypatch):
    _patch(monkeypatch)
    sink = TakTlsSink("tak-server")

    sink.close()  # never connected

    assert sink.sock is None
