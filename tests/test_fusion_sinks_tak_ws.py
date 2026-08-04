"""Tests for TakWsSink.

The sink opens a real WSS connection to a TAK Server and fetches a Keycloak
token over HTTPS. We replace websocket.WebSocket and requests.post so no real
socket, handshake, or HTTP call occurs; takproto framing runs for real (it is a
pure byte transform) so we assert the wire frame is genuine STREAM protobuf.
"""

import base64
import json
import time
from unittest.mock import MagicMock

import pytest
import requests
import websocket

from context_foundry.fusion.sinks.tak_ws import TakWsSink

TOKEN_URL = "https://iam.example/realms/rain-realm/protocol/openid-connect/token"


def _jwt(exp: float, groups=None) -> str:
    """Build an unsigned JWT whose payload carries exp (and optional groups)."""
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    claims = {"exp": exp}
    if groups is not None:
        claims["groups"] = groups
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    return f"{header}.{payload}.sig"


def _patch_ws(monkeypatch):
    """Install a fake websocket.WebSocket; return the shared mock instance."""
    ws = MagicMock()
    monkeypatch.setattr(
        "context_foundry.fusion.sinks.tak_ws.websocket.WebSocket", lambda *a, **kw: ws
    )
    return ws


def _patch_token(monkeypatch, access_token, expires_in=7200):
    """Install a fake requests.post returning a client_credentials token body."""
    resp = MagicMock()
    resp.json.return_value = {"access_token": access_token, "expires_in": expires_in}
    resp.raise_for_status.return_value = None
    calls = []

    def fake_post(url, data=None, verify=None, timeout=None):
        calls.append({"url": url, "data": data, "verify": verify})
        return resp

    monkeypatch.setattr("context_foundry.fusion.sinks.tak_ws.requests.post", fake_post)
    return calls


def test_requires_some_auth():
    with pytest.raises(ValueError, match="static_token or"):
        TakWsSink("tak")


def test_static_token_send_connects_and_writes_stream_frame(monkeypatch, caplog):
    ws = _patch_ws(monkeypatch)
    token = _jwt(time.time() + 3600, groups=["team-a"])
    sink = TakWsSink("tak-server", static_token=token)

    with caplog.at_level("INFO"):
        sink.send("<event uid='x'/>")

    # Upgrade carried the bearer header at the pinned WebTAK path.
    url, kwargs = ws.connect.call_args[0][0], ws.connect.call_args[1]
    assert url == "wss://tak-server:8446/takproto/1"
    assert kwargs["header"] == [f"Authorization: Bearer {token}"]
    # The wire frame is a real TAK-proto v1 STREAM frame (magic byte 0xbf).
    (frame,), _ = ws.send_binary.call_args
    assert isinstance(frame, bytes) and frame[0] == 0xBF
    assert any("connected to wss://tak-server:8446/takproto/1" in r.message for r in caplog.records)


def test_client_credentials_token_is_fetched_and_used(monkeypatch):
    ws = _patch_ws(monkeypatch)
    token = _jwt(time.time() + 7200, groups=["team-a"])
    calls = _patch_token(monkeypatch, token)

    sink = TakWsSink(
        "tak-server",
        token_url=TOKEN_URL,
        client_id="cf-a",
        client_secret="s3cret",
        scope="openid",
    )
    sink.send("<event/>")

    assert calls[0]["url"] == TOKEN_URL
    assert calls[0]["data"] == {
        "grant_type": "client_credentials",
        "client_id": "cf-a",
        "client_secret": "s3cret",
        "scope": "openid",
    }
    assert calls[0]["verify"] is False
    assert ws.connect.call_args[1]["header"] == [f"Authorization: Bearer {token}"]


def test_connection_is_reused_across_sends(monkeypatch):
    ws = _patch_ws(monkeypatch)
    sink = TakWsSink("tak", static_token=_jwt(time.time() + 3600))

    sink.send("<event uid='a'/>")
    sink.send("<event uid='b'/>")

    assert ws.connect.call_count == 1
    assert ws.send_binary.call_count == 2


def test_handshake_401_refreshes_token_and_retries(monkeypatch, caplog):
    ws = _patch_ws(monkeypatch)
    good = _jwt(time.time() + 7200)
    calls = _patch_token(monkeypatch, good)
    # First connect 401s, second succeeds.
    ws.connect.side_effect = [
        websocket.WebSocketBadStatusException("Handshake status %s", 401),
        None,
    ]

    sink = TakWsSink("tak", token_url=TOKEN_URL, client_id="cf-a")
    with caplog.at_level("WARNING"):
        sink.send("<event/>")

    assert ws.connect.call_count == 2
    assert len(calls) == 2  # token fetched again after the 401
    assert ws.send_binary.call_count == 1
    assert any("401" in r.message for r in caplog.records)


def test_non_401_handshake_error_drops_socket(monkeypatch, caplog):
    ws = _patch_ws(monkeypatch)
    ws.connect.side_effect = websocket.WebSocketBadStatusException("Handshake status %s", 403)
    sink = TakWsSink("tak", static_token=_jwt(time.time() + 3600))

    with caplog.at_level("ERROR"):
        sink.send("<event/>")

    assert sink.ws is None
    assert any("TAK WS stream error" in r.message for r in caplog.records)


def test_send_error_drops_socket_and_reconnects(monkeypatch, caplog):
    ws = _patch_ws(monkeypatch)
    ws.send_binary.side_effect = [websocket.WebSocketConnectionClosedException(), None]
    sink = TakWsSink("tak", static_token=_jwt(time.time() + 3600), retry_delay=0.0)

    with caplog.at_level("ERROR"):
        sink.send("<event uid='a'/>")  # fails, closes the socket
    assert sink.ws is None
    assert any("TAK WS stream error" in r.message for r in caplog.records)

    sink.send("<event uid='b'/>")  # reconnects
    assert ws.connect.call_count == 2


def test_send_backs_off_instead_of_reconnecting_per_event(monkeypatch, caplog):
    """A TAK Server that is down must not cost a TLS handshake per event."""
    ws = _patch_ws(monkeypatch)
    ws.send_binary.side_effect = websocket.WebSocketConnectionClosedException()
    sink = TakWsSink("tak", static_token=_jwt(time.time() + 3600), retry_delay=60.0)

    with caplog.at_level("ERROR"):
        for _ in range(5):
            sink.send("<event uid='a'/>")

    # One connect for the first send; the rest are swallowed by the backoff.
    assert ws.connect.call_count == 1


def test_an_expired_static_token_does_not_force_a_reconnect(monkeypatch):
    """A static token cannot be refreshed, so dropping the working socket the
    moment it expires would only buy a 401 per event.
    """
    ws = _patch_ws(monkeypatch)
    sink = TakWsSink("tak", static_token=_jwt(time.time() - 10))

    sink.send("<event uid='a'/>")
    sink.send("<event uid='b'/>")

    assert ws.connect.call_count == 1
    assert ws.send_binary.call_count == 2


def test_token_near_expiry_forces_reconnect_with_fresh_token(monkeypatch):
    ws = _patch_ws(monkeypatch)
    near = _jwt(time.time() + 5)  # inside the refresh leeway
    fresh = _jwt(time.time() + 7200)
    tokens = [near, fresh]

    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.side_effect = lambda: {"access_token": tokens.pop(0), "expires_in": 7200}
    monkeypatch.setattr(
        "context_foundry.fusion.sinks.tak_ws.requests.post",
        lambda *a, **kw: resp,
    )

    sink = TakWsSink("tak", token_url=TOKEN_URL, client_id="cf-a")
    sink.send("<event uid='a'/>")  # near-exp token
    sink.send("<event uid='b'/>")  # exp within leeway -> reconnect + refresh

    assert ws.connect.call_count == 2
    assert ws.close.call_count >= 1
    assert ws.connect.call_args_list[-1][1]["header"] == [f"Authorization: Bearer {fresh}"]


def test_verify_tls_toggles_sslopt(monkeypatch):
    captured = {}

    def fake_ctor(*a, **kw):
        captured["sslopt"] = kw.get("sslopt")
        return MagicMock()

    monkeypatch.setattr("context_foundry.fusion.sinks.tak_ws.websocket.WebSocket", fake_ctor)

    TakWsSink("tak", static_token=_jwt(time.time() + 3600)).send("<event/>")
    assert captured["sslopt"] == {"cert_reqs": __import__("ssl").CERT_NONE}

    TakWsSink("tak", static_token=_jwt(time.time() + 3600), verify_tls=True).send("<event/>")
    assert captured["sslopt"] == {}


def test_opaque_token_exp_falls_back(monkeypatch):
    _patch_ws(monkeypatch)
    # A non-JWT static token can't be decoded; the sink must still send.
    sink = TakWsSink("tak", static_token="opaque-not-a-jwt")
    sink.send("<event/>")
    assert sink._token_exp > time.time()


def test_close_without_connection_is_noop(monkeypatch):
    _patch_ws(monkeypatch)
    sink = TakWsSink("tak", static_token=_jwt(time.time() + 3600))
    sink.close()  # never connected
    assert sink.ws is None


def test_token_endpoint_failure_is_caught_and_backed_off(monkeypatch, caplog):
    """A Keycloak that is down must not take the fusion loop with it."""
    _patch_ws(monkeypatch)
    monkeypatch.setattr(
        "context_foundry.fusion.sinks.tak_ws.requests.post",
        MagicMock(side_effect=requests.ConnectionError("keycloak unreachable")),
    )
    sink = TakWsSink("tak", token_url=TOKEN_URL, client_id="cf-a", retry_delay=60.0)

    with caplog.at_level("ERROR"):
        sink.send("<event uid='a'/>")

    assert sink.ws is None
    assert any("keycloak unreachable" in r.message for r in caplog.records)


def test_events_dropped_during_an_outage_are_reported_on_reconnect(monkeypatch, caplog):
    ws = _patch_ws(monkeypatch)
    ws.send_binary.side_effect = [websocket.WebSocketConnectionClosedException(), None, None]
    sink = TakWsSink("tak", static_token=_jwt(time.time() + 3600), retry_delay=0.05)

    sink.send("<event uid='a'/>")  # fails, arms the backoff
    sink.send("<event uid='b'/>")  # dropped while backing off
    time.sleep(0.06)
    with caplog.at_level("WARNING"):
        sink.send("<event uid='c'/>")  # reconnects

    # Both the send that failed and the one suppressed behind it are counted.
    assert any("2 CoT event(s) were dropped" in r.message for r in caplog.records)


def test_connect_bounds_the_socket_timeout(monkeypatch):
    """Against the real library, not a mock: the constructor swallows unknown
    kwargs, so passing the timeout there leaves the socket unbounded and a TAK
    host that blackholes parks the fusion loop for the kernel's SYN budget.
    """
    opened = {}

    def fake_connect(url, sock_opt, proxy, socket):
        opened["timeout"] = sock_opt.timeout
        raise OSError("no network in tests")

    monkeypatch.setattr("websocket._core.connect", fake_connect)
    sink = TakWsSink("tak", static_token=_jwt(time.time() + 3600), socket_timeout=7.5)

    sink.send("<event/>")  # the OSError is caught and backed off

    assert opened["timeout"] == 7.5


def test_the_failing_event_is_counted_as_dropped(monkeypatch, caplog):
    ws = _patch_ws(monkeypatch)
    ws.send_binary.side_effect = [websocket.WebSocketConnectionClosedException(), None]
    sink = TakWsSink("tak", static_token=_jwt(time.time() + 3600), retry_delay=0.0)

    sink.send("<event uid='a'/>")  # the event that failed is itself lost
    with caplog.at_level("WARNING"):
        sink.send("<event uid='b'/>")

    assert any("1 CoT event(s) were dropped" in r.message for r in caplog.records)
