# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# src/context_foundry/fusion/sinks/tak_ws.py

import base64
import binascii
import json
import logging
import ssl
import time

import requests
import takproto
import websocket

from .base import CotSink

logger = logging.getLogger(__name__)

# Seconds to wait on the Keycloak token endpoint. The fetch is synchronous inside
# a send, so this is time the fusion loop spends not reading its sensor sockets.
DEFAULT_TOKEN_TIMEOUT = 5.0


class TakWsSink(CotSink):
    """
    Streams CoT to a TAK Server over the WebTAK streaming WebSocket.

    The 8446 connector exposes `wss://<host>:8446/takproto/1`, the transport
    WebTAK itself uses. A client authenticates with an `Authorization: Bearer`
    header on the upgrade and writes TAK-protocol-v1 STREAM-framed protobuf; CoT
    sent over it is tagged with the authenticated caller's Keycloak `groups`, so
    a per-group producer's events are confined to its one group.

    Token acquisition is either a Keycloak `client_credentials` grant (preferred,
    refreshed before `exp` and on a 401 handshake) or a static bearer token. The
    socket is opened lazily on the first send and re-opened on the next send after
    any failure.

    Two separate TLS connections are involved, and the trust knobs deliberately
    apply to only one each:

    * The **token endpoint** is always verified against the system trust store.
      It carries the confidential client secret -- the credential the whole
      per-group isolation rests on -- to a public host with a publicly trusted
      certificate, so there is nothing to be gained by weakening it and no knob
      that does.
    * The **TAK connection** is governed by `ca` and `verify_tls`. `ca` verifies
      the server's chain against that bundle and takes precedence over
      `verify_tls`; it does not require the hostname to match, because TAK is
      commonly reached at an internal name while its certificate names the
      ingress. `verify_tls` on its own verifies against the system trust store,
      hostname included. Neither means the server is not verified at all, which
      is for a self-signed dev stack only.
    """

    def __init__(
        self,
        host: str,
        port: int = 8446,
        *,
        token_url: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        static_token: str | None = None,
        scope: str | None = None,
        verify_tls: bool = False,
        ca: str | None = None,
        path: str = "/takproto/1",
        refresh_leeway: int = 30,
        retry_delay: float = 5.0,
        socket_timeout: float = 10.0,
        token_timeout: float = DEFAULT_TOKEN_TIMEOUT,
    ):
        if not static_token and not (token_url and client_id):
            raise ValueError(
                "TakWsSink needs either static_token or (token_url + client_id) for auth"
            )
        self.host = host
        self.port = port
        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.static_token = static_token
        self.scope = scope
        self.verify_tls = verify_tls
        self.ca = ca
        self.path = path
        self.refresh_leeway = refresh_leeway
        self.retry_delay = retry_delay
        self.socket_timeout = socket_timeout
        self.token_timeout = token_timeout
        self.ws = None
        self._token = None
        self._token_exp = 0.0
        self._retry_after = 0.0
        self._suppressed = 0

    @staticmethod
    def _decode_exp(token: str) -> float:
        """Read the `exp` claim from a JWT without verifying it (framing only)."""
        try:
            payload = token.split(".")[1]
            payload += "=" * (-len(payload) % 4)
            claims = json.loads(base64.urlsafe_b64decode(payload))
            return float(claims.get("exp", 0.0))
        except (IndexError, ValueError, binascii.Error, json.JSONDecodeError):
            # Opaque / undecodable token: treat as long-lived, refresh only on 401.
            return time.time() + 3600.0

    def _fetch_token(self) -> None:
        if self.static_token:
            self._token = self.static_token
            self._token_exp = self._decode_exp(self.static_token)
            return
        data = {"grant_type": "client_credentials", "client_id": self.client_id}
        if self.client_secret:
            data["client_secret"] = self.client_secret
        if self.scope:
            data["scope"] = self.scope
        # verify is pinned on: the TAK knobs must not reach the secret's channel.
        # A blocking post stalls the fusion loop with the sensor sockets unread,
        # so an unreachable Keycloak has to give up in seconds, not tens of them.
        resp = requests.post(self.token_url, data=data, verify=True, timeout=self.token_timeout)
        resp.raise_for_status()
        body = resp.json()
        self._token = body["access_token"]
        # Prefer the JWT exp; fall back to expires_in if the token is opaque.
        exp = self._decode_exp(self._token)
        if "expires_in" in body:
            exp = min(exp, time.time() + float(body["expires_in"]))
        self._token_exp = exp
        logger.info(
            "Fetched Keycloak token for %s (exp in %ds)", self.client_id, int(exp - time.time())
        )

    def _token_valid(self) -> bool:
        return bool(self._token) and time.time() < self._token_exp - self.refresh_leeway

    def _url(self) -> str:
        return f"wss://{self.host}:{self.port}{self.path}"

    def _sslopt(self) -> dict:
        """TLS options for the WSS handshake only; the token endpoint is separate.

        A CA bundle is verification, so it overrides `verify_tls`, but with the
        hostname check off: the certificate of a TAK reached at an in-cluster
        Service name names the ingress instead, and failing that check is what
        pushes an operator into turning verification off altogether.
        """
        if self.ca:
            return {
                "cert_reqs": ssl.CERT_REQUIRED,
                "ca_certs": self.ca,
                "check_hostname": False,
            }
        if self.verify_tls:
            return {}  # library default: system trust store, hostname checked
        return {"cert_reqs": ssl.CERT_NONE}

    def _connect(self) -> None:
        """Open the WS; refresh the token once and retry if the handshake 401s."""
        for attempt in (1, 2):
            if not self._token_valid():
                self._fetch_token()
            ws = websocket.WebSocket(sslopt=self._sslopt())
            try:
                # The timeout belongs here, not on the constructor, which
                # swallows unknown kwargs. Unbounded, a TAK host that blackholes
                # (a Service with no endpoints, a packet filter that drops) parks
                # the fusion loop for the kernel's SYN retry budget -- minutes,
                # silently, with the sensor sockets unread behind it. It stays on
                # the socket afterwards, so a wedged send is bounded too; both
                # surface as the usual retry.
                ws.connect(
                    self._url(),
                    header=[f"Authorization: Bearer {self._token}"],
                    timeout=self.socket_timeout,
                )
            except websocket.WebSocketBadStatusException as e:
                if e.status_code == 401 and attempt == 1 and not self.static_token:
                    logger.warning("TAK WS handshake 401; refreshing token and retrying")
                    self._token = None  # force a fresh fetch on the retry
                    continue
                raise
            self.ws = ws
            logger.info("CoT-over-WS stream connected to %s", self._url())
            return

    def send(self, cot_payload: str) -> None:
        try:
            # A token nearing exp forces a reconnect with a fresh one. A static
            # token cannot be refreshed, so reconnecting on its expiry would only
            # buy a 401 per event; keep using the socket until it actually fails.
            if self.ws is not None and not self.static_token and not self._token_valid():
                self.close()
            if self.ws is None:
                if time.monotonic() < self._retry_after:
                    self._suppressed += 1  # still backing off from a failed connect
                    return
                self._connect()
                if self._suppressed:
                    logger.warning(
                        "Reconnected to TAK; %d CoT event(s) were dropped while it was "
                        "unreachable.",
                        self._suppressed,
                    )
                    self._suppressed = 0
            frame = takproto.xml2proto(cot_payload, takproto.TAKProtoVer.STREAM)
            self.ws.send_binary(bytes(frame))
        except (websocket.WebSocketException, OSError, requests.RequestException) as e:
            # A TAK Server that is down, restarting, or rejecting the token would
            # otherwise get a full TLS handshake per event, forever.
            self._suppressed += 1
            self._retry_after = time.monotonic() + self.retry_delay
            logger.error("TAK WS stream error: %s (retrying in %gs)", e, self.retry_delay)
            self.close()  # drop the socket so a later send reconnects

    def close(self) -> None:
        if self.ws is not None:
            try:
                self.ws.close()
            finally:
                self.ws = None
