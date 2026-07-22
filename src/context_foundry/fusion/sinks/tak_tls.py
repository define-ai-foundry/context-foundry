# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# src/context_foundry/fusion/sinks/tak_tls.py

import logging
import socket
import ssl

from .base import CotSink

logger = logging.getLogger(__name__)


class TakTlsSink(CotSink):
    """
    Streams CoT to a TAK Server over a persistent TCP+TLS connection.

    TAK's `stdssl` input (default port 8089) is a mutual-TLS stream: the client
    presents a CA-signed certificate and writes newline-delimited CoT <event>
    documents. The connection is opened lazily on the first send and re-opened
    on the next send after any failure.
    """

    def __init__(
        self,
        host: str,
        port: int = 8089,
        cert: str | None = None,
        key: str | None = None,
        ca: str | None = None,
    ):
        self.host = host
        self.port = port
        self.cert = cert
        self.key = key
        self.ca = ca
        self.sock = None

    def _build_context(self) -> ssl.SSLContext:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        # TAK server certs are trusted via the TAK CA, not matched to a hostname.
        context.check_hostname = False
        if self.ca:
            context.load_verify_locations(self.ca)
        else:
            # No CA supplied (dev / self-signed): trust the server without verification.
            context.verify_mode = ssl.CERT_NONE
        if self.cert:
            context.load_cert_chain(certfile=self.cert, keyfile=self.key)
        return context

    def _connect(self) -> None:
        raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock = self._build_context().wrap_socket(raw, server_hostname=self.host)
        self.sock.connect((self.host, self.port))
        logger.info(f"CoT-over-TLS stream connected to {self.host}:{self.port}")

    def send(self, cot_payload: str) -> None:
        try:
            if self.sock is None:
                self._connect()
            self.sock.sendall(cot_payload.encode("utf-8") + b"\n")
        except (OSError, ssl.SSLError) as e:
            logger.error(f"TAK TLS stream error: {e}")
            self.close()  # drop the socket so the next send reconnects

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None
