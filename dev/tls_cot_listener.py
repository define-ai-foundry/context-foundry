#!/usr/bin/env python3
# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# Minimal mutual-TLS TCP listener that logs newline-framed CoT events.
# Stand-in for a TAK Server's stdssl input (8089) to validate TakTlsSink over a
# real socket, without the heavy TAK + Postgres stack. Dev tooling only.

import socket
import ssl

HOST = "0.0.0.0"  # noqa: S104 dev listener inside a container
PORT = 8089
CERT = "/certs/server.pem"
KEY = "/certs/server.key"
CA = "/certs/ca.pem"


def main():
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(CERT, KEY)
    # Require a client cert signed by our CA, exactly like TAK's stdssl input.
    context.load_verify_locations(CA)
    context.verify_mode = ssl.CERT_REQUIRED

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((HOST, PORT))
    listener.listen()
    print(f"TLS CoT listener ready on {HOST}:{PORT} (mutual TLS)", flush=True)

    while True:
        conn, addr = listener.accept()
        try:
            tls = context.wrap_socket(conn, server_side=True)
        except ssl.SSLError as e:
            print(f"handshake failed from {addr}: {e}", flush=True)
            conn.close()
            continue

        print(f"client connected: {addr}", flush=True)
        buffer = b""
        with tls:
            while True:
                data = tls.recv(4096)
                if not data:
                    break
                buffer += data
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    print(f"CoT received: {line.decode('utf-8', 'replace')}", flush=True)
        print(f"client disconnected: {addr}", flush=True)


if __name__ == "__main__":
    main()
