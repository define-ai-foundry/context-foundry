#!/usr/bin/env python3
# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# Subscribes to a TAK Server's WebTAK streaming WebSocket as a given user and
# prints the CoT that user is allowed to see. Stands in for a WebTAK browser
# session, so per-group delivery can be asserted from a script. Dev tooling only.

import argparse
import ssl
import sys
import time

import requests
import takproto
import websocket


def fetch_token(host, port, username, password, verify):
    resp = requests.post(
        f"https://{host}:{port}/oauth/token",
        data={"grant_type": "password", "username": username, "password": password},
        verify=verify,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def main():
    parser = argparse.ArgumentParser(description="Tail the CoT a WebTAK user receives")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8446)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", default="Fusion-Demo-2026!")
    parser.add_argument("--seconds", type=float, default=30.0, help="How long to listen")
    parser.add_argument("--verify-tls", action="store_true")
    args = parser.parse_args()

    token = fetch_token(args.host, args.port, args.user, args.password, args.verify_tls)
    sslopt = {} if args.verify_tls else {"cert_reqs": ssl.CERT_NONE}
    ws = websocket.WebSocket(sslopt=sslopt)
    ws.connect(
        f"wss://{args.host}:{args.port}/takproto/1",
        header=[f"Authorization: Bearer {token}"],
    )
    print(f"[{args.user}] subscribed to wss://{args.host}:{args.port}/takproto/1", flush=True)

    deadline = time.monotonic() + args.seconds
    received = 0
    while time.monotonic() < deadline:
        ws.settimeout(max(0.1, deadline - time.monotonic()))
        try:
            frame = ws.recv()
        except websocket.WebSocketTimeoutException:
            break
        except websocket.WebSocketConnectionClosedException:
            print(f"[{args.user}] connection closed by server", flush=True)
            break
        if not frame:
            continue
        if isinstance(frame, str):
            frame = frame.encode("utf-8")
        received += 1
        try:
            event = takproto.parse_proto(bytearray(frame))
        except Exception as e:
            print(f"[{args.user}] undecodable frame ({e}): {frame[:80]!r}", flush=True)
            continue
        print(f"[{args.user}] {event}", flush=True)

    ws.close()
    print(f"[{args.user}] received {received} frames", flush=True)
    # Non-zero when nothing arrived, so a shell check can assert delivery.
    sys.exit(0 if received else 1)


if __name__ == "__main__":
    main()
