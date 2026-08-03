#!/usr/bin/env python3
# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

# Emits a recorded scenario as live UDP traffic, so the engine's live ingress
# paths (--enable-sapient / --enable-cot) can be exercised without real sensors.
# Stands in for an ASM: each scenario detection is re-stamped to the present and
# sent at the scenario's own cadence, as a SAPIENT protobuf datagram (:5000) or a
# CoT XML datagram (:6969). Dev tooling only.

import argparse
import json
import socket
import time
from datetime import datetime, timedelta, timezone
from xml.sax.saxutils import quoteattr

from google.protobuf.json_format import ParseDict

from sapient_msg.bsi_flex_335_v2_0.sapient_message_pb2 import SapientMessage

SAPIENT_PORT = 5000
COT_PORT = 6969

# CoT type for a hostile airborne track: the sensor-side classification a
# CoT-speaking ASM would put on the wire.
COT_TYPE = "a-h-A-M-F-Q"


def load_messages(path):
    """Return the scenario's detection reports as (timestamp, envelope) pairs, in time order."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    events = []
    for entry in raw:
        envelope = entry.get("sapientMessage", {})
        if "detectionReport" not in envelope:
            continue
        timestamp = datetime.fromisoformat(envelope["timestamp"].replace("Z", "+00:00"))
        events.append((timestamp, envelope))

    events.sort(key=lambda pair: pair[0])
    return events


def as_sapient_datagram(envelope, timestamp):
    msg = SapientMessage()
    ParseDict(envelope, msg)
    msg.timestamp.FromDatetime(timestamp)
    return msg.SerializeToString()


def as_cot_datagram(envelope, timestamp, stale_seconds):
    """Render a scenario detection as the CoT an ASM would emit for it."""
    report = envelope["detectionReport"]
    location = report.get("location")
    if location is None:
        return None  # range/bearing reports have no lat/lon to put in a CoT point

    stale = timestamp + timedelta(seconds=stale_seconds)
    # One CoT uid per tracked object, so repeated reports update one marker.
    uid = f"{envelope['nodeId']}.{report.get('objectId', 'unknown')}"

    def stamp(value):
        return value.strftime("%Y-%m-%dT%H:%M:%S.") + f"{value.microsecond // 1000:03d}Z"

    xml = (
        f'<event version="2.0" uid={quoteattr(uid)} type="{COT_TYPE}" '
        f'time="{stamp(timestamp)}" start="{stamp(timestamp)}" stale="{stamp(stale)}" '
        f'how="m-s">'
        f'<point lat="{location["y"]:.6f}" lon="{location["x"]:.6f}" '
        f'hae="{location.get("z", 0.0):.1f}" ce="10.0" le="10.0"/>'
        f"</event>"
    )
    return xml.encode("utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Replay a scenario file as live UDP sensor traffic"
    )
    parser.add_argument("--scenario", default="data/examples/sapient_messages.json")
    parser.add_argument(
        "--protocol",
        choices=("sapient", "cot"),
        default="sapient",
        help="sapient: binary SAPIENT protobuf; cot: CoT XML",
    )
    parser.add_argument("--host", default="127.0.0.1", help="UDP destination host")
    parser.add_argument("--port", type=int, help="UDP destination port (default: per protocol)")
    parser.add_argument(
        "--realtime-factor",
        type=float,
        default=1.0,
        help="Emission speed relative to the scenario's own clock (0 = as fast as possible)",
    )
    parser.add_argument(
        "--use-scenario-timestamps",
        action="store_true",
        help="Send the file's own timestamps instead of re-stamping the scenario to start now",
    )
    parser.add_argument(
        "--cot-stale-seconds", type=float, default=30.0, help="stale window on emitted CoT"
    )
    parser.add_argument("--limit", type=int, help="Stop after N datagrams")
    parser.add_argument("--loop", action="store_true", help="Restart the scenario when it drains")
    args = parser.parse_args()

    port = args.port or (SAPIENT_PORT if args.protocol == "sapient" else COT_PORT)
    events = load_messages(args.scenario)
    if not events:
        raise SystemExit(f"no detection reports in {args.scenario}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    span = (events[-1][0] - events[0][0]).total_seconds()
    print(
        f"Feeding {len(events)} {args.protocol} detections ({span:.0f}s of scenario time) "
        f"to udp://{args.host}:{port} at {args.realtime_factor or float('inf')}x",
        flush=True,
    )

    while True:
        scenario_start = events[0][0]
        offset = (
            timedelta(0)
            if args.use_scenario_timestamps
            else datetime.now(timezone.utc) - scenario_start
        )
        wall_start = time.monotonic()
        sent = 0

        for scenario_time, envelope in events:
            if args.realtime_factor > 0:
                due = (scenario_time - scenario_start).total_seconds() / args.realtime_factor
                behind = time.monotonic() - (wall_start + due)
                if behind < 0:
                    time.sleep(-behind)

            timestamp = scenario_time + offset
            if args.protocol == "sapient":
                datagram = as_sapient_datagram(envelope, timestamp)
            else:
                datagram = as_cot_datagram(envelope, timestamp, args.cot_stale_seconds)
            if datagram is None:
                continue

            sock.sendto(datagram, (args.host, port))
            sent += 1
            if sent % 100 == 0:
                print(f"  sent {sent}/{len(events)}", flush=True)
            if args.limit and sent >= args.limit:
                print(f"Sent {sent} datagrams (--limit reached).", flush=True)
                return

        print(f"Scenario drained ({sent} datagrams).", flush=True)
        if not args.loop:
            return


if __name__ == "__main__":
    main()
