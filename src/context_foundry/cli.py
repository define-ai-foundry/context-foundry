# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import signal
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from context_foundry.fusion import config
from context_foundry.fusion.augmentor import TacticalContextAugmentor
from context_foundry.fusion.serializers import (
    DEFAULT_STALE_SECONDS,
    CotSerializer,
    SapientSerializer,
)
from context_foundry.fusion.sinks.file import FileCotSink
from context_foundry.fusion.sinks.tak_tls import TakTlsSink
from context_foundry.fusion.sinks.tak_ws import DEFAULT_TOKEN_TIMEOUT, TakWsSink
from context_foundry.fusion.sources.cot_stream import CotNetworkStream
from context_foundry.fusion.sources.frames import DEFAULT_FRAME_WINDOW_SECONDS
from context_foundry.fusion.sources.json_file import JsonSapientSource
from context_foundry.fusion.sources.multiplex import MultiplexedSource
from context_foundry.fusion.sources.offset import OffsetReplaySource
from context_foundry.fusion.sources.paced import RealtimeReplaySource
from context_foundry.fusion.sources.stream import NetworkSapientStream
from context_foundry.fusion.tracker import SapientAsynchronousTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FusionEngine")

# How often to report events dropped for arriving out of step. A skewed sensor
# clock produces one per packet, so this is a rate limit, not a one-shot.
LATE_EVENT_REPORT_SECONDS = 60.0

# How far ahead of now a live event may be stamped. Live sensors report the
# present, so anything beyond this is a broken clock or a spoofed datagram --
# and without the bound one such packet would become the high-water mark that
# every healthy event afterwards is measured against and dropped.
FUTURE_HORIZON_SECONDS = 10.0

# How far behind the newest event seen a live event may still be fused. A sensor
# stamps before it transmits, so every event arrives already a little in the past
# -- flight time, frame assembly, one fusion pass -- while the newest timestamp a
# live run will accept is pinned to the present. A gate with no tolerance
# therefore drops the ordinary case: measured, healthy sensors landed 0.3s behind
# a mark an out-of-step sender had pulled up to now, and every one was refused.
# Beyond this it is a clock that disagrees rather than latency, and the rewind it
# would cost the tracker is a fraction of one update interval either way.
LATE_EVENT_TOLERANCE_SECONDS = 2.0

# How often to log the running fusion summary. Without it an engine that is up
# and fusing nothing logs the same lines as a healthy one and then nothing at
# all, so there is no way to tell a quiet sensor network from a broken ingress.
SUMMARY_REPORT_SECONDS = 60.0


def _sensor_ids(detections):
    """The sensors an event came from, for reporting which one is out of step."""
    ids = {getattr(det, "metadata", {}).get("nodeId") for det in detections}
    return ", ".join(sorted(str(node_id) for node_id in ids if node_id)) or "an unnamed sensor"


def _start_summary_reporter(counters):
    """Report the summary every SUMMARY_REPORT_SECONDS; returns the stop switch.

    A daemon thread rather than a check inside the fusion loop, because the loop
    only runs when an event arrives: the state worth reporting most loudly -- an
    ingress that has stopped producing anything -- is exactly the one a
    loop-driven report cannot describe. The counters are only read for a log
    line, so a torn read costs at most a slightly stale number.
    """
    stop = threading.Event()

    def report():
        while not stop.wait(SUMMARY_REPORT_SECONDS):
            _log_summary(counters, "Fusion summary")

    threading.Thread(target=report, name="summary-reporter", daemon=True).start()
    return stop


def _log_summary(counters, prefix):
    """Report the running totals an operator needs to place the engine's state.

    All four failure modes read differently here: nothing arriving leaves every
    counter at zero, arriving-and-dropped moves only the drop count, healthy
    fusion moves all of them, and a TAK server refusing the stream shows CoT
    still being sent while the sink logs its own connection warnings.
    """
    logger.info(
        "%s: %d event(s) fused, %d detection(s) ingested, %d CoT message(s) sent, "
        "%d event(s) dropped out of step.",
        prefix,
        counters["events"],
        counters["detections"],
        counters["cot"],
        counters["dropped"],
    )


def _install_shutdown_handler():
    """Make SIGTERM end the run instead of being ignored.

    A live source never terminates on its own, so the only way the engine ever
    stops is a signal -- and as PID 1 in a container it gets no default action
    for SIGTERM, so an unhandled one leaves it running until the runtime loses
    patience and SIGKILLs it, skipping every sink's close().
    """

    def shutdown(signum, _frame):
        logger.info("Received %s; shutting down.", signal.Signals(signum).name)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, shutdown)


def fusion_main():
    # 1. Parse Command-Line Arguments
    parser = argparse.ArgumentParser(description="DEFINE Edge Fusion Engine")
    parser.add_argument(
        "--enable-sapient", action="store_true", help="Enable live SAPIENT UDP stream"
    )
    parser.add_argument("--enable-cot", action="store_true", help="Enable live CoT UDP stream")
    parser.add_argument("--replay-file", type=str, help="Path to JSON scenario file")
    parser.add_argument(
        "--loop",
        action="store_true",
        help="When replaying a --replay-file, restart the replay continuously instead of "
        "exiting (turns a finite replay into a long-running feed).",
    )
    parser.add_argument(
        "--loop-delay",
        type=float,
        default=60.0,
        help="Seconds to wait between replay iterations when --loop is set (default: 60).",
    )
    parser.add_argument(
        "--realtime-factor",
        type=float,
        default=1.0,
        help="Speed at which a --replay-file is emitted, relative to the scenario's own "
        "timeline: 1.0 (default) replays at real time, 5 replays five times faster. "
        "0 disables pacing entirely and drains the file as fast as it can be fused.",
    )
    parser.add_argument(
        "--use-scenario-timestamps",
        action="store_true",
        help="Stamp replayed events with the timestamps in the file instead of shifting the "
        "scenario to start now. Recorded scenarios are then emitted with their original "
        "clock, which TAK clients may treat as too old or too far ahead to display.",
    )
    parser.add_argument(
        "--cot-stale-seconds",
        type=float,
        default=DEFAULT_STALE_SECONDS,
        help=f"How long a CoT marker stays live in TAK after the event it was built from "
        f"(default: {DEFAULT_STALE_SECONDS:g}). Raise it above the interval between a "
        "track's updates to stop markers expiring between them.",
    )
    parser.add_argument(
        "--log-to-file",
        action="store_true",
        help="Write fused CoT to 'fused_tracks_debug.xml' for offline validation",
    )
    parser.add_argument(
        "--tak-tls-host", help="TAK Server host to stream CoT to over TCP+TLS (enables the sink)"
    )
    parser.add_argument(
        "--tak-tls-port", type=int, default=8089, help="TAK Server TLS port (default 8089)"
    )
    parser.add_argument("--tak-tls-cert", help="Client certificate (PEM) for mutual TLS")
    parser.add_argument("--tak-tls-key", help="Client private key (PEM)")
    parser.add_argument("--tak-tls-ca", help="CA bundle (PEM) to verify the TAK Server")
    parser.add_argument(
        "--tak-ws-host",
        help="TAK Server host for the WebTAK streaming WebSocket sink (bearer-authenticated; "
        "CoT is tagged with the token's Keycloak groups). Enables the sink.",
    )
    parser.add_argument(
        "--tak-ws-port", type=int, default=8446, help="TAK WebTAK WebSocket port (default 8446)"
    )
    parser.add_argument(
        "--tak-ws-verify-tls",
        action="store_true",
        help="Verify the TAK Server TLS cert against the system trust store. A CA issued for a "
        "name the sink does not dial by (an in-cluster Service name) fails here; use "
        "--tak-ws-ca instead. Never affects the Keycloak token exchange, which is always "
        "verified.",
    )
    parser.add_argument(
        "--tak-ws-ca",
        help="CA bundle (PEM) used to verify the TAK Server on the WebSocket sink. Takes "
        "precedence over --tak-ws-verify-tls.",
    )
    parser.add_argument(
        "--tak-ws-token-timeout",
        type=float,
        default=DEFAULT_TOKEN_TIMEOUT,
        help=f"Seconds to wait on the Keycloak token endpoint (default: "
        f"{DEFAULT_TOKEN_TIMEOUT:g}). The fetch blocks the fusion loop, so the sensor sockets "
        "go unread for its duration.",
    )
    parser.add_argument(
        "--frame-window-seconds",
        type=float,
        default=DEFAULT_FRAME_WINDOW_SECONDS,
        help=f"How long a live source holds one sensor-instant open to collect the rest of its "
        f"datagrams before handing it on as a single event (default: "
        f"{DEFAULT_FRAME_WINDOW_SECONDS:g}). It must exceed both how spread out one sweep's "
        "datagrams arrive and how long one fusion pass takes, or sweeps split into "
        "single-detection events. Costs that much latency per live detection.",
    )
    parser.add_argument(
        "--keycloak-token-url",
        help="Keycloak token endpoint (.../protocol/openid-connect/token) for client_credentials",
    )
    parser.add_argument(
        "--oidc-client-id", help="Keycloak client id for the client_credentials grant"
    )
    parser.add_argument("--oidc-client-secret", help="Keycloak client secret (confidential client)")
    parser.add_argument(
        "--tak-bearer-token", help="Static bearer token, as an alternative to a Keycloak grant"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to sensor config JSON")
    args = parser.parse_args()

    if args.loop and not args.replay_file:
        logger.warning("--loop has no effect without --replay-file; live sources never terminate.")

    if args.replay_file and (args.enable_sapient or args.enable_cot):
        # A replay runs on the scenario's clock -- shifted to start now and then
        # running minutes or hours ahead of it, or (with
        # --use-scenario-timestamps) on whatever date the file was recorded.
        # Either way it does not share a timeline with a live sensor, and the
        # tracker cannot fuse two timelines: whichever is behind is discarded.
        # --loop is a further casualty, since the replay only restarts once
        # every source has run dry and a live one never does.
        parser.error(
            "--replay-file cannot be combined with --enable-sapient/--enable-cot: a replay "
            "does not share a timeline with live sensors. Run one or the other."
        )

    if args.realtime_factor < 0:
        parser.error("--realtime-factor must be >= 0 (0 disables pacing)")

    if args.realtime_factor != 1.0 and not args.replay_file:
        logger.warning(
            "--realtime-factor has no effect without --replay-file; live sources "
            "already arrive in real time."
        )

    if args.use_scenario_timestamps and not args.replay_file:
        logger.warning(
            "--use-scenario-timestamps has no effect without --replay-file; live sources "
            "are already stamped with the present."
        )

    # A negative sleep would raise; clamp once and use the local everywhere below.
    loop_delay = max(0.0, args.loop_delay)

    # Load the sensor network from the file
    try:
        config.load_sensor_network(sensor_config_path=args.config)
    except Exception as e:
        # Non-zero, or the container exits "Completed" and a misconfiguration
        # looks exactly like a clean finish to the runtime and to alerting.
        logger.error(f"Failed to load sensor config: {e}")
        raise SystemExit(1) from e

    # 2. Setup Sources
    sources = []
    if args.replay_file:
        replay_source = JsonSapientSource(Path(args.replay_file))
        # Offset first, pace second: the pacer sleeps against the intervals
        # between events, which the offset leaves untouched.
        if not args.use_scenario_timestamps:
            replay_source = OffsetReplaySource(replay_source)
        else:
            logger.info(f"Replaying {args.replay_file} with the scenario's own timestamps")
        if args.realtime_factor > 0:
            replay_source = RealtimeReplaySource(replay_source, factor=args.realtime_factor)
            logger.info(
                f"Replaying {args.replay_file} at {args.realtime_factor}x real time "
                "(--realtime-factor 0 to drain as fast as possible)"
            )
        else:
            logger.info(f"Replaying {args.replay_file} unpaced, as fast as events can be fused")
        sources.append(replay_source)
    if args.enable_sapient:
        sources.append(
            NetworkSapientStream(port=5000, frame_window_seconds=args.frame_window_seconds)
        )
    if args.enable_cot:
        sources.append(CotNetworkStream(port=6969, frame_window_seconds=args.frame_window_seconds))

    if not sources:
        logger.error("No sources enabled! Use --enable-sapient, --enable-cot, or --replay-file")
        raise SystemExit(1)

    if len(sources) > 1:
        # A live source blocks until its own next packet, so the loop below would
        # never reach the sources after it. Read them all concurrently instead.
        sources = [MultiplexedSource(sources)]
        logger.info("Reading all enabled sources concurrently")

    # 3. Initialize Core Components
    tracker = SapientAsynchronousTracker()
    augmentor = TacticalContextAugmentor()
    serializers = {
        "TAK": CotSerializer(stale_seconds=args.cot_stale_seconds),
        "SAPIENT": SapientSerializer(),
    }

    # Setup output sinks. At least one is required; file and TLS can be combined.
    sinks = []
    if args.log_to_file:
        sinks.append(FileCotSink("fused_tracks_debug.xml"))
        logger.info("Logging fused CoT to 'fused_tracks_debug.xml' for offline validation")
    if args.tak_tls_host:
        sinks.append(
            TakTlsSink(
                args.tak_tls_host,
                args.tak_tls_port,
                cert=args.tak_tls_cert,
                key=args.tak_tls_key,
                ca=args.tak_tls_ca,
            )
        )
        logger.info(f"Streaming CoT to TAK Server {args.tak_tls_host}:{args.tak_tls_port} over TLS")
    if args.tak_ws_host:
        sinks.append(
            TakWsSink(
                args.tak_ws_host,
                args.tak_ws_port,
                token_url=args.keycloak_token_url,
                client_id=args.oidc_client_id,
                client_secret=args.oidc_client_secret,
                static_token=args.tak_bearer_token,
                verify_tls=args.tak_ws_verify_tls,
                ca=args.tak_ws_ca,
                token_timeout=args.tak_ws_token_timeout,
            )
        )
        logger.info(
            f"Streaming CoT to TAK Server {args.tak_ws_host}:{args.tak_ws_port} "
            "over the WebTAK WebSocket (group-tagged by token)"
        )

    if not sinks:
        logger.error(
            "No output sink configured. Add --tak-tls-host <host> to stream CoT to a TAK Server "
            "over TLS, --tak-ws-host <host> for the group-tagged WebTAK WebSocket, "
            "or --log-to-file to write CoT to a file for offline validation."
        )
        raise SystemExit(1)

    _install_shutdown_handler()
    logger.info("Fusion loop started. Listening for targets...")

    # 4. The Main Loop
    latest_timestamp = None
    # Which sensor's event set the current watermark, so a drop report can name
    # the sensor that caused it and not only the one that suffered it.
    watermark_sensor = "no event accepted yet"
    counters = {"events": 0, "detections": 0, "cot": 0, "dropped": 0}
    next_skew_report = 0.0
    # Reported off a timer rather than from the loop below: an ingress that has
    # gone quiet is the case an operator most needs to see, and a loop-driven
    # report says nothing at all precisely then, so silence would have to be read
    # as either "no sensor is sending" or "the process is wedged".
    stop_summary = _start_summary_reporter(counters)
    # Only a live run has a present to be measured against; see the gate below.
    future_horizon = FUTURE_HORIZON_SECONDS if not args.replay_file else None
    try:
        while True:
            processed_any_events = False
            for source in sources:
                for timestamp, detections in source.iter_events():
                    processed_any_events = True

                    # Stone Soup cannot predict backwards: feeding it an event
                    # older than the last one rewinds every track's timestamp and
                    # re-broadcasts the lot with a CoT time that moves back, which
                    # TAK draws as markers jumping into the past -- for every
                    # track, not just the late sensor's. So events are gated to
                    # the highest timestamp seen so far.
                    #
                    # That makes the gate only as good as the highest timestamp,
                    # which is why a live event stamped in the future is refused
                    # the mark: one bad clock or spoofed datagram would otherwise
                    # set a watermark every healthy sensor then falls behind, and
                    # the engine would go quiet for good. A replay legitimately
                    # runs ahead of the wall clock, and cannot be combined with a
                    # live source, so the horizon only applies to live runs.
                    skew = None
                    now = None
                    if future_horizon is not None:
                        now = datetime.now(timezone.utc)
                        ahead = (timestamp - now).total_seconds()
                        if ahead > future_horizon:
                            skew = f"{ahead:.1f}s ahead of the present"
                    if skew is None and latest_timestamp is not None:
                        behind = (latest_timestamp - timestamp).total_seconds()
                        if behind > LATE_EVENT_TOLERANCE_SECONDS:
                            skew = f"{behind:.1f}s behind the newest event seen"

                    if skew is not None:
                        counters["dropped"] += 1
                        if time.monotonic() >= next_skew_report:
                            next_skew_report = time.monotonic() + LATE_EVENT_REPORT_SECONDS
                            logger.warning(
                                "Dropped %d event(s) whose timestamps are out of step with the "
                                "rest; the latest is %s, from %s, against a mark set by %s. "
                                "Fusing it would drag every track's time with it. Check the "
                                "sensors' clocks.",
                                counters["dropped"],
                                skew,
                                _sensor_ids(detections),
                                watermark_sensor,
                            )
                        continue

                    # The event is fused with its own timestamp, but the watermark
                    # it sets is clamped to the present on a live run: a sensor a
                    # few seconds fast is inside the future horizon, and letting it
                    # mark the future would put the gate ahead of what the healthy
                    # sensors report, so their events -- not its -- get dropped.
                    mark = timestamp if now is None else min(timestamp, now)
                    if latest_timestamp is None or mark > latest_timestamp:
                        latest_timestamp = mark
                        watermark_sensor = _sensor_ids(detections)

                    counters["events"] += 1
                    counters["detections"] += len(detections)
                    active_tracks = tracker.process_async_event(timestamp, set(detections))
                    broadcast = 0

                    # Serialization / Output
                    for track in active_tracks:
                        # ONLY broadcast if this track was updated during this specific event
                        # timestamp; this prevents re-broadcasting tracks that haven't changed
                        if track.state.timestamp == timestamp:
                            tactical_track = augmentor.extract_tactical_track(track)
                            cot_payload = serializers["TAK"].serialize(tactical_track)

                            for sink in sinks:
                                sink.send(cot_payload)

                            counters["cot"] += 1
                            broadcast += 1
                            # Per track per event: 20 lines a second on a busy
                            # picture, which buries the drop warnings. The
                            # per-event line below carries the same information at
                            # a rate an operator can read.
                            logger.debug(
                                f"Broadcast Update for Track {tactical_track.track_id[-4:]} | Threat: {tactical_track.threat_level.upper()}"
                            )

                    logger.info(
                        "Fused event at %s: %d detection(s) in, %d track(s) broadcast.",
                        timestamp.isoformat(timespec="seconds"),
                        len(detections),
                        broadcast,
                    )

            if processed_any_events:
                continue

            # Every source ran dry on this pass. A live source's read loop
            # swallows its own errors and never returns, so reaching this means
            # something unforeseen took the ingress down; re-polling would spin.
            # Failing loudly beats a container that looks healthy and reads
            # nothing.
            if args.enable_sapient or args.enable_cot:
                logger.error("Every live source ended; nothing left to read. Exiting.")
                raise SystemExit(1)

            # Exit logic for replay files
            if args.replay_file:
                if args.loop:
                    logger.info(f"Replay drained. Looping again in {loop_delay}s.")
                    for source in sources:
                        source.reset()
                    # A restart is a new scenario, not a continuation, so track state must
                    # not carry over. Required outright under --use-scenario-timestamps:
                    # timestamps jump backwards there and Stone Soup cannot predict
                    # backwards.
                    tracker = SapientAsynchronousTracker()
                    # The new iteration re-anchors the clock, so the previous
                    # one's timestamps must not gate it.
                    latest_timestamp = None
                    time.sleep(loop_delay)
                    continue
                logger.info("Replay file processing complete. Exiting.")
                break
    finally:
        if counters["dropped"]:
            # The periodic report can end mid-window, so a short run would
            # otherwise finish under-reporting what it dropped.
            logger.warning(
                "Dropped %d event(s) in total for timestamps out of step with the rest.",
                counters["dropped"],
            )
        stop_summary.set()
        _log_summary(counters, "Fusion totals at shutdown")
        for sink in sinks:
            sink.close()


if __name__ == "__main__":
    fusion_main()
