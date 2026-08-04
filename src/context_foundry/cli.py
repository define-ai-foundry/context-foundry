# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import signal
import time
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
from context_foundry.fusion.sinks.tak_ws import TakWsSink
from context_foundry.fusion.sources.cot_stream import CotNetworkStream
from context_foundry.fusion.sources.json_file import JsonSapientSource
from context_foundry.fusion.sources.multiplex import MultiplexedSource
from context_foundry.fusion.sources.offset import OffsetReplaySource
from context_foundry.fusion.sources.paced import RealtimeReplaySource
from context_foundry.fusion.sources.stream import NetworkSapientStream
from context_foundry.fusion.tracker import SapientAsynchronousTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FusionEngine")

# How often to report events dropped for arriving out of order. A skewed sensor
# clock produces one per packet, so this is a rate limit, not a one-shot.
LATE_EVENT_REPORT_SECONDS = 60.0


def _sensor_ids(detections):
    """The sensors an event came from, for reporting which one is misbehaving."""
    ids = {getattr(det, "metadata", {}).get("nodeId") for det in detections}
    return ", ".join(sorted(str(node_id) for node_id in ids if node_id)) or "an unnamed sensor"


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
        help="Verify the TAK Server TLS cert (default: skip, for self-signed dev)",
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
        logger.error(f"Failed to load sensor config: {e}")
        return

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
        sources.append(NetworkSapientStream(port=5000))
    if args.enable_cot:
        sources.append(CotNetworkStream(port=6969))

    if not sources:
        logger.error("No sources enabled! Use --enable-sapient, --enable-cot, or --replay-file")
        return

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
        return

    _install_shutdown_handler()
    logger.info("Fusion loop started. Listening for targets...")

    # 4. The Main Loop
    latest_timestamp = None
    late_events = 0
    next_late_report = 0.0
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
                    # track, not just the late sensor's. UDP reordering produces
                    # the odd one; a sensor whose clock is behind produces nothing
                    # but, and is effectively out of the fusion picture until its
                    # clock is fixed, so keep saying so.
                    if latest_timestamp is not None and timestamp < latest_timestamp:
                        late_events += 1
                        if time.monotonic() >= next_late_report:
                            next_late_report = time.monotonic() + LATE_EVENT_REPORT_SECONDS
                            logger.warning(
                                "Dropped %d event(s) stamped before the last one processed; "
                                "the latest is %.1fs behind, from %s. Its sensor's clock is "
                                "out of step and its detections are not being fused.",
                                late_events,
                                (latest_timestamp - timestamp).total_seconds(),
                                _sensor_ids(detections),
                            )
                        continue
                    latest_timestamp = timestamp

                    active_tracks = tracker.process_async_event(timestamp, set(detections))

                    # Serialization / Output
                    for track in active_tracks:
                        # ONLY broadcast if this track was updated during this specific event
                        # timestamp; this prevents re-broadcasting tracks that haven't changed
                        if track.state.timestamp == timestamp:
                            tactical_track = augmentor.extract_tactical_track(track)
                            cot_payload = serializers["TAK"].serialize(tactical_track)

                            for sink in sinks:
                                sink.send(cot_payload)

                            logger.info(
                                f"Broadcast Update for Track {tactical_track.track_id[-4:]} | Threat: {tactical_track.threat_level.upper()}"
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
        for sink in sinks:
            sink.close()


if __name__ == "__main__":
    fusion_main()
