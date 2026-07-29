# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import time
from pathlib import Path

from context_foundry.fusion import config
from context_foundry.fusion.augmentor import TacticalContextAugmentor
from context_foundry.fusion.serializers import CotSerializer, SapientSerializer
from context_foundry.fusion.sinks.file import FileCotSink
from context_foundry.fusion.sinks.tak_tls import TakTlsSink
from context_foundry.fusion.sinks.tak_ws import TakWsSink
from context_foundry.fusion.sources.cot_stream import CotNetworkStream
from context_foundry.fusion.sources.json_file import JsonSapientSource
from context_foundry.fusion.sources.paced import RealtimeReplaySource
from context_foundry.fusion.sources.stream import NetworkSapientStream
from context_foundry.fusion.tracker import SapientAsynchronousTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FusionEngine")


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

    if args.realtime_factor < 0:
        parser.error("--realtime-factor must be >= 0 (0 disables pacing)")

    if args.realtime_factor != 1.0 and not args.replay_file:
        logger.warning(
            "--realtime-factor has no effect without --replay-file; live sources "
            "already arrive in real time."
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

    # 3. Initialize Core Components
    tracker = SapientAsynchronousTracker()
    augmentor = TacticalContextAugmentor()
    serializers = {"TAK": CotSerializer(), "SAPIENT": SapientSerializer()}

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

    logger.info("Fusion loop started. Listening for targets...")

    # 4. The Main Loop
    try:
        while True:
            processed_any_events = False
            for source in sources:
                for timestamp, detections in source.iter_events():
                    processed_any_events = True
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

            # Exit logic for replay files
            if (
                args.replay_file
                and not args.enable_sapient
                and not args.enable_cot
                and not processed_any_events
            ):
                if args.loop:
                    logger.info(f"Replay drained. Looping again in {loop_delay}s.")
                    for source in sources:
                        source.reset()
                    # Replay timestamps jump backwards on restart and Stone Soup cannot
                    # predict backwards, so a fresh tracker per pass is required.
                    tracker = SapientAsynchronousTracker()
                    time.sleep(loop_delay)
                    continue
                logger.info("Replay file processing complete. Exiting.")
                break
    finally:
        for sink in sinks:
            sink.close()


if __name__ == "__main__":
    fusion_main()
