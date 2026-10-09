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
from context_foundry.fusion.engine import FusionEngine, sensor_ids
from context_foundry.fusion.publish import (
    DEFAULT_COAST_SECONDS,
    DEFAULT_WINDOW_SECONDS,
    POLICIES,
    PublishPolicy,
    is_predicted,
    recent_sources,
)
from context_foundry.fusion.serializers import (
    DEFAULT_STALE_SECONDS,
    CotSerializer,
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
from context_foundry.fusion.tracker import (
    CONFIRM_WINDOW_SECONDS,
    DEFAULT_CONFIRM_HITS,
    MAX_COAST_SECONDS,
    MAX_EVENT_DETECTIONS,
    MAX_LIVE_TRACKS,
    MAX_TRACK_HISTORY,
    SapientAsynchronousTracker,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FusionEngine")

# How often to report events dropped for arriving out of step. A skewed sensor
# clock produces one per packet, so this is a rate limit, not a one-shot.
LATE_EVENT_REPORT_SECONDS = 60.0

# How often to log the running fusion summary. Without it an engine that is up
# and fusing nothing logs the same lines as a healthy one and then nothing at
# all, so there is no way to tell a quiet sensor network from a broken ingress.
SUMMARY_REPORT_SECONDS = 60.0

# Where Linux publishes per-socket receive-drop counters. Both families are read
# because a source binds one socket and the operator does not necessarily know
# which table it lands in.
PROC_UDP_FILES = ("/proc/net/udp", "/proc/net/udp6")


def _kernel_udp_drops(ports):
    """Packets the kernel discarded per bound local port, as {port: drops}.

    This is the loss the engine's own counters cannot see. Under sustained
    overload the sockets fill and the kernel drops datagrams before anything
    reads them, so every in-process counter stays consistent while most of the
    feed is gone.

    Only read once per summary, never per event: it re-reads the whole table.

    Ports with no counter are left out and a table that cannot be read yields
    nothing, so a host without /proc/net/udp -- anything that is not Linux --
    degrades to a summary without this clause rather than to an error. The two
    fields wanted are taken from the ends of each row after checking the header
    names them there, because the columns are not one per header name:
    tx_queue:rx_queue and tr:tm->when are each printed as one colon-joined field,
    so counting header columns finds the wrong values.
    """
    wanted = set(ports)
    drops = {}
    if not wanted:
        return drops
    for path in PROC_UDP_FILES:
        try:
            lines = Path(path).read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        if not lines:
            continue
        header = lines[0].split()
        if len(header) < 3 or header[1] != "local_address" or header[-1] != "drops":
            continue
        for line in lines[1:]:
            fields = line.split()
            if len(fields) < 3:
                continue
            try:
                port = int(fields[1].rsplit(":", 1)[1], 16)
                count = int(fields[-1])
            except (IndexError, ValueError):
                continue
            if port in wanted:
                # One port can appear more than once (several bound sockets).
                drops[port] = drops.get(port, 0) + count
    return drops


def _start_summary_reporter(counters, live_ports=()):
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
            _log_summary(counters, "Fusion summary", live_ports)

    threading.Thread(target=report, name="summary-reporter", daemon=True).start()
    return stop


def _log_summary(counters, prefix, live_ports=()):
    """Report the running totals an operator needs to place the engine's state.

    All five failure modes read differently here: nothing arriving leaves every
    counter at zero, arriving-and-dropped moves only the drop count, healthy
    fusion moves all of them, a TAK server refusing the stream shows CoT still
    being sent while the sink logs its own connection warnings -- and a feed being
    lost in the kernel before ingest moves only the socket drop counts, which
    every in-process counter is blind to.
    """
    drops = _kernel_udp_drops(live_ports)
    per_port = ", ".join(f"UDP {port}: {count}" for port, count in sorted(drops.items()))
    sockets = f" Discarded by the kernel before ingest -- {per_port}." if per_port else ""
    logger.info(
        "%s: %d event(s) fused, %d detection(s) ingested, %d CoT message(s) sent, "
        "%d event(s) dropped out of step.%s",
        prefix,
        counters["events"],
        counters["detections"],
        counters["cot"],
        counters["dropped"],
        sockets,
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
        "--kafka-topic",
        help="Read sapient-raw records from this Kafka topic (the records sapient-ingest "
        "writes). Fused on the records' own event_time, like a replay.",
    )
    parser.add_argument(
        "--kafka-bootstrap",
        default="localhost:9092",
        help="Kafka bootstrap servers for --kafka-topic (default: localhost:9092)",
    )
    parser.add_argument(
        "--kafka-group",
        default="context-foundry-fusion",
        help="Kafka consumer group for --kafka-topic (default: context-foundry-fusion)",
    )
    parser.add_argument(
        "--kafka-output-topic",
        help="Publish each fused track as a sapient-raw record to this Kafka topic "
        "(enables the sink). Uses --kafka-bootstrap and --kafka-security-protocol.",
    )
    parser.add_argument(
        "--publish",
        choices=POLICIES,
        default="window",
        help="Which fused tracks each event sends to every output: 'window' (the default: only "
        "states a sensor report made, at most one per track per --publish-window-seconds, so "
        "fewer messages go out than reports come in and no prediction is sent), 'all' (every "
        "track the event moved, predictions included), 'updates' (every state a report made), "
        "'coast' (updates, plus a predicted state for a track not sent for --coast-seconds) or "
        "'predicted' (only predicted states; for research).",
    )
    parser.add_argument(
        "--publish-window-seconds",
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
        help=f"With --publish window, the shortest time between two messages for one track "
        f"(default: {DEFAULT_WINDOW_SECONDS:g}). Longer means fewer messages and older positions.",
    )
    parser.add_argument(
        "--coast-seconds",
        type=float,
        default=DEFAULT_COAST_SECONDS,
        help=f"With --publish coast, how long a track may go unpublished before a predicted "
        f"state is sent (default: {DEFAULT_COAST_SECONDS:g}).",
    )
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="List the sensor reports behind each published track (associated_detection): "
        "each sensor and sensor object id in the track's retained history, newest sighting.",
    )
    parser.add_argument(
        "--label-predicted",
        action="store_true",
        help="Mark each published track as measured or predicted (object_info fusionUpdate).",
    )
    parser.add_argument(
        "--pipeline-id",
        default="local",
        help="pipeline_id header on published records (default: local)",
    )
    parser.add_argument(
        "--producer-stage",
        default="fusion",
        help="producer_stage header on published records (default: fusion)",
    )
    parser.add_argument(
        "--kafka-security-protocol",
        default="PLAINTEXT",
        help="Kafka security.protocol for --kafka-topic (default: PLAINTEXT)",
    )
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
        "--max-live-tracks",
        type=int,
        default=MAX_LIVE_TRACKS,
        help=f"Live tracks held at once; unassociated detections stop starting new ones past it "
        f"(default: {MAX_LIVE_TRACKS}). The default is where one fusion pass reaches the frame "
        "window on one core, so raising it needs --frame-window-seconds raised with it or the "
        "engine falls behind the feed and never recovers.",
    )
    parser.add_argument(
        "--max-track-history",
        type=int,
        default=MAX_TRACK_HISTORY,
        help=f"States retained per track (default: {MAX_TRACK_HISTORY}). Nothing downstream reads "
        "more than the newest, so this is memory: lower it to afford more live tracks.",
    )
    parser.add_argument(
        "--max-coast-seconds",
        type=float,
        default=MAX_COAST_SECONDS,
        help=f"How long a track may coast on prediction alone, without a detection, before it is "
        f"dropped (default: {MAX_COAST_SECONDS:g}). Size it against the revisit interval of the "
        "sensors that see a track: it has to exceed one revisit, and two or three of them to ride "
        "out sweeps that miss, or tracks expire between sweeps and every sweep re-initiates them. "
        "Every second above that is a track whose uncertainty -- and so whose validation gate -- "
        "keeps widening, until it takes detections away from tracks that still know where their "
        "target is; such tracks are dropped on uncertainty instead, with a warning.",
    )
    parser.add_argument(
        "--max-event-detections",
        type=int,
        default=MAX_EVENT_DETECTIONS,
        help=f"Detections a single event is associated with; the rest are shed, with a warning "
        f"(default: {MAX_EVENT_DETECTIONS}). With --max-live-tracks this is what bounds one "
        "fusion pass: cost is the product of the two. The default is nearly four times the "
        "densest instant the reference network produces, so reaching it means a sensor is "
        "reporting more than one sweep at a time -- raise it only with the frame window.",
    )
    parser.add_argument(
        "--confirm-hits",
        type=int,
        default=DEFAULT_CONFIRM_HITS,
        help=f"Detections a new track needs within {CONFIRM_WINDOW_SECONDS:g} s before it is "
        f"confirmed and published (default: {DEFAULT_CONFIRM_HITS}). A track that does not "
        "confirm in time is dropped, so a single stray report never becomes a track. 1 turns "
        "confirmation off.",
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
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a sensor config JSON. Optional: without one, sensors are learned from "
        "their SAPIENT Registration and status reports (Kafka input), and a declared value "
        "always wins over the file's.",
    )
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

    if args.kafka_topic and (args.replay_file or args.enable_sapient or args.enable_cot):
        # Records on the topic carry the sensors' own event_time, which need not be
        # the present (a backlog, or a replayed scenario), so they share a timeline
        # with neither a replay file nor a live socket.
        parser.error(
            "--kafka-topic cannot be combined with --replay-file/--enable-sapient/--enable-cot: "
            "they do not share a timeline. Run one or the other."
        )

    if args.realtime_factor < 0:
        parser.error("--realtime-factor must be >= 0 (0 disables pacing)")

    # A window of zero or less expires every frame the moment it opens, so each
    # datagram becomes its own single-detection event: the per-datagram
    # degradation frames exist to remove, and with no error to show for it.
    if args.frame_window_seconds <= 0:
        parser.error("--frame-window-seconds must be > 0; a sweep needs a window to assemble in")

    # Zero live tracks fuses nothing, and zero retained states leaves a track with
    # no current state for the augmentor to read.
    if args.max_live_tracks < 1:
        parser.error("--max-live-tracks must be >= 1")

    if args.max_track_history < 1:
        parser.error("--max-track-history must be >= 1")

    # A track that cannot coast at all is dropped the moment the sweep that made
    # it ends, so every sweep re-initiates the whole picture and nothing is ever
    # fused across sensors.
    if args.max_coast_seconds <= 0:
        parser.error("--max-coast-seconds must be > 0; a track has to survive between sweeps")

    # Zero detections an event may carry sheds every sweep in full.
    if args.coast_seconds <= 0:
        parser.error("--coast-seconds must be > 0")

    if args.publish_window_seconds <= 0:
        parser.error("--publish-window-seconds must be > 0")

    if args.confirm_hits < 1:
        parser.error("--confirm-hits must be >= 1")

    if args.max_event_detections < 1:
        parser.error("--max-event-detections must be >= 1")

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

    # Load the sensor network from the file, when there is one
    if args.config:
        try:
            config.load_sensor_network(sensor_config_path=args.config)
        except Exception as e:
            # Non-zero, or the container exits "Completed" and a misconfiguration
            # looks exactly like a clean finish to the runtime and to alerting.
            logger.error(f"Failed to load sensor config: {e}")
            raise SystemExit(1) from e
    else:
        logger.info("No sensor config: sensors are learned from Registration and status reports")

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
    if args.kafka_topic:
        from context_foundry.fusion.sapient_raw import fusion_node_id
        from context_foundry.fusion.sources.kafka import KafkaSapientSource

        sources.append(
            KafkaSapientSource(
                args.kafka_bootstrap,
                args.kafka_topic,
                args.kafka_group,
                security_protocol=args.kafka_security_protocol,
                # Fused tracks routed back to the input are not fused again.
                own_node_id=fusion_node_id(args.pipeline_id, args.producer_stage),
            )
        )
    if args.enable_sapient:
        sources.append(
            NetworkSapientStream(port=5000, frame_window_seconds=args.frame_window_seconds)
        )
    if args.enable_cot:
        sources.append(CotNetworkStream(port=6969, frame_window_seconds=args.frame_window_seconds))

    if not sources:
        logger.error(
            "No sources enabled! Use --enable-sapient, --enable-cot, --replay-file or --kafka-topic"
        )
        raise SystemExit(1)

    kafka_source = next((s for s in sources if hasattr(s, "consumer")), None)

    # The sockets whose kernel drop counters the summary reports. Each live source
    # knows the port it bound; a replay source has none. Collected before the
    # multiplexer wraps them out of reach.
    live_ports = sorted({port for port in (getattr(s, "port", None) for s in sources) if port})

    if len(sources) > 1:
        # A live source blocks until its own next packet, so the loop below would
        # never reach the sources after it. Read them all concurrently instead.
        sources = [MultiplexedSource(sources)]
        logger.info("Reading all enabled sources concurrently")

    if live_ports and not _kernel_udp_drops(live_ports):
        # Said once, at startup: the summary then omits the socket counts, and an
        # operator reading it would otherwise take their absence for zero loss.
        logger.warning(
            "No kernel receive-drop counter found for UDP %s; the summary cannot report packets "
            "discarded before ingest (read from %s, so Linux only).",
            ", ".join(str(port) for port in live_ports),
            PROC_UDP_FILES[0],
        )

    # 3. Initialize Core Components
    tracker = SapientAsynchronousTracker(
        max_live_tracks=args.max_live_tracks,
        max_track_history=args.max_track_history,
        max_coast_seconds=args.max_coast_seconds,
        max_event_detections=args.max_event_detections,
        confirm_hits=args.confirm_hits,
    )
    augmentor = TacticalContextAugmentor()
    serializers = {
        "TAK": CotSerializer(stale_seconds=args.cot_stale_seconds),
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
        try:
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
        except ValueError as e:
            # Reported and exited like every other startup misconfiguration here;
            # unhandled it reaches the operator as a traceback.
            logger.error(
                "Cannot authenticate to the WebTAK WebSocket: %s. Supply --keycloak-token-url "
                "and --oidc-client-id (plus --oidc-client-secret for a confidential client), "
                "or --tak-bearer-token.",
                e,
            )
            raise SystemExit(1) from e
        logger.info(
            f"Streaming CoT to TAK Server {args.tak_ws_host}:{args.tak_ws_port} "
            "over the WebTAK WebSocket (group-tagged by token)"
        )

    kafka_sink = None
    publish_policy = PublishPolicy(args.publish, args.coast_seconds, args.publish_window_seconds)
    if args.kafka_output_topic:
        from context_foundry.fusion.sinks.kafka import KafkaSapientRawSink

        kafka_sink = KafkaSapientRawSink(
            args.kafka_bootstrap,
            args.kafka_output_topic,
            args.pipeline_id,
            args.producer_stage,
            security_protocol=args.kafka_security_protocol,
        )

    if not sinks and kafka_sink is None:
        logger.error(
            "No output sink configured. Add --tak-tls-host <host> to stream CoT to a TAK Server "
            "over TLS, --tak-ws-host <host> for the group-tagged WebTAK WebSocket, "
            "--kafka-output-topic <topic> to publish fused tracks as sapient-raw, "
            "or --log-to-file to write CoT to a file for offline validation."
        )
        raise SystemExit(1)

    _install_shutdown_handler()
    logger.info("Fusion loop started. Listening for targets...")

    # 4. The Main Loop
    engine = FusionEngine(tracker, publish_policy, augmentor)
    counters = {"events": 0, "detections": 0, "cot": 0, "dropped": 0}
    next_skew_report = 0.0
    # Reported off a timer rather than from the loop below: an ingress that has
    # gone quiet is the case an operator most needs to see, and a loop-driven
    # report says nothing at all precisely then, so silence would have to be read
    # as either "no sensor is sending" or "the process is wedged".
    stop_summary = _start_summary_reporter(counters, live_ports)
    # Only a live run has a present to be measured against. A replay legitimately
    # runs ahead of the wall clock, and cannot be combined with a live source; Kafka
    # records keep their sensors' event_time, which a backlog or a replayed scenario
    # leaves far from the present. Both are gated on their own timestamps alone.
    live = not (args.replay_file or args.kafka_topic)
    try:
        while True:
            processed_any_events = False
            for source in sources:
                for timestamp, detections in source.iter_events():
                    processed_any_events = True
                    now = datetime.now(timezone.utc) if live else None
                    result = engine.process(timestamp, detections, now)

                    if result.dropped:
                        counters["dropped"] += 1
                        if time.monotonic() >= next_skew_report:
                            next_skew_report = time.monotonic() + LATE_EVENT_REPORT_SECONDS
                            logger.warning(
                                "Dropped %d event(s) whose timestamps are out of step with the "
                                "rest; the latest is %s, from %s, against a mark set by %s. "
                                "Fusing it would drag every track's time with it. Check the "
                                "sensors' clocks.",
                                counters["dropped"],
                                result.skew,
                                sensor_ids(detections),
                                engine.watermark_sensor,
                            )
                        continue

                    counters["events"] += 1
                    counters["detections"] += len(detections)

                    # Serialization / Output
                    for track, tactical_track in result.published:
                        cot_payload = serializers["TAK"].serialize(tactical_track)
                        for sink in sinks:
                            sink.send(cot_payload)
                        if kafka_sink is not None:
                            kafka_sink.send_track(
                                tactical_track,
                                track.metadata,
                                predicted=is_predicted(track) if args.label_predicted else None,
                                sources=recent_sources(track) if args.list_sources else None,
                            )
                        counters["cot"] += 1
                        # Per track per event: 20 lines a second on a busy picture,
                        # which buries the drop warnings. The per-event line below
                        # carries the same information at a rate an operator can read.
                        logger.debug(
                            f"Broadcast Update for Track {tactical_track.track_id[-4:]} | Threat: {tactical_track.threat_level.upper()}"
                        )

                    logger.info(
                        "Fused event at %s: %d detection(s) in, %d track(s) broadcast.",
                        result.mark.isoformat(timespec="seconds"),
                        len(detections),
                        len(result.published),
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
                    engine.reset(
                        SapientAsynchronousTracker(
                            max_live_tracks=args.max_live_tracks,
                            max_track_history=args.max_track_history,
                            max_coast_seconds=args.max_coast_seconds,
                            max_event_detections=args.max_event_detections,
                            confirm_hits=args.confirm_hits,
                        )
                    )
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
        _log_summary(counters, "Fusion totals at shutdown", live_ports)
        for sink in sinks:
            sink.close()
        if kafka_sink is not None:
            kafka_sink.close()
        if kafka_source is not None:
            kafka_source.close()


if __name__ == "__main__":
    fusion_main()
