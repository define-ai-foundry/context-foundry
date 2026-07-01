# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import socket
from pathlib import Path
from context_foundry.fusion.sources.json_file import JsonSapientSource
from context_foundry.fusion.sources.stream import NetworkSapientStream
from context_foundry.fusion.sources.cot_stream import CotNetworkStream
from context_foundry.fusion.tracker import SapientAsynchronousTracker
from context_foundry.fusion.augmentor import TacticalContextAugmentor
from context_foundry.fusion.serializers import CotSerializer, SapientSerializer
from context_foundry.fusion import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FusionEngine")

def fusion_main():
    # 1. Parse Command-Line Arguments
    parser = argparse.ArgumentParser(description="DEFINE Edge Fusion Engine")
    parser.add_argument("--enable-sapient", action="store_true", help="Enable live SAPIENT UDP stream")
    parser.add_argument("--enable-cot", action="store_true", help="Enable live CoT UDP stream")
    parser.add_argument("--replay-file", type=str, help="Path to JSON scenario file")
    parser.add_argument("--log-to-file", action="store_true", help="Log CoT payloads to 'fused_tracks_debug.xml' instead of network broadcast")
    parser.add_argument("--tak-ip", default="239.2.3.1")
    parser.add_argument("--tak-port", type=int, default=6969)
    parser.add_argument("--config", type=str, required=True, help="Path to sensor config JSON")
    args = parser.parse_args()

    # Load the sensor network from the file
    try:
        config.load_sensor_network(sensor_config_path=args.config)
    except Exception as e:
        logger.error(f"Failed to load sensor config: {e}")
        return
    
    # 2. Setup Sources
    sources = []
    if args.replay_file:
        sources.append(JsonSapientSource(Path(args.replay_file)))
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

    # Setup UDP Socket only if not logging to file
    udp_sock = None
    if not args.log_to_file:
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        udp_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        logger.info(f"TAK Multicast Broadcast active on {args.tak_ip}:{args.tak_port}")
    else:
        logger.info("Validation Mode: Logging fused tracks to 'fused_tracks_debug.xml'")

    logger.info("Fusion loop started. Listening for targets...")

    # 4. The Main Loop
    while True:
            processed_any_events = False
            for source in sources:
                for timestamp, detections in source.iter_events():
                    processed_any_events = True
                    active_tracks = tracker.process_async_event(timestamp, set(detections))
                    
                    # Serialization / Output
                    for track in active_tracks:
                        # ONLY broadcast if this track was updated during this specific event timestamp
                        # This prevents re-broadcasting tracks that haven't changed
                        if track.state.timestamp == timestamp:
                            tactical_track = augmentor.extract_tactical_track(track)
                            cot_payload = serializers["TAK"].serialize(tactical_track)
                            
                            if args.log_to_file:
                                with open("fused_tracks_debug.xml", "a") as f:
                                    f.write(cot_payload + "\n")
                            else:
                                try:
                                    if udp_sock:
                                        udp_sock.sendto(cot_payload.encode('utf-8'), (args.tak_ip, args.tak_port))
                                except OSError as e:
                                    logger.error(f"Network error: {e}")
                            
                            logger.info(f"Broadcast Update for Track {tactical_track.track_id[-4:]} | Threat: {tactical_track.threat_level.upper()}")

            # Exit logic for replay files
            if args.replay_file and not args.enable_sapient and not args.enable_cot:
                if not processed_any_events:
                    logger.info("Replay file processing complete. Exiting.")
                    break


if __name__ == "__main__":
    fusion_main()