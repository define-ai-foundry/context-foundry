# Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import socket
from pathlib import Path

# Absolute imports work perfectly now because the package will be installed!
from context_foundry.fusion.sources.json_file import JsonSapientSource
from context_foundry.fusion.sources.stream import NetworkSapientStream
from context_foundry.fusion.tracker import SapientAsynchronousTracker
from context_foundry.fusion.augmentor import TacticalContextAugmentor
from context_foundry.fusion.serializers import CotSerializer, SapientSerializer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FusionEngine")

def fusion_main():
    """Entry point for the context-foundry-fusion command."""
    parser = argparse.ArgumentParser(description="DEFINE Edge Fusion Engine")
    parser.add_argument("--mode", choices=["replay", "live"], default="replay", help="Operation mode")
    parser.add_argument("--file", type=str, default="data/examples/sapient_messages.json", help="Path to JSON scenario")
    parser.add_argument("--tak-ip", type=str, default="239.2.3.1", help="TAK Multicast IP")
    parser.add_argument("--tak-port", type=int, default=6969, help="TAK Multicast Port")
    args = parser.parse_args()

    # 1. Initialize Source based on mode
    if args.mode == "replay":
        scenario_path = Path(args.file)
        if not scenario_path.exists():
            logger.error(f"Scenario file not found: {scenario_path}")
            return
        logger.info(f"Starting in REPLAY mode using {scenario_path}")
        source = JsonSapientSource(scenario_path)
    else:
        logger.info("Starting in LIVE network mode on port 5000.")
        source = NetworkSapientStream(ip="0.0.0.0", port=5000)

    # 2. Initialize Core Components
    tracker = SapientAsynchronousTracker()
    augmentor = TacticalContextAugmentor()
    serializers = {
        "TAK": CotSerializer(),
        "SAPIENT": SapientSerializer()
    }

    # 3. Setup Network
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    udp_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    
    logger.info(f"TAK Multicast Broadcast active on {args.tak_ip}:{args.tak_port}")
    logger.info("Fusion loop started. Listening for targets...")

    # 4. The Main Loop
    for timestamp, detections in source.iter_events():
        for det in detections:
            active_tracks = tracker.process_async_event(det)
            
            for track in active_tracks:
                state = augmentor.extract_tactical_state(track)
                
                # Push Cursor on Target (CoT) to TAK
                cot_payload = serializers["TAK"].serialize(state)
                udp_sock.sendto(cot_payload.encode('utf-8'), (args.tak_ip, args.tak_port))
                
                logger.info(f"Broadcast Fused Track {state.track_id[-4:]} | Threat: {state.threat_level.upper()}")

if __name__ == "__main__":
    fusion_main()