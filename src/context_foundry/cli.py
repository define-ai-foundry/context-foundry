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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FusionEngine")

def fusion_main():
    parser = argparse.ArgumentParser(description="DEFINE Edge Fusion Engine")
    
    # 1. Multi-protocol support
    parser.add_argument("--enable-sapient", action="store_true", help="Enable live SAPIENT UDP stream")
    parser.add_argument("--enable-cot", action="store_true", help="Enable live CoT UDP stream")
    parser.add_argument("--replay-file", type=str, help="Path to JSON scenario file")
    
    parser.add_argument("--tak-ip", default="239.2.3.1")
    parser.add_argument("--tak-port", type=int, default=6969)
    args = parser.parse_args()

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

    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    udp_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    
    logger.info(f"TAK Multicast Broadcast active on {args.tak_ip}:{args.tak_port}")
    logger.info("Fusion loop started. Listening for targets...")

    # 4. The Main Loop
    while True:
        for source in sources:
            # Note: For non-blocking/concurrent reading, you would typically 
            # use asyncio or threading here. 
            for timestamp, detections in source.iter_events():
                # Corrected Loop: Pass the set of detections for JPDA
                active_tracks = tracker.process_async_event(timestamp, set(detections))
                
                # Serialization / Output
                for track in active_tracks:
                    tactical_track = augmentor.extract_tactical_track(track)
                    
                    # Push Cursor on Target (CoT) to TAK
                    cot_payload = serializers["TAK"].serialize(tactical_track)
                    udp_sock.sendto(cot_payload.encode('utf-8'), (args.tak_ip, args.tak_port))
                    
                    logger.info(f"Broadcast Fused Track {tactical_track.track_id[-4:]} | Threat: {tactical_track.threat_level.upper()}")

if __name__ == "__main__":
    fusion_main()