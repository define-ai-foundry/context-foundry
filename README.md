# DEFINE Edge Fusion Engine (Context Foundry)
<p align="center">
  <img src="docs/images/winter_swarm_header.jpg" align="right" width="350" alt="Drone swarm over Finnish forest">
</p>

High-performance, format-agnostic Tactical Edge Sensor Fusion Gateway

The DEFINE Edge Fusion Engine is a modular tactical sensor fusion gateway designed for edge operations. It ingests heterogeneous sensor telemetry from radars, acoustic arrays, and EO/IR systems, performs real-time multi-sensor track fusion using advanced mathematical filtering, and delivers operationally relevant situational awareness directly to mission users and C2 systems.

## Core Capabilities

### Universal C2 Gateway Architecture

Format-agnostic architecture with decoupled ingestion, fusion, and dissemination pipelines, enabling rapid integration of new sensors, protocols, and C2 ecosystems.

### Dual-Mode Execution

Supports both:

- Scenario Replay (```--mode replay```) — analysis of recorded JSON-based sensor scenarios.
- Live Edge (```--mode live```) — real-time processing of tactical telemetry over UDP networks.

### Advanced Multi-Sensor Track Fusion

Built on the Stone Soup framework, leveraging:
- 9D Unscented Kalman Filters (UKF) for nonlinear state estimation.
- Joint Probabilistic Data Association (JPDA) for multi-sensor track correlation.
- Persistent tracking of dynamic targets, including UAV and swarm scenarios.

<br clear="right"/>

### Multi-Format Sensor Ingestion

Supports validated tactical data flows including:
- BSI Flex 335 SAPIENT JSON telemetry for replay and simulation.
- SAPIENT Protobuf streams for live low-latency operation.

### Tactical Track Augmentation & Dissemination

Transforms sensor data into operational tracks by:
- Converting kinematic vectors into WGS84 geospatial coordinates.
- Adding contextual metadata.
- Broadcasting standardized Cursor-on-Target (CoT) XML to tactical clients such as ATAK and WinTAK.

### Edge-Native Integration

Designed for distributed tactical environments, enabling low-latency sensor-to-user information flow across heterogeneous systems.

## Architecture

The system operates on a "Rosetta Stone" methodology. Incoming data is immediately mapped to a pure mathematical state space, tracked, and then serialized back out to whatever format the downstream consumer requires.

```text
[ ASMs / Sensors ]         [ The Core Engine ]          [ C2 Consumers ]
                                                                 
SAPIENT (JSON)   ──┐   ┌──> Pydantic Validation ──┐   ┌──> Cursor on Target (ATAK)
                   ├───┤                          ├───┤
SAPIENT (Binary) ──┘   └──> Stone Soup Tracker  ──┘   └──> SAPIENT (HLDMM)
                            (UKF + JPDA)
```

## Quick Start

Because all SAPIENT standard .proto files are pre-vendored and compiled within the repository, the engine is fully capable of being built and deployed on secure, air-gapped tactical edge nodes.

### 1. Installation
Clone the repository and install the package in editable mode. This automatically configures your environment and registers the global Command Line Interface (CLI).

```bash
git clone [https://github.com/define-ai-foundry/context-foundry.git]
cd context-foundry

# Create and activate a virtual environment (Recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install the package and CLI
pip install -e .
```

### 2. Verify with Replay Mode
Test the math engine and tracking logic by replaying a validated scenario file. Ensure your local ATAK device is open on the same network to see the fused tracks appear dynamically.

```bash
context-foundry-fusion --mode replay --file data/examples/sapient_messages.json
```

## CLI Reference

The installation exposes the context-foundry-fusion command globally.

| Flag / Argument | Type | Description | Default / Required |
| :--- | :--- | :--- | :--- |
| `-m`, `--mode` | `Choice` | Execution mode. Valid options are `replay` or `live`. | `replay` |
| `-f`, `--file` | `String` | Path to the JSON scenario file. | *Required if mode is replay* |
| `--tak-ip` | `String` | The multicast IP address for Cursor on Target broadcasts. | `239.2.3.1` |
| `--tak-port` | `Integer` | The UDP multicast port for Cursor on Target broadcasts. | `6969` |
| `-v`, `--verbose` | `Flag` | Enables debug-level logging output. | `False` |

Example:

```bash
context-foundry-fusion --mode live --tak-ip 192.168.1.255 --tak-port 4242
```

## Repository Structure

```text
fusion-engine/
├── compile_protos.sh           # Utility script to re-compile SAPIENT schemas
├── pyproject.toml              # Modern Python packaging configuration
├── protos/                     # Vendored BSI Flex 335 source files
├── data/
│   └── examples/               # Generated test scenarios
└── src/
    └── context_foundry/
        ├── cli.py              # Main execution interface
        └── fusion/
            ├── augmentor.py    # Converts 9D math to WGS84 Tactical State
            ├── schemas.py      # BSI Flex 335 Pydantic validation rules
            ├── serializers.py  # CoT and SAPIENT egress formatters
            ├── tracker.py      # Stone Soup UKF/JPDA core logic
            └── sources/        # Ingress Adapters
                ├── json_file.py
                ├── stream.py
                └── sapient_msg/# Pre-compiled _pb2 Protobuf Python bindings
```

## Extending the Gateway

Due to the format-agnostic core, adding support for new networks (e.g., Link 16, ASTERIX) is incredibly simple:

- Create a new class extending ```BaseSerializer``` inside ```src/context_foundry/fusion/serializers.py.```

- Map the universal ```TacticalState``` parameters (e.g., ```state.lat, state.speed_m_s```) to your target format.

- Register your new serializer in the ```cli.py``` routing dictionary. The tracker logic remains completely untouched!



Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry. SPDX-License-Identifier: Apache-2.0