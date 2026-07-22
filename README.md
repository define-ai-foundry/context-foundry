# DEFINE Edge Fusion Engine (Context Foundry)
<p align="center">
  <img src="winter_swarm_header.jpg" align="right" width="350" alt="Drone swarm over Finnish forest">
</p>

High-performance, format-agnostic Tactical Edge Sensor Fusion Gateway

The DEFINE Edge Fusion Engine is a modular tactical sensor fusion gateway designed for edge operations. It ingests heterogeneous sensor telemetry from radars, acoustic arrays, and EO/IR systems, performs real-time multi-sensor track fusion using advanced mathematical filtering, and delivers operationally relevant situational awareness directly to mission users and C2 systems.

## Core Capabilities

### Universal C2 Gateway Architecture

Format-agnostic architecture with decoupled ingestion, fusion, and dissemination pipelines, enabling rapid integration of new sensors, protocols, and C2 ecosystems.

### Dual-Mode Execution

Supports both:

- Scenario Replay (```--replay-file```) — analysis of recorded JSON-based sensor scenarios.
- Live Edge (```--enable-sapient``` / ```--enable-cot```) — real-time processing of tactical telemetry over UDP networks.

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
- Streaming standardized Cursor-on-Target (CoT) XML to a TAK Server over TCP+TLS, which disseminates it to connected ATAK/WinTAK clients — or writing it to file for offline validation.

### Edge-Native Integration

Designed for distributed tactical environments, enabling low-latency sensor-to-user information flow across heterogeneous systems.

## Architecture

The system operates on a "Rosetta Stone" methodology. Incoming data is immediately mapped to a pure mathematical state space, tracked, and then serialized back out to whatever format the downstream consumer requires.

```text
[ ASMs / Sensors ]         [ The Core Engine ]          [ C2 Consumers ]
                                                                 
SAPIENT (JSON)   ──┐   ┌──> Pydantic Validation ──┐   ┌──> Cursor on Target (TAK/TLS)
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
Test the math engine and tracking logic by replaying a validated scenario file. `--log-to-file` writes the fused CoT to `fused_tracks_debug.xml` for offline inspection — no network required.

```bash
context-foundry-fusion --replay-file data/examples/sapient_messages.json --config config/sensors/joensuu.json --log-to-file
```

## CLI Reference

The installation exposes the context-foundry-fusion command globally. `--config` is always required, and at least one source (`--replay-file`, `--enable-sapient`, or `--enable-cot`) must be provided. At least one output sink is also required — `--log-to-file`, `--tak-tls-host`, or both.

| Flag / Argument | Type | Description | Default / Required |
| :--- | :--- | :--- | :--- |
| `--config` | `String` | Path to the sensor network config JSON. | *Required* |
| `--replay-file` | `String` | Path to a JSON scenario file to replay. | *One source required* |
| `--enable-sapient` | `Flag` | Enable the live SAPIENT UDP stream (port 5000). | `False` |
| `--enable-cot` | `Flag` | Enable the live CoT UDP stream (port 6969). | `False` |
| `--log-to-file` | `Flag` | Write fused CoT to `fused_tracks_debug.xml` for offline validation. | *One sink required* |
| `--tak-tls-host` | `String` | TAK Server host to stream CoT to over TCP+TLS; providing it enables the TLS sink. | *One sink required* |
| `--tak-tls-port` | `Integer` | TAK Server TLS port. | `8089` |
| `--tak-tls-cert` | `String` | Client certificate (PEM) for mutual TLS. | *Optional* |
| `--tak-tls-key` | `String` | Client private key (PEM). | *Optional* |
| `--tak-tls-ca` | `String` | CA bundle (PEM) used to verify the TAK Server. | *Optional* |

The TAK Server `stdssl` CoT input (port 8089) is a mutual-TLS stream: supply a client certificate signed by the TAK CA (`--tak-tls-cert`/`--tak-tls-key`) and the CA bundle (`--tak-tls-ca`). Omitting `--tak-tls-ca` skips server verification (useful only for local/self-signed testing).

Example:

```bash
context-foundry-fusion --enable-sapient --config config/sensors/joensuu.json \
  --tak-tls-host tak.example.mil --tak-tls-cert client.pem --tak-tls-key client.key --tak-tls-ca ca.pem
```

## Repository Structure

```text
context-foundry/
├─ config/
│  ├─ sensors/
│  │  └─ joensuu.json
│  └─ locations.yaml
├─ data/
│  ├─ examples/
│  │  └─ sapient_messages.json
│  └─ generated_input/
│     └─ joensuu_messages.json
├─ docs/
│  ├─ 01-concepts/
│  └─ 02-architecture/
├─ protos/
│  ├─ cot/
│  │  ├─ CoT Base-Event Schema  (PUBLIC RELEASE).xsd
│  │  ├─ CoT Contact Schema (PUBLIC RELEASE).xsd
│  │  ├─ CoT Flow-Tags Schema  (PUBLIC RELEASE).xsd
│  │  ├─ CoT Image Schema  (PUBLIC RELEASE).xsd
│  │  ├─ CoT Link Schema  (PUBLIC RELEASE).xsd
│  │  ├─ CoT Remarks Schema  (PUBLIC RELEASE).xsd
│  │  ├─ CoT Request Schema  (PUBLIC RELEASE).xsd
│  │  ├─ CoT Sensor Schema  (PUBLIC RELEASE).xsd
│  │  ├─ CoT Shape Schema  (PUBLIC RELEASE).xsd
│  │  ├─ CoT Spatial Schema  (PUBLIC RELEASE).xsd
│  │  ├─ CoT Track Schema  (PUBLIC RELEASE).xsd
│  │  └─ CoT Uid Schema  (PUBLIC RELEASE).xsd
│  └─ sapient_msg/
│     ├─ bsi_flex_335_v2_0/
│     │  ├─ alert_ack.proto
│     │  ├─ alert.proto
│     │  ├─ associated_detection.proto
│     │  ├─ associated_file.proto
│     │  ├─ detection_report.proto
│     │  ├─ error.proto
│     │  ├─ follow.proto
│     │  ├─ location.proto
│     │  ├─ range_bearing.proto
│     │  ├─ registration_ack.proto
│     │  ├─ registration.proto
│     │  ├─ sapient_message.proto
│     │  ├─ status_report.proto
│     │  ├─ task_ack.proto
│     │  ├─ task.proto
│     │  └─ velocity.proto
│     └─ proto_options.proto
├─ src/
│  ├─ context_foundry/
│  │  ├─ fusion/
│  │  │  ├─ sources/
│  │  │  │  ├─ base.py
│  │  │  │  ├─ cot_stream.py
│  │  │  │  ├─ json_file.py
│  │  │  │  └─ stream.py
│  │  │  ├─ validators/
│  │  │  │  ├─ base.py
│  │  │  │  ├─ cot.py
│  │  │  │  └─ sapient.py
│  │  │  ├─ augmentor.py
│  │  │  ├─ config.py
│  │  │  ├─ models.py
│  │  │  ├─ schemas.py
│  │  │  ├─ serializers.py
│  │  │  └─ tracker.py
│  │  └─ cli.py
│  └─ sapient_msg/
│     └─ bsi_flex_335_v2_0/
│        ├─ alert_ack_pb2.py
│        ├─ alert_pb2.py
│        ├─ associated_detection_pb2.py
│        ├─ associated_file_pb2.py
│        ├─ detection_report_pb2.py
│        ├─ error_pb2.py
│        ├─ follow_pb2.py
│        ├─ location_pb2.py
│        ├─ range_bearing_pb2.py
│        ├─ registration_ack_pb2.py
│        ├─ registration_pb2.py
│        ├─ sapient_message_pb2.py
│        ├─ status_report_pb2.py
│        ├─ task_ack_pb2.py
│        ├─ task_pb2.py
│        └─ velocity_pb2.py
├─ tests/
├─ .github/
│  └─ workflows/
│     └─ ci.yml
├─ compile_protos.sh
├─ winter_swarm_header.jpg
├─ LICENSE
├─ pyproject.toml
└─ README.md
```

## Extending the Gateway

Due to the format-agnostic core, adding support for new networks (e.g., Link 16, ASTERIX) is incredibly simple:

- Create a new class extending ```BaseSerializer``` inside ```src/context_foundry/fusion/serializers.py.```

- Map the universal ```TacticalState``` parameters (e.g., ```state.lat, state.speed_m_s```) to your target format.

- Register your new serializer in the ```cli.py``` routing dictionary. The tracker logic remains completely untouched!

## Development

Install the project together with the development tooling (Ruff, pytest, pytest-cov):

```bash
pip install -e ".[dev]"
```

### Linting

Linting and formatting use [Ruff](https://docs.astral.sh/ruff/). Its configuration lives in `pyproject.toml`:

```bash
ruff check .           # report lint issues
ruff format --check .  # report formatting issues

ruff check --fix .     # auto-fix lint issues
ruff format .          # apply formatting
```

### Testing

Tests use `pytest` with branch coverage via `pytest-cov`. The coverage settings and the 95% minimum gate are configured in `pyproject.toml`, so a plain invocation runs the full suite and prints a coverage report:

```bash
pytest
```

The run fails if coverage drops below the configured threshold. Tests never touch the network or `data/**` — all fixtures are synthesized.

### Continuous Integration

All of the above run automatically on GitHub Actions (`.github/workflows/ci.yml`) for every push to `main`/`master` and every pull request: Ruff lint, Ruff format check, and the test suite with the coverage gate.



Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry. SPDX-License-Identifier: Apache-2.0