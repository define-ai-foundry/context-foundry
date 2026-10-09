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
- Global nearest-neighbour association over a Mahalanobis gate, solved as a 2D assignment, for multi-sensor track correlation.
- Persistent tracking of dynamic targets, including UAV and swarm scenarios.

Association cost is polynomial in tracks × detections, and every term of one fusion pass is bounded — the live track population, the detections one event may carry, and how long a track may coast. That matters at density: exact joint association is exponential once coasting tracks' gates overlap into a single cluster, which at 9 nodes and 80 objects meant one call that never returned and allocated 9.8 GiB.

<br clear="right"/>

### Multi-Format Sensor Ingestion

Supports validated tactical data flows including:
- BSI Flex 335 SAPIENT JSON telemetry for replay and simulation.
- SAPIENT Protobuf streams for live low-latency operation.

### Tactical Track Augmentation & Dissemination

Transforms sensor data into operational tracks by:
- Converting kinematic vectors into WGS84 geospatial coordinates.
- Adding contextual metadata.
- Delivering standardized Cursor-on-Target (CoT) XML to a TAK Server — over the WebTAK streaming WebSocket (bearer-authenticated; tracks tagged with the producer's Keycloak groups) or a raw TCP+TLS stream — for dissemination to connected WebTAK/ATAK/WinTAK clients, or writing it to file for offline validation.

### Edge-Native Integration

Designed for distributed tactical environments, enabling low-latency sensor-to-user information flow across heterogeneous systems.

## Architecture

The system operates on a "Rosetta Stone" methodology. Incoming data is immediately mapped to a pure mathematical state space, tracked, and then serialized back out to whatever format the downstream consumer requires.

```text
[ ASMs / Sensors ]         [ The Core Engine ]          [ C2 Consumers ]
                                                                 
SAPIENT (JSON)   ──┐   ┌──> Pydantic Validation ──┐   ┌──> Cursor on Target → TAK (WS / TLS)
                   ├───┤                          ├───┤
SAPIENT (Binary) ──┘   └──> Stone Soup Tracker  ──┘   └──> SAPIENT (HLDMM)
                            (UKF + GNN)
```

## Quick Start

Because all SAPIENT standard .proto files are pre-vendored and compiled within the repository, the engine is fully capable of being built and deployed on secure, air-gapped tactical edge nodes.

### 1. Installation
Clone the repository and install the package in editable mode. This automatically configures your environment and registers the global Command Line Interface (CLI).

```bash
git clone https://github.com/define-ai-foundry/context-foundry.git
cd context-foundry

# Create and activate a virtual environment (Recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install the package and CLI
pip install -e .
```

### 2. Verify with Replay Mode
Test the math engine and tracking logic by replaying a validated scenario file. `--log-to-file` writes the fused CoT to `fused_tracks_debug.xml` for offline inspection — no network required.

By default a replay is paced to the scenario's own clock (`--realtime-factor 1.0`), so `data/examples/sapient_messages.json` (37.8 minutes of scenario time) takes 37.8 minutes to drain. Use `--realtime-factor N` to run N times faster (`5` replays a 38-minute scenario in ~7.6 minutes), or `--realtime-factor 0` to disable pacing and drain the file as fast as it can be fused — the fast path for offline math validation:

```bash
context-foundry-fusion --replay-file data/examples/sapient_messages.json --config config/sensors/joensuu.json --log-to-file --realtime-factor 0
```

Replayed events are stamped so the scenario **starts now**, keeping the original spacing between them. A recorded file therefore surfaces as if it were happening live, which is what consumers assume — TAK will not draw a marker for an event dated months away from the present. This is independent of `--realtime-factor`: the factor changes how fast events are emitted, never what they are stamped with. Pass `--use-scenario-timestamps` to emit the file's own clock instead, e.g. when comparing output against a recorded run.

`--cot-stale-seconds` (default `15`) sets how long a CoT marker stays live in TAK after the event it was built from. Raise it above the interval between a track's updates, or each marker expires just before its own next update and flickers on the map.

A finite replay exits once drained. Add `--loop` to restart it continuously (waiting `--loop-delay` seconds, default `60`, between iterations), turning a one-shot scenario into a long-running feed — ideal as a persistent Deployment demo streaming into TAK. Each iteration re-stamps the scenario to the new present, and starts a fresh tracker: a restart is a new scenario rather than a continuation, and under `--use-scenario-timestamps` the timestamps jump backwards, which Stone Soup cannot predict through. `--loop` only applies with `--replay-file`; live sources never terminate.

```bash
context-foundry-fusion --replay-file data/examples/sapient_messages.json --config config/sensors/joensuu.json --log-to-file --loop --loop-delay 30
```

## Running in a Container

The engine ships a container image and a `make`-driven dev loop that works with **Docker or Podman** (auto-detected; override with `make ENGINE=podman …`).

```bash
make build     # build the runtime + dev images
make demo      # replay the example scenario, log fused CoT, clean exit
make test      # run the suite in-container (matches CI, 95% coverage gate)
make lint      # ruff check + format --check in-container
make shell     # shell into the dev image with the worktree mounted
```

`make test`/`make lint` bind-mount the live worktree, so edits need no rebuild. `make demo` runs the **baked** runtime image — re-run `make build` after changing `src/`.

### Validate against a real TAK Server + WebTAK (per-group isolation)

A `tak` compose profile stands up a real `pvarki/tak-server` + PostGIS locally and demonstrates the group-correct path: two groups, each with its own producer, tracks visible **only** to that group's WebTAK users.

```bash
make tak-up      # start TAK Server + Postgres (first boot ~2–3 min)
make tak-users   # create one File user per group: 'alpha' (group alpha), 'bravo' (group bravo)
make tak-demo                        # replay into group 'alpha' over the WebTAK WebSocket sink
make tak-demo TAK_DEMO_GROUP=bravo   # ...or into group 'bravo'
make tak-demo TAK_DEMO_FACTOR=10     # ...at 10x real time (~4 min instead of ~38)
make tak-down    # stop (keeps volumes);  make tak-clean wipes them
```

`make tak-demo` fetches a TAK `/oauth/token` bearer for that group's File user and runs `TakWsSink` against `wss://…:8446/takproto/1`, so the CoT is tagged with the user's single TAK group. It streams at scenario cadence (~38 min for the example scenario) and holds the terminal until drained — raise `TAK_DEMO_FACTOR` to compress it. Log into `https://localhost:8446` as `alpha` / `Fusion-Demo-2026!` — with the map open **before** you run `make tak-demo` (TAK does not backfill history to a viewer that connects after the stream) — and you will see the tracks; log in as `bravo` and you will not. This mirrors the production model (one producer per group), using TAK File-user tokens locally in place of Keycloak; the Keycloak-token equivalent has been verified against a real TAK + Keycloak cluster. A raw TCP+TLS sink is also available (`make tls-demo`).

### Validate live ingestion (UDP sensor traffic, no file)

The demos above replay a file. `make live-demo` exercises the other mode: the engine reads sensor traffic **off the network** (`--enable-sapient` on UDP 5000, `--enable-cot` on UDP 6969) and streams the fused result into the same per-group WebTAK sink. `dev/udp_feeder.py` stands in for the ASMs — it replays `data/examples/sapient_messages.json` onto those sockets as either binary SAPIENT protobuf or CoT XML, stamped with the present at the moment of sending — one stamp per sweep, since a node that sees five objects at once reports all five with the same timestamp.

```bash
make tak-up && make tak-users            # once, as above
make tak-tail &                          # watch what group 'alpha' receives (start it first)
make live-demo                           # SAPIENT protobuf over UDP, at scenario cadence (~38 min)
make live-demo LIVE_PROTOCOL=cot         # ...CoT XML instead
make live-demo LIVE_FACTOR=10            # ...10x faster: an ingest smoke test, see below
make tak-tail TAK_DEMO_GROUP=bravo       # confirm 'bravo' receives none (during a feed)
make live-down                           # stop the engine (leaves TAK up)
```

`make live-demo` starts the engine detached and runs the feeder in the foreground, so the run lasts as long as the feed. Both live sources are read concurrently, so `--enable-sapient --enable-cot` together work: each source gets its own reader thread, and events are fused in arrival order. An event whose timestamp is **out of step with the rest** is dropped: the tracker cannot predict backwards, so letting a late one through rewinds *every* track's timestamp and redraws its TAK marker in the past. Events are therefore gated to the newest timestamp seen — and a live event stamped more than ten seconds **ahead of the present** is refused that mark, or one bad clock (or one spoofed datagram) would set a watermark every healthy sensor then falls behind, silencing the engine for good. Drops are re-reported once a minute with a running count and the node id involved, rather than once per run. **Keep the sensors' clocks in step.** If fusion falls behind the incoming rate the readers stall rather than discard, so nothing is lost in-process; sustained overload then overruns the kernel's socket receive buffer (visible as the drop counter for the socket in `/proc/net/udp`).

A live socket sees one datagram at a time, but a sensor that reports five objects at one instant sends five of them, and the fused result depends on that instant arriving as **one event** — the contract a replay file's grouping gives for free, and what the associator needs in order to match more than one detection per track update. The engine therefore assembles the datagrams of one sensor-instant into a single event, bounded by a short window (`--frame-window-seconds`, 250 ms) from the instant's first datagram: every live detection pays at least that window in latency — a frame surfaces on the first read that returns after it expires, so a slow consumer adds to it — and a sweep whose datagrams are spread wider than the window splits across events. Several sensors' frames are assembled independently and in parallel, so two nodes whose sweeps overlap interleave on the wire without either losing detections, and each is handed on when its own window is up, in the order the datagrams arrived.

The window bounds a sweep's **arrival spread**, which is why 250 ms is enough for sweeps that a sensor emits as a burst: measured at 9 nodes × 80 objects, every sweep arrived whole with no splits and nothing discarded, while one fusion pass took 0.44 s. A pass that overlaps a frame still being assembled can split it, so if the per-event log line starts reporting partial sweeps, raise the window. The two things worth watching in the summary are that and the kernel discard counter.

A replay file and a live source cannot be combined — the engine refuses the pair. A scenario runs on its own clock, minutes or hours away from the sensors', and the gate above would silently discard whichever is behind.

Because the feeder stamps at send time, `LIVE_FACTOR` compresses the timeline the engine sees rather than pushing timestamps into the future — the same targets appear to move that many times faster. **Fusion does not survive that.** The tracker's association gate is sized by the interval between events, so a compressed feed makes it too tight to match anything and every detection starts its own track: the scenario's first 40 detections fuse into 16 tracks at `LIVE_FACTOR=1` and 40 — one per detection, i.e. no association at all — at `LIVE_FACTOR=10`. Use a factor above 1 to check that ingest works, and 1 (the default) to judge the fused picture. (16 tracks for those 40 detections is itself more than the objects behind them; association quality on the scenario is a separate, pre-existing matter.) The engine's own `--replay-file` pacing has no such limit: it compresses emission while keeping the scenario's spacing, which is exactly what a replay can do and a live sensor cannot.

The feeder's CoT mode emits a fixed hostile type for every track and carries no classification or swarm count (a CoT `<event>` has nowhere standard to put them), so `LIVE_PROTOCOL=cot` exercises the ingest path, not the classification carry-through that SAPIENT reports drive.

`make tak-tail` is the scriptable stand-in for a WebTAK browser session: it subscribes to `wss://…:8446/takproto/1` as a group's File user and prints the CoT that user is allowed to see. Run it (or open WebTAK) **before** the feed starts — TAK does not backfill to a late subscriber. Receiving nothing is the expected result for a group the producer is not in, so the target says so rather than failing; `dev/webtak_ws_tail.py` itself exits non-zero on an empty tail, for a script that wants to assert delivery.

Run the feeder against a non-containerised engine the same way:

```bash
context-foundry-fusion --enable-sapient --config config/sensors/joensuu.json --log-to-file &
python dev/udp_feeder.py --protocol sapient --realtime-factor 10
```

A live run only ends when it is signalled; SIGTERM (`docker stop`, a pod deletion) closes the sinks and exits.

## CLI Reference

The installation exposes the `context-foundry-fusion` command globally. `--config` is always required, and at least one source (`--replay-file`, `--enable-sapient`, or `--enable-cot`) must be provided. `--replay-file` and the live sources are **mutually exclusive** — a scenario's clock does not line up with a sensor's, and the engine refuses the pair rather than silently discarding whichever is behind. The two live sources can be used together, provided their sensors agree on the time. At least one output sink is also required — `--log-to-file`, `--tak-ws-host`, `--tak-tls-host`, or any combination.

| Flag / Argument | Type | Description | Default / Required |
| :--- | :--- | :--- | :--- |
| `--config` | `String` | Path to the sensor network config JSON. | *Required* |
| `--replay-file` | `String` | Path to a JSON scenario file to replay. Cannot be combined with a live source. | *One source required* |
| `--loop` | `Flag` | With `--replay-file`, restart the replay continuously instead of exiting — turns a one-shot replay into a continuous feed (ideal as a long-running Deployment demo streaming into TAK). No effect without `--replay-file`. | `False` |
| `--loop-delay` | `Float` | Seconds to wait between replay iterations when `--loop` is set. | `60` |
| `--realtime-factor` | `Float` | With `--replay-file`, the speed to emit events at relative to the scenario's own timeline: `1.0` replays at real time, `5` replays five times faster, `0` disables pacing and drains as fast as events can be fused. No effect without `--replay-file`. Must be `>= 0`. | `1.0` |
| `--use-scenario-timestamps` | `Flag` | With `--replay-file`, stamp events with the timestamps in the file instead of shifting the scenario to start now. TAK may treat the result as too old or too far ahead to display. No effect without `--replay-file`. | `False` |
| `--cot-stale-seconds` | `Float` | How long a CoT marker stays live in TAK after the event it was built from. | `15` |
| `--enable-sapient` | `Flag` | Enable the live SAPIENT UDP stream (port 5000). Cannot be combined with `--replay-file`. | `False` |
| `--enable-cot` | `Flag` | Enable the live CoT UDP stream (port 6969). Cannot be combined with `--replay-file`. | `False` |
| `--log-to-file` | `Flag` | Write fused CoT to `fused_tracks_debug.xml` for offline validation. | *One sink required* |
| `--tak-ws-host` | `String` | TAK host for the WebTAK WebSocket sink (bearer-auth; CoT tagged with the token's Keycloak groups). Enables the sink. | *One sink required* |
| `--tak-ws-port` | `Integer` | WebTAK WebSocket port. | `8446` |
| `--keycloak-token-url` | `String` | Keycloak token endpoint for the `client_credentials` grant. | *WS: unless `--tak-bearer-token`* |
| `--oidc-client-id` | `String` | Keycloak client id — the producer identity, whose group its CoT lands in. | *WS: with token URL* |
| `--oidc-client-secret` | `String` | Keycloak client secret (confidential client). | *Optional* |
| `--tak-bearer-token` | `String` | Static bearer token, as an alternative to a Keycloak grant. | *Optional* |
| `--tak-ws-verify-tls` | `Flag` | Verify the TAK server against the system trust store. A server dialled by a name its certificate does not carry — an in-cluster Service name — fails here; use `--tak-ws-ca`. Never affects the Keycloak token exchange, which is always verified. | `False` |
| `--tak-ws-ca` | `String` | CA bundle (PEM) verifying the TAK server on the WebSocket sink. The production trust setting; takes precedence over `--tak-ws-verify-tls`. | *Optional* |
| `--tak-ws-token-timeout` | `Float` | Seconds to wait on the Keycloak token endpoint. The fetch blocks the fusion loop, so the sensor sockets go unread for its duration. | `5` |
| `--frame-window-seconds` | `Float` | How long a live source holds one sensor-instant open to collect the rest of its datagrams before handing it on as one event. Must exceed both the spread of a sweep's datagrams and one fusion pass, or sweeps split into single-detection events. Costs that much latency per live detection. | `0.25` |
| `--max-live-tracks` | `Integer` | Live tracks held at once. Detections that associate with nothing each start a track, so clutter or an injected feed grow the population; past this bound new tracks are refused and reported, which keeps the targets already being followed. | `120` |
| `--max-track-history` | `Integer` | States retained per track. Nothing downstream reads history; it is history times the track bound that has to fit the memory limit. | `20` |
| `--max-event-detections` | `Integer` | Largest number of detections one event carries into association — the only cost term a sender controls. Past it detections are shed in a deterministic order and reported. | `300` |
| `--max-coast-seconds` | `Float` | How long a track may coast without a real detection before it is dropped. Must exceed the interval at which a target is re-detected, or tracks die between revisits (on the reference scenario, `20` gives 274 tracks against 88 at the default). | `45` |
| `--tak-tls-host` | `String` | TAK host to stream CoT to over raw TCP+TLS (port 8089); enables the TLS sink. | *One sink required* |
| `--tak-tls-port` | `Integer` | TAK TLS port. | `8089` |
| `--tak-tls-cert` | `String` | Client certificate (PEM) for mutual TLS. | *Optional* |
| `--tak-tls-key` | `String` | Client private key (PEM). | *Optional* |
| `--tak-tls-ca` | `String` | CA bundle (PEM) used to verify the TAK Server. | *Optional* |

### Delivering CoT to a TAK Server

TAK only shows a track to viewers who share a **group** with it, so the choice of sink is really a choice of how the producer's group is assigned:

- **WebTAK WebSocket + Keycloak (`--tak-ws-host`, recommended).** The producer authenticates to `wss://<host>:8446/takproto/1` with a Keycloak bearer (a `client_credentials` grant via `--keycloak-token-url`/`--oidc-client-id`/`--oidc-client-secret`, or a static `--tak-bearer-token`). **CoT is tagged with the token's Keycloak `groups`** — so run **one producer per group** with a Keycloak identity in that single group, and only that group's WebTAK users see its tracks. No client certificate. This is the group-correct path for a multi-tenant TAK, verified end-to-end against a real TAK + Keycloak cluster.

  ```bash
  context-foundry-fusion --replay-file data/examples/sapient_messages.json \
    --config config/sensors/joensuu.json \
    --tak-ws-host tak.example.mil \
    --keycloak-token-url https://iam.example.mil/realms/rain-realm/protocol/openid-connect/token \
    --oidc-client-id cf-team-viper --oidc-client-secret "$CF_CLIENT_SECRET"
  ```

- **Raw TCP+TLS stream (`--tak-tls-host`).** Streams newline-delimited CoT to the TAK `stdssl` input (port 8089). With `authRequired="true"` it is mutual TLS — supply a CA-signed client cert (`--tak-tls-cert`/`--tak-tls-key`) and the CA bundle (`--tak-tls-ca`); omitting `--tak-tls-ca` skips server verification (local/self-signed only). **Group assignment is server-side, not from the cert's OU** — TAK's `x509groups` OU→group mapping does not work on the stock pvarki image, so the group comes from a `<filtergroup>` on the input or from registering the cert as a group user (`certmod -g`, which requires a messaging restart).

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
│  │  │  │  ├─ multiplex.py
│  │  │  │  ├─ offset.py
│  │  │  │  ├─ paced.py
│  │  │  │  └─ stream.py
│  │  │  ├─ validators/
│  │  │  │  ├─ base.py
│  │  │  │  ├─ cot.py
│  │  │  │  └─ sapient.py
│  │  │  ├─ sinks/
│  │  │  │  ├─ base.py
│  │  │  │  ├─ file.py
│  │  │  │  ├─ tak_tls.py
│  │  │  │  └─ tak_ws.py
│  │  │  ├─ augmentor.py
│  │  │  ├─ config.py
│  │  │  ├─ models.py
│  │  │  ├─ schemas.py
│  │  │  ├─ serializers.py
│  │  │  ├─ timeutil.py
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
├─ Dockerfile
├─ .dockerignore
├─ compose.yaml
├─ Makefile
├─ dev/                    # UDP sensor feeder, WebTAK tail, TLS listener, real-TAK helpers (certs gitignored)
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

### Pipeline stage

The engine is also published as the `fusion` stage type of the pipeline platform: an image built from the Dockerfile's `stage` target, pushed to `rainaiacr.azurecr.io/context-foundry/fusion`, and described to the platform by `stage/fusion/descriptor.yaml` in the [`pipeline-catalog`](https://github.com/define-ai-foundry/pipeline-catalog). `stage/fusion/conformance.yaml` and `stage/fusion/samples/` are the stage type's conformance profile, the records the harness feeds it and what it expects back. The standalone image that `.github/workflows/publish-image.yaml` pushes is unaffected.

`.github/workflows/stage.yaml` does the rest. On every pull request it builds the stage image locally, fills the descriptor with a placeholder digest into a checkout of the catalog's `main`, runs the catalog's own descriptor checks on it, and runs the conformance harness of [`pipeline-stages`](https://github.com/define-ai-foundry/pipeline-stages) against the image; the report is uploaded as the `conformance-report-fusion` artifact. The harness is taken from `pipeline-stages` `main`, unpinned, so it checks the contract as it is now; when it fails, the job names the `pipeline-stages` and catalog commits it ran with.

A merge to `main`, or a manual run from `main`, publishes only when `[project].version` in `pyproject.toml` ranks above (semantic version precedence) the `image.version` the catalog's `main` pins for `fusion`, or the catalog has no `fusion` yet; otherwise the run is skipped with a notice. To publish, raise the version in the pull request. A publish pushes the image tagged `<version>` and `sha-<short sha>`, runs conformance against the pushed digest, and opens a catalog pull request from the branch `publish/fusion-<version>`, or updates the open one, carrying the descriptor pinned to that digest, the unverified verification record and the conformance report in its body. A failed conformance run opens nothing.

Verification is the platform team's, in a separate pull request on the catalog: the publish only ever resets `verification.yaml` to unverified, and the stage runs as its own pod (`runtime: deployment`) whatever its verification.

The workflow is off until the repository variable `FUSION_STAGE_CI` is `true`; every job checks it, so the workflow changes nothing until the stage implements the packaging contract. It needs two GitHub Apps beside the Azure login `publish-image.yaml` already uses: `pipeline-catalog-publisher` (`CATALOG_APP_ID` variable, `CATALOG_APP_PRIVATE_KEY` secret), installed on `pipeline-catalog` with contents and pull requests read and write, whose token is narrowed to read wherever a job only reads the catalog; and `pipeline-stages-reader` (`STAGES_APP_ID` variable, `STAGES_APP_PRIVATE_KEY` secret), with contents read on `pipeline-stages`, for the harness.

`stage/publish.py` holds the few steps `pipeline-stages`' `scripts/publish_descriptor.py` cannot take for an image outside `pipeline-stages/`; the workflow runs that script, from its checkout, for the rest. To run conformance locally, build the image with `docker build --target stage -t context-foundry/fusion:local .`, install `pipeline-stages[conformance]` from a checkout, and run `pipeline-conformance --image context-foundry/fusion:local --stage-dir stage/fusion --catalog <pipeline-catalog checkout> --runtime docker`.



Copyright 2026 Lempea Edge Oy / DEFINE AI Foundry. SPDX-License-Identifier: Apache-2.0
