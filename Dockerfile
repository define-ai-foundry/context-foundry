# syntax=docker/dockerfile:1

# Build stage: install the package (and its deps) into an isolated venv.
FROM python:3.13-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# pyproject reads README.md; deps resolve from PyPI wheels (numpy/scipy/stonesoup/
# grpcio-tools all ship manylinux wheels, so no compiler is needed here).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# Runtime stage: carry only the venv plus the data/config the CLI reads at run time.
FROM python:3.13-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Run as an unprivileged user; /app is writable so --log-to-file works by default.
RUN useradd --create-home --uid 10001 fusion && mkdir -p /app && chown fusion:fusion /app
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=fusion:fusion config ./config
COPY --chown=fusion:fusion data ./data
# CotValidator loads the MITRE base-event XSD by relative path on construction,
# so --enable-cot cannot start without it.
COPY --chown=fusion:fusion protos/cot ./protos/cot

USER fusion

ENTRYPOINT ["context-foundry-fusion"]
# Default to the replay demo, unpaced so a bare `docker run` drains fast; override in compose / on the CLI.
CMD ["--replay-file", "data/examples/sapient_messages.json", \
     "--config", "config/sensors/joensuu.json", "--log-to-file", "--realtime-factor", "0"]

# Pipeline stage: the runtime image started as a stage under the packaging contract.
FROM runtime AS stage
# TODO: run the stage entrypoint once the engine implements the packaging contract; until
# then the image exits at start, so conformance fails.
ENTRYPOINT ["sh", "-c", "echo 'fusion: the pipeline stage entrypoint is not implemented yet' >&2; exit 64"]
CMD []

# Dev stage: editable install with dev extras for in-container tests + lint (CI parity).
# The worktree is bind-mounted over /app at run time; the venv lives outside /app.
FROM python:3.13-slim AS dev

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

RUN python -m venv /opt/venv
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e ".[dev]"

# tests/, config/, data/ arrive via the bind mount; run the same checks CI runs.
CMD ["pytest"]
