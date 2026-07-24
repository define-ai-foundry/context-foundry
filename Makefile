# Container dev loop for context-foundry.
# Auto-detects the container engine: prefers docker (what most people use),
# falls back to podman. Override with `make ENGINE=podman <target>`.

ENGINE ?= $(shell command -v docker >/dev/null 2>&1 && echo docker || echo podman)
COMPOSE := $(ENGINE) compose

.DEFAULT_GOAL := help

.PHONY: help build demo test lint shell certs tls-demo down \
	tak-up tak-users tak-demo tak-down tak-clean

# Local per-group isolation demo: one TAK File user per group, each in exactly
# one group. A user's /oauth/token bearer both drives its producer (TakWsSink)
# and logs into WebTAK, where it sees only its own group's CoT.
TAK_GROUPS ?= alpha bravo
TAK_DEMO_PASS ?= Fusion-Demo-2026!
TAK_DEMO_GROUP ?= alpha

# Resolve the tak-server container id via the compose labels (portable across
# docker/podman; `compose ps -q <svc>` is unreliable under the podman provider).
TAK_CID = $(ENGINE) ps -q \
	--filter label=com.docker.compose.project=context-foundry \
	--filter label=com.docker.compose.service=tak-server

help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'
	@echo "Using engine: $(ENGINE)"

build: ## Build the runtime + dev images
	$(COMPOSE) build

demo: ## Run the replay demo (fuse example scenario, log fused tracks)
	$(COMPOSE) run --rm fusion

test: ## Run the test suite in-container (matches CI, 95% coverage gate)
	$(COMPOSE) run --rm test

lint: ## Run ruff lint + format check in-container (matches CI)
	$(COMPOSE) run --rm lint

shell: ## Open a shell in the dev image with the worktree mounted
	$(COMPOSE) run --rm test sh

certs: ## Generate throwaway TLS certs for the local CoT listener (openssl runs in-container)
	$(COMPOSE) run --rm --no-deps test sh dev/gen-certs.sh

tls-demo: certs ## Replay -> TLS -> local listener; prints the CoT it receives
	$(COMPOSE) --profile tls up -d --build tls-listener
	$(COMPOSE) --profile tls run --rm fusion-tls; status=$$?; \
		echo "=== tls-listener received ==="; \
		$(COMPOSE) --profile tls logs tls-listener; \
		$(COMPOSE) --profile tls down >/dev/null 2>&1; exit $$status

down: ## Stop and remove all compose resources (keeps named volumes)
	$(COMPOSE) --profile tls --profile tak down --remove-orphans

# --- TAK Server validation (tak profile): CoT -> WebTAK WebSocket -> per-group isolation in WebTAK ---

tak-up: ## Start the local TAK Server + Postgres; waits until it is ready (~2-3 min first boot)
	$(COMPOSE) --profile tak up -d tak-db tak-server
	@echo "Waiting for TAK Server (first boot self-generates CA + certs + schema)..."
	@for i in $$(seq 1 72); do \
		code=$$(curl -sk -o /dev/null -w '%{http_code}' https://localhost:8446/ 2>/dev/null); \
		if [ "$$code" != "000" ]; then echo "TAK Server ready (8446 -> HTTP $$code)"; exit 0; fi; \
		sleep 5; \
	done; echo "TAK Server did not become ready in time (see: $(COMPOSE) --profile tak logs tak-server)" >&2; exit 1

tak-users: ## Create one WebTAK/producer File user per demo group ($(TAK_GROUPS))
	@id=$$($(TAK_CID)); \
	[ -n "$$id" ] || { echo "tak-server not running; run 'make tak-up' first" >&2; exit 1; }; \
	for g in $(TAK_GROUPS); do \
		$(ENGINE) exec -e JDK_JAVA_OPTIONS= $$id bash -c \
			"cd /opt/tak && java -jar /opt/tak/utils/UserManager.jar usermod -g $$g -p '$(TAK_DEMO_PASS)' $$g" \
			2>&1 | grep -viE "add-opens|WARNING|NOTE:|TOTALRAM|_MAX_HEAP" || true; \
	done
	@echo "WebTAK https://localhost:8446 — users [$(TAK_GROUPS)] (pass $(TAK_DEMO_PASS)); each is in one group and sees only that group's CoT"

tak-demo: ## Replay into group $(TAK_DEMO_GROUP) via the WebTAK WebSocket sink (override: make tak-demo TAK_DEMO_GROUP=bravo)
	@id=$$($(TAK_CID)); \
	[ -n "$$id" ] || { echo "tak-server not running; run 'make tak-up' first" >&2; exit 1; }; \
	tok=$$(curl -sk -X POST https://localhost:8446/oauth/token \
		-d grant_type=password -d username=$(TAK_DEMO_GROUP) -d password='$(TAK_DEMO_PASS)' \
		| python3 -c 'import sys,json;print(json.load(sys.stdin).get("access_token",""))' 2>/dev/null); \
	[ -n "$$tok" ] || { echo "no token for user '$(TAK_DEMO_GROUP)' — run 'make tak-users' first" >&2; exit 1; }; \
	CF_BEARER=$$tok $(COMPOSE) --profile tak run --rm fusion-tak
	@echo "Streamed into group '$(TAK_DEMO_GROUP)'. In WebTAK, only user '$(TAK_DEMO_GROUP)' sees these tracks."

tak-down: ## Stop the TAK stack (keeps the tak-data volume, so the CA + users persist)
	$(COMPOSE) --profile tak down --remove-orphans

tak-clean: ## Stop the TAK stack and delete its volumes (wipes CA, certs, users, DB)
	$(COMPOSE) --profile tak down -v --remove-orphans
