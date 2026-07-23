# Container dev loop for context-foundry.
# Auto-detects the container engine: prefers docker (what most people use),
# falls back to podman. Override with `make ENGINE=podman <target>`.

ENGINE ?= $(shell command -v docker >/dev/null 2>&1 && echo docker || echo podman)
COMPOSE := $(ENGINE) compose

.DEFAULT_GOAL := help

.PHONY: help build demo test lint shell certs tls-demo down

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

down: ## Stop and remove compose resources
	$(COMPOSE) --profile tls down --remove-orphans
