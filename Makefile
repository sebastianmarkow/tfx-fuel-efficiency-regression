PWD:=$(shell pwd)

RUN_ARGS=--remove-orphans --renew-anon-volumes

.DEFAULT_GOAL: help
.PHONY: help
help:
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "%-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: pull
pull: ## Pull prebuilt docker images
	docker compose pull

.PHONY: build
build: ## Build docker images
	docker compose build

.PHONY: run
run: build ## Run app with dashboard and service dependencies
	docker compose up app dashboard ${RUN_ARGS}

.PHONY: run-app
run-app: pull build ## Run only app
	docker compose up app ${RUN_ARGS}

.PHONY: run-dashboard
run-dashboard: pull build ## Run only dashboard
	docker compose up dashboard ${RUN_ARGS}

.PHONY: run-mlmd
run-mlmd: pull build ## Run only mlmd services
	docker compose up mlmddb mlmdgrpc ${RUN_ARGS}

.PHONY: run-postgres
run-postgres: pull build ## Run only postgres
	docker compose up postgres dataloader ${RUN_ARGS}

.PHONY: run-jupyter
run-jupyter: pull build ## Run only jupyter lab
	docker compose up jupyter ${RUN_ARGS}
