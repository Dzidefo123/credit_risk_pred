# Run the equivalent uv commands directly on Windows without make.
.PHONY: install test lint format-check check-config build docker-build docker-smoke
install:
	uv sync --locked

test:
	uv run --locked pytest

lint:
	uv run --locked ruff check src tests scripts api

check-config:
	uv run --locked credit-risk-lab check-config --config-dir configs

build:
	uv build

format-check:
	uv run --locked ruff format --check src tests scripts api

docker-build:
	docker build --tag credit-risk-lab:local .

docker-smoke: docker-build
	python scripts/smoke_container.py --image credit-risk-lab:local
