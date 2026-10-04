# Run the equivalent uv commands directly on Windows without make.
.PHONY: install test lint check-config build
install:
	uv sync --locked

test:
	uv run --locked pytest

lint:
	uv run --locked ruff check src tests

check-config:
	uv run --locked credit-risk-lab check-config --config-dir configs

build:
	uv build
