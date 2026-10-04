# syntax=docker/dockerfile:1
ARG PYTHON_IMAGE=python:3.11.14-slim-bookworm
FROM ghcr.io/astral-sh/uv:0.10.11 AS uv
FROM ${PYTHON_IMAGE} AS base
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app

FROM base AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/opt/venv
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY src/ ./src/
COPY docs/architecture.md ./docs/architecture.md
RUN uv sync --locked --no-dev --no-editable

FROM base AS runtime
RUN groupadd --gid 10001 risk && useradd --uid 10001 --gid risk --no-create-home risk
COPY --from=builder /opt/venv /opt/venv
COPY configs/ ./configs/
ENV PATH="/opt/venv/bin:$PATH" \
    CREDIT_RISK_SERVING_CONFIG=/app/configs/serving.container.yaml
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]
CMD ["python", "-m", "uvicorn", "credit_risk.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
