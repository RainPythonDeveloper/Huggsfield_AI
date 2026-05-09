# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ── builder ────────────────────────────────────────────────────────────────
FROM base AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install --prefix=/install \
        "fastapi>=0.115.0" \
        "uvicorn[standard]>=0.32.0" \
        "pydantic>=2.9.0" \
        "pydantic-settings>=2.5.0" \
        "asyncpg>=0.30.0" \
        "httpx[http2]>=0.27.0" \
        "tiktoken>=0.8.0" \
        "tenacity>=9.0.0" \
        "python-json-logger>=2.0.7"

# ── runtime ────────────────────────────────────────────────────────────────
FROM base AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY src/ ./src/

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
    CMD curl -fsS http://localhost:8080/health || exit 1

CMD ["uvicorn", "memory.main:app", "--host", "0.0.0.0", "--port", "8080", "--app-dir", "src"]
