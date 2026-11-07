# syntax=docker/dockerfile:1.7
# ---- build stage: resolve and install dependencies into an isolated virtualenv ------------
FROM python:3.11-slim AS builder
ARG EXTRAS="pg,vertex,anthropic,openai"
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[${EXTRAS}]"

# ---- runtime stage: slim image, non-root user, no build tooling ---------------------------
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH="/opt/venv/bin:$PATH" \
    CORPUS_PATH=/app/data/fixtures/corpus_fixture.jsonl CACHE_DIR=/app/data/cache \
    RAW_DIR=/app/data/raw PORT=8080
RUN groupadd --system --gid 10001 app && useradd --system --uid 10001 --gid app --no-create-home app
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=app:app data/fixtures ./data/fixtures
RUN mkdir -p /app/data/cache /app/data/raw && chown -R app:app /app/data
USER app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8080\")}/health', timeout=4)"
# Cloud Run injects $PORT; shell form so it is expanded.
CMD exec uvicorn eu_ai_act_rag.api.main:app_factory --factory --host 0.0.0.0 --port ${PORT} --proxy-headers
