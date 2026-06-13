FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_INDEX_URL=https://download.pytorch.org/whl/cpu \
    UV_EXTRA_INDEX_URL=https://pypi.org/simple \
    UV_INDEX_STRATEGY=unsafe-best-match \
    UV_CACHE_DIR=/tmp/uv-cache \
    HF_HOME=/tmp/huggingface \
    TRANSFORMERS_CACHE=/tmp/huggingface

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency metadata
COPY pyproject.toml uv.lock README.md ./

# Install locked production dependencies with CPU-only torch and no caches.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && uv sync --frozen --no-dev --no-install-project --no-cache \
    && rm -rf /tmp/uv-cache /tmp/huggingface /root/.cache
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY app/ ./app/
COPY main.py ./
COPY gunicorn.conf.py ./

# Create non-root user
RUN useradd -m -u 1000 appuser \
    && mkdir -p /models/modernbert \
    && chown -R appuser:appuser /app /models
USER appuser

# Expose the port GKE expects
EXPOSE 8080

# Run ASGI app with gunicorn + Uvicorn workers
CMD ["gunicorn", "-c", "gunicorn.conf.py", "main:app"]
