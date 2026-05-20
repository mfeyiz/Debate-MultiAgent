FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency metadata
COPY pyproject.toml uv.lock README.md ./

# Install locked production dependencies with CPU-only torch.
# `uv pip install -e .` intentionally is not used here because it bypasses
# uv.lock resolution and can pick incompatible newest transitive releases.
ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN uv sync --frozen --no-dev --no-install-project
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY app/ ./app/
COPY main.py ./
COPY gunicorn.conf.py ./
COPY scripts/check_model_artifacts.py ./scripts/check_model_artifacts.py

# Copy only the production model weights (~1.14 GB)
COPY models/component_classifier/final/ ./models/component_classifier/final/
COPY models/relation_classifier/final/ ./models/relation_classifier/final/
RUN python scripts/check_model_artifacts.py

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose the port GKE expects
EXPOSE 8080

# Run ASGI app with gunicorn + Uvicorn workers
CMD ["gunicorn", "-c", "gunicorn.conf.py", "main:app"]
