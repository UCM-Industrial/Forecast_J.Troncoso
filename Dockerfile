# ── RECAST Pipeline Image ──
FROM python:3.13-slim

WORKDIR /app

# System dependencies for cfgrib / geopandas
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libeccodes0 \
        libgeos-dev \
        libproj-dev \
        gdal-bin \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency file first for layer caching
COPY pyproject.toml uv.lock* ./
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY flows/ ./flows/
COPY scripts/ ./scripts/

# Create directories
RUN mkdir -p logs data tmp

CMD ["python", "-m", "flows.ingestion_flow"]
