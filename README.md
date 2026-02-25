# ⚡ RECAST — Renewable Energy Scenario Forecasting Toolkit

Professional end-to-end data pipeline for **solar and wind energy generation forecasting** using XGBoost and ECMWF weather data.

## Architecture

```
┌────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  ECMWF API │────>│   Ingestion  │────>│ Preprocessor │────>│   Training   │
│ (aifs-sing)│     │  (download)  │     │ (features)   │     │  (XGBoost)   │
└────────────┘     └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
                          │                    │                    │
                     ┌────▼────────────────────▼────────────────────▼────┐
                     │              Google Cloud Storage                  │
                     │  raw/ │ processed/ │ models/ │ predictions/       │
                     └───────────────────────┬──────────────────────────┘
                                             │
                    ┌────────────────────────▼──────────────────┐
                    │          FastAPI Prediction Service        │
                    │   GET /predictions/latest?technology=wind  │
                    └───────────────────────────────────────────┘
```

**Orchestrated with [Prefect](https://www.prefect.io/)** — daily ingestion, preprocessing, and prediction flows with automatic retries.

## Features

- 🌤️ **ECMWF Open Data** ingestion (aifs-single model)
- 🗺️ **Geospatial processing** with regionmask
- 🤖 **XGBoost** forecasting with time-series cross-validation
- 📦 **Versioned storage** on GCS (raw → processed → models → predictions)
- 🔄 **Prefect orchestration** with retries and logging
- 🌐 **FastAPI** serving layer for predictions
- 🐳 **Docker Compose** for local development
- ✅ **Tested** with pytest

## Quick Start

### Prerequisites

- Python ≥ 3.13
- [uv](https://docs.astral.sh/uv/) package manager
- Docker & Docker Compose (for containerised setup)

### Local Development

```bash
# 1. Clone and install
git clone https://github.com/your-user/Forecast_J.Troncoso.git
cd Forecast_J.Troncoso
uv sync

# 2. Configure environment
cp .env.example .env
# Edit .env with your ECMWF API key and GCS credentials

# 3. Run tests
uv run pytest tests/ -v

# 4. Run a pipeline step locally
uv run python scripts/run_local_pipeline.py --step ingestion --date 20260225

# 5. Start the API
uv run uvicorn src.prediction.api:app --reload --port 8080
```

### Docker Compose

```bash
# Build and start all services
docker compose up --build

# Services:
#   - Prefect UI:  http://localhost:4200
#   - Prediction API: http://localhost:8080
#   - Health check:   http://localhost:8080/health
```

## Project Structure

```
├── config/                  # YAML configuration
│   ├── settings.yaml        # Pipeline parameters
│   └── logging.yaml         # Logging configuration
├── src/                     # Application source code
│   ├── ingestion/           # ECMWF data download
│   ├── preprocessing/       # Feature engineering
│   ├── training/            # XGBoost model
│   ├── prediction/          # Batch + API serving
│   └── utils/               # Config, GCS, logging, validators
├── flows/                   # Prefect flow definitions
├── tests/                   # pytest test suite
├── scripts/                 # Utility scripts
├── Dockerfile               # Pipeline image
├── Dockerfile.api           # API image (Cloud Run)
└── docker-compose.yml       # Local development setup
```

## Configuration

All pipeline parameters are centralised in `config/settings.yaml`. Environment variables in `.env` override YAML values:

| Variable | Description |
|---|---|
| `ECMWF_API_KEY` | ECMWF Open Data API key |
| `GCS_BUCKET_NAME` | Google Cloud Storage bucket |
| `STORAGE_BACKEND` | `local` or `gcs` |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP service account key |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/predictions/latest?technology=wind` | Latest predictions |
| `GET` | `/predictions/{date}?technology=solar` | Predictions by date |
| `GET` | `/predictions/dates/available?technology=wind` | Available dates |

## Tech Stack

| Category | Tool |
|---|---|
| ML | XGBoost, scikit-learn |
| Data | pandas, xarray, cfgrib, regionmask |
| Orchestration | Prefect |
| API | FastAPI, uvicorn |
| Cloud | Google Cloud Storage |
| Config | Pydantic Settings, YAML |
| Containers | Docker, Docker Compose |
| Quality | Ruff, pytest, ty |

## License

MIT
