"""RECAST — Run pipeline locally (without Prefect).

Utility script for development and debugging.  Executes the
pipeline steps sequentially without the orchestrator.

Usage:
    uv run python scripts/run_local_pipeline.py --step ingestion --date 20260225
    uv run python scripts/run_local_pipeline.py --step preprocessing --date 20260225
    uv run python scripts/run_local_pipeline.py --step training --wind-features data/features/wind.csv
    uv run python scripts/run_local_pipeline.py --step prediction --date 20260225
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import setup_logging  # noqa: E402


def main() -> None:
    """Parse arguments and run the selected pipeline step."""
    parser = argparse.ArgumentParser(description="RECAST — Local pipeline runner")
    parser.add_argument(
        "--step",
        choices=["ingestion", "preprocessing", "training", "prediction"],
        required=True,
        help="Pipeline step to execute",
    )
    parser.add_argument("--date", help="Date in YYYYMMDD format")
    parser.add_argument("--wind-features", help="Path to wind features CSV (training)")
    parser.add_argument(
        "--solar-features", help="Path to solar features CSV (training)"
    )
    parser.add_argument("--target", default="generation_mwh", help="Target column name")

    args = parser.parse_args()

    logger = setup_logging()
    logger.info("Running pipeline step: %s", args.step)

    if args.step == "ingestion":
        from src.ingestion.downloader import download_and_store_forecast

        result = download_and_store_forecast(date=args.date)
        logger.info("Result: %s", result)

    elif args.step == "preprocessing":
        from flows.preprocessing_flow import preprocessing_flow

        results = preprocessing_flow(date=args.date)
        for tech, path in results.items():
            logger.info("  %s → %s", tech, path)

    elif args.step == "training":
        from src.training.trainer import train_pipeline

        if args.wind_features:
            results = train_pipeline(
                features_path=args.wind_features,
                target_col=args.target,
                model_output_path=f"data/models/wind/{args.date or 'latest'}/model",
            )
            logger.info("Wind results: %s", results["test_metrics"])

        if args.solar_features:
            results = train_pipeline(
                features_path=args.solar_features,
                target_col=args.target,
                model_output_path=f"data/models/solar/{args.date or 'latest'}/model",
            )
            logger.info("Solar results: %s", results["test_metrics"])

    elif args.step == "prediction":
        from src.prediction.batch import run_batch_prediction

        for tech in ["wind", "solar"]:
            try:
                output = run_batch_prediction(date=args.date, technology=tech)
                logger.info("  %s predictions → %s", tech, output)
            except FileNotFoundError as e:
                logger.warning("  Skipping %s: %s", tech, e)

    logger.info("Done.")


if __name__ == "__main__":
    main()
