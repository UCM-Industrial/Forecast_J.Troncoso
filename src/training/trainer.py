"""RECAST — Training pipeline.

End-to-end training workflow: load data → split → train → evaluate → save.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.training.evaluate import evaluate_regression_metrics
from src.training.model import XGBoostForecaster
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger("training.trainer")


def train_pipeline(
    features_path: str | Path,
    target_col: str,
    model_output_path: str | Path,
    *,
    feature_cols: list[str] | None = None,
    test_size: float | None = None,
    cv_folds: int | None = None,
) -> dict:
    """Run the full training pipeline.

    Steps:
        1. Load feature CSV.
        2. Split into train/test (temporal split).
        3. Cross-validate on the training set.
        4. Fit final model on the full training set.
        5. Evaluate on the held-out test set.
        6. Save model + metadata.

    Args:
        features_path: Path to the CSV with features and target.
        target_col: Name of the target column.
        model_output_path: Base path for saving the model.
        feature_cols: Feature columns to use.  If ``None``, uses all
            numeric columns except the target.
        test_size: Fraction of data for testing.  Defaults to config.
        cv_folds: Number of cross-validation folds.  Defaults to config.

    Returns:
        Dict with training results: metrics, feature importance, paths.
    """
    settings = get_settings()
    test_size = test_size or settings.training.test_size
    cv_folds = cv_folds or settings.training.cv_folds

    # ── Step 1: Load data ────────────────────────────────────
    logger.info("Loading features from %s", features_path)
    from src.preprocessing.features import read_csv_with_datetime

    df = read_csv_with_datetime(features_path)

    if feature_cols is None:
        numeric_cols = df.select_dtypes("number").columns.tolist()
        feature_cols = [c for c in numeric_cols if c != target_col]

    x = df[feature_cols]
    y = df[target_col]

    logger.info(
        "Data loaded: %d samples, %d features, target='%s'",
        len(df),
        len(feature_cols),
        target_col,
    )

    # ── Step 2: Temporal split ───────────────────────────────
    split_idx = int(len(df) * (1 - test_size))
    x_train, x_test = x.iloc[:split_idx], x.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(
        "Split: train=%d, test=%d (%.0f%% test)",
        len(x_train),
        len(x_test),
        test_size * 100,
    )

    # ── Step 3: Cross-validate ───────────────────────────────
    cv_scores = cross_validate(x_train, y_train, cv_folds=cv_folds)
    logger.info(
        "CV results — MAE: %.4f ± %.4f, R²: %.4f",
        cv_scores["mae_mean"],
        cv_scores["mae_std"],
        cv_scores["r2_mean"],
    )

    # ── Step 4: Train final model ────────────────────────────
    model = XGBoostForecaster()
    model.fit(x_train, y_train)

    # ── Step 5: Evaluate on test set ─────────────────────────
    y_pred = model.predict(x_test)
    test_metrics = evaluate_regression_metrics(y_test, y_pred)
    logger.info(
        "Test metrics — MAE: %.4f, MSE: %.4f, R²: %.4f",
        test_metrics["mae"],
        test_metrics["mse"],
        test_metrics["r2"],
    )

    # Add metrics to model metadata
    model._metadata["test_metrics"] = test_metrics
    model._metadata["cv_scores"] = cv_scores

    # ── Step 6: Save ─────────────────────────────────────────
    model_path = model.save(model_output_path)
    logger.info("Training pipeline complete. Model saved to %s", model_path)

    return {
        "model_path": str(model_path),
        "test_metrics": test_metrics,
        "cv_scores": cv_scores,
        "feature_importance": model.get_feature_importance(),
        "n_train": len(x_train),
        "n_test": len(x_test),
        "features": feature_cols,
    }


def cross_validate(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    cv_folds: int = 5,
) -> dict[str, float]:
    """Run time-series cross-validation.

    Args:
        x: Feature DataFrame.
        y: Target Series.
        cv_folds: Number of folds.

    Returns:
        Dict with mean and std of MAE, MSE, and R² across folds.
    """
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    scores: dict[str, list[float]] = {"mae": [], "mse": [], "r2": []}

    for fold, (train_idx, val_idx) in enumerate(tscv.split(x), start=1):
        x_tr, x_val = x.iloc[train_idx], x.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_model = XGBoostForecaster()
        fold_model.fit(x_tr, y_tr)
        y_pred = fold_model.predict(x_val)

        metrics = evaluate_regression_metrics(y_val, y_pred)
        for key in scores:
            scores[key].append(metrics[key])

        logger.debug(
            "Fold %d/%d — MAE: %.4f, R²: %.4f",
            fold,
            cv_folds,
            metrics["mae"],
            metrics["r2"],
        )

    import numpy as np

    return {
        "mae_mean": float(np.mean(scores["mae"])),
        "mae_std": float(np.std(scores["mae"])),
        "mse_mean": float(np.mean(scores["mse"])),
        "mse_std": float(np.std(scores["mse"])),
        "r2_mean": float(np.mean(scores["r2"])),
        "r2_std": float(np.std(scores["r2"])),
        "cv_folds": cv_folds,
    }
