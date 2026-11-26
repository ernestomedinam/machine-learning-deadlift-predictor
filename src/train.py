"""Train and evaluate a deadlift regression model.

The script recreates the notebook's lifter-grouped train/validation/test split,
compares ready-to-use scikit-learn regressors on validation data, then refits
preprocessing and the selected model on train + validation before its single
final evaluation on the untouched test split. The fitted pipeline is saved for
use by app.py.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "openpowerlifting-2021-06-26-027dc895.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "deadlift_model.joblib"

FEATURE_COLUMNS = [
    "Sex",
    "Equipment",
    "Age",
    "BodyweightKg",
    "Event",
    "max_squat_kg",
    "max_bench_kg",
]
CATEGORICAL_FEATURES = ["Sex", "Equipment", "Event"]
NUMERIC_FEATURES = ["Age", "BodyweightKg", "max_squat_kg", "max_bench_kg"]
TARGET_COLUMN = "max_deadlift_kg"
GROUP_COLUMN = "Name"


def _positive_max_from_columns(
    frame: pd.DataFrame,
    attempt_pattern: str,
    best_column: str,
) -> pd.Series:
    """Return each row's highest successful lift across attempts and summary."""
    attempt_columns = [
        column for column in frame.columns if re.fullmatch(attempt_pattern, column)
    ]
    source_columns = attempt_columns.copy()
    if best_column in frame.columns:
        source_columns.append(best_column)

    if not source_columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")

    lifts = frame[source_columns].apply(pd.to_numeric, errors="coerce")
    return lifts.where(lifts > 0).max(axis=1)


def load_modeling_data(csv_path: Path) -> pd.DataFrame:
    """Load source records and derive the target and model features."""
    frame = pd.read_csv(csv_path, low_memory=False)

    frame[TARGET_COLUMN] = _positive_max_from_columns(
        frame,
        r"Deadlift\d+Kg",
        "Best3DeadliftKg",
    )
    frame["max_squat_kg"] = _positive_max_from_columns(
        frame,
        r"Squat\d+Kg",
        "Best3SquatKg",
    )
    frame["max_bench_kg"] = _positive_max_from_columns(
        frame,
        r"Bench\d+Kg",
        "Best3BenchKg",
    )

    required = FEATURE_COLUMNS + [TARGET_COLUMN, GROUP_COLUMN]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Required columns are missing from the source CSV: {missing}")

    # Successful deadlift is required for a supervised target; Name is required
    # to guarantee lifter-group isolation between dataset splits.
    frame = frame.loc[
        frame[TARGET_COLUMN].notna() & (frame[TARGET_COLUMN] > 0)
    ].dropna(subset=[GROUP_COLUMN])

    return frame[FEATURE_COLUMNS + [TARGET_COLUMN, GROUP_COLUMN]].copy()


def make_preprocessor() -> ColumnTransformer:
    """Build the same train-fitted imputation/encoding/scaling used in exploration."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def make_pipeline(estimator: Any) -> Pipeline:
    """Combine feature preprocessing and an estimator into one deployable unit."""
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("regressor", estimator),
        ]
    )


def regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    """Compute standard regression metrics, with error values in kilograms."""
    return {
        "mae_kg": float(mean_absolute_error(actual, predicted)),
        "rmse_kg": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
    }


def split_by_lifter(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return raw feature/target train, validation, and test frames by Name."""
    X = frame[FEATURE_COLUMNS]
    y = frame[TARGET_COLUMN]
    groups = frame[GROUP_COLUMN]

    test_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_validation_indices, test_indices = next(
        test_splitter.split(X, y, groups=groups)
    )

    X_train_validation = X.iloc[train_validation_indices]
    y_train_validation = y.iloc[train_validation_indices]
    groups_train_validation = groups.iloc[train_validation_indices]

    validation_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=43)
    train_indices, validation_indices = next(
        validation_splitter.split(
            X_train_validation,
            y_train_validation,
            groups=groups_train_validation,
        )
    )

    train = frame.iloc[train_validation_indices].iloc[train_indices]
    validation = frame.iloc[train_validation_indices].iloc[validation_indices]
    test = frame.iloc[test_indices]

    train_names = set(train[GROUP_COLUMN])
    validation_names = set(validation[GROUP_COLUMN])
    test_names = set(test[GROUP_COLUMN])
    if not train_names.isdisjoint(validation_names):
        raise AssertionError("A lifter appears in both train and validation splits.")
    if not train_names.isdisjoint(test_names):
        raise AssertionError("A lifter appears in both train and test splits.")
    if not validation_names.isdisjoint(test_names):
        raise AssertionError("A lifter appears in both validation and test splits.")

    return train, validation, test


def train_and_evaluate(
    csv_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    """Compare models, refit the winner on train+validation, test, and save it."""
    frame = load_modeling_data(csv_path)
    train, validation, test = split_by_lifter(frame)

    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
    X_validation, y_validation = validation[FEATURE_COLUMNS], validation[TARGET_COLUMN]
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN]

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    validation_predictions = {
        "Dummy mean baseline": baseline.predict(X_validation),
    }

    candidate_estimators = {
        "Linear regression": LinearRegression(),
        "Ridge regression (alpha=1.0)": Ridge(alpha=1.0),
    }
    candidate_pipelines: dict[str, Pipeline] = {}
    for name, estimator in candidate_estimators.items():
        candidate_pipeline = make_pipeline(estimator)
        candidate_pipeline.fit(X_train, y_train)
        candidate_pipelines[name] = candidate_pipeline
        validation_predictions[name] = candidate_pipeline.predict(X_validation)

    validation_results = {
        name: regression_metrics(y_validation, predictions)
        for name, predictions in validation_predictions.items()
    }
    selected_model_name = min(
        candidate_estimators,
        key=lambda name: validation_results[name]["mae_kg"],
    )

    # Important improvement: refit preprocessing and the selected estimator
    # together on all development data, then evaluate once on untouched test.
    X_train_validation = pd.concat([X_train, X_validation], axis=0)
    y_train_validation = pd.concat([y_train, y_validation], axis=0)
    final_pipeline = make_pipeline(candidate_estimators[selected_model_name])
    final_pipeline.fit(X_train_validation, y_train_validation)
    test_predictions = final_pipeline.predict(X_test)
    test_metrics = regression_metrics(y_test, test_predictions)

    artifact: dict[str, Any] = {
        "pipeline": final_pipeline,
        "model_name": selected_model_name,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "validation_metrics": validation_results,
        "test_metrics": test_metrics,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "preprocessing_fit_rows": int(len(X_train_validation)),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    summary = {
        "model_path": str(model_path),
        "selected_model": selected_model_name,
        "rows": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "validation_metrics": validation_results,
        "test_metrics": test_metrics,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Raw OpenPowerlifting CSV (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Output path for the trained model artifact (default: {DEFAULT_MODEL_PATH})",
    )
    args = parser.parse_args()

    summary = train_and_evaluate(args.data, args.model_output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
