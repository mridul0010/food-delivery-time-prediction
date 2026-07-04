from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import typer
import yaml
from loguru import logger
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor

from src.config import MODELS_DIR, TRAIN_OUTPUT_PATH, PROJ_ROOT, TARGET_COLUMN

app = typer.Typer()


def _load_params(params_path: Path) -> dict[str, Any]:
    if not params_path.exists():
        raise FileNotFoundError(f"Parameters file not found: {params_path}")

    with params_path.open("r", encoding="utf-8") as file_handle:
        params = yaml.safe_load(file_handle) or {}

    if "Train" not in params:
        raise ValueError(f"Missing 'Train' section in {params_path}")

    return params


def _load_training_frame(features_path: Path, labels_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)

    if features_df.empty:
        raise ValueError(f"Features file is empty: {features_path}")
    if labels_df.empty:
        raise ValueError(f"Labels file is empty: {labels_path}")
    if TARGET_COLUMN not in labels_df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in {labels_path}")
    if len(features_df) != len(labels_df):
        raise ValueError(
            "Feature and label row counts do not match: "
            f"{len(features_df)} features vs {len(labels_df)} labels"
        )

    return features_df, labels_df[TARGET_COLUMN]


def _build_model(train_params: dict[str, Any]) -> Pipeline:
    rf_params = dict(train_params.get("best_rf_params", {}))
    xgb_params = dict(train_params.get("best_xgb_params", {}))

    if not rf_params:
        raise ValueError("Missing 'best_rf_params' under Train in params.yaml")
    if not xgb_params:
        raise ValueError("Missing 'best_xgb_params' under Train in params.yaml")

    rf_params.setdefault("random_state", 42)
    rf_params.setdefault("n_jobs", -1)

    xgb_params.setdefault("random_state", 42)
    xgb_params.setdefault("n_jobs", -1)
    xgb_params.setdefault("objective", "reg:squarederror")

    stacked_model = StackingRegressor(
        estimators=[
            ("rf", RandomForestRegressor(**rf_params)),
            ("xgb", XGBRegressor(**xgb_params)),
        ],
        final_estimator=LinearRegression(),
        cv=5,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            (
                "model",
                TransformedTargetRegressor(
                    regressor=stacked_model,
                    transformer=PowerTransformer(),
                ),
            ),
        ]
    )


@app.command()
def main(
    train_features_path: Path = TRAIN_OUTPUT_PATH / "train_trans.csv",
    train_labels_path: Path = TRAIN_OUTPUT_PATH / "train_labels.csv",
    model_path: Path = MODELS_DIR / "model.joblib",
    params_path: Path = PROJ_ROOT / "params.yaml",
):
    try:
        logger.info("Loading preprocessed training data...")
        X_train, y_train = _load_training_frame(train_features_path, train_labels_path)
        
        logger.info("Loading training parameters from {}", params_path)
        params = _load_params(params_path)
        train_params = params["Train"]

        model = _build_model(train_params)

        logger.info("Fitting the final stacking regressor model pipeline...")
        model.fit(X_train, y_train)

        model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving trained model artifact to {}", model_path)
        joblib.dump(model, model_path)

        logger.success("Training run completed successfully! Model pipeline artifact generated.")

    except FileNotFoundError as exc:
        logger.exception("Training input missing: {}", exc)
        raise typer.Exit(code=1)
    except (pd.errors.ParserError, ValueError, KeyError, OSError) as exc:
        logger.exception("Training pipeline execution failed: {}", exc)
        raise typer.Exit(code=1)
    except Exception:
        logger.exception("Unexpected error occurred while training the model")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()