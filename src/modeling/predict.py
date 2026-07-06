from dotenv import load_dotenv  
import os
from pathlib import Path


load_dotenv()

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from loguru import logger
import typer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

from src.config import MODELS_DIR, TARGET_COLUMN, TEST_OUTPUT_PATH, TRAIN_OUTPUT_PATH, PROJ_ROOT

app = typer.Typer()

try:
    import mlflow
    import mlflow.models
    import mlflow.data
except ImportError:  # pragma: no cover
    mlflow = None


def _resolve_tracking_uri() -> str | None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        return tracking_uri

    dagshub_owner = os.getenv("DAGSHUB_REPO_OWNER")
    dagshub_repo = os.getenv("DAGSHUB_REPO_NAME")
    if dagshub_owner and dagshub_repo:
        return f"https://dagshub.com/{dagshub_owner}/{dagshub_repo}.mlflow"

    return None


def _configure_mlflow() -> bool:
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "DVC Pipeline")

    if not tracking_uri or not repo_owner or not repo_name:
        print("MLflow/DagsHub environment variables are missing from your .env file!")
        return False

    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME", repo_owner)
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN", "")

    import dagshub
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return True


def _load_dataset_splits(
    features_path: Path, labels_path: Path
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)

    if features_df.empty or labels_df.empty:
        raise ValueError("Provided training or testing evaluation splits are empty.")
    if TARGET_COLUMN not in labels_df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in {labels_path}")
    if len(features_df) != len(labels_df):
        raise ValueError("Feature and label row counts do not match.")

    full_df = features_df.copy()
    full_df[TARGET_COLUMN] = labels_df[TARGET_COLUMN].values

    return features_df, labels_df[TARGET_COLUMN], full_df


def _load_model(model_path: Path) -> Any:
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    model = joblib.load(model_path)
    if not hasattr(model, "predict"):
        raise TypeError(f"Loaded artifact from {model_path} does not expose predict().")

    return model


def _compute_cross_val_score(
    model: Any, X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[float, list[float]]:
    logger.info("Evaluating robust cross-validation scores across 5 training splits...")
    
    cv_scores = cross_val_score(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    individual_maes = [-float(score) for score in cv_scores]
    mean_mae = float(np.mean(individual_maes))
    
    return mean_mae, individual_maes


def _save_run_information(
    save_json_path: Path,
    run_id: str,
    artifact_path: str,
    model_name: str,
):
    info_dict = {
        "run_id": run_id,
        "artifact_path": artifact_path,
        "model_name": model_name,
        "model_artifact_path": model_name,
    }
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(info_dict, f, indent=4)
    logger.info("Run execution metadata safely saved to {}", save_json_path)


@app.command()
def main(
    test_features_path: Path = TEST_OUTPUT_PATH / "test_trans.csv",
    test_labels_path: Path = TEST_OUTPUT_PATH / "test_labels.csv",
    train_features_path: Path = TRAIN_OUTPUT_PATH / "train_trans.csv",
    train_labels_path: Path = TRAIN_OUTPUT_PATH / "train_labels.csv",
    model_path: Path = MODELS_DIR / "model.joblib",
    # Fixed explicitly to match the external path requirements expected by the stage tracker
    predictions_path: Path = PROJ_ROOT / "data" / "external" / "test_predictions.csv",
    metrics_path: Path = PROJ_ROOT / "data" / "external" / "evaluation_metrics.json",
    run_info_path: Path = PROJ_ROOT / "run_information.json",
):
    try:
        use_mlflow = _configure_mlflow()

        logger.info("Loading validation/testing dataset splits...")
        X_test, y_test, test_full_df = _load_dataset_splits(test_features_path, test_labels_path)

        logger.info("Loading training dataset splits for full pipeline reporting...")
        X_train, y_train, train_full_df = _load_dataset_splits(train_features_path, train_labels_path)

        logger.info("Loading target pipeline model artifact from {}", model_path)
        model = _load_model(model_path)

        logger.info("Executing batch inference across dataset splits...")
        y_train_pred = np.asarray(model.predict(X_train))
        y_test_pred = np.asarray(model.predict(X_test))

        train_mae = float(mean_absolute_error(y_train, y_train_pred))
        test_mae = float(mean_absolute_error(y_test, y_test_pred))
        train_r2 = float(r2_score(y_train, y_train_pred))
        test_r2 = float(r2_score(y_test, y_test_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
        bias = float(np.mean(y_test_pred - y_test.to_numpy()))

        mean_cv_mae, cv_folds = _compute_cross_val_score(model, X_train, y_train)

        # Unified metric log payload: includes mean cross-validation score securely
        metrics = {
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "test_rmse": rmse,
            "prediction_bias": bias,
            "cv_score_mae": mean_cv_mae
        }
        
        predictions_df = pd.DataFrame(
            {
                "actual": y_test.to_numpy(),
                "prediction": y_test_pred,
                "residual": y_test.to_numpy() - y_test_pred,
            }
        )

        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Exporting evaluation test predictions onto {}", predictions_path)
        predictions_df.to_csv(predictions_path, index=False)

        logger.info("Exporting generated metric payload onto {}", metrics_path)
        pd.Series(metrics).to_json(metrics_path, indent=2)

        if use_mlflow:
            model_name = "delivery_time_pred_model"
            with mlflow.start_run(run_name=os.getenv("MLFLOW_RUN_NAME", "evaluation")) as run:
                mlflow.set_tag("model", "Food Delivery Time Regressor")

                mlflow.log_param("model_path", str(model_path))
                mlflow.log_param("test_features_path", str(test_features_path))
                mlflow.log_param("test_labels_path", str(test_labels_path))
                
                # Extract and log hyperparameter structures out of the internal StackingRegressor
                if hasattr(model, "regressor_"):
                    all_params = model.regressor_.get_params()
                elif hasattr(model, "regressor") and model.regressor is not None:
                    all_params = model.regressor.get_params()
                else:
                    all_params = model.get_params()

                # Clean the parameters dictionary to verify primitives are passed to MLflow
                filtered_params = {
                    k: str(v) if isinstance(v, (list, tuple, dict, object)) and v is not None else v
                    for k, v in all_params.items()
                }
                mlflow.log_params(filtered_params)

                mlflow.log_metric("train_mae", train_mae)
                mlflow.log_metric("test_mae", test_mae)
                mlflow.log_metric("train_r2", train_r2)
                mlflow.log_metric("test_r2", test_r2)
                mlflow.log_metric("test_rmse", rmse)
                mlflow.log_metric("prediction_bias", bias)
                mlflow.log_metric("mean_cv_score", mean_cv_mae)

                for i, fold_score in enumerate(cv_folds):
                    mlflow.log_metric(f"CV {i}", fold_score)
                    
                train_data_input = mlflow.data.from_pandas(train_full_df, targets=TARGET_COLUMN)
                test_data_input = mlflow.data.from_pandas(test_full_df, targets=TARGET_COLUMN)
                mlflow.log_input(dataset=train_data_input, context="training")
                mlflow.log_input(dataset=test_data_input, context="validation")

                sample_input = X_train.sample(20, random_state=42)
                sample_output = np.asarray(model.predict(sample_input))
                model_signature = mlflow.models.infer_signature(model_input=sample_input, model_output=sample_output)

                trusted_types = [
                    "sklearn.utils._bunch.Bunch",
                    "xgboost.core.Booster",
                    "xgboost.sklearn.XGBRegressor",
                ]

                # Fixed parameter mapping from name= to artifact_path= to resolve registration lookups
                mlflow.sklearn.log_model(
                    sk_model=model, 
                    artifact_path=model_name, 
                    signature=model_signature,
                    skops_trusted_types=trusted_types
                )
                
                mlflow.log_artifact(str(predictions_path))
                mlflow.log_artifact(str(metrics_path))
                
                artifact_uri = mlflow.get_artifact_uri()
                run_id = run.info.run_id
                
            _save_run_information(
                save_json_path=run_info_path,
                run_id=run_id,
                artifact_path=artifact_uri,
                model_name=model_name
            )

        logger.success(
            "Evaluation complete! Test MAE={:.4f}, Test R2={:.4f}, CV Mean MAE={:.4f}",
            metrics["test_mae"],
            metrics["test_r2"],
            metrics["cv_score_mae"],
        )
    except FileNotFoundError as exc:
        logger.exception("Evaluation routing execution halted due to missing path resource: {}", exc)
        raise typer.Exit(code=1)
    except (pd.errors.ParserError, ValueError, TypeError, OSError) as exc:
        logger.exception("Evaluation phase validation process failed: {}", exc)
        raise typer.Exit(code=1)
    except Exception:
        logger.exception("Unexpected system exception thrown during evaluation execution run loop.")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()