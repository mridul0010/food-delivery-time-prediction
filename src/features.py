from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
import typer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import (
    PROCESSED_DATA_DIR,
    TARGET_COLUMN,
    FEATURES_OUTPUT_PATH,
    LABELS_OUTPUT_PATH,
    NUMERICAL_COLUMNS,
    ONE_HOT_COLUMNS,
    ORDINAL_COLUMNS,
    ROAD_TRAFFIC_DENSITY,
    VEHICLE_CONDITION ,
    FESTIVAL, 
    DELIVERY_RATING_GROUP,
    AGE_GROUP ,
    DISTANCE_GROUP ,
)

app = typer.Typer()


def _required_feature_columns() -> set[str]:
    return set(ONE_HOT_COLUMNS + ORDINAL_COLUMNS + NUMERICAL_COLUMNS) | {
        "Delivery_person_Ratings",
        "Delivery_person_Age",
        "distance_km",
        "Order_Datetime",
        "Pickup_Datetime",
        "Delivery_Agent",
    }

def _create_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        return OneHotEncoder(sparse=False, handle_unknown="ignore")


def _safe_qcut_bins(series: pd.Series, q: int) -> np.ndarray:
    clean_series = pd.to_numeric(series, errors="coerce").dropna()
    if clean_series.empty:
        return np.array([0.0, 1.0])

    quantiles = np.linspace(0, 1, q + 1)
    bins = np.unique(np.quantile(clean_series.to_numpy(), quantiles))

    if len(bins) < 2:
        value = float(clean_series.iloc[0])
        return np.array([value - 0.5, value + 0.5])

    if len(bins) == 2:
        bins = np.array([bins[0] - 0.5, bins[1] + 0.5])

    return bins.astype(str)


class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.rating_bins: np.ndarray | None = None

    def fit(self, X, y=None):
        X = X.copy()
        self.rating_bins = _safe_qcut_bins(X["Delivery_person_Ratings"], q=3)
        return self

    def transform(self, X):
        X = X.copy()

        X["Order_Datetime"] = pd.to_datetime(X["Order_Datetime"], errors="coerce")
        X["Pickup_Datetime"] = pd.to_datetime(X["Pickup_Datetime"], errors="coerce")

        if self.rating_bins is None:
            raise ValueError("FeatureEngineering must be fitted before calling transform().")

        X["delivery_rating_group"] = pd.cut(
            X["Delivery_person_Ratings"],
            bins=self.rating_bins,
            labels=DELIVERY_RATING_GROUP[: len(self.rating_bins) - 1],
            include_lowest=True,
        )

        X["age_group"] = pd.cut(
            X["Delivery_person_Age"],
            bins=[14, 25, 35, 60],
            labels=AGE_GROUP,
            include_lowest=True,
        )

        X["distance_group"] = pd.cut(
            X["distance_km"],
            bins=[0, 5, 10, 25],
            labels=DISTANCE_GROUP,
            include_lowest=True,
        )

        X["Prep_Time(min)"] = (
            X["Pickup_Datetime"] - X["Order_Datetime"]
        ).dt.total_seconds() / 60

        X["Order_hour"] = X["Order_Datetime"].dt.hour
        X["Order_day"] = X["Order_Datetime"].dt.day_name()
        X["isWeekend"] = X["Order_day"].isin(["Saturday", "Sunday"]).astype(int)

        X["Time_Of_Day"] = pd.cut(
            X["Order_hour"],
            bins=[0, 6, 12, 18, 24],
            labels=["Night", "Morning", "Afternoon", "Evening"],
            include_lowest=True,
            right=True,
        )

        X = X.drop(columns=["Order_Datetime", "Pickup_Datetime", "Delivery_Agent"], errors="ignore")
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features


def _build_pipeline() -> Pipeline:
    transformer = ColumnTransformer(
        transformers=[
            ("ohe", _create_one_hot_encoder(), ONE_HOT_COLUMNS),
            (
                "oe",
                OrdinalEncoder(
                    categories=[
                        ROAD_TRAFFIC_DENSITY,
                        VEHICLE_CONDITION,
                        FESTIVAL,
                        DELIVERY_RATING_GROUP,
                        AGE_GROUP,
                        DISTANCE_GROUP,
                    ]
                ),
                ORDINAL_COLUMNS,
            ),
            ("scaling", StandardScaler(), NUMERICAL_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineering()),
            ("column_transformer", transformer),
        ]
    )


def _to_dataframe(transformed_features, feature_names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(transformed_features, columns=feature_names)


def _validate_input_frame(df: pd.DataFrame, input_path: Path) -> None:
    required_columns = _required_feature_columns() | {TARGET_COLUMN}
    missing_columns = sorted(required_columns.difference(df.columns))

    if missing_columns:
        raise ValueError(
            f"Missing required columns in {input_path}: {', '.join(missing_columns)}"
        )


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "Cleaned Delivery Dataset.csv",
    features_path: Path = FEATURES_OUTPUT_PATH,
    labels_path: Path = LABELS_OUTPUT_PATH,
):
    try:
        logger.info("Reading cleaned dataset from {}", input_path)
        df = pd.read_csv(input_path)

        _validate_input_frame(df, input_path)

        logger.info(
            "Preparing feature matrix from {} rows and {} columns",
            df.shape[0],
            df.shape[1],
        )
        labels = df[TARGET_COLUMN].copy()
        features = df.drop(columns=[TARGET_COLUMN])

        pipeline = _build_pipeline()
        transformed = pipeline.fit_transform(features)

        column_transformer = pipeline.named_steps["column_transformer"]
        feature_names = list(column_transformer.get_feature_names_out())
        feature_frame = _to_dataframe(transformed, feature_names)

        features_path.parent.mkdir(parents=True, exist_ok=True)
        labels_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Writing engineered features to {}", features_path)
        feature_frame.to_csv(features_path, index=False)

        logger.info("Writing labels to {}", labels_path)
        labels.to_frame(name=TARGET_COLUMN).to_csv(labels_path, index=False)

        logger.success(
            "Feature engineering complete: {} feature columns and {} labels",
            feature_frame.shape[1],
            labels.shape[0],
        )
    except FileNotFoundError:
        logger.exception("Input file not found: {}", input_path)
        raise typer.Exit(code=1)
    except (pd.errors.ParserError, ValueError, KeyError, OSError) as exc:
        logger.exception("Feature engineering failed: {}", exc)
        raise typer.Exit(code=1)
    except Exception:
        logger.exception("Unexpected error while engineering features")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
