from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from loguru import logger
import typer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import joblib

from src.config import (
    AGE_GROUP,
    DELIVERY_RATING_GROUP,
    DISTANCE_GROUP,
    TRAIN_OUTPUT_PATH,  
    TEST_OUTPUT_PATH,
    FESTIVAL,
    INTERIM_DATA_DIR,
    NUMERICAL_COLUMNS,
    ONE_HOT_COLUMNS,
    ORDINAL_COLUMNS,
    ROAD_TRAFFIC_DENSITY,
    TARGET_COLUMN,
    VEHICLE_CONDITION,
)

app = typer.Typer()


def _required_feature_columns() -> set[str]:
    """
    Returns only the raw columns that must exist in the input CSV files.
    Engineered features (like 'Order_day', 'Prep_Time(min)', etc.) are excluded 
    here since they don't exist until FeatureEngineering().transform() runs.
    """
    # 1. Gather all columns expected by the ColumnTransformer
    config_columns = set(ONE_HOT_COLUMNS + ORDINAL_COLUMNS + NUMERICAL_COLUMNS)
    
    # 2. Define the columns created dynamically during feature engineering
    engineered_columns = {
        "delivery_rating_group",
        "age_group",
        "distance_group",
        "Prep_Time(min)",
        "Order_hour",
        "Order_day",
        "isWeekend",
        "Time_Of_Day"
    }
    
    # 3. Define essential raw tracking fields required by the transformer logic
    raw_required_fields = {
        "Delivery_person_Ratings",
        "Delivery_person_Age",
        "distance_km",
        "Order_Datetime",
        "Pickup_Datetime",
        "Delivery_Agent",
    }
    
    # Raw columns required = (Configured columns - Engineered columns) + Raw required fields
    return (config_columns - engineered_columns) | raw_required_fields


def _create_one_hot_encoder() -> OneHotEncoder:
    return OneHotEncoder(sparse_output=False, handle_unknown="ignore")


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

    return bins


class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.rating_bins: np.ndarray | None = None
        self.feature_names_out_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        self.rating_bins = _safe_qcut_bins(X["Delivery_person_Ratings"], q=3)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if self.rating_bins is None:
            raise ValueError("FeatureEngineering must be fitted before calling transform().")

        X["Order_Datetime"] = pd.to_datetime(X["Order_Datetime"], errors="coerce")
        X["Pickup_Datetime"] = pd.to_datetime(X["Pickup_Datetime"], errors="coerce")

        X["delivery_rating_group"] = pd.cut(
            X["Delivery_person_Ratings"],
            bins=self.rating_bins,
            labels=DELIVERY_RATING_GROUP[: len(self.rating_bins) - 1],
            include_lowest=True,
        ).astype(str)

        X["age_group"] = pd.cut(
            X["Delivery_person_Age"],
            bins=[14, 25, 35, 60],
            labels=AGE_GROUP,
            include_lowest=True,
        ).astype(str)

        X["distance_group"] = pd.cut(
            X["distance_km"],
            bins=[0, 5, 10, 25],
            labels=DISTANCE_GROUP,
            include_lowest=True,
        ).astype(str)

        X["Prep_Time(min)"] = (
            X["Pickup_Datetime"] - X["Order_Datetime"]
        ).dt.total_seconds() / 60

        X["Order_hour"] = X["Order_Datetime"].dt.hour.fillna(-1).astype(int)
        X["Order_day"] = X["Order_Datetime"].dt.day_name().fillna("Unknown")
        X["isWeekend"] = X["Order_day"].isin(["Saturday", "Sunday"]).astype(int)

        X["Time_Of_Day"] = pd.cut(
            X["Order_hour"],
            bins=[-1, 6, 12, 18, 24],
            labels=["Night", "Morning", "Afternoon", "Evening"],
            include_lowest=True,
            right=True,
        ).astype(str)

        X = X.drop(columns=["Order_Datetime", "Pickup_Datetime", "Delivery_Agent"], errors="ignore")
        
        self.feature_names_out_ = X.columns.tolist()
        return X

    def get_feature_names_out(self, input_features=None):
        if self.feature_names_out_ is None:
            raise ValueError("Transformer must be fitted/transformed first.")
        return np.array(self.feature_names_out_, dtype=object)


def _build_pipeline() -> Pipeline:
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categories_config = [
        ROAD_TRAFFIC_DENSITY,
        VEHICLE_CONDITION,
        FESTIVAL,
        DELIVERY_RATING_GROUP,
        AGE_GROUP,
        DISTANCE_GROUP,
    ]

    transformer = ColumnTransformer(
        transformers=[
            ("ohe", _create_one_hot_encoder(), ONE_HOT_COLUMNS),
            (
                "oe",
                OrdinalEncoder(
                    categories=categories_config,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ORDINAL_COLUMNS,
            ),
            ("scaling", numerical_transformer, NUMERICAL_COLUMNS),
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


def _process_split(df: pd.DataFrame, path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    _validate_input_frame(df, path)
    labels = df[TARGET_COLUMN].copy()
    features = df.drop(columns=[TARGET_COLUMN])
    return features, labels


def save_transformer(transformer, save_dir: Path, transformer_name: str):
    save_location = save_dir / transformer_name
    joblib.dump(value=transformer, filename=save_location)
    logger.info("Successfully serialized transformer state to {}", save_location)


@app.command()
def main(
    train_input_path: Path = INTERIM_DATA_DIR / "train.csv",
    test_input_path: Path = INTERIM_DATA_DIR / "test.csv",
    train_output_dir: Path = TRAIN_OUTPUT_PATH,
    test_output_dir: Path = TEST_OUTPUT_PATH,
):
    try:
        logger.info("Reading interim datasets...")
        train_df = pd.read_csv(train_input_path)
        test_df = pd.read_csv(test_input_path)

        logger.info("Validating and parsing train/test splits...")
        X_train, y_train = _process_split(train_df, train_input_path)
        X_test, y_test = _process_split(test_df, test_input_path)

        pipeline = _build_pipeline()

        logger.info("Fitting pipeline and transforming training data...")
        train_transformed = pipeline.fit_transform(X_train)

        logger.info("Transforming test data using fitted states...")
        test_transformed = pipeline.transform(X_test)

        column_transformer = pipeline.named_steps["column_transformer"]
        feature_names = list(column_transformer.get_feature_names_out())
        
        train_features_frame = _to_dataframe(train_transformed, feature_names)
        test_features_frame = _to_dataframe(test_transformed, feature_names)

        train_output_dir.mkdir(parents=True, exist_ok=True)
        test_output_dir.mkdir(parents=True, exist_ok=True)

        train_feat_path = train_output_dir / "train_trans.csv"
        test_feat_path = test_output_dir / "test_trans.csv"
        train_lbl_path = train_output_dir / "train_labels.csv"
        test_lbl_path = test_output_dir / "test_labels.csv"

        logger.info("Saving engineered features to disk...")
        train_features_frame.to_csv(train_feat_path, index=False)
        test_features_frame.to_csv(test_feat_path, index=False)

        transformer_filename = "preprocessor.joblib"
        try:
            root_path = Path(__file__).resolve().parent.parent
        except NameError:
            root_path = Path(".").resolve()
            
        transformer_save_dir = root_path / "models"
        transformer_save_dir.mkdir(parents=True, exist_ok=True)
        
        save_transformer(
            transformer=pipeline,
            save_dir=transformer_save_dir,
            transformer_name=transformer_filename
        )

        logger.info("Saving target labels to disk...")
        y_train.to_frame(name=TARGET_COLUMN).to_csv(train_lbl_path, index=False)
        y_test.to_frame(name=TARGET_COLUMN).to_csv(test_lbl_path, index=False)

        logger.success(
            "Feature engineering complete! Train: {} rows. Test: {} rows.",
            train_features_frame.shape[0],
            test_features_frame.shape[0],
        )

    except FileNotFoundError as exc:
        logger.exception("Input data split file missing: {}", exc.filename)
        raise typer.Exit(code=1)
    except (pd.errors.ParserError, ValueError, KeyError, OSError) as exc:
        logger.exception("Feature engineering transformation process failed: {}", exc)
        raise typer.Exit(code=1)
    except Exception:
        logger.exception("Unexpected exception caught during execution run loop.")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()