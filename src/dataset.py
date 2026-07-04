from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from loguru import logger 
import typer

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

CITY_MAPPING = {
    "INDO": "Indore", "BANG": "Bangalore", "COIMB": "Coimbatore",
    "CHEN": "Chennai", "HYD": "Hyderabad", "RANCHI": "Ranchi",
    "MYS": "Mysore", "DEH": "Dehradun", "KOC": "Kochi",
    "PUNE": "Pune", "LUDH": "Ludhiana", "KNP": "Kanpur",
    "MUM": "Mumbai", "KOL": "Kolkata", "JAP": "Jaipur",
    "SUR": "Surat", "SU": "Surat", "GOA": "Goa",
    "AURG": "Aurangabad", "AGR": "Agra", "AG": "Agra",
    "VAD": "Vadodara", "ALH": "Prayagraj", "BHP": "Bhopal",
}


def _safe_mode_vectorized(series: pd.Series, default: Any = "Unknown") -> Any:
    """Safely retrieves the most frequent categorical value (mode) without crashing."""
    mode_vals = series.mode().dropna()
    return mode_vals.iloc[0] if not mode_vals.empty else default


def extract_delivery_person_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts and clean vectorized City, Zone, and Agent information from the ID."""
    logger.info("Extracting delivery person metadata from ID")
    raw_id = df["Delivery_person_ID"].astype(str)
    
    # Extract & map City
    city_code = raw_id.str[:-10].str.rstrip("R")
    df.insert(1, "City", city_code.map(CITY_MAPPING).fillna(city_code))
    
    # Extract & clean Zone
    zone_code = raw_id.str[-10:-5].str.strip()
    zone_code = zone_code.str.replace("ES010", "Zone10", regex=False).str.replace("RES", "Zone", regex=False)
    df.insert(2, "Zone", zone_code)
    
    # Extract Delivery Agent numeric suffix
    delivery_agent = raw_id.str[-2:].str.strip()
    df.insert(3, "Delivery_Agent", pd.to_numeric(delivery_agent, errors="coerce").astype("Int32"))
    return df


def process_timestamps_and_prep_time(df: pd.DataFrame, global_median_prep: float) -> pd.DataFrame:
    """Parses timestamps, fixes midnight boundaries, and calculates prep times."""
    logger.info("Processing time-series transformations and date features")
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")

    # Fast cleanup for overnight shifts (e.g., '24:15' -> '00:15')
    time_ordered_str = df["Time_Orderd"].astype(str).str.strip()
    time_picked_str = df["Time_Order_picked"].astype(str).str.strip()
    time_picked_str = np.where(time_picked_str.str.startswith("24:"), "00:" + time_picked_str.str.split(":").str[1], time_picked_str)

    # Convert string arrays directly to timedelta offsets
    td_ordered = pd.to_timedelta(time_ordered_str + ":00", errors="coerce")
    td_picked = pd.to_timedelta(time_picked_str + ":00", errors="coerce")

    # Combine Date + Time
    df.insert(10, "Order_Datetime", df["Order_Date"] + td_ordered)
    df.insert(11, "Pickup_Datetime", df["Order_Date"] + td_picked)

    # FIXED: Replaced += with explicit assignment to eliminate DeprecationWarning
    overnight_mask = df["Pickup_Datetime"] < df["Order_Datetime"]
    logger.info("Adjusting {} overnight pickup timestamps", int(overnight_mask.sum()))
    df.loc[overnight_mask, "Pickup_Datetime"] = df.loc[overnight_mask, "Pickup_Datetime"] + pd.Timedelta(days=1)

    df = df.drop(columns=["Order_Date", "Time_Orderd", "Time_Order_picked"])

    # Calculate operational Prep Time
    df.insert(12, "Prep_Time(min)", (df["Pickup_Datetime"] - df["Order_Datetime"]).dt.total_seconds() / 60)

    # Drop unrecoverable records missing both points
    df = df.dropna(subset=["Order_Datetime", "Pickup_Datetime"], how="all").reset_index(drop=True)

    # Impute single missing datetimes using median prep windows
    valid_prep_times = df["Prep_Time(min)"].dropna()
    median_prep_time = valid_prep_times.median() if not valid_prep_times.empty else global_median_prep
    
    logger.info("Imputing missing datetimes using prep time window of {} minutes", round(float(median_prep_time), 2))
    order_missing = df["Order_Datetime"].isna()
    pickup_missing = df["Pickup_Datetime"].isna()
    
    # FIXED: Replaced -= and += with explicit assignments to eliminate DeprecationWarning
    df.loc[order_missing, "Order_Datetime"] = df.loc[order_missing, "Pickup_Datetime"] - pd.Timedelta(minutes=median_prep_time)
    df.loc[pickup_missing, "Pickup_Datetime"] = df.loc[pickup_missing, "Order_Datetime"] + pd.Timedelta(minutes=median_prep_time)

    # Re-compute exact delta values
    df["Prep_Time(min)"] = (df["Pickup_Datetime"] - df["Order_Datetime"]).dt.total_seconds() / 60
    return df


def impute_missing_features(df: pd.DataFrame, refs: Dict[str, Any]) -> pd.DataFrame:
    """Applies conditional mode and median group imputations across missing fields."""
    logger.info("Dropping rows missing major target-side features")
    target_side_null_mask = df["Delivery_person_Ratings"].isna() & df["Weather_conditions"].isna() & df["Road_traffic_density"].isna()
    df = df[~target_side_null_mask].reset_index(drop=True)

    logger.info("Applying multi-level operational categorical group imputations")
    df["City_Type"] = df.groupby("City")["City_Type"].transform(lambda x: x.fillna(_safe_mode_vectorized(x, "Unknown")))

    # Treat out-of-bounds rating anomaly (6.0) as NaN and fill with median
    df.loc[df["Delivery_person_Ratings"] == 6.0, "Delivery_person_Ratings"] = np.nan
    df["Delivery_person_Ratings"] = df.groupby("City_Type")["Delivery_person_Ratings"].transform(lambda x: x.fillna(x.median()))

    # Regional Modes
    df["Weather_conditions"] = df.groupby("City")["Weather_conditions"].transform(lambda x: x.fillna(_safe_mode_vectorized(x, "Cloudy")))
    df["Road_traffic_density"] = df.groupby("City_Type")["Road_traffic_density"].transform(lambda x: x.fillna(_safe_mode_vectorized(x, "Medium")))

    # Baseline Static Fallbacks
    df["Delivery_person_Age"] = df["Delivery_person_Age"].fillna(refs["mean_age"])
    df["multiple_deliveries"] = df["multiple_deliveries"].fillna(refs["median_mult_del"])
    df["Festival"] = df["Festival"].fillna(refs["mode_festival"])
    return df


def compute_geospatial_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates fully vectorized Haversine distance from spatial coordinates."""
    logger.info("Computing geospatial distance matrix")
    radius_km = 6371.0088
    
    lat1, lon1 = np.radians(df["Restaurant_latitude"]), np.radians(df["Restaurant_longitude"])
    lat2, lon2 = np.radians(df["Delivery_location_latitude"]), np.radians(df["Delivery_location_longitude"])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distance_km = np.round(radius_km * c, 2)

    df.insert(df.columns.get_loc("Delivery_location_longitude") + 1, "distance_km", distance_km)
    return df


def cast_and_finalize_datatypes(df: pd.DataFrame) -> pd.DataFrame:
    """Applies memory optimized strict downcasting types and clears temp attributes."""
    logger.info("Applying strict type conversions and final column optimization drops")
    df["Delivery_Agent"] = df["Delivery_Agent"].astype("Int32")
    df["Delivery_person_Age"] = df["Delivery_person_Age"].astype(int)
    df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].astype("float32")
    df["Vehicle_condition"] = df["Vehicle_condition"].map({0: "poor", 1: "Average", 2: "Good", 3: "Excellent"}).fillna("Average")
    df["multiple_deliveries"] = df["multiple_deliveries"].astype("Int8")
    
    # FIXED: Round the values first, then apply the Int16 cast safely
    df["Prep_Time(min)"] = pd.to_numeric(df["Prep_Time(min)"], errors="coerce").round().astype("Int16")

    df = df.drop(columns=["Prep_Time(min)", "Delivery_person_ID"], errors="ignore")
    return df


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Master pipeline managing isolated component functions."""
    logger.info("Starting preprocessing for {} rows and {} columns", df.shape[0], df.shape[1])
    df = df.copy()

    # Drop missing core ID records
    df = df.dropna(subset=["Delivery_person_ID"]).reset_index(drop=True)

    # 1. Coordinate Uniformity
    df["Restaurant_latitude"] = df["Restaurant_latitude"].abs()
    df["Restaurant_longitude"] = df["Restaurant_longitude"].abs()

    # 2. Extract Metadata from ID Strings
    df = extract_delivery_person_metadata(df)

    # Build reference tracking stats to secure pipeline mapping steps against data leakage
    imputation_refs = {
        "mean_age": df["Delivery_person_Age"].mean(),
        "median_mult_del": df["multiple_deliveries"].median(),
        "mode_festival": _safe_mode_vectorized(df["Festival"], "No"),
        "fallback_prep": 15.0
    }

    # 3. Dynamic Date Transformation Routing
    df = process_timestamps_and_prep_time(df, global_median_prep=imputation_refs["fallback_prep"])

    # 4. Fill Holes / Gaps via Segment Tracking
    df = impute_missing_features(df, imputation_refs)

    # 5. Geolocation Math Logic
    df = compute_geospatial_distance(df)

    # 6. Type Enforcement Optimization
    df = cast_and_finalize_datatypes(df)

    logger.info("Preprocessing finished with {} rows and {} columns", df.shape[0], df.shape[1])
    return df


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "Delivery Dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "Cleaned Delivery Dataset.csv",
):
    logger.info("Reading raw dataset from {}", input_path)
    df = pd.read_csv(input_path)

    cleaned_df = preprocess_dataset(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing cleaned dataset to {}", output_path)
    cleaned_df.to_csv(output_path, index=False)
    logger.success("Dataset preprocessing complete")


if __name__ == "__main__":
    app()