from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


# Features file constants
TARGET_COLUMN = "Time_taken (min)"
TRAIN_OUTPUT_PATH = PROCESSED_DATA_DIR / "train_features"
TEST_OUTPUT_PATH = PROCESSED_DATA_DIR / "test_features"

NUMERICAL_COLUMNS = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "multiple_deliveries",
    "distance_km",
    "Prep_Time(min)",
    "Order_hour",
    "isWeekend",
]

ONE_HOT_COLUMNS = [
    "City",
    "Zone",
    "Weather_conditions",
    "Type_of_order",
    "Type_of_vehicle",
    "City_Type",
    "Order_day",
    "Time_Of_Day",
]

ORDINAL_COLUMNS = [
    "Road_traffic_density",
    "Vehicle_condition",
    "Festival",
    "delivery_rating_group",
    "age_group",
    "distance_group",
]

ROAD_TRAFFIC_DENSITY = ["Low", "Medium", "High", "Jam"]
VEHICLE_CONDITION = ["poor", "Average", "Good", "Excellent"]
FESTIVAL = ["No", "Yes"]
DELIVERY_RATING_GROUP = ["Low", "Medium", "High"]
AGE_GROUP = ["Young", "Adult", "Senior"]
DISTANCE_GROUP = ["Short Distance", "Medium Distance", "Long Distance"]

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
