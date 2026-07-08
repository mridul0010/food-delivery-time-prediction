from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config import MODELS_DIR


APP_TITLE = "Food Delivery Time Prediction API"
APP_DESCRIPTION = "Predict delivery time in minutes for the food-delivery project."
APP_VERSION = "1.0.0"
MODEL_PATH = MODELS_DIR / "model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"


class DeliveryRequest(BaseModel):
	model_config = ConfigDict(populate_by_name=True)

	city: str = Field(alias="City", min_length=1)
	zone: str = Field(alias="Zone", min_length=1)
	delivery_agent: int = Field(alias="Delivery_Agent", ge=0)
	delivery_person_age: int = Field(alias="Delivery_person_Age", ge=0)
	delivery_person_ratings: float = Field(alias="Delivery_person_Ratings", ge=0, le=5)
	restaurant_latitude: float = Field(alias="Restaurant_latitude", ge=-90, le=90)
	restaurant_longitude: float = Field(alias="Restaurant_longitude", ge=-180, le=180)
	delivery_location_latitude: float = Field(
		alias="Delivery_location_latitude", ge=-90, le=90
	)
	delivery_location_longitude: float = Field(
		alias="Delivery_location_longitude", ge=-180, le=180
	)
	distance_km: float = Field(alias="distance_km", ge=0)
	order_datetime: datetime = Field(alias="Order_Datetime")
	pickup_datetime: datetime = Field(alias="Pickup_Datetime")
	weather_conditions: str = Field(alias="Weather_conditions", min_length=1)
	road_traffic_density: str = Field(alias="Road_traffic_density", min_length=1)
	vehicle_condition: str = Field(alias="Vehicle_condition", min_length=1)
	type_of_order: str = Field(alias="Type_of_order", min_length=1)
	type_of_vehicle: str = Field(alias="Type_of_vehicle", min_length=1)
	multiple_deliveries: int = Field(alias="multiple_deliveries", ge=0)
	festival: str = Field(alias="Festival", min_length=1)
	city_type: str = Field(alias="City_Type", min_length=1)

	@model_validator(mode="after")
	def validate_timestamps(self) -> DeliveryRequest:
		if self.pickup_datetime < self.order_datetime:
			raise ValueError("Pickup_Datetime must be greater than or equal to Order_Datetime.")
		return self


class PredictionResponse(BaseModel):
	model_config = ConfigDict(populate_by_name=True)

	predicted_time_taken_min: float
	model_loaded: bool


app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION, version=APP_VERSION)


def _request_to_frame(payload: DeliveryRequest) -> pd.DataFrame:
	record = payload.model_dump(by_alias=True)
	return pd.DataFrame(
		[
			{
				"City": record["City"],
				"Zone": record["Zone"],
				"Delivery_Agent": record["Delivery_Agent"],
				"Delivery_person_Age": record["Delivery_person_Age"],
				"Delivery_person_Ratings": record["Delivery_person_Ratings"],
				"Restaurant_latitude": record["Restaurant_latitude"],
				"Restaurant_longitude": record["Restaurant_longitude"],
				"Delivery_location_latitude": record["Delivery_location_latitude"],
				"Delivery_location_longitude": record["Delivery_location_longitude"],
				"distance_km": record["distance_km"],
				"Order_Datetime": record["Order_Datetime"],
				"Pickup_Datetime": record["Pickup_Datetime"],
				"Weather_conditions": record["Weather_conditions"],
				"Road_traffic_density": record["Road_traffic_density"],
				"Vehicle_condition": record["Vehicle_condition"],
				"Type_of_order": record["Type_of_order"],
				"Type_of_vehicle": record["Type_of_vehicle"],
				"multiple_deliveries": record["multiple_deliveries"],
				"Festival": record["Festival"],
				"City_Type": record["City_Type"],
			}
		]
	)


@lru_cache(maxsize=1)
def _load_artifacts() -> tuple[Any, Any]:
	if not MODEL_PATH.exists():
		raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
	if not PREPROCESSOR_PATH.exists():
		raise FileNotFoundError(f"Preprocessor artifact not found: {PREPROCESSOR_PATH}")

	preprocessor = joblib.load(PREPROCESSOR_PATH)
	model = joblib.load(MODEL_PATH)
	return preprocessor, model


def _predict(payload: DeliveryRequest) -> float:
	try:
		preprocessor, model = _load_artifacts()
	except FileNotFoundError as exc:
		raise HTTPException(
			status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
			detail=str(exc),
		) from exc

	frame = _request_to_frame(payload)

	try:
		transformed = preprocessor.transform(frame)
		prediction = model.predict(transformed)
	except Exception as exc:  # pragma: no cover - defensive API boundary
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail=f"Prediction failed: {exc}",
		) from exc

	return float(pd.Series(prediction).iloc[0])


@app.get("/", tags=["health"])
def root() -> dict[str, str]:
	return {
		"message": "Food Delivery Time Prediction API",
		"docs": "/docs",
		"health": "/health",
		"predict": "/predict",
	}


@app.get("/health", tags=["health"])
def health() -> dict[str, Any]:
	model_loaded = MODEL_PATH.exists() and PREPROCESSOR_PATH.exists()
	return {
		"status": "ok" if model_loaded else "degraded",
		"model_loaded": model_loaded,
		"model_path": str(MODEL_PATH),
		"preprocessor_path": str(PREPROCESSOR_PATH),
	}


@app.post(
	"/predict",
	response_model=PredictionResponse,
	tags=["prediction"],
	summary="Predict food delivery time",
)
def predict(payload: DeliveryRequest) -> PredictionResponse:
	predicted_minutes = _predict(payload)
	return PredictionResponse(
		predicted_time_taken_min=predicted_minutes,
		model_loaded=True,
	)


if __name__ == "__main__":
	import uvicorn

	uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
