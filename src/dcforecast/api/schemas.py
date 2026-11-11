"""Request and response models for the API (rendered in the Swagger docs)."""

from __future__ import annotations

import datetime as dt
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Health(BaseModel):
    status: str
    model_loaded: bool
    model_trained_at: Optional[str] = None
    version: str


class Site(BaseModel):
    site_id: str
    site_class: str
    location: str
    timezone: str
    mean_load_MW: float
    peak_load_MW: float
    mean_PUE: float
    regimes: list[str]


class DailyKPI(BaseModel):
    date: dt.date
    regime: str
    energy_total_MWh: float
    energy_it_MWh: float
    energy_cooling_MWh: float
    peak_MW: float
    daily_PUE: float = Field(description="Energy-weighted PUE for the day")
    load_factor: float = Field(description="Mean load / peak load")
    cooling_share_of_total: float
    max_temp_C: float
    heatwave_hours: int


class HourlyWeather(BaseModel):
    temp_C: float = Field(ge=-40, le=50)
    dewpoint_C: float = Field(ge=-50, le=40)
    rh_pct: float = Field(ge=0, le=100)
    heatwave_intensity: float = Field(0.0, ge=0)
    heatwave_flag: int = Field(0, ge=0, le=1)


class ForecastRequest(BaseModel):
    site_id: str = Field(examples=["SITE_004"])
    date: dt.date = Field(examples=["2018-08-01"])
    weather: Optional[list[HourlyWeather]] = Field(
        None, min_length=24, max_length=24,
        description="24 hourly weather values for the day. Optional for days inside the dataset "
                    "(observed weather is used); required for 2018-08-01, the day after the data ends.",
    )
    temp_offset_c: float = Field(0.0, ge=-15, le=15, description="What-if shift applied to temperature")

    model_config = ConfigDict(json_schema_extra={"examples": [{
        "site_id": "SITE_004",
        "date": "2018-08-01",
        "weather": [{"temp_C": 22 + 6 * (h >= 10 and h <= 18), "dewpoint_C": 16, "rh_pct": 65,
                     "heatwave_intensity": 0, "heatwave_flag": 0} for h in range(24)],
        "temp_offset_c": 0,
    }]})


class Band(BaseModel):
    forecast: float
    lower: float
    upper: float
    actual: Optional[float] = None


class HourForecast(BaseModel):
    timestamp: dt.datetime
    P_total_MW: Band
    PUE: Band


class DaySummary(BaseModel):
    energy_MWh: float
    peak_MW: float
    peak_hour: int
    daily_PUE: float


class TargetEvaluation(BaseModel):
    MAE: float
    MAPE_pct: float
    baseline_lag24_MAE: float


class ForecastResponse(BaseModel):
    site_id: str
    date: dt.date
    data_cutoff: dt.datetime = Field(description="Latest load observation the forecast used")
    in_sample: bool = Field(description="True if the final model saw this day in training")
    weather_source: str
    temp_offset_c: float
    interval_coverage: float = Field(description="Nominal coverage of the lower/upper band")
    summary: DaySummary
    evaluation: Optional[dict[str, TargetEvaluation]] = Field(
        None, description="Accuracy against actuals; present for historical days without what-if changes")
    hours: list[HourForecast]


class PredictRow(BaseModel):
    """One site-hour with every model input supplied explicitly."""
    site_id: str
    timestamp: dt.datetime
    temp_C: float
    dewpoint_C: float
    rh_pct: float
    heatwave_intensity: float = 0.0
    heatwave_flag: int = 0
    P_total_MW_lag24: float
    P_total_MW_lag48: float
    P_total_MW_lag168: float
    P_total_MW_prevday_mean: float
    P_total_MW_prevday_max: float
    P_total_MW_prevday_last: float
    PUE_lag24: float
    PUE_lag48: float
    PUE_lag168: float
    PUE_prevday_mean: float
    PUE_prevday_max: float
    PUE_prevday_last: float

    model_config = ConfigDict(json_schema_extra={"examples": [{
        "site_id": "SITE_002", "timestamp": "2018-07-20T15:00:00",
        "temp_C": 29.4, "dewpoint_C": 18.3, "rh_pct": 51.0, "heatwave_intensity": 0.0, "heatwave_flag": 0,
        "P_total_MW_lag24": 66.1, "P_total_MW_lag48": 64.9, "P_total_MW_lag168": 63.2,
        "P_total_MW_prevday_mean": 60.4, "P_total_MW_prevday_max": 68.0, "P_total_MW_prevday_last": 55.7,
        "PUE_lag24": 1.47, "PUE_lag48": 1.46, "PUE_lag168": 1.45,
        "PUE_prevday_mean": 1.45, "PUE_prevday_max": 1.49, "PUE_prevday_last": 1.43,
    }]})


class PredictRequest(BaseModel):
    rows: list[PredictRow] = Field(min_length=1, max_length=1000)


class PredictResult(BaseModel):
    site_id: str
    timestamp: dt.datetime
    P_total_MW: float
    PUE: float


class PredictResponse(BaseModel):
    predictions: list[PredictResult]
