"""FastAPI service exposing the gold layer and the forecasting models.

Run locally:
    uvicorn dcforecast.api.main:app --reload
Then open http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse

from .. import __version__
from ..predictor import ForecastError, Forecaster
from . import schemas

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["forecaster"] = Forecaster()
    yield
    state.clear()


app = FastAPI(
    title="Data Centre Load Forecasting API",
    version=__version__,
    description=(
        "Day-ahead forecasts of total power demand (MW) and PUE for six data centre sites, "
        "served from a bronze → silver → gold pipeline.\n\n"
        "**Try it:** `GET /forecast/SITE_004?date=2018-07-20` compares the forecast with what actually "
        "happened. Add `temp_offset_c=5` for a heat what-if. `POST /forecast` forecasts 2018-08-01, the day "
        "after the data ends, from weather you supply."
    ),
    lifespan=lifespan,
)


def fc() -> Forecaster:
    return state["forecaster"]


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health", response_model=schemas.Health, tags=["service"])
def health():
    f = state.get("forecaster")
    return schemas.Health(status="ok", model_loaded=f is not None,
                          model_trained_at=f.meta["trained_at"] if f else None, version=__version__)


@app.get("/model", tags=["service"], summary="Model card: setup, split, assumptions and test metrics")
def model_info():
    m = fc().meta
    return {
        "algorithm": m["algorithm"],
        "forecast_setting": m["forecast_setting"],
        "split": m["split"],
        "assumptions": m["assumptions"],
        "forecastable_dates": [str(fc().first_forecast_date), str(fc().last_forecast_date)],
        "targets": {
            t: {
                "features": v["features"],
                "test_metrics": v["metrics"]["model"],
                "baseline_lag24": v["metrics"]["baseline_lag24"],
                "baseline_lag168": v["metrics"]["baseline_lag168"],
                "skill_vs_lag24_pct": v["metrics"]["skill_vs_lag24_pct"],
                "interval": v["metrics"]["interval"],
                "top_features": dict(list(v["metrics"]["feature_importance_gain"].items())[:8]),
            }
            for t, v in m["targets"].items()
        },
    }


@app.get("/sites", response_model=list[schemas.Site], tags=["data"])
def sites():
    rows = fc().sites.to_dict(orient="records")
    for r in rows:
        r["regimes"] = r["regimes"].split(",")
    return rows


@app.get("/sites/{site_id}/kpis", response_model=list[schemas.DailyKPI], tags=["data"],
         summary="Daily operational KPIs from the gold layer")
def site_kpis(site_id: str, start: Optional[date] = None, end: Optional[date] = None):
    try:
        k = fc().site_kpis(site_id, start, end)
    except ForecastError as e:
        raise HTTPException(404, str(e))
    k = k.assign(date=k["date"].dt.date, heatwave_hours=k["heatwave_hours"].astype(int))
    return k.to_dict(orient="records")


@app.get("/forecast/{site_id}", response_model=schemas.ForecastResponse, tags=["forecast"],
         summary="Day-ahead forecast for a day in the dataset, with actuals for comparison")
def forecast_get(
    site_id: str,
    date: date = Query(..., examples=["2018-07-20"], description="Day to forecast (2018-05-08 to 2018-07-31)"),
    temp_offset_c: float = Query(0.0, ge=-15, le=15, description="What-if temperature shift in °C"),
):
    try:
        return fc().forecast_day(site_id, date, temp_offset_c=temp_offset_c)
    except ForecastError as e:
        raise HTTPException(status_code=404 if "Unknown site" in str(e) else 422, detail=str(e))


@app.post("/forecast", response_model=schemas.ForecastResponse, tags=["forecast"],
          summary="Day-ahead forecast with caller-supplied weather (e.g. for 2018-08-01)")
def forecast_post(req: schemas.ForecastRequest):
    weather = [w.model_dump() for w in req.weather] if req.weather else None
    try:
        return fc().forecast_day(req.site_id, req.date, weather=weather, temp_offset_c=req.temp_offset_c)
    except ForecastError as e:
        raise HTTPException(status_code=404 if "Unknown site" in str(e) else 422, detail=str(e))


@app.post("/predict", response_model=schemas.PredictResponse, tags=["forecast"],
          summary="Score raw feature rows directly (model-level testing)")
def predict(req: schemas.PredictRequest):
    rows = pd.DataFrame([r.model_dump() for r in req.rows])
    rows["timestamp"] = pd.to_datetime(rows["timestamp"]).dt.tz_localize(None)
    try:
        preds = fc().predict_rows(rows)
    except ForecastError as e:
        raise HTTPException(status_code=422, detail=str(e))
    out = pd.concat([rows[["site_id", "timestamp"]], preds], axis=1)
    return {"predictions": out.to_dict(orient="records")}
