"""Model serving logic, independent of the web framework.

The Forecaster rebuilds features for a requested day with the same functions used
in the gold layer, so training and serving cannot drift apart.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from functools import cached_property

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config, features, gold

WEATHER_INPUTS = ["temp_C", "dewpoint_C", "rh_pct", "heatwave_intensity", "heatwave_flag"]


class ForecastError(ValueError):
    """Raised for requests the model cannot answer (unknown site, date out of range...)."""


@dataclass
class TargetModel:
    name: str
    booster: lgb.Booster
    features: list[str]
    half_width: float


class Forecaster:
    def __init__(self) -> None:
        meta_path = config.MODELS_DIR / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError("No trained model found. Run: python -m dcforecast.pipeline")
        self.meta = json.loads(meta_path.read_text())
        self.categories: dict[str, list[str]] = self.meta["categories"]
        self.models = {
            t: TargetModel(
                name=t,
                booster=lgb.Booster(model_file=str(config.MODELS_DIR / m["file"])),
                features=m["features"],
                half_width=m["interval_half_width"],
            )
            for t, m in self.meta["targets"].items()
        }
        self.hourly = gold.read_features()
        self.kpis = gold.read_kpis()
        self.sites = gold.read_sites()
        self.trained_through = pd.Timestamp(self.meta["split"]["final_model_trained_through"]).normalize()

    # ------------------------------------------------------------------ ranges
    @cached_property
    def first_forecast_date(self) -> date:
        ok = self.hourly.loc[self.hourly["history_complete"], "date"]
        return ok.min().date()

    @cached_property
    def last_observed_date(self) -> date:
        return self.hourly["date"].max().date()

    @property
    def last_forecast_date(self) -> date:
        """One day past the data: the latest day whose D-1 history exists."""
        return self.last_observed_date + timedelta(days=1)

    def site_ids(self) -> list[str]:
        return sorted(self.sites["site_id"].tolist())

    def _check_site(self, site_id: str) -> None:
        if site_id not in self.site_ids():
            raise ForecastError(f"Unknown site_id '{site_id}'. Valid: {', '.join(self.site_ids())}")

    # --------------------------------------------------------------- forecast
    def forecast_day(
        self,
        site_id: str,
        day: date,
        weather: list[dict] | None = None,
        temp_offset_c: float = 0.0,
    ) -> dict:
        self._check_site(site_id)
        if not (self.first_forecast_date <= day <= self.last_forecast_date):
            raise ForecastError(
                f"date must be between {self.first_forecast_date} and {self.last_forecast_date} "
                f"(the model needs 7 days of history before the forecast day)."
            )

        site_hist = self.hourly[self.hourly["site_id"] == site_id]
        day_ts = pd.Timestamp(day)
        history = site_hist[site_hist["date"] < day_ts]
        observed = site_hist[site_hist["date"] == day_ts]

        timestamps = pd.date_range(day_ts, periods=24, freq="h")
        frame = pd.DataFrame({"timestamp": timestamps, "date": day_ts, "site_id": site_id})
        frame["site_class"] = self.sites.set_index("site_id").at[site_id, "site_class"]

        if weather is not None:
            if len(weather) != 24:
                raise ForecastError("weather must contain exactly 24 hourly entries (00:00 to 23:00).")
            w = pd.DataFrame(weather)
            for col in WEATHER_INPUTS:
                frame[col] = w[col].to_numpy() if col in w else 0.0
            weather_source = "request"
        elif len(observed) == 24:
            frame[WEATHER_INPUTS] = observed[WEATHER_INPUTS].to_numpy()
            weather_source = "observed (perfect-forecast proxy)"
        else:
            raise ForecastError(
                f"No observed weather for {day}. Supply 24 hourly weather values in the request body "
                f"(POST /forecast)."
            )

        frame["temp_C"] = frame["temp_C"] + temp_offset_c
        frame["dewpoint_C"] = np.minimum(frame["dewpoint_C"] + temp_offset_c, frame["temp_C"])

        base_cols = ["site_id", "site_class", "timestamp", "date"] + WEATHER_INPUTS + config.TARGETS
        combined = pd.concat([history[base_cols], frame], ignore_index=True)
        combined = features.add_calendar_features(combined)
        combined = features.add_weather_features(combined)
        for t in config.TARGETS:
            combined = features.add_lag_features(combined, t)
        day_rows = combined[combined["date"] == day_ts].reset_index(drop=True)

        out = pd.DataFrame({"timestamp": timestamps})
        for t, m in self.models.items():
            X = features.align_categories(day_rows[m.features].copy(),
                                          {k: v for k, v in self.categories.items() if k in m.features})
            pred = m.booster.predict(X)
            out[f"{t}_forecast"] = pred
            out[f"{t}_lower"] = pred - m.half_width
            out[f"{t}_upper"] = pred + m.half_width
            out[f"{t}_baseline_lag24"] = day_rows[f"{t}_lag24"].to_numpy()
            if len(observed) == 24:
                out[f"{t}_actual"] = observed[t].to_numpy()
        out["PUE_lower"] = out["PUE_lower"].clip(lower=1.0)
        out["P_total_MW_lower"] = out["P_total_MW_lower"].clip(lower=0.0)

        return self._package(site_id, day, out, weather_source, temp_offset_c,
                             has_actuals=len(observed) == 24 and weather is None and temp_offset_c == 0)

    def _package(self, site_id, day, out, weather_source, temp_offset_c, has_actuals) -> dict:
        load = out["P_total_MW_forecast"]
        it_energy = (load / out["PUE_forecast"]).sum()
        summary = {
            "energy_MWh": float(load.sum()),
            "peak_MW": float(load.max()),
            "peak_hour": int(load.idxmax()),
            "daily_PUE": float(load.sum() / it_energy),
        }
        evaluation = None
        if has_actuals:
            evaluation = {}
            for t in config.TARGETS:
                err = out[f"{t}_forecast"] - out[f"{t}_actual"]
                base_err = out[f"{t}_baseline_lag24"] - out[f"{t}_actual"]
                evaluation[t] = {
                    "MAE": float(err.abs().mean()),
                    "MAPE_pct": float((err.abs() / out[f"{t}_actual"]).mean() * 100),
                    "baseline_lag24_MAE": float(base_err.abs().mean()),
                }

        hours = []
        for _, r in out.iterrows():
            row = {"timestamp": r["timestamp"].isoformat()}
            for t in config.TARGETS:
                row[t] = {
                    "forecast": round(float(r[f"{t}_forecast"]), 4),
                    "lower": round(float(r[f"{t}_lower"]), 4),
                    "upper": round(float(r[f"{t}_upper"]), 4),
                    "actual": round(float(r[f"{t}_actual"]), 4) if has_actuals else None,
                }
            hours.append(row)

        return {
            "site_id": site_id,
            "date": day.isoformat(),
            "data_cutoff": (pd.Timestamp(day) - pd.Timedelta(hours=1)).isoformat(),
            "in_sample": pd.Timestamp(day) <= self.trained_through,
            "weather_source": weather_source,
            "temp_offset_c": temp_offset_c,
            "interval_coverage": self.meta["targets"]["P_total_MW"]["metrics"]["interval"]["nominal_coverage"],
            "summary": summary,
            "evaluation": evaluation,
            "hours": hours,
        }

    # -------------------------------------------------------------- raw rows
    def predict_rows(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Score rows that already carry every model feature (used by POST /predict)."""
        for sid in rows["site_id"].unique():
            self._check_site(sid)
        rows = rows.copy()
        rows["site_class"] = rows["site_id"].map(self.sites.set_index("site_id")["site_class"])
        rows = features.add_calendar_features(rows)
        rows = features.add_weather_features(rows)
        out = pd.DataFrame(index=rows.index)
        for t, m in self.models.items():
            missing = [f for f in m.features if f not in rows.columns]
            if missing:
                raise ForecastError(f"Missing inputs for {t}: {missing}")
            X = features.align_categories(rows[m.features].copy(),
                                          {k: v for k, v in self.categories.items() if k in m.features})
            out[t] = m.booster.predict(X)
        return out

    # ------------------------------------------------------------------- KPIs
    def site_kpis(self, site_id: str, start: date | None, end: date | None) -> pd.DataFrame:
        self._check_site(site_id)
        k = self.kpis[self.kpis["site_id"] == site_id]
        if start:
            k = k[k["date"] >= pd.Timestamp(start)]
        if end:
            k = k[k["date"] <= pd.Timestamp(end)]
        return k
