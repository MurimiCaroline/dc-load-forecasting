"""Feature engineering shared by the gold layer, training and the API.

Forecast setting: a day-ahead forecast issued at 00:00 local time for the 24 hours
of day D, using only information known at that moment:
- load history up to and including D-1 23:00 (lags of 24h or more, previous-day stats)
- calendar features
- site attributes and the planned workload regime for day D
- the weather forecast for day D (the dataset has observed weather; we use it as a
  perfect-forecast proxy and say so in the README)

Same-hour measurements such as u_it_frac, P_IT_MW or P_cool_MW are never used as
inputs: they are part of the answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["timestamp"].dt.hour
    out["day_of_week"] = out["timestamp"].dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    return out


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cooling_degree_hours"] = (out["temp_C"] - config.COOLING_BASE_TEMP_C).clip(lower=0)
    return out


def add_lag_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Lag and previous-day features for one target. Expects a complete hourly
    grid per site, sorted by site_id and timestamp."""
    out = df.copy()
    grouped = out.groupby("site_id", observed=True)[target]
    for lag in (24, 48, 168):
        out[f"{target}_lag{lag}"] = grouped.shift(lag)

    daily = (
        out.groupby(["site_id", "date"], observed=True)[target]
        .agg(prevday_mean="mean", prevday_max="max", prevday_last="last")
        .reset_index()
    )
    daily["date"] = daily["date"] + pd.Timedelta(days=1)  # stats of D-1 become features for D
    daily = daily.rename(columns={c: f"{target}_{c}" for c in ("prevday_mean", "prevday_max", "prevday_last")})
    out = out.merge(daily, on=["site_id", "date"], how="left")
    return out


def to_model_frame(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Select model inputs in a stable order and set categorical dtypes."""
    X = df[config.feature_columns(target)].copy()
    for c in config.CATEGORICAL_FEATURES:
        X[c] = X[c].astype("category")
    return X


def align_categories(X: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Force categorical columns to the exact category sets seen in training."""
    X = X.copy()
    for c, cats in categories.items():
        X[c] = pd.Categorical(X[c].astype(str), categories=cats)
    return X
