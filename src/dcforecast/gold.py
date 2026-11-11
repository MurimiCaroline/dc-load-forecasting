"""Gold layer: business-ready tables.

- features_hourly: one row per site-hour with targets and every model feature
- site_daily_kpis: daily operational KPIs per site (energy, peak, PUE, cooling share)
- dim_sites: site dimension with summary statistics
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import config, features, silver

log = logging.getLogger(__name__)

FEATURES_TABLE = "features_hourly.parquet"
KPI_TABLE = "site_daily_kpis.parquet"
SITES_TABLE = "dim_sites.parquet"


def _assemble_hourly(load: pd.DataFrame, sites: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    df = load.merge(sites[["site_id", "site_class", "location"]], on="site_id", how="left")
    df = df.merge(weather, on=["location", "timestamp"], how="left", validate="many_to_one")
    return df.sort_values(["site_id", "timestamp"]).reset_index(drop=True)


def build_feature_table(load: pd.DataFrame, sites: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    df = _assemble_hourly(load, sites, weather)
    df = features.add_calendar_features(df)
    df = features.add_weather_features(df)
    for target in config.TARGETS:
        df = features.add_lag_features(df, target)
    df["history_complete"] = df[[c for t in config.TARGETS for c in config.lag_features(t)]].notna().all(axis=1)
    return df


def build_daily_kpis(hourly: pd.DataFrame) -> pd.DataFrame:
    g = hourly.groupby(["site_id", "date"], observed=True)
    kpi = g.agg(
        regime=("regime", "first"),
        hours=("P_total_MW", "size"),
        energy_total_MWh=("P_total_MW", "sum"),
        energy_it_MWh=("P_IT_MW", "sum"),
        energy_cooling_MWh=("P_cool_MW", "sum"),
        energy_other_MWh=("P_other_MW", "sum"),
        peak_MW=("P_total_MW", "max"),
        min_MW=("P_total_MW", "min"),
        mean_MW=("P_total_MW", "mean"),
        max_hourly_PUE=("PUE", "max"),
        mean_it_utilisation=("u_it_frac", "mean"),
        max_temp_C=("temp_C", "max"),
        heatwave_hours=("heatwave_flag", "sum"),
    ).reset_index()
    # Energy-weighted daily PUE (the correct way to aggregate PUE).
    kpi["daily_PUE"] = kpi["energy_total_MWh"] / kpi["energy_it_MWh"]
    kpi["load_factor"] = kpi["mean_MW"] / kpi["peak_MW"]
    kpi["cooling_share_of_total"] = kpi["energy_cooling_MWh"] / kpi["energy_total_MWh"]
    return kpi


def build_dim_sites(sites: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    stats = hourly.groupby("site_id").agg(
        mean_load_MW=("P_total_MW", "mean"),
        peak_load_MW=("P_total_MW", "max"),
        mean_PUE=("PUE", "mean"),
        regimes=("regime", lambda s: ",".join(sorted(s.astype(str).unique()))),
    ).reset_index()
    return sites.merge(stats, on="site_id", how="left")


def build(out_dir: Path = config.GOLD_DIR) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    load, sites, weather = silver.read_load(), silver.read_sites(), silver.read_weather()

    hourly = build_feature_table(load, sites, weather)
    kpis = build_daily_kpis(hourly)
    dim = build_dim_sites(sites, hourly)

    hourly.to_parquet(out_dir / FEATURES_TABLE, index=False)
    kpis.to_parquet(out_dir / KPI_TABLE, index=False)
    dim.to_parquet(out_dir / SITES_TABLE, index=False)
    log.info("Gold: %s feature rows (%s with full history), %s KPI rows",
             len(hourly), int(hourly["history_complete"].sum()), len(kpis))
    return {"features": hourly, "kpis": kpis, "sites": dim}


def read_features() -> pd.DataFrame:
    return pd.read_parquet(config.GOLD_DIR / FEATURES_TABLE)


def read_kpis() -> pd.DataFrame:
    return pd.read_parquet(config.GOLD_DIR / KPI_TABLE)


def read_sites() -> pd.DataFrame:
    return pd.read_parquet(config.GOLD_DIR / SITES_TABLE)
