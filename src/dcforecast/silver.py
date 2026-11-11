"""Silver layer: typed, validated, conformed data.

Responsibilities
- enforce the expected schema and cast types
- validate every row (parsing, duplicates, physical ranges, energy identities,
  calendar consistency); failing rows go to a quarantine table, not the trusted table
- check that every site has a complete hourly grid
- split the wide source into conformed tables: a site dimension, an hourly
  weather table and an hourly load fact table
- write a data quality report
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import bronze, config

log = logging.getLogger(__name__)

LOAD_TABLE = "load_profiles.parquet"
SITES_TABLE = "sites.parquet"
WEATHER_TABLE = "weather_hourly.parquet"
QUARANTINE_TABLE = "quarantine.parquet"
DQ_REPORT = "dq_report.json"


class SchemaError(ValueError):
    pass


def _check_schema(df: pd.DataFrame) -> None:
    missing = [c for c in config.EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Bronze table is missing expected columns: {missing}")


def _cast(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    for c in config.INT_COLUMNS + config.FLOAT_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in config.CATEGORY_COLUMNS:
        out[c] = out[c].astype("string").str.strip()
    return out


def _row_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Return a boolean frame: one column per rule, True where the row FAILS."""
    fails = pd.DataFrame(index=df.index)

    numeric = config.INT_COLUMNS + config.FLOAT_COLUMNS
    fails["unparseable_value"] = df["timestamp"].isna() | df[numeric].isna().any(axis=1)
    fails["missing_category"] = df[config.CATEGORY_COLUMNS].isna().any(axis=1) | (
        df[config.CATEGORY_COLUMNS] == ""
    ).any(axis=1)
    fails["duplicate_site_timestamp"] = df.duplicated(["site_id", "timestamp"], keep="first")

    out_of_range = pd.Series(False, index=df.index)
    for col, (lo, hi) in config.RANGE_CHECKS.items():
        out_of_range |= ~df[col].between(lo, hi)
    fails["out_of_range"] = out_of_range & ~fails["unparseable_value"]

    tol = config.IDENTITY_TOLERANCE
    component_sum = df["P_IT_MW"] + df["P_cool_MW"] + df["P_other_MW"]
    fails["total_ne_components"] = (df["P_total_MW"] - component_sum).abs() > tol * df["P_total_MW"].abs().clip(lower=1)
    fails["pue_ne_total_over_it"] = (df["PUE"] - df["P_total_MW"] / df["P_IT_MW"]).abs() > tol
    overhead = df["P_cool_MW"] + df["P_other_MW"]
    fails["cooling_share_inconsistent"] = (df["cooling_share_overhead"] - df["P_cool_MW"] / overhead).abs() > tol

    fails["calendar_mismatch"] = (
        (df["hour"] != df["timestamp"].dt.hour)
        | (df["day_of_week"] != df["timestamp"].dt.dayofweek)
        | (df["is_weekend"] != (df["timestamp"].dt.dayofweek >= 5).astype(int))
    ) & df["timestamp"].notna()

    return fails.fillna(True).astype(bool)


def _grid_completeness(df: pd.DataFrame) -> dict:
    result = {}
    for site, d in df.groupby("site_id"):
        full = pd.date_range(d["timestamp"].min(), d["timestamp"].max(), freq="h")
        missing = full.difference(pd.DatetimeIndex(d["timestamp"]))
        result[site] = {
            "start": str(d["timestamp"].min()),
            "end": str(d["timestamp"].max()),
            "expected_hours": int(len(full)),
            "present_hours": int(d["timestamp"].nunique()),
            "missing_hours": int(len(missing)),
            "first_missing": [str(t) for t in missing[:10]],
        }
    return result


def _to_utc(df: pd.DataFrame) -> pd.Series:
    """Localise naive local timestamps per site timezone and convert to UTC."""
    parts = []
    for tz, d in df.groupby("timezone"):
        local = d["timestamp"].dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
        parts.append(local.dt.tz_convert("UTC"))
    return pd.concat(parts).reindex(df.index)


def build(bronze_df: pd.DataFrame | None = None, out_dir: Path = config.SILVER_DIR) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = bronze.read() if bronze_df is None else bronze_df
    _check_schema(raw)

    typed = _cast(raw)
    fails = _row_checks(typed)
    bad = fails.any(axis=1)

    quarantine = typed[bad].copy()
    rule_names = fails[bad].columns.to_numpy()
    quarantine["failed_rules"] = [",".join(rule_names[row]) for row in fails[bad].to_numpy()]
    clean = typed[~bad].copy()

    clean["timestamp_utc"] = _to_utc(clean)
    clean = clean.sort_values(["site_id", "timestamp"]).reset_index(drop=True)
    clean["date"] = clean["timestamp"].dt.normalize()

    # Conformed tables -------------------------------------------------------
    sites = (
        clean.groupby("site_id")
        .agg(site_class=("site_class", "first"), location=("location", "first"), timezone=("timezone", "first"),
             first_seen=("timestamp", "min"), last_seen=("timestamp", "max"))
        .reset_index()
    )

    weather_cols = ["temp_C", "dewpoint_C", "rh_pct", "heatwave_intensity", "heatwave_flag"]
    weather = clean[["location", "timestamp"] + weather_cols]
    weather_conflicts = int((weather.groupby(["location", "timestamp"])[weather_cols].nunique() > 1).any(axis=1).sum())
    weather = weather.drop_duplicates(["location", "timestamp"]).sort_values(["location", "timestamp"]).reset_index(drop=True)

    load = clean[[
        "site_id", "timestamp", "timestamp_utc", "date", "hour", "day_of_week", "is_weekend", "regime",
        "u_it_frac", "rplus_it_frac", "PUE", "cooling_share_overhead",
        "P_IT_MW", "P_cool_MW", "P_other_MW", "P_total_MW",
    ]]

    load.to_parquet(out_dir / LOAD_TABLE, index=False)
    sites.to_parquet(out_dir / SITES_TABLE, index=False)
    weather.to_parquet(out_dir / WEATHER_TABLE, index=False)
    quarantine.to_parquet(out_dir / QUARANTINE_TABLE, index=False)

    report = {
        "rows_in": int(len(raw)),
        "rows_passed": int(len(clean)),
        "rows_quarantined": int(bad.sum()),
        "failures_by_rule": {k: int(v) for k, v in fails.sum().items()},
        "unconvertible_utc_timestamps": int(clean["timestamp_utc"].isna().sum()),
        "weather_conflicts_across_sites": weather_conflicts,
        "grid_completeness": _grid_completeness(clean),
        "sites": int(sites.shape[0]),
        "time_span": [str(clean["timestamp"].min()), str(clean["timestamp"].max())],
    }
    (out_dir / DQ_REPORT).write_text(json.dumps(report, indent=2))
    log.info("Silver: %s passed, %s quarantined", report["rows_passed"], report["rows_quarantined"])
    return {"load": load, "sites": sites, "weather": weather, "quarantine": quarantine, "report": report}


def read_load() -> pd.DataFrame:
    return pd.read_parquet(config.SILVER_DIR / LOAD_TABLE)


def read_sites() -> pd.DataFrame:
    return pd.read_parquet(config.SILVER_DIR / SITES_TABLE)


def read_weather() -> pd.DataFrame:
    return pd.read_parquet(config.SILVER_DIR / WEATHER_TABLE)
