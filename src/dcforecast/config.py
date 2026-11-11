"""Central configuration: paths, schema, split dates and feature lists."""

from __future__ import annotations

import os
from pathlib import Path

# Project root = two levels above this file (src/dcforecast/config.py -> project root).
PROJECT_ROOT = Path(os.environ.get("DCF_PROJECT_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_FILE = RAW_DIR / "3-Month Data Center Load Profiles.csv"

# ---------------------------------------------------------------------------
# Source schema (from the dataset README, Table 4)
# ---------------------------------------------------------------------------
EXPECTED_COLUMNS = [
    "timestamp", "site_id", "location", "timezone", "site_class", "regime",
    "hour", "day_of_week", "is_weekend",
    "temp_C", "dewpoint_C", "rh_pct", "heatwave_intensity", "heatwave_flag",
    "u_it_frac", "rplus_it_frac", "PUE", "cooling_share_overhead",
    "P_IT_MW", "P_cool_MW", "P_other_MW", "P_total_MW",
]

INT_COLUMNS = ["hour", "day_of_week", "is_weekend", "heatwave_flag"]
FLOAT_COLUMNS = [
    "temp_C", "dewpoint_C", "rh_pct", "heatwave_intensity",
    "u_it_frac", "rplus_it_frac", "PUE", "cooling_share_overhead",
    "P_IT_MW", "P_cool_MW", "P_other_MW", "P_total_MW",
]
CATEGORY_COLUMNS = ["site_id", "location", "timezone", "site_class", "regime"]

# Plausible physical ranges used by silver-layer validation. Deliberately a little
# wider than the README summary statistics so they catch errors, not normal extremes.
RANGE_CHECKS = {
    "temp_C": (-40.0, 50.0),
    "dewpoint_C": (-50.0, 40.0),
    "rh_pct": (0.0, 100.0),
    "heatwave_intensity": (0.0, 5.0),
    "u_it_frac": (0.0, 1.5),
    "rplus_it_frac": (0.0, 1.0),
    "PUE": (1.0, 3.0),
    "cooling_share_overhead": (0.0, 1.0),
    "P_IT_MW": (0.0, 500.0),
    "P_cool_MW": (0.0, 500.0),
    "P_other_MW": (0.0, 500.0),
    "P_total_MW": (0.0, 1000.0),
}
IDENTITY_TOLERANCE = 1e-6

# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
TARGETS = ["P_total_MW", "PUE"]

# Chronological split (local site time). The first 7 days are consumed by the
# 168-hour lag features, so training effectively starts on 2018-05-08.
TRAIN_END = "2018-06-30 23:00"
VALID_END = "2018-07-15 23:00"   # validation: 1-15 July (early stopping)
# test: 16-31 July (held out, reported once)

HISTORY_HOURS = 168  # history needed before a forecast day

CATEGORICAL_FEATURES = ["site_id", "site_class"]
CALENDAR_FEATURES = ["hour", "day_of_week", "is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
WEATHER_FEATURES = ["temp_C", "dewpoint_C", "rh_pct", "heatwave_intensity", "heatwave_flag", "cooling_degree_hours"]


def lag_features(target: str) -> list[str]:
    """Features derived from a target's own history; all known at forecast origin."""
    return [
        f"{target}_lag24",
        f"{target}_lag48",
        f"{target}_lag168",
        f"{target}_prevday_mean",
        f"{target}_prevday_max",
        f"{target}_prevday_last",
    ]


def feature_columns(target: str) -> list[str]:
    return CATEGORICAL_FEATURES + CALENDAR_FEATURES + WEATHER_FEATURES + lag_features(target)


SITE_ATTRIBUTES = ["site_id", "site_class", "location", "timezone"]
COOLING_BASE_TEMP_C = 18.0  # base temperature for cooling degree hours


def ensure_dirs() -> None:
    for d in (BRONZE_DIR, SILVER_DIR, GOLD_DIR, MODELS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
