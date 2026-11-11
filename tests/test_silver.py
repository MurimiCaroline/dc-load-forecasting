import pandas as pd
import pytest

from dcforecast import config, silver


def _good_rows(n=48, site="SITE_X"):
    ts = pd.date_range("2018-05-01", periods=n, freq="h")
    it = 50.0
    cool, other = 10.0, 5.0
    total = it + cool + other
    df = pd.DataFrame({
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "site_id": site, "location": "Syracuse, NY", "timezone": "America/New_York",
        "site_class": "modern", "regime": "business_peak",
        "hour": ts.hour, "day_of_week": ts.dayofweek, "is_weekend": (ts.dayofweek >= 5).astype(int),
        "temp_C": 20.0, "dewpoint_C": 12.0, "rh_pct": 60.0, "heatwave_intensity": 0.0, "heatwave_flag": 0,
        "u_it_frac": 0.8, "rplus_it_frac": 0.0, "PUE": total / it, "cooling_share_overhead": cool / (cool + other),
        "P_IT_MW": it, "P_cool_MW": cool, "P_other_MW": other, "P_total_MW": total,
    })
    return df.astype(str)


def test_clean_data_passes(tmp_path):
    out = silver.build(_good_rows(), out_dir=tmp_path)
    assert out["report"]["rows_quarantined"] == 0
    assert out["report"]["grid_completeness"]["SITE_X"]["missing_hours"] == 0


@pytest.mark.parametrize("column,value,rule", [
    ("P_total_MW", "999.0", "total_ne_components"),
    ("rh_pct", "150", "out_of_range"),
    ("temp_C", "not-a-number", "unparseable_value"),
    ("hour", "7", "calendar_mismatch"),
])
def test_bad_rows_are_quarantined(tmp_path, column, value, rule):
    df = _good_rows()
    df.loc[5, column] = value
    out = silver.build(df, out_dir=tmp_path)
    assert out["report"]["rows_quarantined"] >= 1
    assert rule in out["quarantine"]["failed_rules"].iloc[0]
    assert len(out["load"]) == len(df) - out["report"]["rows_quarantined"]


def test_duplicates_and_gaps_detected(tmp_path):
    df = _good_rows()
    df = pd.concat([df, df.iloc[[3]]]).drop(index=10)  # one duplicate, one missing hour
    out = silver.build(df, out_dir=tmp_path)
    assert out["report"]["failures_by_rule"]["duplicate_site_timestamp"] == 1
    assert out["report"]["grid_completeness"]["SITE_X"]["missing_hours"] == 1


def test_missing_column_fails_fast(tmp_path):
    with pytest.raises(silver.SchemaError):
        silver.build(_good_rows().drop(columns=["PUE"]), out_dir=tmp_path)
