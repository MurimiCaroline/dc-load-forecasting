import numpy as np
import pandas as pd

from dcforecast import config, features


def _series(days=10):
    ts = pd.date_range("2018-05-01", periods=24 * days, freq="h")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"site_id": "S1", "timestamp": ts, "date": ts.normalize(),
                       "P_total_MW": rng.normal(60, 5, len(ts))})
    return df


def test_lags_use_only_past_data():
    """Changing the target on day D must not change any feature on day D."""
    df = _series()
    day = pd.Timestamp("2018-05-09")
    before = features.add_lag_features(df, "P_total_MW")

    tampered = df.copy()
    tampered.loc[tampered["date"] == day, "P_total_MW"] += 1000
    after = features.add_lag_features(tampered, "P_total_MW")

    cols = config.lag_features("P_total_MW")
    mask = before["date"] == day
    pd.testing.assert_frame_equal(before.loc[mask, cols], after.loc[mask, cols])


def test_lag_values():
    df = _series()
    out = features.add_lag_features(df, "P_total_MW")
    row = out.iloc[24 * 8 + 5]
    assert row["P_total_MW_lag24"] == df.iloc[24 * 7 + 5]["P_total_MW"]
    assert row["P_total_MW_lag168"] == df.iloc[24 * 1 + 5]["P_total_MW"]
    prev_day = df.iloc[24 * 7: 24 * 8]["P_total_MW"]
    assert np.isclose(row["P_total_MW_prevday_mean"], prev_day.mean())
    assert row["P_total_MW_prevday_last"] == prev_day.iloc[-1]


def test_no_same_hour_measurements_in_features():
    leaky = {"u_it_frac", "rplus_it_frac", "P_IT_MW", "P_cool_MW", "P_other_MW", "cooling_share_overhead",
             "P_total_MW", "PUE"}
    for target in config.TARGETS:
        assert leaky.isdisjoint(config.feature_columns(target))
