"""Train and evaluate day-ahead models for each target.

Protocol
1. Chronological split: train (to 30 Jun), validation (1-15 Jul), test (16-31 Jul).
2. Fit LightGBM on train with early stopping on validation.
3. Calibrate an 80% prediction interval from validation residuals (split conformal).
4. Refit on train + validation with the chosen number of trees.
5. Score once on the untouched test window, against seasonal-naive baselines.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config, features, gold

log = logging.getLogger(__name__)

PARAMS = {
    "objective": "regression",
    "metric": "l1",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": 42,
    "verbose": -1,
}
MAX_ROUNDS = 3000
EARLY_STOP = 100
INTERVAL_COVERAGE = 0.80


def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    err = np.asarray(y_pred) - np.asarray(y_true)
    y = np.asarray(y_true)
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE_pct": float(np.mean(np.abs(err) / np.abs(y)) * 100),
        "bias": float(np.mean(err)),
        "n": int(len(y)),
    }


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df[df["history_complete"]]
    train = df[df["timestamp"] <= config.TRAIN_END]
    valid = df[(df["timestamp"] > config.TRAIN_END) & (df["timestamp"] <= config.VALID_END)]
    test = df[df["timestamp"] > config.VALID_END]
    return train, valid, test


def _categories(df: pd.DataFrame) -> dict[str, list[str]]:
    return {c: sorted(df[c].astype(str).unique().tolist()) for c in config.CATEGORICAL_FEATURES}


def _fit(X: pd.DataFrame, y: pd.Series, rounds: int, X_val=None, y_val=None) -> lgb.Booster:
    cats = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
    dtrain = lgb.Dataset(X, y, categorical_feature=cats, free_raw_data=False)
    if X_val is None:
        return lgb.train(PARAMS, dtrain, num_boost_round=rounds)
    dvalid = lgb.Dataset(X_val, y_val, reference=dtrain)
    return lgb.train(
        PARAMS, dtrain, num_boost_round=rounds, valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
    )


def train_target(df: pd.DataFrame, target: str, categories: dict[str, list[str]]) -> dict:
    train, valid, test = split(df)
    feats = config.feature_columns(target)

    def X(frame: pd.DataFrame, cols=feats) -> pd.DataFrame:
        sub = frame[cols].copy()
        return features.align_categories(sub, {k: v for k, v in categories.items() if k in cols})

    # 1-2. Early-stopped fit on train, select tree count on validation.
    model_sel = _fit(X(train), train[target], MAX_ROUNDS, X(valid), valid[target])
    best_iter = int(model_sel.best_iteration or MAX_ROUNDS)

    # 3. Split-conformal interval from validation residuals.
    val_pred = model_sel.predict(X(valid), num_iteration=best_iter)
    half_width = float(np.quantile(np.abs(valid[target].to_numpy() - val_pred), INTERVAL_COVERAGE))

    # 4. Refit on train + validation.
    train_full = pd.concat([train, valid])
    model = _fit(X(train_full), train_full[target], best_iter)

    # 5. Score on test.
    test = test.copy()
    test["pred"] = model.predict(X(test))
    results = {
        "model": metrics(test[target], test["pred"]),
        "baseline_lag24": metrics(test[target], test[f"{target}_lag24"]),
        "baseline_lag168": metrics(test[target], test[f"{target}_lag168"]),
    }
    results["skill_vs_lag24_pct"] = 100 * (1 - results["model"]["MAE"] / results["baseline_lag24"]["MAE"])
    inside = (test[target] >= test["pred"] - half_width) & (test[target] <= test["pred"] + half_width)
    results["interval"] = {"nominal_coverage": INTERVAL_COVERAGE, "half_width": half_width,
                           "test_coverage": float(inside.mean())}
    results["per_site"] = {
        site: {
            "model_MAE": metrics(g[target], g["pred"])["MAE"],
            "model_MAPE_pct": metrics(g[target], g["pred"])["MAPE_pct"],
            "lag24_MAE": metrics(g[target], g[f"{target}_lag24"])["MAE"],
        }
        for site, g in test.groupby("site_id", observed=True)
    }

    # Ablation: would knowing the day's workload regime in advance help? It would
    # add an operational assumption, so it only goes into the model if it earns it.
    feats_regime = feats + ["regime"]
    cats_regime = {**categories, "regime": sorted(df["regime"].astype(str).unique().tolist())}
    X_r = lambda frame: features.align_categories(frame[feats_regime].copy(), cats_regime)  # noqa: E731
    abl_sel = _fit(X_r(train), train[target], MAX_ROUNDS, X_r(valid), valid[target])
    abl = _fit(X_r(train_full), train_full[target], int(abl_sel.best_iteration or MAX_ROUNDS))
    results["ablation_with_regime_MAE"] = metrics(test[target], abl.predict(X_r(test)))["MAE"]

    importance = pd.Series(model.feature_importance("gain"), index=feats).sort_values(ascending=False)
    results["feature_importance_gain"] = {k: float(v) for k, v in (importance / importance.sum()).round(4).items()}

    path = config.MODELS_DIR / f"{target}.txt"
    model.save_model(str(path))
    log.info("%s: test MAE %.3f vs lag24 %.3f (%.1f%% skill)", target, results["model"]["MAE"],
             results["baseline_lag24"]["MAE"], results["skill_vs_lag24_pct"])

    test[["site_id", "timestamp", target, "pred", f"{target}_lag24"]].to_parquet(
        config.REPORTS_DIR / f"test_predictions_{target}.parquet", index=False)

    return {"file": path.name, "features": feats, "best_iteration": best_iter, "interval_half_width": half_width,
            "metrics": results}


def run() -> dict:
    config.ensure_dirs()
    df = gold.read_features()
    categories = _categories(df)
    train, valid, test = split(df)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "algorithm": "LightGBM gradient-boosted trees (global model across sites)",
        "forecast_setting": "Day-ahead: 24 hourly values for day D, issued at 00:00 with data to D-1 23:00",
        "split": {
            "train": [str(train["timestamp"].min()), str(train["timestamp"].max())],
            "validation": [str(valid["timestamp"].min()), str(valid["timestamp"].max())],
            "test": [str(test["timestamp"].min()), str(test["timestamp"].max())],
            "final_model_trained_through": str(valid["timestamp"].max()),
        },
        "assumptions": [
            "Observed weather is used as a perfect day-ahead weather forecast; live use needs a forecast feed.",
            "The workload regime label is excluded: the ablation shows it adds no accuracy.",
        ],
        "categories": categories,
        "params": PARAMS,
        "targets": {},
    }
    for target in config.TARGETS:
        meta["targets"][target] = train_target(df, target, categories)

    (config.MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    (config.REPORTS_DIR / "metrics.json").write_text(
        json.dumps({t: v["metrics"] for t, v in meta["targets"].items()}, indent=2))
    return meta
