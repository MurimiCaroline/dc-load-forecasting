"""Build the README figures from the test-window predictions.

    python -m dcforecast.report        (needs matplotlib: pip install -e ".[dev]")
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402

FIG_DIR = config.REPORTS_DIR / "figures"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
FORECAST, BASELINE = "#2a78d6", "#a3a29c"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def forecast_vs_actual(site: str = "SITE_004", start: str = "2018-07-16", days: int = 7) -> None:
    meta = json.loads((config.MODELS_DIR / "metadata.json").read_text())
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True, facecolor=SURFACE)
    labels = {"P_total_MW": "Total demand (MW)", "PUE": "PUE"}
    for ax, target in zip(axes, config.TARGETS):
        p = pd.read_parquet(config.REPORTS_DIR / f"test_predictions_{target}.parquet")
        p = p[(p["site_id"] == site) & (p["timestamp"] >= start)
              & (p["timestamp"] < pd.Timestamp(start) + pd.Timedelta(days=days))]
        hw = meta["targets"][target]["interval_half_width"]
        ax.fill_between(p["timestamp"], p["pred"] - hw, p["pred"] + hw, color=FORECAST, alpha=0.14,
                        linewidth=0, label="80% interval")
        ax.plot(p["timestamp"], p[target], color=INK, linewidth=1.6, label="Actual")
        ax.plot(p["timestamp"], p["pred"], color=FORECAST, linewidth=2, label="Day-ahead forecast")
        ax.set_ylabel(labels[target], color=INK_2, fontsize=10)
        _style(ax)
    axes[0].legend(loc="upper left", ncols=3, frameon=False, fontsize=9, labelcolor=INK_2)
    fig.suptitle(f"{site}: day-ahead forecast vs actual, held-out test week ({start} onward)",
                 x=0.06, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "forecast_vs_actual.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def per_site_error() -> None:
    m = json.loads((config.REPORTS_DIR / "metrics.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=SURFACE)
    units = {"P_total_MW": "MAE (MW)", "PUE": "MAE (PUE)"}
    for ax, target in zip(axes, config.TARGETS):
        ps = pd.DataFrame(m[target]["per_site"]).T.sort_index(ascending=False)
        y = range(len(ps))
        ax.barh([i + 0.2 for i in y], ps["lag24_MAE"], height=0.36, color=BASELINE, label="Same hour yesterday")
        ax.barh([i - 0.2 for i in y], ps["model_MAE"], height=0.36, color=FORECAST, label="Model")
        ax.set_yticks(list(y), ps.index)
        ax.set_xlabel(units[target], color=INK_2, fontsize=10)
        ax.set_title(f"{target}: {m[target]['skill_vs_lag24_pct']:.0f}% lower error than baseline",
                     loc="left", color=INK, fontsize=11)
        _style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    axes[1].legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_site_error.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    forecast_vs_actual()
    per_site_error()
    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
