"""
STL Decomposition + Inverter Failure Forecasting.

Pipeline
--------
1. Aggregate daily mean Performance Ratio (PR) and anomaly metrics.
2. STL decomposition on daily mean PR (period = 7, weekly).
3. Evaluate STL fitted values vs. naïve-mean baseline → MAE / RMSE.
4. Extrapolate trend + seasonal component for the next HORIZON days.
5. Survival model: convert each inverter's latest risk_score into a
   per-day failure probability, summed across inverters.
6. Export Figure 1 (STL), Figure 2 (forecast), forecast_table.csv.

Inputs  : input_data/iforest_risk.csv  (run iforest_risk.py with no date
          filter first to include all historical dates)
Outputs :
  output/figures/stl_decomposition.png
  output/figures/forecast.png
  input_data/forecast_table.csv

Usage
-----
python preprocess/stl_forecast.py               # default 30-day horizon
python preprocess/stl_forecast.py --horizon 14
python preprocess/stl_forecast.py --horizon 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import STL

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "input_data" / "iforest_risk.csv"
FIGURES_DIR = PROJECT_ROOT / "output" / "figures"
FORECAST_TABLE_PATH = PROJECT_ROOT / "input_data" / "forecast_table.csv"

STL_PERIOD = 7          # weekly seasonality
STL_MIN_POINTS = 14     # require at least 2 full periods
TREND_FIT_WINDOW = 14   # days used to extrapolate the trend linearly
BASE_HAZARD = 0.005     # P(fail per day) at risk_score = 100  (0.5 %)
STYLE = "seaborn-v0_8-whitegrid"


# ---------------------------------------------------------------------------
# 1. Data preparation
# ---------------------------------------------------------------------------

def load_and_aggregate(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load iforest_risk.csv and return two DataFrames:
      - daily_pr   : daily mean PR  (Feb dates only, PR available)
      - daily_all  : daily anomaly metrics for all dates
    """
    df = pd.read_csv(path, parse_dates=["date"])

    daily_all = (
        df.groupby("date")
        .agg(
            mean_risk_score=("risk_score", "mean"),
            anomaly_count=("is_anomaly", "sum"),
            total_inverters=("is_anomaly", "count"),
        )
        .reset_index()
    )
    daily_all["anomaly_rate"] = (
        daily_all["anomaly_count"] / daily_all["total_inverters"]
    )
    daily_all = daily_all.sort_values("date").reset_index(drop=True)

    # PR only available when irradiation exists
    daily_pr = (
        df[df["performance_ratio"].notna()]
        .groupby("date")
        .agg(mean_pr=("performance_ratio", "mean"))
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )

    return daily_pr, daily_all


# ---------------------------------------------------------------------------
# 2. STL decomposition + baseline evaluation
# ---------------------------------------------------------------------------

def run_stl(series: pd.Series) -> STL:
    """Fit STL with robust weighting (handles outlier days like maintenance)."""
    return STL(series, period=STL_PERIOD, robust=True).fit()


def evaluate_baseline(observed: np.ndarray, fitted: np.ndarray) -> dict[str, float]:
    """
    Compare STL in-sample fit vs. naïve-mean baseline.
    Returns a dict with MAE and RMSE for both methods.
    """
    baseline = np.full_like(observed, fill_value=observed.mean())
    return {
        "baseline_mae": mean_absolute_error(observed, baseline),
        "baseline_rmse": np.sqrt(mean_squared_error(observed, baseline)),
        "stl_mae": mean_absolute_error(observed, fitted),
        "stl_rmse": np.sqrt(mean_squared_error(observed, fitted)),
    }


# ---------------------------------------------------------------------------
# 3. Trend + seasonal forecast
# ---------------------------------------------------------------------------

def extrapolate_trend(trend: np.ndarray, horizon: int) -> np.ndarray:
    """Fit a linear model on the last TREND_FIT_WINDOW points and extend."""
    window = min(TREND_FIT_WINDOW, len(trend))
    x = np.arange(window)
    y = trend[-window:]
    coeffs = np.polyfit(x, y, deg=1)
    future_x = np.arange(window, window + horizon)
    return np.polyval(coeffs, future_x)


def repeat_seasonal(seasonal: np.ndarray, horizon: int) -> np.ndarray:
    """Tile the last full seasonal cycle to cover the forecast horizon."""
    last_cycle = seasonal[-STL_PERIOD:]
    repeats = -(-horizon // STL_PERIOD)            # ceiling division
    return np.tile(last_cycle, repeats)[:horizon]


def forecast_pr(stl_result, horizon: int) -> dict[str, np.ndarray]:
    """Return point forecast + 95 % CI for the next `horizon` days."""
    trend_hat = extrapolate_trend(stl_result.trend, horizon)
    seasonal_hat = repeat_seasonal(stl_result.seasonal, horizon)
    point = trend_hat + seasonal_hat

    residual_std = float(np.std(stl_result.resid))
    ci_half = 1.96 * residual_std

    return {
        "point": point,
        "lower": point - ci_half,
        "upper": point + ci_half,
    }


# ---------------------------------------------------------------------------
# 4. Survival model  (risk_score → failure probability)
# ---------------------------------------------------------------------------

def compute_survival_forecast(
    df_risk: pd.DataFrame,
    last_date: pd.Timestamp,
    horizon: int,
) -> pd.DataFrame:
    """
    For each inverter take its most-recent risk_score and apply a
    discrete-time survival model:

        h_i  = BASE_HAZARD × (risk_score_i / 100)         daily hazard
        P(fail on day t | survive to t) = h_i              constant hazard
        E[new failures on day t]  = Σ_i  h_i × (1 - h_i)^(t-1)
        E[cumulative by day T]    = Σ_i  [1 - (1 - h_i)^T]

    Returns a DataFrame with one row per forecast day.
    """
    # Latest risk_score per inverter
    latest = (
        df_risk.sort_values("date")
        .groupby("device_name")
        .last()
        .reset_index()[["device_name", "risk_score"]]
    )
    hazards = (BASE_HAZARD * latest["risk_score"] / 100).values   # shape (n,)

    days = np.arange(1, horizon + 1)                              # 1 … horizon
    forecast_dates = [last_date + pd.Timedelta(days=int(d)) for d in days]

    # E[new failures on day t] = Σ h_i (1-h_i)^(t-1)
    survival = (1 - hazards)[None, :]                             # (1, n)
    daily_new = hazards[None, :] * (survival ** (days[:, None] - 1))   # (T, n)
    expected_new = daily_new.sum(axis=1)

    # E[cumulative by day T] = Σ [1 - (1-h_i)^T]
    expected_cum = (1 - (1 - hazards)[None, :] ** days[:, None]).sum(axis=1)

    # 95 % CI via Poisson approximation: ±1.96 √ (expected_new)
    ci_half = 1.96 * np.sqrt(np.maximum(expected_new, 1e-6))

    return pd.DataFrame(
        {
            "forecast_date": forecast_dates,
            "day_number": days,
            "expected_new_failures": expected_new.round(2),
            "cumulative_failures": expected_cum.round(2),
            "lower_bound": np.maximum(0, expected_new - ci_half).round(2),
            "upper_bound": (expected_new + ci_half).round(2),
            "within_14d": days <= 14,
        }
    )


def assign_risk_level(expected: pd.Series, total_inverters: int) -> pd.Series:
    rate = expected / total_inverters
    return pd.cut(
        rate,
        bins=[-np.inf, 0.02, 0.05, 0.10, np.inf],
        labels=["Low", "Medium", "High", "Critical"],
    )


# ---------------------------------------------------------------------------
# 5. Figure 1 – STL decomposition
# ---------------------------------------------------------------------------

def plot_stl(
    daily_pr: pd.DataFrame,
    stl_result,
    metrics: dict[str, float],
    out_path: Path,
) -> None:
    plt.style.use(STYLE)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    fig.suptitle("Figure 1: STL Decomposition – Daily Mean Performance Ratio", fontsize=14, fontweight="bold")

    dates = daily_pr["date"]
    components = [
        ("Observed PR", stl_result.observed, "#2c7bb6"),
        ("Trend",       stl_result.trend,    "#d7191c"),
        ("Seasonal",    stl_result.seasonal,  "#1a9641"),
        ("Residual",    stl_result.resid,     "#756bb1"),
    ]

    for ax, (label, values, colour) in zip(axes, components):
        if label == "Residual":
            ax.bar(dates, values, color=colour, alpha=0.7, width=0.8)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        else:
            ax.plot(dates, values, color=colour, linewidth=1.6)
        ax.set_ylabel(label, fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(axis="x", rotation=30)

    # Annotate with metrics
    metrics_text = (
        f"In-sample fit (period = {STL_PERIOD} days)\n"
        f"Naïve-mean  MAE: {metrics['baseline_mae']:.4f}   RMSE: {metrics['baseline_rmse']:.4f}\n"
        f"STL model   MAE: {metrics['stl_mae']:.4f}   RMSE: {metrics['stl_rmse']:.4f}"
    )
    axes[0].text(
        0.01, 0.97, metrics_text,
        transform=axes[0].transAxes,
        fontsize=8, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.85),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 1 saved -> {out_path}")


# ---------------------------------------------------------------------------
# 6. Figure 2 – Forecast
# ---------------------------------------------------------------------------

def plot_forecast(
    daily_all: pd.DataFrame,
    survival_df: pd.DataFrame,
    stl_forecast_df: pd.DataFrame | None,
    total_inverters: int,
    horizon: int,
    out_path: Path,
) -> None:
    plt.style.use(STYLE)
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    fig.suptitle(
        f"Figure 2: Inverter Failure Forecast – Next {horizon} Days",
        fontsize=14, fontweight="bold",
    )

    # ── Top panel: historical anomaly count + survival forecast ──────────
    ax1 = axes[0]
    ax1.bar(
        daily_all["date"], daily_all["anomaly_count"],
        color="#74add1", alpha=0.8, width=0.8, label="Historical anomaly count",
    )
    ax1.plot(
        survival_df["forecast_date"], survival_df["expected_new_failures"],
        color="#d73027", linewidth=2, marker="o", markersize=4,
        label="Survival model – expected new failures",
    )
    ax1.fill_between(
        survival_df["forecast_date"],
        survival_df["lower_bound"],
        survival_df["upper_bound"],
        color="#d73027", alpha=0.15, label="95 % CI",
    )
    ax1.axvline(
        daily_all["date"].max(), color="gray", linewidth=1.2,
        linestyle="--", label="Forecast start",
    )
    ax1.set_ylabel("Inverters flagged per day", fontsize=10)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.tick_params(axis="x", rotation=30)

    # 14-day window shading
    d14 = survival_df[survival_df["within_14d"]]["forecast_date"]
    if not d14.empty:
        ax1.axvspan(d14.min(), d14.max(), color="orange", alpha=0.08, label="14-day window")

    # ── Bottom panel: cumulative expected failures ────────────────────────
    ax2 = axes[1]
    ax2.plot(
        survival_df["forecast_date"], survival_df["cumulative_failures"],
        color="#4dac26", linewidth=2.2, marker="s", markersize=4,
        label="Cumulative expected failures",
    )
    ax2.axvline(
        survival_df[survival_df["within_14d"]]["forecast_date"].max(),
        color="orange", linewidth=1.2, linestyle="--", label="14-day mark",
    )
    ax2.axvline(
        survival_df["forecast_date"].max(),
        color="#d73027", linewidth=1.2, linestyle="--", label="30-day mark",
    )

    # Annotate 14-day and 30-day totals
    cum_14 = survival_df[survival_df["within_14d"]]["cumulative_failures"].max()
    cum_30 = survival_df["cumulative_failures"].max()
    ax2.annotate(
        f"14d: {cum_14:.1f}",
        xy=(survival_df[survival_df["within_14d"]]["forecast_date"].max(), cum_14),
        xytext=(10, 5), textcoords="offset points", fontsize=9,
        color="darkorange", fontweight="bold",
    )
    ax2.annotate(
        f"30d: {cum_30:.1f}",
        xy=(survival_df["forecast_date"].max(), cum_30),
        xytext=(-60, 5), textcoords="offset points", fontsize=9,
        color="#d73027", fontweight="bold",
    )
    ax2.set_ylabel("Cumulative expected inverter failures", fontsize=10)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.tick_params(axis="x", rotation=30)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 2 saved -> {out_path}")


# ---------------------------------------------------------------------------
# 7. Main orchestrator
# ---------------------------------------------------------------------------

def run_stl_forecast(
    input_path: Path = INPUT_PATH,
    figures_dir: Path = FIGURES_DIR,
    forecast_table_path: Path = FORECAST_TABLE_PATH,
    horizon: int = 30,
) -> Path:
    df_risk = pd.read_csv(input_path, parse_dates=["date"])
    daily_pr, daily_all = load_and_aggregate(input_path)

    total_inverters = int(daily_all["total_inverters"].max())
    last_date = daily_all["date"].max()

    # ── STL on daily mean PR ─────────────────────────────────────────────
    if len(daily_pr) < STL_MIN_POINTS:
        raise ValueError(
            f"Need at least {STL_MIN_POINTS} days of PR data for STL. "
            f"Found {len(daily_pr)}. Run iforest_risk.py on the full dataset first."
        )

    stl_result = run_stl(daily_pr["mean_pr"])
    stl_fitted = stl_result.trend + stl_result.seasonal
    metrics = evaluate_baseline(daily_pr["mean_pr"].values, stl_fitted)

    print("\n── STL Evaluation (in-sample, mean PR series) ──")
    print(f"  Naïve-mean baseline  MAE: {metrics['baseline_mae']:.4f}   RMSE: {metrics['baseline_rmse']:.4f}")
    print(f"  STL model            MAE: {metrics['stl_mae']:.4f}   RMSE: {metrics['stl_rmse']:.4f}")
    improvement = (1 - metrics["stl_rmse"] / metrics["baseline_rmse"]) * 100
    print(f"  RMSE improvement: {improvement:.1f} %")

    # ── Survival model forecast ──────────────────────────────────────────
    survival_df = compute_survival_forecast(df_risk, last_date, horizon)
    survival_df["risk_level"] = assign_risk_level(
        survival_df["expected_new_failures"], total_inverters
    )

    # ── STL PR forecast (for reference in table) ─────────────────────────
    pr_fc = forecast_pr(stl_result, horizon)
    pr_forecast_dates = [last_date + pd.Timedelta(days=int(d)) for d in range(1, horizon + 1)]

    # ── Forecast table ───────────────────────────────────────────────────
    forecast_table = survival_df.copy()
    forecast_table["stl_forecast_pr"] = np.clip(pr_fc["point"], 0, 1).round(4)
    forecast_table["stl_pr_lower"] = np.clip(pr_fc["lower"], 0, 1).round(4)
    forecast_table["stl_pr_upper"] = np.clip(pr_fc["upper"], 0, 1).round(4)

    forecast_table_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_table.to_csv(forecast_table_path, index=False, encoding="utf-8-sig")
    print(f"\nForecast table saved -> {forecast_table_path}")

    # ── Print summary ─────────────────────────────────────────────────────
    cum_14 = forecast_table[forecast_table["within_14d"]]["cumulative_failures"].max()
    cum_30 = forecast_table["cumulative_failures"].max()
    print("\n── Failure Forecast Summary ──")
    print(f"  Total inverters tracked  : {total_inverters}")
    print(f"  Expected failures in 14d : {cum_14:.1f}")
    print(f"  Expected failures in 30d : {cum_30:.1f}")
    print(f"  Last data date           : {last_date.date()}")
    print(f"  Forecast window          : {pr_forecast_dates[0].date()} – {pr_forecast_dates[-1].date()}")
    print()
    print("Daily breakdown (first 14 days):")
    preview = forecast_table[forecast_table["within_14d"]][
        ["forecast_date", "expected_new_failures", "cumulative_failures", "risk_level"]
    ]
    print(preview.to_string(index=False))

    # ── Figures ──────────────────────────────────────────────────────────
    plot_stl(
        daily_pr, stl_result, metrics,
        out_path=figures_dir / "stl_decomposition.png",
    )
    plot_forecast(
        daily_all, survival_df,
        stl_forecast_df=None,
        total_inverters=total_inverters,
        horizon=horizon,
        out_path=figures_dir / "forecast.png",
    )

    return forecast_table_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="STL decomposition + inverter failure forecasting."
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=30,
        choices=[14, 30],
        help="Forecast horizon in days (14 or 30, default: 30).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_stl_forecast(horizon=args.horizon)


if __name__ == "__main__":
    main()
