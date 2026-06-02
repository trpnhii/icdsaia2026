"""
Generate STL and replacement-risk forecast figures from 2-year daily risk data.

Inputs:
  input_data/2_years/risk_scores.csv
  input_data/2_years/transformed.csv

Outputs:
  output/figures/2_years/stl_decomposition.png
  output/figures/2_years/forecast.png
  input_data/2_years/forecast_table.csv
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
RISK_PATH = PROJECT_ROOT / "input_data" / "2_years" / "risk_scores.csv"
TRANSFORMED_PATH = PROJECT_ROOT / "input_data" / "2_years" / "transformed.csv"
FIGURES_DIR = PROJECT_ROOT / "output" / "figures" / "2_years"
FORECAST_TABLE_PATH = PROJECT_ROOT / "input_data" / "2_years" / "forecast_table.csv"

STL_PERIOD = 7
STL_MIN_POINTS = 14
TREND_FIT_WINDOW = 14
BASE_HAZARD = 0.005
STYLE = "seaborn-v0_8-whitegrid"


def load_daily_series(
    risk_path: Path = RISK_PATH,
    transformed_path: Path = TRANSFORMED_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    risk = pd.read_csv(risk_path, parse_dates=["risk_date"])
    transformed = pd.read_csv(transformed_path, parse_dates=["date"])

    daily_risk = (
        risk.groupby("risk_date")
        .agg(
            mean_risk_score=("risk_score", "mean"),
            candidate_count=("replacement_candidate", "sum"),
            total_inverters=("replacement_candidate", "count"),
        )
        .reset_index()
        .rename(columns={"risk_date": "date"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily_risk["candidate_rate"] = daily_risk["candidate_count"] / daily_risk["total_inverters"]

    daily_pr = (
        transformed[
            transformed["performance_ratio"].notna()
            & transformed["irradiation_kwh_m2"].notna()
        ]
        .groupby("date")
        .agg(mean_pr=("performance_ratio", "mean"))
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Keep PR dates aligned to scored risk dates.
    daily_pr = daily_pr[daily_pr["date"].isin(daily_risk["date"])].reset_index(drop=True)
    return risk, daily_pr, daily_risk


def evaluate_baseline(observed: np.ndarray, fitted: np.ndarray) -> dict[str, float]:
    baseline = np.full_like(observed, fill_value=observed.mean())
    return {
        "baseline_mae": mean_absolute_error(observed, baseline),
        "baseline_rmse": np.sqrt(mean_squared_error(observed, baseline)),
        "stl_mae": mean_absolute_error(observed, fitted),
        "stl_rmse": np.sqrt(mean_squared_error(observed, fitted)),
    }


def extrapolate_trend(trend: np.ndarray, horizon: int) -> np.ndarray:
    window = min(TREND_FIT_WINDOW, len(trend))
    x = np.arange(window)
    coeffs = np.polyfit(x, trend[-window:], deg=1)
    future_x = np.arange(window, window + horizon)
    return np.polyval(coeffs, future_x)


def repeat_seasonal(seasonal: np.ndarray, horizon: int) -> np.ndarray:
    last_cycle = seasonal[-STL_PERIOD:]
    repeats = -(-horizon // STL_PERIOD)
    return np.tile(last_cycle, repeats)[:horizon]


def forecast_pr(stl_result, horizon: int) -> dict[str, np.ndarray]:
    point = extrapolate_trend(stl_result.trend, horizon) + repeat_seasonal(
        stl_result.seasonal, horizon
    )
    ci_half = 1.96 * float(np.std(stl_result.resid))
    return {"point": point, "lower": point - ci_half, "upper": point + ci_half}


def compute_survival_forecast(
    risk: pd.DataFrame,
    last_date: pd.Timestamp,
    horizon: int,
) -> pd.DataFrame:
    latest = (
        risk.sort_values("risk_date")
        .groupby(["zone", "device_name"])
        .last()
        .reset_index()[["zone", "device_name", "risk_score"]]
    )
    hazards = (BASE_HAZARD * latest["risk_score"] / 100).to_numpy(dtype=float)

    days = np.arange(1, horizon + 1)
    forecast_dates = [last_date + pd.Timedelta(days=int(d)) for d in days]
    daily_new = hazards[None, :] * ((1 - hazards)[None, :] ** (days[:, None] - 1))
    expected_new = daily_new.sum(axis=1)
    expected_cum = (1 - (1 - hazards)[None, :] ** days[:, None]).sum(axis=1)
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


def plot_stl(
    daily_pr: pd.DataFrame,
    stl_result,
    metrics: dict[str, float],
    out_path: Path,
) -> None:
    plt.style.use(STYLE)
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    fig.suptitle("Figure 1: STL Decomposition - 2-Year Daily Mean PR", fontsize=14, fontweight="bold")

    components = [
        ("Observed PR", stl_result.observed, "#2c7bb6"),
        ("Trend", stl_result.trend, "#d7191c"),
        ("Seasonal", stl_result.seasonal, "#1a9641"),
        ("Residual", stl_result.resid, "#756bb1"),
    ]
    for ax, (label, values, color) in zip(axes, components):
        if label == "Residual":
            ax.bar(daily_pr["date"], values, color=color, alpha=0.7, width=0.8)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        else:
            ax.plot(daily_pr["date"], values, color=color, linewidth=1.5)
        ax.set_ylabel(label)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=30)

    metrics_text = (
        f"In-sample fit (period = {STL_PERIOD} days)\n"
        f"Naive mean MAE: {metrics['baseline_mae']:.4f}  RMSE: {metrics['baseline_rmse']:.4f}\n"
        f"STL model  MAE: {metrics['stl_mae']:.4f}  RMSE: {metrics['stl_rmse']:.4f}"
    )
    axes[0].text(
        0.01,
        0.97,
        metrics_text,
        transform=axes[0].transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.85),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 1 saved -> {out_path}")


def plot_forecast(
    daily_risk: pd.DataFrame,
    survival_df: pd.DataFrame,
    horizon: int,
    out_path: Path,
) -> None:
    plt.style.use(STYLE)
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    fig.suptitle(
        f"Figure 2: Replacement-Risk Forecast - Next {horizon} Days",
        fontsize=14,
        fontweight="bold",
    )

    ax1 = axes[0]
    ax1.bar(
        daily_risk["date"],
        daily_risk["candidate_count"],
        color="#74add1",
        alpha=0.8,
        width=0.8,
        label="Historical replacement candidates",
    )
    ax1.plot(
        survival_df["forecast_date"],
        survival_df["expected_new_failures"],
        color="#d73027",
        linewidth=2,
        marker="o",
        markersize=4,
        label="Expected new failures",
    )
    ax1.fill_between(
        survival_df["forecast_date"],
        survival_df["lower_bound"],
        survival_df["upper_bound"],
        color="#d73027",
        alpha=0.15,
        label="95% CI",
    )
    ax1.axvline(daily_risk["date"].max(), color="gray", linewidth=1.2, linestyle="--", label="Forecast start")
    ax1.set_ylabel("Inverters flagged per day")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.tick_params(axis="x", rotation=30)

    d14 = survival_df[survival_df["within_14d"]]["forecast_date"]
    if not d14.empty:
        ax1.axvspan(d14.min(), d14.max(), color="orange", alpha=0.08)

    ax2 = axes[1]
    ax2.plot(
        survival_df["forecast_date"],
        survival_df["cumulative_failures"],
        color="#4dac26",
        linewidth=2.2,
        marker="s",
        markersize=4,
        label="Cumulative expected failures",
    )
    ax2.axvline(
        survival_df[survival_df["within_14d"]]["forecast_date"].max(),
        color="orange",
        linewidth=1.2,
        linestyle="--",
        label="14-day mark",
    )
    ax2.axvline(
        survival_df["forecast_date"].max(),
        color="#d73027",
        linewidth=1.2,
        linestyle="--",
        label=f"{horizon}-day mark",
    )
    cum_14 = survival_df[survival_df["within_14d"]]["cumulative_failures"].max()
    cum_h = survival_df["cumulative_failures"].max()
    ax2.annotate(f"14d: {cum_14:.1f}", xy=(d14.max(), cum_14), xytext=(10, 5), textcoords="offset points")
    ax2.annotate(
        f"{horizon}d: {cum_h:.1f}",
        xy=(survival_df["forecast_date"].max(), cum_h),
        xytext=(-70, 5),
        textcoords="offset points",
        color="#d73027",
        fontweight="bold",
    )
    ax2.set_ylabel("Cumulative expected inverter failures")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax2.tick_params(axis="x", rotation=30)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 2 saved -> {out_path}")


def run(horizon: int = 30) -> Path:
    risk, daily_pr, daily_risk = load_daily_series()
    if len(daily_pr) < STL_MIN_POINTS:
        raise ValueError(f"Need at least {STL_MIN_POINTS} days of PR data. Found {len(daily_pr)}.")

    stl_result = STL(daily_pr["mean_pr"], period=STL_PERIOD, robust=True).fit()
    stl_fitted = stl_result.trend + stl_result.seasonal
    metrics = evaluate_baseline(daily_pr["mean_pr"].to_numpy(), stl_fitted)

    last_date = daily_risk["date"].max()
    total_inverters = int(daily_risk["total_inverters"].max())
    survival_df = compute_survival_forecast(risk, last_date, horizon)
    survival_df["risk_level"] = assign_risk_level(survival_df["expected_new_failures"], total_inverters)

    pr_fc = forecast_pr(stl_result, horizon)
    forecast_table = survival_df.copy()
    forecast_table["stl_forecast_pr"] = np.clip(pr_fc["point"], 0, 1).round(4)
    forecast_table["stl_pr_lower"] = np.clip(pr_fc["lower"], 0, 1).round(4)
    forecast_table["stl_pr_upper"] = np.clip(pr_fc["upper"], 0, 1).round(4)

    FORECAST_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    forecast_table.to_csv(FORECAST_TABLE_PATH, index=False, encoding="utf-8-sig")

    plot_stl(daily_pr, stl_result, metrics, FIGURES_DIR / "stl_decomposition.png")
    plot_forecast(daily_risk, survival_df, horizon, FIGURES_DIR / "forecast.png")

    cum_14 = forecast_table[forecast_table["within_14d"]]["cumulative_failures"].max()
    cum_h = forecast_table["cumulative_failures"].max()
    improvement = (1 - metrics["stl_rmse"] / metrics["baseline_rmse"]) * 100
    print()
    print("STL evaluation")
    print(f"  Naive mean baseline MAE: {metrics['baseline_mae']:.4f}  RMSE: {metrics['baseline_rmse']:.4f}")
    print(f"  STL model           MAE: {metrics['stl_mae']:.4f}  RMSE: {metrics['stl_rmse']:.4f}")
    print(f"  RMSE improvement: {improvement:.1f}%")
    print()
    print("Failure forecast summary")
    print(f"  Total inverters tracked: {total_inverters}")
    print(f"  Last scored date: {last_date.date()}")
    print(f"  Forecast window: {(last_date + pd.Timedelta(days=1)).date()} - {(last_date + pd.Timedelta(days=horizon)).date()}")
    print(f"  Expected failures in 14d: {cum_14:.1f}")
    print(f"  Expected failures in {horizon}d: {cum_h:.1f}")
    print(f"Forecast table saved -> {FORECAST_TABLE_PATH}")
    return FORECAST_TABLE_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 2-year STL and replacement-risk figures.")
    parser.add_argument("--horizon", type=int, default=30, choices=[14, 30])
    args = parser.parse_args()
    run(horizon=args.horizon)


if __name__ == "__main__":
    main()
