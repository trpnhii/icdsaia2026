"""
Daily risk scoring and replacement-candidate ranking for the 2-year dataset.

Input:
  input_data/2_years/transformed.csv

Outputs:
  input_data/2_years/risk_scores.csv
  input_data/2_years/replacement_candidates.csv
  input_data/2_years/risk_watchlist.csv
  input_data/2_years/daily/YYYY-MM-DD/risk_scores.csv
  input_data/2_years/daily/YYYY-MM-DD/replacement_candidates.csv
  input_data/2_years/daily/YYYY-MM-DD/risk_watchlist.csv

The score is designed for triage. It does not prove hardware failure by itself;
it highlights inverter-days with persistent recent underperformance versus
same-day fleet peers and deteriorating short-term trend.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "input_data" / "2_years" / "transformed.csv"
OUTPUT_DIR = PROJECT_ROOT / "input_data" / "2_years"
RISK_PATH = OUTPUT_DIR / "risk_scores.csv"
CANDIDATES_PATH = OUTPUT_DIR / "replacement_candidates.csv"
WATCHLIST_PATH = OUTPUT_DIR / "risk_watchlist.csv"
DAILY_OUTPUT_DIR = OUTPUT_DIR / "daily"

MIN_IRRADIATION = 0.5


def _slope_per_day(group: pd.DataFrame, value_col: str) -> float:
    group = group.dropna(subset=[value_col]).sort_values("date")
    if len(group) < 5:
        return 0.0
    x = (group["date"] - group["date"].min()).dt.days.to_numpy(dtype=float)
    y = group[value_col].to_numpy(dtype=float)
    if np.unique(x).size < 2:
        return 0.0
    return float(np.polyfit(x, y, deg=1)[0])


def _window_metrics(df: pd.DataFrame, score_date: pd.Timestamp, days: int) -> pd.DataFrame:
    lo = score_date - pd.Timedelta(days=days - 1)
    w = df[df["date"].between(lo, score_date)].copy()
    return (
        w.groupby(["zone", "device_name"], as_index=False)
        .agg(
            **{
                f"valid_days_{days}d": ("date", "nunique"),
                f"mean_pr_{days}d": ("performance_ratio", "mean"),
                f"mean_relative_pr_{days}d": ("relative_pr", "mean"),
                f"mean_deficit_{days}d": ("peer_deficit", "mean"),
                f"moderate_low_rate_{days}d": ("moderate_low", "mean"),
                f"severe_low_rate_{days}d": ("severe_low", "mean"),
            }
        )
    )


def _prepare_valid_rows(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, parse_dates=["date"])

    # Logger-level rows do not represent replaceable inverter hardware.
    df = df[
        df["installed_capacity_kwp"].notna()
        & ~df["device_name"].str.startswith("Inverter(COM", na=False)
    ].copy()

    # Low/zero irradiation days make PR unstable or undefined for this purpose.
    valid = df[
        df["performance_ratio"].notna()
        & df["irradiation_kwh_m2"].notna()
        & (df["irradiation_kwh_m2"] >= MIN_IRRADIATION)
    ].copy()

    valid["date_median_pr"] = valid.groupby("date")["performance_ratio"].transform("median")
    valid = valid[valid["date_median_pr"] > 0].copy()
    valid["relative_pr"] = valid["performance_ratio"] / valid["date_median_pr"]
    valid["relative_pr"] = valid["relative_pr"].clip(lower=0, upper=2)
    valid["peer_deficit"] = (1 - valid["relative_pr"]).clip(lower=0, upper=1)
    valid["moderate_low"] = ((valid["relative_pr"] < 0.70) | (valid["performance_ratio"] < 0.40)).astype(int)
    valid["severe_low"] = ((valid["relative_pr"] < 0.50) | (valid["performance_ratio"] < 0.25)).astype(int)
    return valid.sort_values(["date", "zone", "device_name"]).reset_index(drop=True)


def _score_one_day(history: pd.DataFrame, score_date: pd.Timestamp) -> pd.DataFrame:
    current = history[history["date"] == score_date].copy()
    if current.empty:
        return pd.DataFrame()

    base = (
        current.sort_values("date")
        .groupby(["zone", "device_name"], as_index=False)
        .agg(
            latest_date=("date", "max"),
            total_valid_days=("date", "nunique"),
            latest_pr=("performance_ratio", "last"),
            latest_relative_pr=("relative_pr", "last"),
            latest_peer_deficit=("peer_deficit", "last"),
        )
    )

    lifetime_days = (
        history.groupby(["zone", "device_name"], as_index=False)
        .agg(total_valid_days=("date", "nunique"))
    )
    base = base.drop(columns=["total_valid_days"]).merge(
        lifetime_days, on=["zone", "device_name"], how="left"
    )

    last_14 = _window_metrics(history, score_date, 14)
    last_30 = _window_metrics(history, score_date, 30)
    trend_30 = (
        history[history["date"].between(score_date - pd.Timedelta(days=29), score_date)]
        .groupby(["zone", "device_name"])
        .apply(lambda g: _slope_per_day(g, "relative_pr"), include_groups=False)
        .rename("relative_pr_slope_30d")
        .reset_index()
    )

    risk = base.merge(last_14, on=["zone", "device_name"], how="left")
    risk = risk.merge(last_30, on=["zone", "device_name"], how="left")
    risk = risk.merge(trend_30, on=["zone", "device_name"], how="left")
    risk = risk.fillna(
        {
            "relative_pr_slope_30d": 0.0,
            "mean_deficit_14d": 0.0,
            "mean_deficit_30d": 0.0,
            "severe_low_rate_14d": 0.0,
            "severe_low_rate_30d": 0.0,
            "moderate_low_rate_14d": 0.0,
            "moderate_low_rate_30d": 0.0,
            "latest_peer_deficit": 0.0,
        }
    )
    risk["risk_date"] = score_date

    risk["trend_decline_30d"] = (-risk["relative_pr_slope_30d"] * 30).clip(lower=0, upper=1)
    risk["projected_relative_pr_14d"] = (
        risk["mean_relative_pr_14d"].fillna(risk["latest_relative_pr"])
        + risk["relative_pr_slope_30d"] * 14
    ).clip(lower=0, upper=2)
    risk["projected_relative_pr_30d"] = (
        risk["mean_relative_pr_30d"].fillna(risk["latest_relative_pr"])
        + risk["relative_pr_slope_30d"] * 30
    ).clip(lower=0, upper=2)

    risk["risk_score"] = (
        100
        * (
            0.25 * risk["mean_deficit_14d"]
            + 0.20 * risk["mean_deficit_30d"]
            + 0.20 * risk["severe_low_rate_14d"]
            + 0.15 * risk["severe_low_rate_30d"]
            + 0.10 * risk["trend_decline_30d"]
            + 0.10 * risk["latest_peer_deficit"]
        )
    ).clip(lower=0, upper=100).round(2)
    risk["fleet_relative_risk_score"] = (
        risk["risk_score"].rank(pct=True, method="average") * 100
    ).round(2)

    replace_14 = (
        (risk["risk_score"] >= 80)
        & (
            (risk["severe_low_rate_14d"] >= 0.50)
            | (risk["mean_deficit_14d"] >= 0.55)
            | (risk["projected_relative_pr_14d"] < 0.45)
        )
    )
    replace_30 = (
        ~replace_14
        & (risk["risk_score"] >= 65)
        & (
            (risk["severe_low_rate_30d"] >= 0.35)
            | (risk["mean_deficit_30d"] >= 0.40)
            | (risk["projected_relative_pr_30d"] < 0.60)
        )
    )

    risk["replacement_window"] = np.select(
        [replace_14, replace_30, risk["risk_score"] >= 45],
        ["within_14_days", "within_30_days", "monitor"],
        default="no_replacement_signal",
    )
    risk["replacement_candidate"] = risk["replacement_window"].isin(["within_14_days", "within_30_days"])
    return risk


def build_risk_scores(input_path: Path = INPUT_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = _prepare_valid_rows(input_path)

    frames: list[pd.DataFrame] = []
    for score_date in valid["date"].drop_duplicates():
        history = valid[valid["date"] <= score_date].copy()
        frames.append(_score_one_day(history, score_date))

    risk = pd.concat(frames, ignore_index=True)
    ordered_cols = [
        "risk_date",
        "zone",
        "device_name",
        "latest_date",
        "risk_score",
        "fleet_relative_risk_score",
        "replacement_window",
        "replacement_candidate",
        "latest_pr",
        "latest_relative_pr",
        "mean_pr_14d",
        "mean_relative_pr_14d",
        "mean_deficit_14d",
        "severe_low_rate_14d",
        "mean_pr_30d",
        "mean_relative_pr_30d",
        "mean_deficit_30d",
        "severe_low_rate_30d",
        "relative_pr_slope_30d",
        "projected_relative_pr_14d",
        "projected_relative_pr_30d",
        "valid_days_14d",
        "valid_days_30d",
        "total_valid_days",
    ]
    risk = risk[ordered_cols].sort_values(
        ["risk_date", "replacement_candidate", "replacement_window", "risk_score", "fleet_relative_risk_score"],
        ascending=[True, False, True, False, False],
    )
    candidates = risk[risk["replacement_candidate"]].copy()
    return risk.reset_index(drop=True), candidates.reset_index(drop=True)


def _top_watchlist(day_risk: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    return (
        day_risk.sort_values(
            ["replacement_candidate", "replacement_window", "risk_score", "fleet_relative_risk_score"],
            ascending=[False, True, False, False],
        )
        .head(limit)
        .copy()
    )


def export_daily_outputs(risk: pd.DataFrame, candidates: pd.DataFrame) -> int:
    DAILY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    written_dates = 0
    for risk_date, day_risk in risk.groupby("risk_date", sort=True):
        date_text = pd.Timestamp(risk_date).strftime("%Y-%m-%d")
        day_dir = DAILY_OUTPUT_DIR / date_text
        day_dir.mkdir(parents=True, exist_ok=True)

        day_candidates = candidates[candidates["risk_date"] == risk_date].copy()
        day_watchlist = _top_watchlist(day_risk)

        day_risk.to_csv(day_dir / "risk_scores.csv", index=False, encoding="utf-8-sig")
        day_candidates.to_csv(day_dir / "replacement_candidates.csv", index=False, encoding="utf-8-sig")
        day_watchlist.to_csv(day_dir / "risk_watchlist.csv", index=False, encoding="utf-8-sig")
        written_dates += 1

    return written_dates


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    risk, candidates = build_risk_scores()
    latest_risk_date = risk["risk_date"].max()
    watchlist = _top_watchlist(risk[risk["risk_date"] == latest_risk_date])

    risk.to_csv(RISK_PATH, index=False, encoding="utf-8-sig")
    candidates.to_csv(CANDIDATES_PATH, index=False, encoding="utf-8-sig")
    watchlist.to_csv(WATCHLIST_PATH, index=False, encoding="utf-8-sig")
    written_dates = export_daily_outputs(risk, candidates)

    print(f"Risk scores -> {RISK_PATH}")
    print(f"Replacement candidates -> {CANDIDATES_PATH}")
    print(f"Watchlist -> {WATCHLIST_PATH}")
    print(f"Daily output folders -> {DAILY_OUTPUT_DIR}")
    print(f"Scored inverter-days: {len(risk)}")
    print(f"Scored dates: {risk['risk_date'].nunique()}")
    print(f"Daily folders written: {written_dates}")
    print(f"Latest scored date: {latest_risk_date.date()}")
    print(f"Replacement candidate-days: {len(candidates)}")
    if not candidates.empty:
        print()
        print(
            candidates[
                [
                    "risk_date",
                    "zone",
                    "device_name",
                    "risk_score",
                    "fleet_relative_risk_score",
                    "replacement_window",
                    "mean_relative_pr_14d",
                    "severe_low_rate_14d",
                    "mean_relative_pr_30d",
                    "severe_low_rate_30d",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
