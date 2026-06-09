"""
Daily risk scoring and replacement-candidate ranking for the 2-year dataset.

Input:
  input_data/base/transformed.csv

Outputs:
  input_data/risk/risk_scores.csv
  input_data/risk/replacement_candidates.csv
  input_data/risk/risk_watchlist.csv
  input_data/risk/daily/YYYY-MM-DD/risk_scores.csv
  input_data/risk/daily/YYYY-MM-DD/replacement_candidates.csv
  input_data/risk/daily/YYYY-MM-DD/risk_watchlist.csv

The score is designed for triage. It does not prove hardware failure by itself;
it highlights inverter-days with persistent recent underperformance versus
same-day fleet peers and deteriorating short-term trend.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_exclusions import apply_exclusions, build_excluded_days


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "input_data" / "base" / "transformed.csv"
THRESHOLDS_PATH = PROJECT_ROOT / "config" / "risk_thresholds.json"
OUTPUT_DIR = PROJECT_ROOT / "input_data" / "risk"
RISK_PATH = OUTPUT_DIR / "risk_scores.csv"
CANDIDATES_PATH = OUTPUT_DIR / "replacement_candidates.csv"
WATCHLIST_PATH = OUTPUT_DIR / "risk_watchlist.csv"
DAILY_OUTPUT_DIR = OUTPUT_DIR / "daily"
GROUND_TRUTH_PATH = OUTPUT_DIR / "ground_truth_replacements.csv"

MIN_IRRADIATION = 0.5


def load_risk_thresholds(config_path: Path = THRESHOLDS_PATH) -> dict:
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    return config["replacement_candidate"]


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

    if "excluded_from_analysis" not in df.columns:
        df = apply_exclusions(df, build_excluded_days(df))

    # Low/zero irradiation days make PR unstable or undefined for this purpose.
    valid = df[
        df["performance_ratio"].notna()
        & df["irradiation_kwh_m2"].notna()
        & (df["irradiation_kwh_m2"] >= MIN_IRRADIATION)
        & ~df["excluded_from_analysis"].fillna(False)
    ].copy()

    valid["date_median_pr"] = valid.groupby("date")["performance_ratio"].transform("median")
    valid = valid[valid["date_median_pr"] > 0].copy()
    valid["relative_pr"] = valid["performance_ratio"] / valid["date_median_pr"]
    valid["relative_pr"] = valid["relative_pr"].clip(lower=0, upper=2)
    valid["peer_deficit"] = (1 - valid["relative_pr"]).clip(lower=0, upper=1)
    valid["moderate_low"] = ((valid["relative_pr"] < 0.70) | (valid["performance_ratio"] < 0.40)).astype(int)
    valid["severe_low"] = ((valid["relative_pr"] < 0.50) | (valid["performance_ratio"] < 0.25)).astype(int)
    return valid.sort_values(["date", "zone", "device_name"]).reset_index(drop=True)


def _score_one_day(
    history: pd.DataFrame,
    score_date: pd.Timestamp,
    thresholds: dict | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = load_risk_thresholds()
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

    acute_cfg = thresholds.get("acute_14d", {})
    replace_14 = (
        (risk["risk_score"] >= thresholds["risk_score_within_14_days"])
        & (
            (risk["severe_low_rate_14d"] >= thresholds["severe_low_rate_14d"])
            | (risk["mean_deficit_14d"] >= thresholds["mean_deficit_14d"])
            | (risk["projected_relative_pr_14d"] < thresholds["projected_relative_pr_14d"])
        )
    )
    replace_30 = (
        ~replace_14
        & (risk["risk_score"] >= thresholds["risk_score_within_30_days"])
        & (
            (risk["severe_low_rate_30d"] >= thresholds["severe_low_rate_30d"])
            | (risk["mean_deficit_30d"] >= thresholds["mean_deficit_30d"])
            | (risk["projected_relative_pr_30d"] < thresholds["projected_relative_pr_30d"])
        )
    )
    replace_acute_14 = pd.Series(False, index=risk.index)
    if acute_cfg.get("enabled", False):
        replace_acute_14 = (
            ~replace_14
            & ~replace_30
            & (risk["severe_low_rate_14d"] >= acute_cfg["severe_low_rate_14d"])
            & (risk["fleet_relative_risk_score"] >= acute_cfg["fleet_relative_risk_score"])
            & (risk["valid_days_14d"] >= acute_cfg["min_valid_days_14d"])
        )

    monitor_score = thresholds.get("monitor_risk_score", 40)
    risk["replacement_window"] = np.select(
        [replace_14, replace_30, replace_acute_14, risk["risk_score"] >= monitor_score],
        ["within_14_days", "within_30_days", "acute_14_days", "monitor"],
        default="no_replacement_signal",
    )
    risk["replacement_candidate"] = risk["replacement_window"].isin(
        ["within_14_days", "within_30_days", "acute_14_days"]
    )
    return risk


def build_risk_scores(
    input_path: Path = INPUT_PATH,
    thresholds: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if thresholds is None:
        thresholds = load_risk_thresholds()
    valid = _prepare_valid_rows(input_path)

    frames: list[pd.DataFrame] = []
    for score_date in valid["date"].drop_duplicates():
        history = valid[valid["date"] <= score_date].copy()
        frames.append(_score_one_day(history, score_date, thresholds))

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
    # Apply persistence filter to reduce spurious single-day replacement flags.
    # Configurable via thresholds.replacement_candidate.persistence (window_days, required_days).
    persistence_cfg = {}
    if thresholds is None:
        thresholds = load_risk_thresholds()
    if isinstance(thresholds, dict):
        persistence_cfg = thresholds.get("persistence", {})
    window_days = int(persistence_cfg.get("window_days", 5))
    required_days = int(persistence_cfg.get("required_days", 3))

    # Compute persistence per device by ordering by device_name -> risk_date,
    # computing a rolling sum of the boolean replacement_candidate flag.
    try:
        tmp = risk.sort_values(["device_name", "risk_date"]).copy()
        tmp["replacement_candidate_flag"] = tmp["replacement_candidate"].astype(int)
        tmp["persistence_count"] = (
            tmp.groupby("device_name")["replacement_candidate_flag"]
            .apply(lambda s: s.rolling(window=window_days, min_periods=1).sum())
            .reset_index(level=0, drop=True)
        )
        # Merge persistence_count back onto the main risk frame
        risk = risk.merge(
            tmp[["device_name", "risk_date", "persistence_count"]],
            on=["device_name", "risk_date"],
            how="left",
        )
        # Only keep replacement candidates that meet persistence requirement
        before = risk["replacement_candidate"].sum()
        risk["replacement_candidate"] = (
            risk["replacement_candidate"] & (risk["persistence_count"] >= required_days)
        )
        after = int(risk["replacement_candidate"].sum())
        if before != after:
            print(f"Applied persistence filter: candidates reduced {before} -> {after} (window={window_days}, required={required_days})")
    except Exception as exc:  # do not fail scoring for persistence computation
        print(f"Warning: could not compute persistence filter: {exc}")
    candidates = risk[risk["replacement_candidate"]].copy()

    # If a ground-truth replacements file exists, merge it in as high-confidence
    # replacement candidates so downstream financial code can use it as truth.
    # Expected columns (flexible): device_name, zone (optional), replacement_date | risk_date | date
    # Or monthly aggregated: calendar_month (YYYY-MM) with a count column 'replacement_candidate_events'.
    if GROUND_TRUTH_PATH.exists():
        try:
            gt = pd.read_csv(GROUND_TRUTH_PATH)
            if gt.empty:
                print(f"Ground-truth file found but empty: {GROUND_TRUTH_PATH}")
            else:
                # Normalize date column
                date_col = None
                for c in ("risk_date", "replacement_date", "date", "replacement_day"):
                    if c in gt.columns:
                        date_col = c
                        break

                if date_col is not None:
                    gt[date_col] = pd.to_datetime(gt[date_col], errors="coerce")
                    gt = gt.dropna(subset=[date_col])

                    # Map numeric site zone from transformed devices. Drop the optional
                    # string zone column in ground-truth to avoid merge suffix conflicts.
                    if "zone" in gt.columns:
                        gt = gt.rename(columns={"zone": "site_zone"})
                    devices = (
                        _prepare_valid_rows(input_path)[["zone", "device_name"]]
                        .drop_duplicates()
                    )
                    gt = gt.merge(devices, on="device_name", how="left")
                    gt = gt.dropna(subset=["zone"]).copy()
                    gt["zone"] = gt["zone"].astype(int)

                    # Build candidate-like rows
                    gt_rows: list[pd.Series] = []
                    for _, row in gt.iterrows():
                        cand = {
                            "risk_date": pd.Timestamp(row[date_col]),
                            "zone": int(row["zone"]),
                            "device_name": row["device_name"],
                            "latest_date": pd.Timestamp(row[date_col]),
                            "risk_score": 100.0,
                            "fleet_relative_risk_score": 100.0,
                            "replacement_window": "ground_truth",
                            "replacement_candidate": True,
                        }
                        gt_rows.append(pd.Series(cand))

                    if gt_rows:
                        gt_df = pd.DataFrame(gt_rows)
                        # Cast columns to align with 'candidates'
                        combined = pd.concat([candidates, gt_df], ignore_index=True, sort=False)
                        # Drop exact duplicates (same device + risk_date)
                        combined = combined.drop_duplicates(subset=["device_name", "risk_date"], keep="last")
                        candidates = combined
                        print(f"Merged {len(gt_df)} ground-truth replacements into candidates from {GROUND_TRUTH_PATH}")
                elif "calendar_month" in gt.columns:
                    # If user provided monthly aggregates, we cannot map to devices here.
                    # We'll surface a warning and skip device-level merge.
                    print(f"Ground-truth file {GROUND_TRUTH_PATH} contains monthly aggregates; device-level mapping required. Skipping merge.")
                else:
                    print(f"Ground-truth file {GROUND_TRUTH_PATH} found but no usable date/device columns.")
        except Exception as exc:  # noqa: BLE001 - we want to surface errors without stopping pipeline
            print(f"Error loading ground-truth replacements from {GROUND_TRUTH_PATH}: {exc}")

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
