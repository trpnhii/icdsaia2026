"""
Risk scoring variants for SolarGuard ablation study.

Each variant removes one architectural component while keeping the rest of the
pipeline comparable:
  - full            : peer-relative PR + STL-adjusted PR + adaptive 14/30d windows
  - no_stl          : skip STL seasonal adjustment on PR before scoring
  - no_peer         : absolute PR only (no same-day fleet peer comparison)
  - no_adaptive     : point-in-time metrics only (no rolling 14/30d windows)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from data_exclusions import apply_exclusions, build_excluded_days
from risk_daily import (
    MIN_IRRADIATION,
    THRESHOLDS_PATH,
    _score_one_day,
    _slope_per_day,
    _window_metrics,
    load_risk_thresholds,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "input_data" / "base" / "transformed.csv"

STL_PERIOD = 7
STL_MIN_POINTS = 14


class AblationVariant(str, Enum):
    FULL = "full"
    NO_STL = "no_stl"
    NO_PEER = "no_peer"
    NO_ADAPTIVE = "no_adaptive_window"


@dataclass(frozen=True)
class AblationConfig:
    variant: AblationVariant
    use_stl_adjustment: bool
    use_peer_relative: bool
    use_adaptive_window: bool
    use_persistence: bool = True


ABLATION_CONFIGS: dict[AblationVariant, AblationConfig] = {
    AblationVariant.FULL: AblationConfig(
        variant=AblationVariant.FULL,
        use_stl_adjustment=True,
        use_peer_relative=True,
        use_adaptive_window=True,
    ),
    AblationVariant.NO_STL: AblationConfig(
        variant=AblationVariant.NO_STL,
        use_stl_adjustment=False,
        use_peer_relative=True,
        use_adaptive_window=True,
    ),
    AblationVariant.NO_PEER: AblationConfig(
        variant=AblationVariant.NO_PEER,
        use_stl_adjustment=True,
        use_peer_relative=False,
        use_adaptive_window=True,
    ),
    AblationVariant.NO_ADAPTIVE: AblationConfig(
        variant=AblationVariant.NO_ADAPTIVE,
        use_stl_adjustment=True,
        use_peer_relative=True,
        use_adaptive_window=False,
        use_persistence=False,
    ),
}


def _fleet_seasonal_fraction(valid: pd.DataFrame) -> pd.Series:
    """Return per-date seasonal fraction from STL on fleet mean PR."""
    daily = (
        valid.groupby("date", as_index=False)
        .agg(mean_pr=("performance_ratio", "mean"))
        .sort_values("date")
    )
    if len(daily) < STL_MIN_POINTS:
        return pd.Series(0.0, index=valid.index)

    observed = daily["mean_pr"].to_numpy(dtype=float)
    stl = STL(observed, period=STL_PERIOD, robust=True).fit()
    seasonal = stl.seasonal
    baseline = np.where(observed != 0, observed, 1.0)
    seasonal_frac = seasonal / baseline
    lookup = pd.Series(seasonal_frac, index=pd.to_datetime(daily["date"]))
    return valid["date"].map(lookup).fillna(0.0)


def _prepare_valid_rows(
    input_path: Path,
    config: AblationConfig,
) -> pd.DataFrame:
    df = pd.read_csv(input_path, parse_dates=["date"])
    df = df[
        df["installed_capacity_kwp"].notna()
        & ~df["device_name"].str.startswith("Inverter(COM", na=False)
    ].copy()

    if "excluded_from_analysis" not in df.columns:
        df = apply_exclusions(df, build_excluded_days(df))

    valid = df[
        df["performance_ratio"].notna()
        & df["irradiation_kwh_m2"].notna()
        & (df["irradiation_kwh_m2"] >= MIN_IRRADIATION)
        & ~df["excluded_from_analysis"].fillna(False)
    ].copy()

    if config.use_stl_adjustment:
        seasonal_frac = _fleet_seasonal_fraction(valid)
        valid["performance_ratio"] = (
            valid["performance_ratio"] / (1.0 + seasonal_frac.clip(-0.5, 0.5))
        ).clip(lower=0)

    if config.use_peer_relative:
        valid["date_median_pr"] = valid.groupby("date")["performance_ratio"].transform("median")
        valid = valid[valid["date_median_pr"] > 0].copy()
        valid["relative_pr"] = valid["performance_ratio"] / valid["date_median_pr"]
        valid["relative_pr"] = valid["relative_pr"].clip(lower=0, upper=2)
        valid["peer_deficit"] = (1 - valid["relative_pr"]).clip(lower=0, upper=1)
        valid["moderate_low"] = (
            (valid["relative_pr"] < 0.70) | (valid["performance_ratio"] < 0.40)
        ).astype(int)
        valid["severe_low"] = (
            (valid["relative_pr"] < 0.50) | (valid["performance_ratio"] < 0.25)
        ).astype(int)
    else:
        # Absolute PR only: compare each inverter to a fixed reference (0.8).
        reference_pr = 0.80
        valid["date_median_pr"] = reference_pr
        valid["relative_pr"] = (valid["performance_ratio"] / reference_pr).clip(lower=0, upper=2)
        valid["peer_deficit"] = (reference_pr - valid["performance_ratio"]).clip(lower=0, upper=1)
        valid["moderate_low"] = (valid["performance_ratio"] < 0.70).astype(int)
        valid["severe_low"] = (valid["performance_ratio"] < 0.50).astype(int)

    return valid.sort_values(["date", "zone", "device_name"]).reset_index(drop=True)


def _score_one_day_with_windows(
    history: pd.DataFrame,
    score_date: pd.Timestamp,
    thresholds: dict,
    window_14: int,
    window_30: int,
    use_trend: bool,
) -> pd.DataFrame:
    """Score one day with configurable rolling windows (full=14/30, ablation=1/1)."""
    current = history[history["date"] == score_date].copy()
    if current.empty:
        return pd.DataFrame()

    base = (
        current.sort_values("date")
        .groupby(["zone", "device_name"], as_index=False)
        .agg(
            latest_date=("date", "max"),
            latest_pr=("performance_ratio", "last"),
            latest_relative_pr=("relative_pr", "last"),
            latest_peer_deficit=("peer_deficit", "last"),
        )
    )
    lifetime_days = (
        history.groupby(["zone", "device_name"], as_index=False)
        .agg(total_valid_days=("date", "nunique"))
    )
    base = base.merge(lifetime_days, on=["zone", "device_name"], how="left")

    last_14 = _window_metrics(history, score_date, window_14).rename(
        columns={
            f"valid_days_{window_14}d": "valid_days_14d",
            f"mean_pr_{window_14}d": "mean_pr_14d",
            f"mean_relative_pr_{window_14}d": "mean_relative_pr_14d",
            f"mean_deficit_{window_14}d": "mean_deficit_14d",
            f"moderate_low_rate_{window_14}d": "moderate_low_rate_14d",
            f"severe_low_rate_{window_14}d": "severe_low_rate_14d",
        }
    )
    last_30 = _window_metrics(history, score_date, window_30).rename(
        columns={
            f"valid_days_{window_30}d": "valid_days_30d",
            f"mean_pr_{window_30}d": "mean_pr_30d",
            f"mean_relative_pr_{window_30}d": "mean_relative_pr_30d",
            f"mean_deficit_{window_30}d": "mean_deficit_30d",
            f"moderate_low_rate_{window_30}d": "moderate_low_rate_30d",
            f"severe_low_rate_{window_30}d": "severe_low_rate_30d",
        }
    )
    if use_trend:
        trend_30 = (
            history[history["date"].between(score_date - pd.Timedelta(days=29), score_date)]
            .groupby(["zone", "device_name"])
            .apply(lambda g: _slope_per_day(g, "relative_pr"), include_groups=False)
            .rename("relative_pr_slope_30d")
            .reset_index()
        )
    else:
        trend_30 = pd.DataFrame(columns=["zone", "device_name", "relative_pr_slope_30d"])

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

    # Prevent single-day snapshots from inflating rolling rates used in thresholds.
    if window_14 < 14:
        scale_14 = np.minimum(1.0, risk["valid_days_14d"] / 14.0)
        risk["severe_low_rate_14d"] *= scale_14
        risk["moderate_low_rate_14d"] *= scale_14
        risk["mean_deficit_14d"] *= scale_14
    if window_30 < 30:
        scale_30 = np.minimum(1.0, risk["valid_days_30d"] / 30.0)
        risk["severe_low_rate_30d"] *= scale_30
        risk["moderate_low_rate_30d"] *= scale_30
        risk["mean_deficit_30d"] *= scale_30

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


def _score_one_day_ablation(
    history: pd.DataFrame,
    score_date: pd.Timestamp,
    config: AblationConfig,
    thresholds: dict,
) -> pd.DataFrame:
    if config.use_adaptive_window:
        return _score_one_day(history, score_date, thresholds)

    return _score_one_day_with_windows(
        history,
        score_date,
        thresholds,
        window_14=1,
        window_30=1,
        use_trend=False,
    )


def _apply_persistence(risk: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    persistence_cfg = thresholds.get("persistence", {})
    window_days = int(persistence_cfg.get("window_days", 5))
    required_days = int(persistence_cfg.get("required_days", 3))

    tmp = risk.sort_values(["device_name", "risk_date"]).copy()
    tmp["replacement_candidate_flag"] = tmp["replacement_candidate"].astype(int)
    tmp["persistence_count"] = (
        tmp.groupby("device_name")["replacement_candidate_flag"]
        .apply(lambda s: s.rolling(window=window_days, min_periods=1).sum())
        .reset_index(level=0, drop=True)
    )
    risk = risk.merge(
        tmp[["device_name", "risk_date", "persistence_count"]],
        on=["device_name", "risk_date"],
        how="left",
    )
    risk["replacement_candidate"] = risk["replacement_candidate"] & (
        risk["persistence_count"] >= required_days
    )
    return risk


def build_ablation_risk_scores(
    variant: AblationVariant,
    input_path: Path = INPUT_PATH,
    thresholds: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = ABLATION_CONFIGS[variant]
    if thresholds is None:
        thresholds = load_risk_thresholds()

    # Load thresholds wrapper for persistence key from full config file.
    with THRESHOLDS_PATH.open(encoding="utf-8") as handle:
        full_config = json.load(handle)
    persistence_thresholds = full_config.get("replacement_candidate", thresholds)
    if "persistence" in full_config.get("replacement_candidate", {}):
        persistence_thresholds = full_config["replacement_candidate"]

    valid = _prepare_valid_rows(input_path, config)
    frames: list[pd.DataFrame] = []
    for score_date in valid["date"].drop_duplicates():
        history = valid[valid["date"] <= score_date].copy()
        if config.use_adaptive_window:
            frames.append(_score_one_day(history, score_date, thresholds))
        else:
            frames.append(_score_one_day_ablation(history, score_date, config, thresholds))

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
        ["risk_date", "replacement_candidate", "replacement_window", "risk_score"],
        ascending=[True, False, True, False],
    )

    if config.use_persistence:
        risk = _apply_persistence(risk, persistence_thresholds)

    candidates = risk[risk["replacement_candidate"]].copy()
    return risk.reset_index(drop=True), candidates.reset_index(drop=True)
