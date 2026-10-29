"""
Isolation Forest anomaly detection for inverter risk scoring.

Input   : input_data/transformed_data.csv
Outputs :
  input_data/iforest_risk.csv   – all rows with risk_score and is_anomaly
  input_data/anomaly_list.csv   – only anomalies, formatted for reporting

Risk score (0–100): higher value means more anomalous / higher risk.
is_anomaly flag   : 1 = anomaly, 0 = normal  (based on IForest prediction).

Usage examples
--------------
# All available data (default)
python preprocess/iforest_risk.py

# Last N days
python preprocess/iforest_risk.py --days 5

# Explicit date range
python preprocess/iforest_risk.py --start-date 2026-02-01 --end-date 2026-02-07

# Combine: last 3 days up to a specific end date
python preprocess/iforest_risk.py --days 3 --end-date 2026-02-20
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "input_data" / "transformed_data.csv"
OUTPUT_PATH = PROJECT_ROOT / "input_data" / "iforest_risk.csv"
ANOMALY_LIST_PATH = PROJECT_ROOT / "input_data" / "anomaly_list.csv"

# Contamination: expected fraction of anomalies in the data.
# 0.05 flags roughly the worst 5 % of inverter-days.
CONTAMINATION = 0.05
RANDOM_STATE = 42

# Risk score thresholds for severity classification.
SEVERITY_THRESHOLDS = {
    "Critical": 80,
    "High": 60,
    "Medium": 40,
    "Low": 0,
}


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features used by the model."""
    df = df.copy()
    # Specific yield normalises by inverter capacity so devices of different
    # sizes are directly comparable (used for rows without irradiation).
    df["specific_yield_kwh_kwp"] = df["yield_kwh"] / df["total_string_capacity_kwp"]
    return df


def _scores_to_risk(raw_scores: np.ndarray) -> np.ndarray:
    """
    Convert IForest raw scores (more negative = more anomalous) to a
    0–100 risk scale (higher = riskier).
    """
    inverted = -raw_scores
    lo, hi = inverted.min(), inverted.max()
    if hi == lo:
        return np.zeros_like(inverted)
    return ((inverted - lo) / (hi - lo) * 100).round(2)


def _assign_severity(risk_score: pd.Series) -> pd.Series:
    """Map numeric risk score to a severity label."""
    conditions = [
        risk_score >= SEVERITY_THRESHOLDS["Critical"],
        risk_score >= SEVERITY_THRESHOLDS["High"],
        risk_score >= SEVERITY_THRESHOLDS["Medium"],
    ]
    choices = ["Critical", "High", "Medium"]
    return pd.Series(
        np.select(conditions, choices, default="Low"),
        index=risk_score.index,
    )


def _assign_anomaly_type(df: pd.DataFrame) -> pd.Series:
    """
    Classify the nature of each anomaly based on available data.

    - Low Performance Ratio : irradiation available, PR is unusually low
    - Yield Anomaly         : no irradiation data (Jan), yield is an outlier
    """
    return pd.Series(
        np.where(
            df["global_irradiation_kwh_m2"].notna(),
            "Low Performance Ratio",
            "Yield Anomaly",
        ),
        index=df.index,
    )


def _build_anomaly_list(result_df: pd.DataFrame, run_timestamp: str) -> pd.DataFrame:
    """Filter to anomalies only and shape into the reporting format."""
    anomalies = result_df[result_df["is_anomaly"] == 1].copy()

    anomalies["inverter_id"] = (
        "Zone" + anomalies["zone"].astype(str) + " - " + anomalies["device_name"]
    )
    anomalies["anomaly_type"] = _assign_anomaly_type(anomalies)
    anomalies["severity"] = _assign_severity(anomalies["risk_score"])
    anomalies["timestamp"] = run_timestamp

    return anomalies[["timestamp", "inverter_id", "anomaly_type", "severity", "risk_score"]].sort_values(
        "risk_score", ascending=False
    )


def _filter_date_range(
    df: pd.DataFrame,
    days: int | None,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    """
    Return rows whose date falls within the requested range.

    Resolution order:
    1. If --start-date / --end-date are given, use them directly.
    2. If --days N is given (with optional --end-date), take N days ending
       at end_date (or the latest date in the dataset).
    3. If nothing is given, return all rows unchanged.
    """
    dates = pd.to_datetime(df["date"])

    if start_date or end_date:
        lo = pd.Timestamp(start_date) if start_date else dates.min()
        hi = pd.Timestamp(end_date) if end_date else dates.max()
    elif days:
        hi = pd.Timestamp(end_date) if end_date else dates.max()
        lo = hi - timedelta(days=days - 1)
    else:
        return df

    mask = (dates >= lo) & (dates <= hi)
    filtered = df[mask]
    if filtered.empty:
        raise ValueError(
            f"No data found between {lo.date()} and {hi.date()}. "
            f"Available range: {dates.min().date()} – {dates.max().date()}"
        )
    return filtered


def run_iforest(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    anomaly_list_path: Path = ANOMALY_LIST_PATH,
    contamination: float = CONTAMINATION,
    days: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[Path, Path]:
    run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    df = pd.read_csv(input_path)
    df = _filter_date_range(df, days, start_date, end_date)
    df = _engineer_features(df)

    # When irradiation is available, PR captures both yield and irradiation
    # in a single normalised metric — a better anomaly signal than raw yield.
    # Specific yield is used as a fallback for rows without irradiation (Jan).
    FEATURES_WITH_IRRADIATION = ["performance_ratio"]
    FEATURES_WITHOUT_IRRADIATION = ["specific_yield_kwh_kwp"]

    has_irr = df["global_irradiation_kwh_m2"].notna()

    result_frames: list[pd.DataFrame] = []

    for mask, feature_cols in [
        (has_irr, FEATURES_WITH_IRRADIATION),
        (~has_irr, FEATURES_WITHOUT_IRRADIATION),
    ]:
        subset = df[mask].copy()
        subset = subset.dropna(subset=feature_cols)
        if subset.empty:
            continue

        X = subset[feature_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(
            contamination=contamination,
            random_state=RANDOM_STATE,
            n_estimators=100,
        )
        model.fit(X_scaled)

        subset["risk_score"] = _scores_to_risk(model.score_samples(X_scaled))
        subset["is_anomaly"] = (model.predict(X_scaled) == -1).astype(int)

        result_frames.append(subset)

    if not result_frames:
        raise ValueError("No data available to run Isolation Forest.")

    result_df = pd.concat(result_frames).sort_values(["zone", "date", "device_name"])

    FULL_OUTPUT_COLS = [
        "zone",
        "date",
        "plant_name",
        "device_name",
        "total_string_capacity_kwp",
        "yield_kwh",
        "specific_yield_kwh_kwp",
        "global_irradiation_kwh_m2",
        "performance_ratio",
        "risk_score",
        "is_anomaly",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df[FULL_OUTPUT_COLS].to_csv(output_path, index=False, encoding="utf-8-sig")

    anomaly_df = _build_anomaly_list(result_df, run_timestamp)
    anomaly_df.to_csv(anomaly_list_path, index=False, encoding="utf-8-sig")

    return output_path, anomaly_list_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Isolation Forest on inverter data to produce a risk-scored anomaly list.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        metavar="N",
        help="Use only the last N days of data (e.g. --days 5).",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Earliest date to include (e.g. --start-date 2026-02-01).",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Latest date to include (e.g. --end-date 2026-02-07).",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=CONTAMINATION,
        metavar="FLOAT",
        help=f"Expected fraction of anomalies (default: {CONTAMINATION}).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    output_path, anomaly_list_path = run_iforest(
        days=args.days,
        start_date=args.start_date,
        end_date=args.end_date,
        contamination=args.contamination,
    )

    result = pd.read_csv(output_path)
    anomalies = pd.read_csv(anomaly_list_path)

    date_range = f"{result['date'].min()} – {result['date'].max()}"
    total = len(result)
    n_anomalies = len(anomalies)

    print(f"Full results   : {output_path}")
    print(f"Anomaly list   : {anomaly_list_path}")
    print(f"Date range     : {date_range}")
    print(f"Total rows     : {total:,}")
    print(f"Anomalies      : {n_anomalies:,}  ({n_anomalies / total * 100:.1f} %)")
    print()
    print("Severity breakdown:")
    print(anomalies["severity"].value_counts().to_string())
    print()
    print("Anomaly type breakdown:")
    print(anomalies["anomaly_type"].value_counts().to_string())
    print()
    print("Top 10 highest-risk anomalies:")
    print(anomalies.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
