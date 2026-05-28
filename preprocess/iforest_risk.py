"""
Isolation Forest anomaly detection for inverter risk scoring.

Input  : input_data/transformed_data.csv
Output : input_data/iforest_risk.csv

Risk score (0–100): higher value means more anomalous / higher risk.
is_anomaly flag   : 1 = anomaly, 0 = normal  (based on IForest prediction).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "input_data" / "transformed_data.csv"
OUTPUT_PATH = PROJECT_ROOT / "input_data" / "iforest_risk.csv"

# Contamination: expected fraction of anomalies in the data.
# 0.05 flags roughly the worst 5 % of inverter-days.
CONTAMINATION = 0.05
RANDOM_STATE = 42


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


def run_iforest(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    contamination: float = CONTAMINATION,
) -> Path:
    df = pd.read_csv(input_path)
    df = _engineer_features(df)

    # When irradiation is available, PR captures both yield and irradiation
    # in a single normalised metric — a better anomaly signal than raw yield.
    # Specific yield is used as a fallback for rows without irradiation (Jan).
    FEATURES_WITH_IRRADIATION = ["performance_ratio"]
    FEATURES_WITHOUT_IRRADIATION = ["specific_yield_kwh_kwp"]

    # Split into rows that have irradiation data and those that do not.
    has_irr = df["global_irradiation_kwh_m2"].notna()

    result_frames: list[pd.DataFrame] = []

    for mask, feature_cols in [
        (has_irr, FEATURES_WITH_IRRADIATION),
        (~has_irr, FEATURES_WITHOUT_IRRADIATION),
    ]:
        subset = df[mask].copy()

        # Drop rows where any required feature is missing.
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
        # IForest returns -1 for anomaly, 1 for normal; convert to 1/0.
        subset["is_anomaly"] = (model.predict(X_scaled) == -1).astype(int)

        result_frames.append(subset)

    if not result_frames:
        raise ValueError("No data available to run Isolation Forest.")

    result_df = pd.concat(result_frames).sort_values(["zone", "date", "device_name"])

    OUTPUT_COLS = [
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
    result_df = result_df[OUTPUT_COLS]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return output_path


def main() -> None:
    output_path = run_iforest()

    result = pd.read_csv(output_path)
    total = len(result)
    anomalies = result["is_anomaly"].sum()

    print(f"Output         : {output_path}")
    print(f"Total rows     : {total:,}")
    print(f"Anomalies      : {anomalies:,}  ({anomalies / total * 100:.1f} %)")
    print(f"Risk score range: {result['risk_score'].min():.1f} – {result['risk_score'].max():.1f}")
    print()
    print("Top 10 highest-risk inverter-days:")
    top = result.nlargest(10, "risk_score")[
        ["zone", "date", "device_name", "yield_kwh", "global_irradiation_kwh_m2", "performance_ratio", "risk_score"]
    ]
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
