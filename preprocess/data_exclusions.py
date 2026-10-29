"""
Mark days to exclude from PR / risk analysis.

Planned whole-site shutdowns (public holidays, fleet-wide low production) distort
peer-relative PR and should not count as inverter underperformance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "data_exclusions.json"


def load_exclusion_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    with config_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _hardware_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df["installed_capacity_kwp"].notna()
        & ~df["device_name"].str.startswith("Inverter(COM", na=False)
    ].copy()


def detect_fleet_shutdown_days(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Days with normal irradiation but most inverters producing almost nothing."""
    rules = config.get("fleet_shutdown_detection", {})
    if not rules.get("enabled", True):
        return pd.DataFrame(columns=["date", "exclusion_reason"])

    min_irr = float(rules.get("min_irradiation_kwh_m2", 2.0))
    max_yield = float(rules.get("max_yield_kwh", 50.0))
    min_devices = int(rules.get("min_devices", 30))
    low_fraction = float(rules.get("low_yield_device_fraction", 0.85))
    reason = str(rules.get("reason", "fleet_wide_low_production"))

    hardware = _hardware_rows(df)
    daily = (
        hardware.groupby("date", as_index=False)
        .agg(
            devices=("device_name", "nunique"),
            low_yield_rate=("yield_kwh", lambda values: (values < max_yield).mean()),
            irradiation_kwh_m2=("irradiation_kwh_m2", "first"),
        )
    )
    flagged = daily[
        (daily["devices"] >= min_devices)
        & daily["irradiation_kwh_m2"].notna()
        & (daily["irradiation_kwh_m2"] >= min_irr)
        & (daily["low_yield_rate"] >= low_fraction)
    ].copy()
    flagged["exclusion_reason"] = reason
    return flagged[["date", "exclusion_reason"]]


def build_excluded_days(df: pd.DataFrame, config_path: Path = DEFAULT_CONFIG_PATH) -> pd.DataFrame:
    config = load_exclusion_config(config_path)
    holiday_dates = pd.to_datetime(config.get("public_holidays", []))
    holiday_rows = pd.DataFrame(
        {
            "date": holiday_dates,
            "exclusion_reason": "public_holiday",
        }
    )

    shutdown_rows = detect_fleet_shutdown_days(df, config)
    excluded = (
        pd.concat([holiday_rows, shutdown_rows], ignore_index=True)
        .drop_duplicates("date", keep="first")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return excluded


def apply_exclusions(df: pd.DataFrame, excluded: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if excluded.empty:
        out["excluded_from_analysis"] = False
        out["exclusion_reason"] = ""
        return out

    excluded = excluded.copy()
    excluded["date"] = pd.to_datetime(excluded["date"])
    reason_map = excluded.set_index("date")["exclusion_reason"].to_dict()
    out["date"] = pd.to_datetime(out["date"])
    out["exclusion_reason"] = out["date"].map(reason_map).fillna("")
    out["excluded_from_analysis"] = out["exclusion_reason"].ne("")
    return out
