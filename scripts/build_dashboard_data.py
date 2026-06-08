"""Build a static data snapshot for the dashboard web page."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
DATA_PATH = DASHBOARD_DIR / "data.js"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def records(path: Path, limit: int | None = None) -> list[dict[str, object]]:
    df = read_csv(path)
    if limit is not None:
        df = df.head(limit)
    return df.where(pd.notna(df), None).to_dict(orient="records")


def latest_week_candidates() -> list[dict[str, object]]:
    path = PROJECT_ROOT / "input_data" / "risk" / "replacement_candidates.csv"
    df = read_csv(path)
    if df.empty:
        return []
    df["risk_date"] = pd.to_datetime(df["risk_date"])
    end = df["risk_date"].max()
    start = end - pd.Timedelta(days=6)
    cols = [
        "risk_date",
        "zone",
        "device_name",
        "risk_score",
        "replacement_window",
        "mean_relative_pr_14d",
        "severe_low_rate_14d",
    ]
    week = df[df["risk_date"].between(start, end)].copy()
    week["risk_date"] = week["risk_date"].dt.strftime("%Y-%m-%d")
    return week[cols].sort_values(["risk_date", "risk_score"], ascending=[True, False]).where(
        pd.notna(week), None
    ).to_dict(orient="records")


def risk_summary() -> dict[str, object]:
    risk = read_csv(PROJECT_ROOT / "input_data" / "risk" / "risk_scores.csv")
    candidates = read_csv(PROJECT_ROOT / "input_data" / "risk" / "replacement_candidates.csv")
    if risk.empty:
        return {}
    latest_date = pd.to_datetime(risk["risk_date"]).max()
    latest = risk[pd.to_datetime(risk["risk_date"]) == latest_date]
    return {
        "latest_scored_date": latest_date.strftime("%Y-%m-%d"),
        "scored_inverter_days": int(len(risk)),
        "scored_dates": int(risk["risk_date"].nunique()),
        "latest_watchlist_count": int(len(latest)),
        "replacement_candidate_days": int(len(candidates)),
    }


def inventory_summary() -> dict[str, object]:
    curve = read_csv(PROJECT_ROOT / "input_data" / "inventory" / "spare_inventory_cost_summary.csv")
    if curve.empty:
        return {}
    optimal = curve.loc[curve["total_cost_vnd"].idxmin()]
    return {
        "optimal_n": int(optimal["stock_n"]),
        "total_cost_vnd": float(optimal["total_cost_vnd"]),
        "expected_generation_loss_vnd": float(optimal["expected_generation_loss_vnd"]),
        "holding_cost_vnd": float(optimal["holding_cost_vnd"]),
        "curve": curve.where(pd.notna(curve), None).to_dict(orient="records"),
    }


def financial_summary() -> dict[str, object]:
    df = read_csv(PROJECT_ROOT / "output_data" / "financial" / "npv_irr_comparison.csv")
    if df.empty:
        return {}
    ai = df[df["scenario"] == "With AI"]
    if ai.empty:
        return {}
    row = ai.iloc[0]
    return {
        "npv_vnd": float(row["npv_vnd"]),
        "irr_monthly": float(row["irr_monthly"]),
        "capex_ai_vnd": float(row["capex_ai_included_vnd"]),
        "break_even_capex_vnd": float(row["max_capex_ai_for_npv_zero_vnd"]),
    }


def main() -> None:
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "risk": risk_summary(),
        "forecast": records(PROJECT_ROOT / "input_data" / "forecast" / "forecast_table.csv", limit=30),
        "weeklyCandidates": latest_week_candidates(),
        "inventory": inventory_summary(),
        "financial": financial_summary(),
        "timeline": records(PROJECT_ROOT / "output_data" / "financial" / "npv_timeline.csv"),
        "operationalSavings": records(
            PROJECT_ROOT / "output_data" / "financial" / "monthly_operational_savings.csv"
        ),
    }
    DATA_PATH.write_text(
        "window.DASHBOARD_DATA = " + json.dumps(payload, indent=2) + ";\n",
        encoding="utf-8",
    )
    print(f"Dashboard data -> {DATA_PATH}")


if __name__ == "__main__":
    main()
