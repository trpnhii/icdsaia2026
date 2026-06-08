"""Build a static data snapshot for the dashboard web page."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data.js"
CONFIG_PATH = PROJECT_ROOT / "config" / "spare_inventory.json"
MONTH_LABELS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


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


def load_inventory_config() -> dict[str, object]:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pr_matrix(limit_devices: int = 9, limit_days: int = 10) -> dict[str, object]:
    transformed = read_csv(PROJECT_ROOT / "input_data" / "base" / "transformed.csv")
    persistence = read_csv(PROJECT_ROOT / "input_data" / "evaluation" / "no_label_candidate_persistence.csv")
    risk = read_csv(PROJECT_ROOT / "input_data" / "risk" / "risk_scores.csv")
    if transformed.empty or persistence.empty:
        return {"dates": [], "rows": []}

    transformed["date"] = pd.to_datetime(transformed["date"])
    top_devices = persistence.head(limit_devices)[["zone", "device_name"]].copy()
    if not risk.empty:
        dates = sorted(pd.to_datetime(risk["risk_date"]).dropna().unique())[-limit_days:]
    else:
        dates = sorted(transformed["date"].dropna().unique())[-limit_days:]
    date_labels = [pd.Timestamp(date).strftime("%d-%b") for date in dates]

    rows: list[dict[str, object]] = []
    for _, device in top_devices.iterrows():
        device_rows = transformed[
            (transformed["zone"] == device["zone"])
            & (transformed["device_name"] == device["device_name"])
            & transformed["date"].isin(dates)
        ].copy()
        values: list[float | None] = []
        for date in dates:
            match = device_rows[device_rows["date"] == date]
            if match.empty or pd.isna(match.iloc[0]["performance_ratio"]):
                values.append(None)
            else:
                values.append(round(float(match.iloc[0]["performance_ratio"]) * 100, 1))
        rows.append(
            {
                "label": str(device["device_name"]).replace(" Inverter ", " INV"),
                "device_name": device["device_name"],
                "zone": int(device["zone"]),
                "values": values,
            }
        )

    return {"dates": date_labels, "rows": rows}


def daily_pr_alarms(limit_days: int = 30) -> list[dict[str, object]]:
    transformed = read_csv(PROJECT_ROOT / "input_data" / "base" / "transformed.csv")
    if transformed.empty:
        return []
    transformed["date"] = pd.to_datetime(transformed["date"])
    valid = transformed[transformed["performance_ratio"].notna()].copy()
    valid["alarm"] = valid["performance_ratio"] < 0.70
    summary = (
        valid.groupby("date")
        .agg(alarm_count=("alarm", "sum"), device_count=("device_name", "nunique"))
        .reset_index()
        .sort_values("date")
        .tail(limit_days)
    )
    summary["date"] = summary["date"].dt.strftime("%Y-%m-%d")
    return summary.where(pd.notna(summary), None).to_dict(orient="records")


def monthly_abnormal_counts() -> list[dict[str, object]]:
    candidates = read_csv(PROJECT_ROOT / "input_data" / "risk" / "replacement_candidates.csv")
    counts = {month: 0 for month in range(1, 13)}
    if not candidates.empty:
        candidates["risk_date"] = pd.to_datetime(candidates["risk_date"])
        grouped = candidates.groupby(candidates["risk_date"].dt.month).size()
        for month, quantity in grouped.items():
            counts[int(month)] = int(quantity)

    return [
        {"month": month, "label": MONTH_LABELS[month - 1], "quantity": counts[month]}
        for month in range(1, 13)
    ]


def equipment_operational_rows(limit: int = 9) -> list[dict[str, object]]:
    candidates = read_csv(PROJECT_ROOT / "input_data" / "risk" / "replacement_candidates.csv")
    transformed = read_csv(PROJECT_ROOT / "input_data" / "base" / "transformed.csv")
    persistence = read_csv(PROJECT_ROOT / "input_data" / "evaluation" / "no_label_candidate_persistence.csv")
    config = load_inventory_config()
    if candidates.empty or transformed.empty or persistence.empty:
        return []

    lead_time_days = float(config.get("lead_time_days", 14))
    generation_hours = float(config.get("generation_hours_per_day", 4.02))
    electricity_price = float(config.get("electricity_price_vnd_per_kwh", 2102.53))
    holding_cost_annual = float(config.get("holding_cost_annual_vnd_per_unit", 10_000_000))

    capacity = (
        transformed[["zone", "device_name", "installed_capacity_kwp"]]
        .dropna()
        .drop_duplicates(["zone", "device_name"])
    )
    device_candidates = candidates.merge(capacity, on=["zone", "device_name"], how="left")
    top_devices = persistence.head(limit)[["zone", "device_name"]]

    rows: list[dict[str, object]] = []
    for _, device in top_devices.iterrows():
        subset = device_candidates[
            (device_candidates["zone"] == device["zone"])
            & (device_candidates["device_name"] == device["device_name"])
        ]
        if subset.empty:
            continue
        capacity_kwp = float(subset["installed_capacity_kwp"].iloc[0])
        pr_pct = round(float(subset["latest_pr"].mean()) * 100, 1)
        baseline_gen_kwh = lead_time_days * generation_hours * capacity_kwp
        baseline_revenue_vnd = baseline_gen_kwh * electricity_price
        baseline_measure_cost_vnd = baseline_revenue_vnd * 0.05
        ai_gen_kwh = baseline_gen_kwh * 0.01
        ai_revenue_vnd = ai_gen_kwh * electricity_price
        ai_measure_cost_vnd = holding_cost_annual / 12
        rows.append(
            {
                "equipment": str(device["device_name"]).replace(" Inverter ", " INV"),
                "pr_pct": pr_pct,
                "baseline_generation_loss_kwh": round(baseline_gen_kwh, 1),
                "baseline_revenue_loss_vnd": round(baseline_revenue_vnd, 0),
                "baseline_measure_cost_vnd": round(baseline_measure_cost_vnd, 0),
                "ai_generation_loss_kwh": round(ai_gen_kwh, 1),
                "ai_revenue_loss_vnd": round(ai_revenue_vnd, 0),
                "ai_measure_cost_vnd": round(ai_measure_cost_vnd, 0),
            }
        )
    return rows


def operational_totals(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {}
    return {
        "baseline_generation_loss_kwh": round(
            sum(float(row["baseline_generation_loss_kwh"]) for row in rows), 1
        ),
        "baseline_revenue_loss_vnd": round(
            sum(float(row["baseline_revenue_loss_vnd"]) for row in rows), 0
        ),
        "baseline_measure_cost_vnd": round(
            sum(float(row["baseline_measure_cost_vnd"]) for row in rows), 0
        ),
        "ai_generation_loss_kwh": round(sum(float(row["ai_generation_loss_kwh"]) for row in rows), 1),
        "ai_revenue_loss_vnd": round(sum(float(row["ai_revenue_loss_vnd"]) for row in rows), 0),
        "ai_measure_cost_vnd": round(sum(float(row["ai_measure_cost_vnd"]) for row in rows), 0),
    }


def scenario_comparison() -> list[dict[str, object]]:
    df = read_csv(PROJECT_ROOT / "output_data" / "financial" / "npv_irr_comparison.csv")
    if df.empty:
        return []
    rows = []
    for _, row in df.iterrows():
        irr_monthly = row.get("irr_monthly")
        irr_annual = row.get("irr_annual_effective")
        rows.append(
            {
                "scenario": row["scenario"],
                "npv_vnd": float(row["npv_vnd"]) if pd.notna(row["npv_vnd"]) else None,
                "irr_monthly": float(irr_monthly) if pd.notna(irr_monthly) else None,
                "irr_annual": float(irr_annual) if pd.notna(irr_annual) else None,
            }
        )
    return rows


def scenario_variance(rows: list[dict[str, object]]) -> dict[str, object]:
    if len(rows) < 2:
        return {}
    baseline = next((row for row in rows if row["scenario"] == "Without AI"), rows[0])
    ai = next((row for row in rows if row["scenario"] == "With AI"), rows[-1])
    npv_delta = (ai.get("npv_vnd") or 0) - (baseline.get("npv_vnd") or 0)
    irr_delta = None
    if ai.get("irr_monthly") is not None and baseline.get("irr_monthly") is not None:
        irr_delta = ai["irr_monthly"] - baseline["irr_monthly"]
    return {"npv_vnd": npv_delta, "irr_monthly": irr_delta}


def inventory_decision() -> dict[str, object]:
    inventory = inventory_summary()
    config = load_inventory_config()
    candidates = read_csv(PROJECT_ROOT / "input_data" / "risk" / "replacement_candidates.csv")
    unique_devices = 0
    if not candidates.empty:
        unique_devices = candidates[["zone", "device_name"]].drop_duplicates().shape[0]

    optimal_n = int(inventory.get("optimal_n", 0))
    current_stock = int(config.get("reference_prestock_units", 10))
    order_qty = max(optimal_n - current_stock, 0)
    return {
        "demand_devices": unique_devices,
        "current_stock_units": current_stock,
        "recommended_stock_units": optimal_n,
        "order_quantity": order_qty,
        "max_daily_candidates": int(candidates.groupby("risk_date").size().max()) if not candidates.empty else 0,
    }


def task_flow_summary() -> dict[str, object]:
    savings = read_csv(PROJECT_ROOT / "output_data" / "financial" / "monthly_operational_savings.csv")
    comparison = scenario_comparison()
    variance = scenario_variance(comparison)
    inventory = inventory_decision()
    total_savings = float(savings["net_cost_delta_excl_revenue"].sum()) if not savings.empty else 0.0
    ai = next((row for row in comparison if row["scenario"] == "With AI"), {})
    return {
        "money_saved_vnd": total_savings,
        "npv_gain_vnd": variance.get("npv_vnd"),
        "irr_gain_monthly": variance.get("irr_monthly"),
        "recommended_order_units": inventory.get("order_quantity"),
        "recommended_stock_units": inventory.get("recommended_stock_units"),
        "break_even_capex_vnd": ai.get("npv_vnd"),
    }


def task_flow_payload() -> dict[str, object]:
    equipment_rows = equipment_operational_rows()
    comparison = scenario_comparison()
    return {
        "prMatrix": pr_matrix(),
        "dailyAlarms": daily_pr_alarms(),
        "monthlyAbnormal": monthly_abnormal_counts(),
        "equipmentRows": equipment_rows,
        "operationalTotals": operational_totals(equipment_rows),
        "scenarioComparison": comparison,
        "scenarioVariance": scenario_variance(comparison),
        "inventoryDecision": inventory_decision(),
        "summary": task_flow_summary(),
    }


def main() -> None:
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
        "taskFlow": task_flow_payload(),
    }
    DATA_PATH.write_text(
        "window.DASHBOARD_DATA = " + json.dumps(payload, indent=2) + ";\n",
        encoding="utf-8",
    )
    print(f"Dashboard data -> {DATA_PATH}")


if __name__ == "__main__":
    main()
