"""Build a static data snapshot for the dashboard web page."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data.js"
CONFIG_PATH = PROJECT_ROOT / "config" / "spare_inventory.json"

# Anchor (inclusive) end date for the dashboard's recent-window views (PR matrix,
# daily alarms, candidate week). The latest scored days (Apr 29-30, 2026) contain
# implausible irradiation spikes that deflate PR fleet-wide, so we end the window
# on a clean date by default. Set to None to always use the latest available date.
DEFAULT_AS_OF = "2026-04-28"
# User-provided reference values for "Without AI" financial baseline.
WITHOUT_AI_IRR_QUARTERLY = 0.1473  # 14.73% per quarter
WITHOUT_AI_NPV_USD = 8517.36
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


def _npv(rate: float, cashflows: list[float]) -> float:
    return sum(cf / ((1 + rate) ** i) for i, cf in enumerate(cashflows))


def _solve_irr(cashflows: list[float]) -> float | None:
    # Simple monotonic root search for monthly IRR in [-99%, +1000%].
    lo, hi = -0.99, 10.0
    flo, fhi = _npv(lo, cashflows), _npv(hi, cashflows)
    if flo * fhi > 0:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        fm = _npv(mid, cashflows)
        if abs(fm) < 1e-7:
            return mid
        if flo * fm <= 0:
            hi = mid
            fhi = fm
        else:
            lo = mid
            flo = fm
    return (lo + hi) / 2


def resolve_as_of(as_of: str | None) -> pd.Timestamp | None:
    if as_of is None:
        return None
    return pd.to_datetime(as_of)


def latest_week_candidates(as_of: pd.Timestamp | None = None) -> list[dict[str, object]]:
    path = PROJECT_ROOT / "input_data" / "risk" / "replacement_candidates.csv"
    df = read_csv(path)
    if df.empty:
        return []
    df["risk_date"] = pd.to_datetime(df["risk_date"])
    end = df["risk_date"].max()
    if as_of is not None:
        end = min(end, as_of)
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


def risk_summary(as_of: pd.Timestamp | None = None) -> dict[str, object]:
    risk = read_csv(PROJECT_ROOT / "input_data" / "risk" / "risk_scores.csv")
    candidates = read_csv(PROJECT_ROOT / "input_data" / "risk" / "replacement_candidates.csv")
    if risk.empty:
        return {}
    risk_dates = pd.to_datetime(risk["risk_date"])
    display_date = risk_dates.max()
    if as_of is not None:
        eligible = risk_dates[risk_dates <= as_of]
        if not eligible.empty:
            display_date = eligible.max()
    latest = risk[risk_dates == display_date]
    return {
        "latest_scored_date": display_date.strftime("%Y-%m-%d"),
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
    missing = read_csv(PROJECT_ROOT / "output_data" / "financial" / "missing_financial_inputs.csv")
    monthly = read_csv(PROJECT_ROOT / "output_data" / "financial" / "monthly_operational_savings.csv")
    row = ai.iloc[0]
    irr_raw_monthly: float | None = float(row["irr_monthly"]) if pd.notna(row["irr_monthly"]) else None
    irr_adjusted_monthly: float | None = irr_raw_monthly
    irr_reliability_note = ""

    coverage_months = int(len(monthly)) if not monthly.empty else 0
    horizon_months = 24
    coverage_ratio = min(1.0, coverage_months / horizon_months) if horizon_months > 0 else 1.0

    if coverage_months > 0 and pd.notna(row.get("capex_ai_included_vnd")):
        capex = float(row["capex_ai_included_vnd"])
        monthly_net = monthly["net_cost_delta_excl_revenue"].astype(float).tolist()
        if coverage_ratio < 1.0:
            adjusted_cashflows = [-capex] + [value * coverage_ratio for value in monthly_net]
            solved = _solve_irr(adjusted_cashflows)
            if solved is not None:
                irr_adjusted_monthly = solved
            irr_reliability_note = (
                f"IRR adjusted by data coverage ({coverage_months}/{horizon_months} months) "
                "to reduce short-window overstatement."
            )
    if irr_adjusted_monthly is None and irr_raw_monthly is not None:
        irr_adjusted_monthly = irr_raw_monthly

    return {
        "npv_vnd": float(row["npv_vnd"]),
        "irr_monthly": irr_adjusted_monthly,
        "irr_raw_monthly": irr_raw_monthly,
        "irr_coverage_months": coverage_months,
        "irr_reliability_note": irr_reliability_note,
        "capex_ai_vnd": float(row["capex_ai_included_vnd"]),
        "break_even_capex_vnd": float(row["max_capex_ai_for_npv_zero_vnd"]),
    }


def load_inventory_config() -> dict[str, object]:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pr_matrix(
    limit_devices: int = 9,
    limit_days: int = 10,
    as_of: pd.Timestamp | None = None,
) -> dict[str, object]:
    transformed = read_csv(PROJECT_ROOT / "input_data" / "base" / "transformed.csv")
    persistence = read_csv(PROJECT_ROOT / "input_data" / "evaluation" / "no_label_candidate_persistence.csv")
    risk = read_csv(PROJECT_ROOT / "input_data" / "risk" / "risk_scores.csv")
    if transformed.empty or persistence.empty:
        return {"dates": [], "rows": []}

    transformed["date"] = pd.to_datetime(transformed["date"])
    top_devices = persistence.head(limit_devices)[["zone", "device_name"]].copy()
    if not risk.empty:
        candidate_dates = pd.to_datetime(risk["risk_date"]).dropna()
    else:
        candidate_dates = transformed["date"].dropna()
    if as_of is not None:
        candidate_dates = candidate_dates[candidate_dates <= as_of]
    dates = sorted(candidate_dates.unique())[-limit_days:]
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


def daily_pr_alarms(limit_days: int = 30, as_of: pd.Timestamp | None = None) -> list[dict[str, object]]:
    transformed = read_csv(PROJECT_ROOT / "input_data" / "base" / "transformed.csv")
    if transformed.empty:
        return []
    transformed["date"] = pd.to_datetime(transformed["date"])
    valid = transformed[transformed["performance_ratio"].notna()].copy()
    if as_of is not None:
        valid = valid[valid["date"] <= as_of]
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
        grouped = (
            candidates.groupby(candidates["risk_date"].dt.month)
            .apply(lambda df: df[["zone", "device_name"]].drop_duplicates().shape[0])
        )
        for month, quantity in grouped.items():
            counts[int(month)] = int(quantity)

    return [
        {"month": month, "label": MONTH_LABELS[month - 1], "quantity": counts[month]}
        for month in range(1, 13)
    ]


def equipment_operational_rows(limit: int = 9, as_of: pd.Timestamp | None = None) -> list[dict[str, object]]:
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
    device_candidates["risk_date"] = pd.to_datetime(device_candidates["risk_date"])
    if as_of is not None:
        device_candidates = device_candidates[device_candidates["risk_date"] <= as_of]
    top_devices = persistence.head(limit)[["zone", "device_name"]]
    if device_candidates.empty:
        return []

    # Cost/impact assumptions for dashboard comparison view.
    inverter_unit_cost = float(config.get("inverter_unit_cost_vnd", 125_000_000))
    diagnostic_cost_per_event = inverter_unit_cost * 0.003  # inspection + dispatch
    ai_loss_residual_ratio = 0.10  # AI + pre-stock avoids ~90% of modeled loss impact

    severity_weight = {
        "within_14_days": 1.0,
        "within_30_days": 0.6,
    }

    weighted_event_days_by_device: dict[tuple[int, str], float] = {}
    for (zone, device_name), group in device_candidates.groupby(["zone", "device_name"]):
        weighted_days = (
            group["replacement_window"]
            .map(severity_weight)
            .fillna(0.5)
            .astype(float)
            .sum()
        )
        weighted_event_days_by_device[(int(zone), str(device_name))] = float(weighted_days)

    total_weighted_event_days = sum(weighted_event_days_by_device.values())

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
        weighted_event_days = weighted_event_days_by_device.get((int(device["zone"]), str(device["device_name"])), 0.0)
        candidate_days = float(subset["risk_date"].nunique())
        baseline_gen_kwh = weighted_event_days * lead_time_days * generation_hours * capacity_kwp
        baseline_revenue_vnd = baseline_gen_kwh * electricity_price
        baseline_measure_cost_vnd = weighted_event_days * diagnostic_cost_per_event
        ai_gen_kwh = baseline_gen_kwh * ai_loss_residual_ratio
        ai_revenue_vnd = ai_gen_kwh * electricity_price
        device_share = (weighted_event_days / total_weighted_event_days) if total_weighted_event_days > 0 else 0
        ai_holding_share = (holding_cost_annual / 12) * device_share
        ai_measure_cost_vnd = ai_holding_share + (weighted_event_days * diagnostic_cost_per_event * 0.4)
        rows.append(
            {
                "equipment": str(device["device_name"]).replace(" Inverter ", " INV"),
                "pr_pct": pr_pct,
                "candidate_days": round(candidate_days, 1),
                "weighted_event_days": round(weighted_event_days, 2),
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
    missing = read_csv(PROJECT_ROOT / "output_data" / "financial" / "missing_financial_inputs.csv")
    monthly = read_csv(PROJECT_ROOT / "output_data" / "financial" / "monthly_operational_savings.csv")
    config = load_inventory_config()
    fx = float(config.get("vnd_per_usd", 25550))
    without_ai_irr_monthly = (1 + WITHOUT_AI_IRR_QUARTERLY) ** (1 / 3) - 1
    without_ai_npv_vnd = WITHOUT_AI_NPV_USD * fx
    adjust_irr = False
    coverage_months = int(len(monthly)) if not monthly.empty else 0
    horizon_months = 24
    coverage_ratio = min(1.0, coverage_months / horizon_months) if horizon_months > 0 else 1.0
    if not missing.empty and "missing_input" in missing.columns:
        missing_inputs = set(missing["missing_input"].dropna().astype(str).tolist())
        adjust_irr = bool(
            missing_inputs.intersection({"24-month operational pipeline data", "official_capex_ai"})
        )
    rows = []
    for _, row in df.iterrows():
        irr_monthly = row.get("irr_monthly")
        irr_annual = row.get("irr_annual_effective")
        npv_vnd = float(row["npv_vnd"]) if pd.notna(row["npv_vnd"]) else None
        note = ""
        if row["scenario"] == "Without AI":
            irr_monthly = without_ai_irr_monthly
            irr_annual = (1 + without_ai_irr_monthly) ** 12 - 1
            npv_vnd = without_ai_npv_vnd
            note = (
                f"Reference provided by team: IRR {WITHOUT_AI_IRR_QUARTERLY * 100:.2f}%/quarter, "
                f"NPV {WITHOUT_AI_NPV_USD:,.2f} USD, FX {fx:,.0f}."
            )
        if row["scenario"] == "With AI" and adjust_irr and coverage_months > 0 and pd.notna(
            row.get("capex_ai_included_vnd")
        ):
            capex = float(row["capex_ai_included_vnd"])
            monthly_net = monthly["net_cost_delta_excl_revenue"].astype(float).tolist()
            adjusted_cashflows = [-capex] + [value * coverage_ratio for value in monthly_net]
            solved = _solve_irr(adjusted_cashflows)
            if solved is not None:
                irr_monthly = solved
                irr_annual = (1 + solved) ** 12 - 1
        rows.append(
            {
                "scenario": row["scenario"],
                "npv_vnd": npv_vnd,
                "irr_monthly": float(irr_monthly) if pd.notna(irr_monthly) else None,
                "irr_annual": float(irr_annual) if pd.notna(irr_annual) else None,
                "note": note,
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


def task_flow_payload(as_of: pd.Timestamp | None = None) -> dict[str, object]:
    equipment_rows = equipment_operational_rows(as_of=as_of)
    comparison = scenario_comparison()
    return {
        "prMatrix": pr_matrix(as_of=as_of),
        "dailyAlarms": daily_pr_alarms(as_of=as_of),
        "monthlyAbnormal": monthly_abnormal_counts(),
        "equipmentRows": equipment_rows,
        "operationalTotals": operational_totals(equipment_rows),
        "scenarioComparison": comparison,
        "scenarioVariance": scenario_variance(comparison),
        "inventoryDecision": inventory_decision(),
        "summary": task_flow_summary(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the dashboard data snapshot (data.js).")
    parser.add_argument(
        "--as-of",
        default=DEFAULT_AS_OF,
        help=(
            "Inclusive end date (YYYY-MM-DD) for recent-window views. "
            "Use 'latest' to always anchor on the latest available date. "
            f"Default: {DEFAULT_AS_OF}."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    as_of_raw = None if str(args.as_of).lower() == "latest" else args.as_of
    as_of = resolve_as_of(as_of_raw)

    payload = {
        "asOf": as_of.strftime("%Y-%m-%d") if as_of is not None else None,
        "risk": risk_summary(as_of=as_of),
        "forecast": records(PROJECT_ROOT / "input_data" / "forecast" / "forecast_table.csv", limit=30),
        "weeklyCandidates": latest_week_candidates(as_of=as_of),
        "inventory": inventory_summary(),
        "financial": financial_summary(),
        "timeline": records(PROJECT_ROOT / "output_data" / "financial" / "npv_timeline.csv"),
        "operationalSavings": records(
            PROJECT_ROOT / "output_data" / "financial" / "monthly_operational_savings.csv"
        ),
        "taskFlow": task_flow_payload(as_of=as_of),
    }
    DATA_PATH.write_text(
        "window.DASHBOARD_DATA = " + json.dumps(payload, indent=2) + ";\n",
        encoding="utf-8",
    )
    print(f"Dashboard data -> {DATA_PATH}")
    print(f"As-of date -> {as_of.strftime('%Y-%m-%d') if as_of is not None else 'latest'}")


if __name__ == "__main__":
    main()
