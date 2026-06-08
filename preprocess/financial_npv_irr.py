"""
Calculate AI vs non-AI NPV/IRR from the team's financial workbook.

This script intentionally does not create financial assumptions. It reads the
workbook fields, validates the required 24-month inputs, and reports missing
items when the workbook is not yet complete enough for final NPV/IRR.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import load_workbook


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKBOOK = PROJECT_ROOT / "input_data" / "Financial_Assumptions.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output_data" / "financial"
DEFAULT_CANDIDATES = PROJECT_ROOT / "input_data" / "risk" / "replacement_candidates.csv"
DEFAULT_DAILY_RISK_DIR = PROJECT_ROOT / "input_data" / "risk" / "daily"
DEFAULT_TRANSFORMED = PROJECT_ROOT / "input_data" / "base" / "transformed.csv"
DEFAULT_INVENTORY_SUMMARY = PROJECT_ROOT / "input_data" / "inventory" / "spare_inventory_cost_summary.csv"


@dataclass(frozen=True)
class Assumption:
    parameter: str
    value: object
    unit: str
    assumption_type: str
    source: str
    note: str


def read_assumptions(workbook_path: Path) -> dict[str, Assumption]:
    wb = load_workbook(workbook_path, data_only=True)
    ws = wb["01_Assumptions"]
    assumptions: dict[str, Assumption] = {}
    for row in ws.iter_rows(min_row=4, values_only=True):
        parameter = row[0]
        if not parameter:
            continue
        parameter_text = str(parameter).strip()
        if parameter_text.lower() == "parameter":
            continue
        if parameter_text.lower().startswith("operational assumptions retained"):
            continue
        assumptions[parameter_text] = Assumption(
            parameter=parameter_text,
            value=row[1],
            unit="" if row[2] is None else str(row[2]),
            assumption_type="" if row[3] is None else str(row[3]),
            source="" if row[4] is None else str(row[4]),
            note="" if row[5] is None else str(row[5]),
        )
    return assumptions


def read_assumption_formulas(workbook_path: Path) -> dict[str, str]:
    wb = load_workbook(workbook_path, data_only=False)
    ws = wb["01_Assumptions"]
    formulas: dict[str, str] = {}
    for row in ws.iter_rows(min_row=4, values_only=True):
        parameter = row[0]
        if not parameter:
            continue
        parameter_text = str(parameter).strip()
        if parameter_text.lower() == "parameter":
            continue
        if parameter_text.lower().startswith("operational assumptions retained"):
            continue
        formulas[parameter_text] = "" if row[5] is None else str(row[5])
    return formulas


def get_number(assumptions: dict[str, Assumption], key: str) -> float | None:
    item = assumptions.get(key)
    if item is None or item.value in (None, ""):
        return None
    if isinstance(item.value, str):
        value = item.value.strip().replace(",", "")
        is_percent = value.endswith("%")
        if is_percent:
            value = value[:-1]
        try:
            number = float(value)
        except ValueError:
            return None
        return number / 100 if is_percent else number
    try:
        return float(item.value)
    except (TypeError, ValueError):
        return None


def get_number_any(assumptions: dict[str, Assumption], keys: list[str]) -> float | None:
    for key in keys:
        value = get_number(assumptions, key)
        if value is not None:
            return value
    return None


def extract_fm_discount_rate(workbook_path: Path) -> float | None:
    wb = load_workbook(workbook_path, data_only=True)
    ws = wb["06_FM_Output_Check"]
    for row in ws.iter_rows(values_only=True):
        metric = row[0]
        if metric == "NPV":
            formula_text = "" if row[5] is None else str(row[5])
            match = re.search(r"XNPV\(([-+]?\d+(?:\.\d+)?)%", formula_text, flags=re.IGNORECASE)
            if match:
                return float(match.group(1)) / 100
    return None


def read_candidate_events(candidates_path: Path, daily_risk_dir: Path) -> pd.DataFrame:
    daily_paths = sorted(daily_risk_dir.glob("*/replacement_candidates.csv"))
    frames = []
    for path in daily_paths:
        df = pd.read_csv(path)
        if not df.empty:
            frames.append(df)
    if not frames and candidates_path.exists():
        frames.append(pd.read_csv(candidates_path))
    if not frames:
        return pd.DataFrame()
    candidates = pd.concat(frames, ignore_index=True)
    candidates = candidates[candidates["replacement_candidate"].astype(bool)].copy()
    candidates["risk_date"] = pd.to_datetime(candidates["risk_date"])
    return candidates.drop_duplicates(["risk_date", "zone", "device_name"])


def load_candidate_capacity(candidates: pd.DataFrame, transformed_path: Path) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    capacity = (
        pd.read_csv(transformed_path)
        [["zone", "device_name", "installed_capacity_kwp"]]
        .dropna()
        .drop_duplicates(["zone", "device_name"])
        .rename(columns={"installed_capacity_kwp": "lost_capacity_kw"})
    )
    df = candidates.merge(capacity, on=["zone", "device_name"], how="left")
    missing = df["lost_capacity_kw"].isna().sum()
    if missing:
        raise ValueError(f"Missing installed capacity for {missing} replacement candidate rows.")
    return df


def resolve_optimal_stock(inventory_summary_path: Path, assumptions: dict[str, Assumption]) -> int:
    if inventory_summary_path.exists():
        curve = pd.read_csv(inventory_summary_path)
        if not curve.empty and {"stock_n", "total_cost_vnd"}.issubset(curve.columns):
            return int(curve.loc[curve["total_cost_vnd"].idxmin(), "stock_n"])
    safety_stock = get_number(assumptions, "safety_stock_units")
    return int(safety_stock or 0)


def build_monthly_deltas_from_pipeline(
    assumptions: dict[str, Assumption],
    candidates_path: Path,
    daily_risk_dir: Path,
    transformed_path: Path,
    inventory_summary_path: Path,
    horizon_months: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    candidates = read_candidate_events(candidates_path, daily_risk_dir)
    candidates = load_candidate_capacity(candidates, transformed_path)

    if candidates.empty:
        return pd.DataFrame(), {"source": "input_data risk pipeline", "coverage_months": 0}

    lead_time_days = get_number(assumptions, "china_lead_time_days") or 0.0
    installation_time_hours = get_number(assumptions, "installation_time_hours") or 0.0
    generation_hours_per_day = get_number(assumptions, "generation_hours_per_day") or 0.0
    price_vnd_per_kwh = get_number_any(
        assumptions,
        [
            "electricity_price_vnd_per_kwh",
            "quick_blended_price_vnd_per_kwh",
            "quick_ppa_price_vnd_per_kwh",
        ],
    ) or 0.0
    current_stock = get_number(assumptions, "current_spare_stock_units") or 0.0
    inverter_unit_cost = get_number(assumptions, "inverter_unit_cost") or 0.0
    holding_rate_annual = get_number(assumptions, "holding_cost_rate_annual") or 0.0
    ai_recurring_monthly = get_number(assumptions, "ai_recurring_cost_monthly") or 0.0
    ai_stock = resolve_optimal_stock(inventory_summary_path, assumptions)

    baseline_downtime_days = lead_time_days + installation_time_hours / 24
    ai_uncovered_downtime_days = baseline_downtime_days
    ai_covered_downtime_days = installation_time_hours / 24
    current_holding_monthly = current_stock * inverter_unit_cost * holding_rate_annual / 12
    ai_holding_monthly = ai_stock * inverter_unit_cost * holding_rate_annual / 12

    candidates["month_period"] = candidates["risk_date"].dt.to_period("M")
    first_period = candidates["month_period"].min()
    rows = []
    for period, month_candidates in candidates.groupby("month_period"):
        month_index = (period.year - first_period.year) * 12 + period.month - first_period.month + 1
        if month_index > horizon_months:
            continue

        baseline_loss_kwh = 0.0
        ai_loss_kwh = 0.0
        downtime_days_avoided = 0.0

        for _, day_candidates in month_candidates.groupby("risk_date"):
            ordered = day_candidates.sort_values(
                ["lost_capacity_kw", "risk_score"],
                ascending=[False, False],
            ).reset_index(drop=True)
            baseline_loss_kwh += (
                ordered["lost_capacity_kw"].sum() * generation_hours_per_day * baseline_downtime_days
            )

            covered = ordered.iloc[:ai_stock]
            uncovered = ordered.iloc[ai_stock:]
            ai_loss_kwh += (
                covered["lost_capacity_kw"].sum() * generation_hours_per_day * ai_covered_downtime_days
            )
            ai_loss_kwh += (
                uncovered["lost_capacity_kw"].sum() * generation_hours_per_day * ai_uncovered_downtime_days
            )
            downtime_days_avoided += len(covered) * max(0.0, lead_time_days)

        baseline_loss_vnd = baseline_loss_kwh * price_vnd_per_kwh
        ai_loss_vnd = ai_loss_kwh * price_vnd_per_kwh
        avoided_loss_vnd = baseline_loss_vnd - ai_loss_vnd
        holding_saved = current_holding_monthly - ai_holding_monthly
        net_cashflow = avoided_loss_vnd + holding_saved - ai_recurring_monthly

        rows.append(
            {
                "Month": month_index,
                "calendar_month": str(period),
                "replacement_candidate_events": len(month_candidates),
                "ai_stock_level_units": ai_stock,
                "downtime_days_avoided": downtime_days_avoided,
                "generation_saved_mwh": (baseline_loss_kwh - ai_loss_kwh) / 1000,
                "baseline_generation_loss_mwh": baseline_loss_kwh / 1000,
                "ai_generation_loss_mwh": ai_loss_kwh / 1000,
                "baseline_generation_loss_vnd": baseline_loss_vnd,
                "ai_generation_loss_vnd": ai_loss_vnd,
                "current_holding_cost_monthly": current_holding_monthly,
                "ai_holding_cost_monthly": ai_holding_monthly,
                "holding_cost_saved_monthly": holding_saved,
                "ai_recurring_cost_monthly": ai_recurring_monthly,
                "spare_purchase_cost_vnd": 0.0,
                "net_cost_delta_excl_revenue": net_cashflow,
            }
        )

    deltas = pd.DataFrame(rows).sort_values("Month")
    metadata = {
        "source": "input_data/risk/daily + input_data/base/transformed.csv",
        "coverage_months": len(deltas),
        "first_candidate_date": candidates["risk_date"].min().date().isoformat(),
        "last_candidate_date": candidates["risk_date"].max().date().isoformat(),
        "ai_stock_level_units": ai_stock,
    }
    return deltas, metadata


def irr(cashflows: list[float]) -> float | None:
    if not any(cf > 0 for cf in cashflows) or not any(cf < 0 for cf in cashflows):
        return None

    def npv_at(rate: float) -> float:
        return sum(cf / ((1 + rate) ** i) for i, cf in enumerate(cashflows))

    low, high = -0.999999, 10.0
    low_value, high_value = npv_at(low), npv_at(high)
    if low_value * high_value > 0:
        return None
    for _ in range(200):
        mid = (low + high) / 2
        mid_value = npv_at(mid)
        if abs(mid_value) < 1e-7:
            return mid
        if low_value * mid_value <= 0:
            high = mid
            high_value = mid_value
        else:
            low = mid
            low_value = mid_value
    return (low + high) / 2


def build_financial_outputs(
    assumptions: dict[str, Assumption],
    formulas: dict[str, str],
    deltas: pd.DataFrame,
    discount_rate_annual: float | None,
    horizon_months: int,
    operational_metadata: dict[str, object],
    capex_ai_override_vnd: float | None,
    capex_ai_source: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly_rate = None if discount_rate_annual is None else (1 + discount_rate_annual) ** (1 / 12) - 1
    capex_ai_from_workbook = get_number(assumptions, "capex_ai")
    capex_ai = capex_ai_override_vnd if capex_ai_override_vnd is not None else capex_ai_from_workbook

    timeline_rows = []
    ai_cashflows = [-(capex_ai or 0.0)]
    no_ai_cashflows = [0.0]

    for _, row in deltas.iterrows():
        month = int(row["Month"])
        net_delta = float(row.get("net_cost_delta_excl_revenue") or 0.0)
        ai_cashflows.append(net_delta)
        no_ai_cashflows.append(0.0)
        discount_factor = math.nan if monthly_rate is None else (1 + monthly_rate) ** month
        timeline_rows.append(
            {
                "month": month,
                "calendar_month": row.get("calendar_month", ""),
                "replacement_candidate_events": row.get("replacement_candidate_events", 0),
                "ai_stock_level_units": row.get("ai_stock_level_units", None),
                "baseline_generation_loss_mwh": row.get("baseline_generation_loss_mwh", 0.0),
                "ai_generation_loss_mwh": row.get("ai_generation_loss_mwh", 0.0),
                "generation_saved_mwh": row.get("generation_saved_mwh", 0.0),
                "baseline_generation_loss_vnd": row.get("baseline_generation_loss_vnd", 0.0),
                "ai_generation_loss_vnd": row.get("ai_generation_loss_vnd", 0.0),
                "holding_cost_saved_monthly": row.get("holding_cost_saved_monthly", 0.0),
                "ai_recurring_cost_monthly": row.get("ai_recurring_cost_monthly", 0.0),
                "cashflow_no_ai_vnd": 0.0,
                "cashflow_ai_vnd": net_delta,
                "incremental_cashflow_vnd": net_delta,
                "discounted_incremental_cashflow_vnd": (
                    math.nan if monthly_rate is None else net_delta / discount_factor
                ),
            }
        )

    timeline = pd.DataFrame(timeline_rows)
    if timeline.empty:
        cumulative_npv = pd.Series(dtype=float)
    elif monthly_rate is None:
        cumulative_npv = pd.Series([math.nan] * len(timeline), index=timeline.index)
    else:
        initial = -(capex_ai or 0.0)
        cumulative_npv = initial + timeline["discounted_incremental_cashflow_vnd"].cumsum()
    timeline["cumulative_incremental_npv_vnd"] = cumulative_npv

    ai_npv = (
        math.nan
        if monthly_rate is None
        else sum(cf / ((1 + monthly_rate) ** i) for i, cf in enumerate(ai_cashflows))
    )
    pv_operating_benefit = math.nan if monthly_rate is None else ai_npv - ai_cashflows[0]

    comparison = pd.DataFrame(
        [
            {
                "scenario": "Without AI",
                "npv_vnd": 0.0 if monthly_rate is not None else math.nan,
                "irr_monthly": None,
                "irr_annual_effective": None,
                "capex_ai_source": "",
                "capex_ai_included_vnd": 0.0,
                "max_capex_ai_for_npv_zero_vnd": None,
                "cashflow_month_0_vnd": 0.0,
                "cashflow_month_1_to_horizon_total_vnd": sum(no_ai_cashflows[1:]),
            },
            {
                "scenario": "With AI",
                "npv_vnd": ai_npv,
                "irr_monthly": irr(ai_cashflows),
                "irr_annual_effective": None,
                "capex_ai_source": capex_ai_source if capex_ai is not None else "missing",
                "capex_ai_included_vnd": capex_ai,
                "max_capex_ai_for_npv_zero_vnd": pv_operating_benefit,
                "cashflow_month_0_vnd": ai_cashflows[0],
                "cashflow_month_1_to_horizon_total_vnd": sum(ai_cashflows[1:]),
            },
        ]
    )
    ai_irr = comparison.loc[comparison["scenario"] == "With AI", "irr_monthly"].iloc[0]
    if ai_irr is not None:
        comparison.loc[comparison["scenario"] == "With AI", "irr_annual_effective"] = (1 + ai_irr) ** 12 - 1

    assumption_rows = []
    for item in assumptions.values():
        assumption_rows.append(
            {
                "parameter": item.parameter,
                "value": item.value,
                "unit": item.unit,
                "type": item.assumption_type,
                "source_owner": item.source,
                "formula_or_note": formulas.get(item.parameter, item.note),
            }
        )
    if discount_rate_annual is not None and "discount_rate_annual" not in assumptions:
        assumption_rows.append(
            {
                "parameter": "discount_rate_annual",
                "value": discount_rate_annual,
                "unit": "rate",
                "type": "Fixed from FM formula",
                "source_owner": "06_FM_Output_Check",
                "formula_or_note": "Parsed from NPV formula =XNPV(7%,...). Confirm with Hà if this is fixed.",
            }
        )

    missing = []
    coverage_months = int(operational_metadata.get("coverage_months") or 0)
    if coverage_months < horizon_months:
        missing.append(
            {
                "missing_input": "24-month operational pipeline data",
                "reason": (
                    f"Task horizon is {horizon_months} months, but input_data currently provides "
                    f"{coverage_months} month(s) of replacement-candidate savings "
                    f"({operational_metadata.get('first_candidate_date')} to "
                    f"{operational_metadata.get('last_candidate_date')})."
                ),
            }
        )
    if discount_rate_annual is None:
        missing.append({"missing_input": "discount_rate_annual", "reason": "Needed for NPV discounting."})
    if capex_ai is None:
        missing.append({"missing_input": "capex_ai", "reason": "Needed for month-0 AI investment cashflow."})
    elif capex_ai_from_workbook is None:
        missing.append(
            {
                "missing_input": "official_capex_ai",
                "reason": f"Current run uses {capex_ai_source}. Hà/team should confirm whether this is the official CAPEX AI.",
            }
        )
    if "baseline_downtime_cost" not in assumptions:
        missing.append(
            {
                "missing_input": "baseline_downtime_cost",
                "reason": "Requested by the task; needed if the FM team wants downtime savings valued directly instead of through generation_saved_mwh.",
            }
        )
    if deltas.empty or deltas.get("generation_saved_mwh", pd.Series(dtype=float)).fillna(0).sum() == 0:
        missing.append(
            {
                "missing_input": "generation_saved_mwh from input_data",
                "reason": "No positive generation savings could be derived from replacement candidates and capacity data.",
            }
        )

    return comparison, timeline, pd.DataFrame(assumption_rows), pd.DataFrame(missing)


def write_chart(timeline: pd.DataFrame, output_path: Path) -> None:
    if timeline.empty or timeline["cumulative_incremental_npv_vnd"].isna().all():
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(
        timeline["month"],
        timeline["cumulative_incremental_npv_vnd"],
        marker="o",
        linewidth=2,
        color="#1f77b4",
    )
    ax.axhline(0, color="#666666", linewidth=1, linestyle="--")
    ax.set_title("Cumulative Incremental NPV: With AI vs Without AI")
    ax.set_xlabel("Month")
    ax.set_ylabel("NPV (VND)")
    ax.ticklabel_format(style="plain", axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_decision_summary(
    comparison: pd.DataFrame,
    timeline: pd.DataFrame,
    missing: pd.DataFrame,
    output_path: Path,
) -> None:
    ai = comparison[comparison["scenario"] == "With AI"].iloc[0]
    total_saved_mwh = timeline["generation_saved_mwh"].sum() if "generation_saved_mwh" in timeline else 0.0
    total_candidates = timeline["replacement_candidate_events"].sum() if "replacement_candidate_events" in timeline else 0
    max_capex = ai.get("max_capex_ai_for_npv_zero_vnd")
    npv = ai.get("npv_vnd")
    capex = ai.get("capex_ai_included_vnd")
    capex_source = ai.get("capex_ai_source")
    irr_monthly = ai.get("irr_monthly")
    irr_annual = ai.get("irr_annual_effective")

    lines = [
        "# AI Financial Effectiveness Summary",
        "",
        "This report uses `Financial_Assumptions.xlsx` for assumptions only. Operational savings are derived from `input_data` pipeline outputs.",
        "",
        "## Current Data-Driven Result",
        "",
        f"- AI NPV after included CAPEX: {npv:,.0f} VND",
        f"- Included AI CAPEX: {capex:,.0f} VND ({capex_source})",
        f"- Break-even AI CAPEX for NPV >= 0: {max_capex:,.0f} VND",
        f"- Monthly IRR: {irr_monthly:.2%}" if pd.notna(irr_monthly) else "- Monthly IRR: unavailable",
        f"- Annual effective IRR: {irr_annual:.2%}" if pd.notna(irr_annual) else "- Annual effective IRR: unavailable",
        f"- Generation saved from detected replacement candidates: {total_saved_mwh:,.2f} MWh",
        f"- Replacement candidate events covered in current data: {int(total_candidates):,}",
        "",
        "## Interpretation",
        "",
        "AI is financially effective under the current pipeline data if confirmed AI CAPEX is below the break-even CAPEX above. If CAPEX is supplied as a scenario override, Hà/team should still confirm whether it is the official CAPEX assumption.",
        "",
        "## Inputs Still Needed For Final 24-Month Proof",
        "",
    ]
    if missing.empty:
        lines.append("- None")
    else:
        for _, row in missing.iterrows():
            lines.append(f"- {row['missing_input']}: {row['reason']}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate 24-month NPV and IRR from Financial_Assumptions.xlsx.")
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--horizon-months", type=int, default=24)
    parser.add_argument("--capex-ai-vnd", type=float, default=None)
    parser.add_argument("--capex-ai-usd", type=float, default=None)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--daily-risk-dir", type=Path, default=DEFAULT_DAILY_RISK_DIR)
    parser.add_argument("--transformed", type=Path, default=DEFAULT_TRANSFORMED)
    parser.add_argument("--inventory-summary", type=Path, default=DEFAULT_INVENTORY_SUMMARY)
    return parser.parse_args()


def resolve_capex_override(args: argparse.Namespace, assumptions: dict[str, Assumption]) -> tuple[float | None, str]:
    if args.capex_ai_vnd is not None and args.capex_ai_usd is not None:
        raise ValueError("Use either --capex-ai-vnd or --capex-ai-usd, not both.")
    if args.capex_ai_vnd is not None:
        return args.capex_ai_vnd, "cli_override_vnd"
    if args.capex_ai_usd is not None:
        fx = get_number_any(assumptions, ["fx_vnd_usd", "fx_usd_vnd"])
        if fx is None:
            raise ValueError("--capex-ai-usd requires fx_vnd_usd or fx_usd_vnd in the assumptions workbook.")
        return args.capex_ai_usd * fx, f"cli_override_usd_{args.capex_ai_usd:g}_fx_{fx:g}"
    if get_number(assumptions, "capex_ai") is not None:
        return None, "workbook_assumption"
    return None, "missing"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    assumptions = read_assumptions(args.workbook)
    formulas = read_assumption_formulas(args.workbook)
    capex_ai_override_vnd, capex_ai_source = resolve_capex_override(args, assumptions)
    discount_rate = get_number(assumptions, "discount_rate_annual") or extract_fm_discount_rate(args.workbook)
    deltas, operational_metadata = build_monthly_deltas_from_pipeline(
        assumptions=assumptions,
        candidates_path=args.candidates,
        daily_risk_dir=args.daily_risk_dir,
        transformed_path=args.transformed,
        inventory_summary_path=args.inventory_summary,
        horizon_months=args.horizon_months,
    )

    comparison, timeline, assumption_notes, missing = build_financial_outputs(
        assumptions=assumptions,
        formulas=formulas,
        deltas=deltas,
        discount_rate_annual=discount_rate,
        horizon_months=args.horizon_months,
        operational_metadata=operational_metadata,
        capex_ai_override_vnd=capex_ai_override_vnd,
        capex_ai_source=capex_ai_source,
    )

    comparison_path = args.output_dir / "npv_irr_comparison.csv"
    timeline_path = args.output_dir / "npv_timeline.csv"
    assumptions_path = args.output_dir / "assumptions_variable_fixed.csv"
    missing_path = args.output_dir / "missing_financial_inputs.csv"
    operational_path = args.output_dir / "monthly_operational_savings.csv"
    chart_path = args.output_dir / "npv_timeline.png"
    summary_path = args.output_dir / "ai_financial_effectiveness_summary.md"

    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    timeline.to_csv(timeline_path, index=False, encoding="utf-8-sig")
    deltas.to_csv(operational_path, index=False, encoding="utf-8-sig")
    assumption_notes.to_csv(assumptions_path, index=False, encoding="utf-8-sig")
    missing.to_csv(missing_path, index=False, encoding="utf-8-sig")
    write_chart(timeline, chart_path)
    write_decision_summary(comparison, timeline, missing, summary_path)

    print(f"Comparison -> {comparison_path}")
    print(f"Timeline -> {timeline_path}")
    print(f"Operational savings from input_data -> {operational_path}")
    print(f"Assumption fixed/variable note -> {assumptions_path}")
    print(f"Missing-input report -> {missing_path}")
    print(f"Decision summary -> {summary_path}")
    if chart_path.exists():
        print(f"NPV chart -> {chart_path}")
    if not missing.empty:
        print("WARNING: Final 24-month NPV/IRR is not complete until missing inputs are filled.")
        for item in missing["missing_input"].tolist():
            print(f"- {item}")


if __name__ == "__main__":
    main()
