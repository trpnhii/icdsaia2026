"""
Spare-inverter inventory cost model for the 2-year risk pipeline.

The model compares:
  1. Expected generation loss when replacement candidates are not pre-stocked.
  2. Holding cost for keeping N spare inverters in inventory.

Formula:
  generation_loss_vnd = lead_time_days * generation_hours_per_day
                        * lost_capacity_kw * electricity_price_vnd_per_kwh

Set generation_hours_per_day=1 to match a simple day*kW*price formula, or set
it to the expected equivalent generation hours per day for a solar-specific
energy-loss estimate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_INPUT_DIR = PROJECT_ROOT / "input_data" / "base"
RISK_INPUT_DIR = PROJECT_ROOT / "input_data" / "risk"
INVENTORY_OUTPUT_DIR = PROJECT_ROOT / "input_data" / "inventory"
CONFIG_PATH = PROJECT_ROOT / "config" / "spare_inventory.json"

CANDIDATES_PATH = RISK_INPUT_DIR / "replacement_candidates.csv"
TRANSFORMED_PATH = BASE_INPUT_DIR / "transformed.csv"
SUMMARY_PATH = INVENTORY_OUTPUT_DIR / "spare_inventory_cost_summary.csv"
RECOMMENDATION_PATH = INVENTORY_OUTPUT_DIR / "spare_inventory_recommendation.md"
COST_CURVE_PATH = PROJECT_ROOT / "output_data" / "figures" / "inventory" / "spare_inventory_cost_curve.png"


def load_candidate_capacity() -> pd.DataFrame:
    candidates = pd.read_csv(CANDIDATES_PATH, parse_dates=["risk_date"])
    capacity = (
        pd.read_csv(TRANSFORMED_PATH)
        [["zone", "device_name", "installed_capacity_kwp"]]
        .dropna()
        .drop_duplicates(["zone", "device_name"])
        .rename(columns={"installed_capacity_kwp": "lost_capacity_kw"})
    )
    df = candidates.merge(capacity, on=["zone", "device_name"], how="left")
    missing = df["lost_capacity_kw"].isna().sum()
    if missing:
        raise ValueError(f"Missing capacity for {missing} candidate rows.")
    return df.sort_values(["risk_date", "risk_score"], ascending=[True, False])


def shortage_loss_for_day(
    day_candidates: pd.DataFrame,
    stock_n: int,
    lead_time_days: float,
    electricity_price_vnd_per_kwh: float,
    generation_hours_per_day: float,
) -> float:
    ordered = day_candidates.sort_values(
        ["lost_capacity_kw", "risk_score"],
        ascending=[False, False],
    )
    unstocked = ordered.iloc[stock_n:]
    lost_capacity_kw = float(unstocked["lost_capacity_kw"].sum())
    return (
        lead_time_days
        * generation_hours_per_day
        * lost_capacity_kw
        * electricity_price_vnd_per_kwh
    )


def build_cost_curve(
    candidates: pd.DataFrame,
    lead_time_days: float,
    electricity_price_vnd_per_kwh: float,
    holding_cost_vnd_per_unit: float,
    generation_hours_per_day: float,
    max_stock: int | None = None,
) -> pd.DataFrame:
    daily_counts = candidates.groupby("risk_date").size()
    if max_stock is None:
        max_stock = int(daily_counts.max()) if not daily_counts.empty else 0

    rows = []
    for stock_n in range(max_stock + 1):
        shortage_loss = 0.0
        for _, day_candidates in candidates.groupby("risk_date"):
            shortage_loss += shortage_loss_for_day(
                day_candidates,
                stock_n,
                lead_time_days,
                electricity_price_vnd_per_kwh,
                generation_hours_per_day,
            )
        holding_cost = stock_n * holding_cost_vnd_per_unit
        rows.append(
            {
                "stock_n": stock_n,
                "expected_generation_loss_vnd": round(shortage_loss, 0),
                "holding_cost_vnd": round(holding_cost, 0),
                "total_cost_vnd": round(shortage_loss + holding_cost, 0),
            }
        )
    curve = pd.DataFrame(rows)
    curve["marginal_loss_reduction_vnd"] = (
        curve["expected_generation_loss_vnd"].shift(1) - curve["expected_generation_loss_vnd"]
    ).fillna(0)
    curve["net_saving_vs_previous_n_vnd"] = curve["marginal_loss_reduction_vnd"] - holding_cost_vnd_per_unit
    return curve


def find_breakeven(curve: pd.DataFrame, holding_cost_vnd_per_unit: float) -> pd.DataFrame:
    breakeven = curve[curve["marginal_loss_reduction_vnd"] >= holding_cost_vnd_per_unit].copy()
    return breakeven[["stock_n", "marginal_loss_reduction_vnd", "net_saving_vs_previous_n_vnd"]]


def plot_cost_curve(curve: pd.DataFrame, optimal_n: int) -> None:
    COST_CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(curve["stock_n"], curve["expected_generation_loss_vnd"], marker="o", label="Expected generation loss")
    ax.plot(curve["stock_n"], curve["holding_cost_vnd"], marker="s", label="Holding cost")
    ax.plot(curve["stock_n"], curve["total_cost_vnd"], marker="^", linewidth=2.2, label="Total cost")
    ax.axvline(optimal_n, color="#d73027", linestyle="--", linewidth=1.3, label=f"Optimal N = {optimal_n}")
    ax.set_title("Spare Inverter Inventory Cost Curve", fontweight="bold")
    ax.set_xlabel("Pre-stocked spare inverters (N)")
    ax.set_ylabel("Cost (VND)")
    ax.ticklabel_format(style="plain", axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(COST_CURVE_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_recommendation(
    curve: pd.DataFrame,
    breakeven: pd.DataFrame,
    optimal: pd.Series,
    candidates: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    max_daily_candidates = int(candidates.groupby("risk_date").size().max()) if not candidates.empty else 0
    unique_candidate_devices = candidates[["zone", "device_name"]].drop_duplicates().shape[0]

    if breakeven.empty:
        breakeven_text = "No positive breakeven stock level was found because each additional spare costs more than the avoided generation-loss estimate."
    else:
        max_breakeven_n = int(breakeven["stock_n"].max())
        breakeven_text = (
            f"Pre-stocking remains economically beneficial through N = {max_breakeven_n}, "
            "where marginal avoided generation loss is still greater than or equal to marginal holding cost."
        )

    report = f"""# Spare Inventory Cost Recommendation

## Inputs

- Electricity price: {args.electricity_price_vnd_per_kwh:,.0f} VND/kWh
- China lead time: {args.lead_time_days:g} days
- Generation hours per day: {args.generation_hours_per_day:g}
- Inverter unit cost: {args.inverter_unit_cost_vnd:,.0f} VND
- WACC: {args.wacc_annual_percent:g}%
- Holding cost formula: {args.holding_cost_formula}
- Holding cost (10 spares / year): {args.reference_prestock_holding_cost_annual_vnd:,.0f} VND
- Holding cost per spare / year: {args.holding_cost_annual_vnd_per_unit:,.0f} VND
- Holding cost per spare over analysis period: {args.holding_cost_vnd_per_unit:,.0f} VND
- Config file: `{CONFIG_PATH.relative_to(PROJECT_ROOT)}`
- Candidate-days: {len(candidates):,}
- Unique candidate devices: {unique_candidate_devices:,}
- Maximum daily simultaneous candidates: {max_daily_candidates:,}

## Optimal Stock Level

- Optimal N: {int(optimal["stock_n"])}
- Expected generation loss at optimal N: {optimal["expected_generation_loss_vnd"]:,.0f} VND
- Holding cost at optimal N: {optimal["holding_cost_vnd"]:,.0f} VND
- Total cost at optimal N: {optimal["total_cost_vnd"]:,.0f} VND

## Breakeven Point

{breakeven_text}

## Interpretation

The optimal N minimizes:

```text
total_cost = expected_generation_loss + holding_cost
```

Expected generation loss is calculated from unstocked replacement candidates:

```text
generation_loss_vnd = lead_time_days * generation_hours_per_day
                    * lost_capacity_kw * electricity_price_vnd_per_kwh
```

If generation_hours_per_day is set to 1, the model follows the simplified formula requested. For solar production, generation_hours_per_day should ideally be set to the site's expected equivalent full-generation hours per day.

## Outputs

- Cost summary: `{SUMMARY_PATH.relative_to(PROJECT_ROOT)}`
- Cost curve: `{COST_CURVE_PATH.relative_to(PROJECT_ROOT)}`
"""
    RECOMMENDATION_PATH.write_text(report, encoding="utf-8")


def load_config(path: Path) -> dict[str, float | int | None]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return {
        "electricity_price_vnd_per_kwh": config.get("electricity_price_vnd_per_kwh"),
        "lead_time_days": config.get("lead_time_days"),
        "holding_cost_vnd_per_unit": config.get("holding_cost_vnd_per_unit"),
        "holding_cost_annual_vnd_per_unit": config.get("holding_cost_annual_vnd_per_unit"),
        "holding_cost_formula": config.get("holding_cost_formula", ""),
        "inverter_unit_cost_vnd": config.get("inverter_unit_cost_vnd"),
        "wacc_annual_percent": config.get("wacc_annual_percent"),
        "reference_prestock_holding_cost_annual_vnd": config.get(
            "reference_prestock_holding_cost_annual_vnd"
        ),
        "generation_hours_per_day": config.get("generation_hours_per_day", 1.0),
        "max_stock": config.get("max_stock"),
    }


def resolve_settings(args: argparse.Namespace) -> argparse.Namespace:
    config = load_config(args.config)
    resolved = argparse.Namespace(
        config=args.config,
        electricity_price_vnd_per_kwh=(
            args.electricity_price_vnd_per_kwh
            if args.electricity_price_vnd_per_kwh is not None
            else config["electricity_price_vnd_per_kwh"]
        ),
        lead_time_days=args.lead_time_days if args.lead_time_days is not None else config["lead_time_days"],
        holding_cost_vnd_per_unit=(
            args.holding_cost_vnd_per_unit
            if args.holding_cost_vnd_per_unit is not None
            else config["holding_cost_vnd_per_unit"]
        ),
        generation_hours_per_day=(
            args.generation_hours_per_day
            if args.generation_hours_per_day is not None
            else config["generation_hours_per_day"]
        ),
        max_stock=args.max_stock if args.max_stock is not None else config["max_stock"],
        holding_cost_annual_vnd_per_unit=config.get("holding_cost_annual_vnd_per_unit"),
        holding_cost_formula=config.get("holding_cost_formula", ""),
        inverter_unit_cost_vnd=config.get("inverter_unit_cost_vnd"),
        wacc_annual_percent=config.get("wacc_annual_percent"),
        reference_prestock_holding_cost_annual_vnd=config.get(
            "reference_prestock_holding_cost_annual_vnd"
        ),
    )

    required = [
        "electricity_price_vnd_per_kwh",
        "lead_time_days",
        "holding_cost_vnd_per_unit",
        "generation_hours_per_day",
    ]
    missing = [name for name in required if resolved.__dict__[name] is None]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            f"Missing inventory-cost settings: {missing_text}. "
            f"Fill them in {args.config} or pass command-line overrides."
        )
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate optimal spare inverter stock level.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--electricity-price-vnd-per-kwh", type=float, default=None)
    parser.add_argument("--lead-time-days", type=float, default=None)
    parser.add_argument("--holding-cost-vnd-per-unit", type=float, default=None)
    parser.add_argument(
        "--generation-hours-per-day",
        type=float,
        default=None,
        help="Use 1.0 to match day*kW*price. Use solar equivalent hours/day for energy-based loss.",
    )
    parser.add_argument("--max-stock", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = resolve_settings(parse_args())
    INVENTORY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = load_candidate_capacity()
    curve = build_cost_curve(
        candidates,
        lead_time_days=args.lead_time_days,
        electricity_price_vnd_per_kwh=args.electricity_price_vnd_per_kwh,
        holding_cost_vnd_per_unit=args.holding_cost_vnd_per_unit,
        generation_hours_per_day=args.generation_hours_per_day,
        max_stock=args.max_stock,
    )
    optimal = curve.loc[curve["total_cost_vnd"].idxmin()]
    breakeven = find_breakeven(curve, args.holding_cost_vnd_per_unit)

    curve.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    plot_cost_curve(curve, int(optimal["stock_n"]))
    write_recommendation(curve, breakeven, optimal, candidates, args)

    print(f"Cost summary -> {SUMMARY_PATH}")
    print(f"Recommendation -> {RECOMMENDATION_PATH}")
    print(f"Cost curve -> {COST_CURVE_PATH}")
    print(f"Optimal N: {int(optimal['stock_n'])}")
    print(f"Total cost at optimal N: {optimal['total_cost_vnd']:,.0f} VND")
    if breakeven.empty:
        print("Breakeven: no positive stock level")
    else:
        print(f"Breakeven through N: {int(breakeven['stock_n'].max())}")


if __name__ == "__main__":
    main()
