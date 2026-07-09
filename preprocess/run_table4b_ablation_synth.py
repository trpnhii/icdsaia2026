"""
Table 4b: Ablation study on the synthetic-event protocol (aligned with Table 11).

Outputs:
  output_data/experimental_results/table4b_ablation_synth.csv
  output_data/experimental_results/table4b_ablation_synth.md
  output_data/experimental_results/figure4b_ablation_synth_cost.png
  output_data/experimental_results/figure4b_ablation_synth_npv.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from run_table11_extended import (
    OUTPUT_DIR,
    TRANSFORMED_PATH,
    MethodMetrics,
    build_base_panel,
    collapse_alert_episodes,
    evaluate_event_alerts,
    inject_synthetic_faults,
)


OUTPUT_CSV = OUTPUT_DIR / "table4b_ablation_synth.csv"
OUTPUT_MD = OUTPUT_DIR / "table4b_ablation_synth.md"
OUTPUT_FIG = OUTPUT_DIR / "figure4b_ablation_synth_cost.png"
OUTPUT_FIG_NPV = OUTPUT_DIR / "figure4b_ablation_synth_npv.png"

STYLE = "seaborn-v0_8-whitegrid"
STL_PERIOD = 7


def _recompute_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    g = work.groupby(["zone", "device_key"], sort=False)
    work["mean_deficit_14d"] = g["peer_deficit"].transform(lambda s: s.rolling(14, min_periods=1).mean())
    work["mean_deficit_30d"] = g["peer_deficit"].transform(lambda s: s.rolling(30, min_periods=1).mean())
    work["severe_low_14d"] = g["relative_pr"].transform(
        lambda s: (s < 0.5).astype(float).rolling(14, min_periods=1).mean()
    )
    work["severe_low_30d"] = g["relative_pr"].transform(
        lambda s: (s < 0.5).astype(float).rolling(30, min_periods=1).mean()
    )
    work["pr_slope_30d"] = g["relative_pr"].transform(
        lambda s: s.rolling(30, min_periods=5).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) >= 5 else 0.0,
            raw=False,
        )
    ).fillna(0.0)
    return work


def _apply_stl_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    daily = work.groupby("date", as_index=False).agg(mean_pr=("performance_ratio", "mean")).sort_values("date")
    if len(daily) < STL_PERIOD * 2:
        return work

    observed = daily["mean_pr"].to_numpy(dtype=float)
    stl = STL(observed, period=STL_PERIOD, robust=True).fit()
    seasonal = stl.seasonal
    baseline = np.where(observed == 0, 1.0, observed)
    seasonal_frac = seasonal / baseline
    lookup = pd.Series(seasonal_frac, index=pd.to_datetime(daily["date"]))
    scale = 1.0 + work["date"].map(lookup).fillna(0.0).clip(-0.5, 0.5)
    work["performance_ratio"] = (work["performance_ratio"] / scale).clip(lower=0, upper=2)

    # Recompute peer-relative columns after STL de-seasonalization.
    date_median = work.groupby("date")["performance_ratio"].transform("median").replace(0, np.nan)
    work["relative_pr"] = (work["performance_ratio"] / date_median).clip(0, 2)
    work["peer_deficit"] = (1 - work["relative_pr"]).clip(0, 1)
    return work


def build_alerts_variant(
    panel: pd.DataFrame,
    *,
    use_stl: bool,
    use_peer: bool,
    use_adaptive: bool,
) -> pd.DataFrame:
    work = panel.copy()
    if use_stl:
        work = _apply_stl_adjustment(work)

    if not use_peer:
        ref_pr = 0.8
        work["relative_pr"] = (work["performance_ratio"] / ref_pr).clip(0, 2)
        work["peer_deficit"] = (ref_pr - work["performance_ratio"]).clip(0, 1)

    work = _recompute_features(work)

    if not use_adaptive:
        # Fair point-in-time variant: remove rolling-window memory while
        # preserving feature scale so this is not underpowered by construction.
        work["mean_deficit_14d"] = work["peer_deficit"]
        work["mean_deficit_30d"] = work["peer_deficit"]
        work["severe_low_14d"] = (work["relative_pr"] < 0.5).astype(float)
        work["severe_low_30d"] = (work["relative_pr"] < 0.5).astype(float)
        work["pr_slope_30d"] = 0.0

    risk_score = (
        100
        * (
            0.25 * work["mean_deficit_14d"]
            + 0.20 * work["mean_deficit_30d"]
            + 0.20 * work["severe_low_14d"]
            + 0.15 * work["severe_low_30d"]
            + 0.10 * (-work["pr_slope_30d"] * 30).clip(0, 1)
            + 0.10 * work["peer_deficit"]
        )
    ).clip(0, 100)
    mask = (risk_score >= 13) & (work["relative_pr"] < 0.85)
    return work.loc[mask, ["zone", "device_key", "date"]].rename(columns={"date": "alert_date"})


def build_threshold_alerts(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.loc[panel["performance_ratio"] < 0.8, ["zone", "device_key", "date"]].rename(
        columns={"date": "alert_date"}
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Table 4b ablation on synthetic-event protocol.")
    p.add_argument("--target-events", type=int, default=300)
    p.add_argument("--lead-window-days", type=int, default=30)
    p.add_argument("--episode-gap-days", type=int, default=7)
    p.add_argument("--fp-inspection-cost-usd", type=float, default=10.0)
    p.add_argument("--fn-miss-cost-usd", type=float, default=1500.0)
    p.add_argument("--random-seed", type=int, default=42)
    return p.parse_args()


def _cost_plot(table: pd.DataFrame) -> None:
    ordered = table.sort_values("expected_maintenance_cost_usd", ascending=True).copy()
    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(ordered))
    colors = ["#1a9641" if "Full SolarGuard" in m else "#636363" for m in ordered["method"]]
    ax.bar(x, ordered["expected_maintenance_cost_usd"], color=colors, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["method"], rotation=18, ha="right")
    ax.set_ylabel("Expected Maintenance Cost (USD)")
    ax.set_title("Table 4b Ablation (Synthetic Protocol): Cost Comparison")
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _npv_plot(table: pd.DataFrame) -> None:
    ordered = table.copy()
    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(ordered))
    colors = ["#1a9641" if "Full SolarGuard" in m else "#636363" for m in ordered["model_configuration"]]
    ax.bar(x, ordered["npv_gain_usd"], color=colors, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["model_configuration"], rotation=18, ha="right")
    ax.set_ylabel("NPV Gain (USD)")
    ax.set_title("Table 4b Ablation (Synthetic Protocol): NPV Gain Comparison")
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG_NPV, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    transformed = pd.read_csv(TRANSFORMED_PATH, parse_dates=["date"])
    panel = build_base_panel(transformed)
    panel, synthetic_events = inject_synthetic_faults(panel, args.target_events, args.random_seed)

    cutoff = panel["date"].quantile(0.7)
    test = panel[panel["date"] > cutoff].copy()
    events_test = synthetic_events[synthetic_events["replacement_date"] > cutoff].copy()
    if len(events_test) == 0:
        raise RuntimeError("No synthetic events in test split.")

    variants = [
        ("Threshold-based (Baseline)", build_threshold_alerts(test)),
        ("w/o Peer-relative PR", build_alerts_variant(test, use_stl=True, use_peer=False, use_adaptive=True)),
        ("w/o STL Decomposition", build_alerts_variant(test, use_stl=False, use_peer=True, use_adaptive=True)),
        ("w/o Adaptive Window", build_alerts_variant(test, use_stl=True, use_peer=True, use_adaptive=False)),
        ("Full SolarGuard", build_alerts_variant(test, use_stl=True, use_peer=True, use_adaptive=True)),
    ]

    metrics: list[MethodMetrics] = []
    for name, alerts in variants:
        collapsed = collapse_alert_episodes(alerts, args.episode_gap_days)
        row = evaluate_event_alerts(
            name,
            collapsed,
            events_test,
            args.lead_window_days,
            args.fp_inspection_cost_usd,
            args.fn_miss_cost_usd,
        )
        metrics.append(row)

    raw = pd.DataFrame([m.__dict__ for m in metrics])

    baseline_cost = float(
        raw.loc[raw["method"] == "Threshold-based (Baseline)", "expected_maintenance_cost_usd"].iloc[0]
    )
    full_cost = float(raw.loc[raw["method"] == "Full SolarGuard", "expected_maintenance_cost_usd"].iloc[0])
    raw["cost_saving_vs_baseline_usd"] = baseline_cost - raw["expected_maintenance_cost_usd"]
    # Keep NPV Gain naming for paper table; this is a test-window proxy derived
    # from operational cost savings versus baseline.
    raw["npv_gain_usd"] = raw["cost_saving_vs_baseline_usd"]
    raw["delta_cost_vs_full_usd"] = raw["expected_maintenance_cost_usd"] - full_cost

    order = [
        "Full SolarGuard",
        "w/o STL Decomposition",
        "w/o Peer-relative PR",
        "w/o Adaptive Window",
        "Threshold-based (Baseline)",
    ]
    raw["order"] = raw["method"].map({k: i for i, k in enumerate(order)})
    table = raw.sort_values("order").reset_index(drop=True)

    export = pd.DataFrame(
        {
            "model_configuration": table["method"],
            "precision": table["precision"].round(4),
            "recall": table["recall"].round(4),
            "f1_score": table["f1"].round(4),
            "npv_gain_usd": table["npv_gain_usd"].round(2),
            "cost_saving_vs_baseline_usd": table["cost_saving_vs_baseline_usd"].round(2),
            "detected_events": table["detected_events"].astype(int),
            "total_events": table["total_events"].astype(int),
            "alert_episodes": table["alert_episodes"].astype(int),
            "avg_early_days": table["avg_early_days"],
            "med_early_days": table["med_early_days"],
            "expected_maintenance_cost_usd": table["expected_maintenance_cost_usd"].round(2),
            "delta_cost_vs_full_usd": table["delta_cost_vs_full_usd"].round(2),
        }
    )
    export.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    _cost_plot(export.rename(columns={"model_configuration": "method"}))
    _npv_plot(export)

    lines = [
        "# Table 4b — Ablation Study on Synthetic-Event Protocol",
        "",
        f"- Synthetic events injected: {len(synthetic_events)}",
        f"- Events in test window: {len(events_test)}",
        f"- Lead window: {args.lead_window_days} days",
        "",
        "| Model Configuration | Precision | Recall | F1-Score | NPV Gain (USD) | Expected Maintenance Cost (USD) | Delta Cost vs Full (USD) | Detected / Total | Alert Episodes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in export.iterrows():
        lines.append(
            f"| {r['model_configuration']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1_score']:.4f} | "
            f"{r['npv_gain_usd']:,.2f} | {r['expected_maintenance_cost_usd']:,.2f} | "
            f"{r['delta_cost_vs_full_usd']:,.2f} | {int(r['detected_events'])}/{int(r['total_events'])} | {int(r['alert_episodes'])} |"
        )
    lines.extend(
        [
            "",
            "\\* NPV Gain (test-window proxy) = baseline expected cost - model expected cost.",
            "\\* Delta Cost vs Full = model expected cost - full-model expected cost (negative = better than full).",
            "",
            "Auxiliary metrics (Avg early, Med early) are retained in the CSV.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Table 4b CSV -> {OUTPUT_CSV}")
    print(f"Table 4b Markdown -> {OUTPUT_MD}")
    print(f"Figure 4b -> {OUTPUT_FIG}")
    print(f"Figure 4b NPV -> {OUTPUT_FIG_NPV}")
    print()
    print(export.to_string(index=False))


if __name__ == "__main__":
    main()
