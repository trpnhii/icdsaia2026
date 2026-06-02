"""
Label-free evaluation for the 2-year inverter risk pipeline.

This script does not measure true prediction accuracy. Instead, it produces
auditable checks that are possible without ground-truth failure labels:

- data-quality summary
- daily risk/candidate summary
- persistence checks for repeated candidate flags
- diagnostic plots for the highest-risk candidate devices
- markdown report
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_INPUT_DIR = PROJECT_ROOT / "input_data" / "base"
RISK_INPUT_DIR = PROJECT_ROOT / "input_data" / "risk"
OUTPUT_DIR = PROJECT_ROOT / "input_data" / "evaluation"
FIGURE_DIR = PROJECT_ROOT / "output_data" / "figures" / "candidate_diagnostics"

TRANSFORMED_PATH = BASE_INPUT_DIR / "transformed.csv"
RISK_PATH = RISK_INPUT_DIR / "risk_scores.csv"
CANDIDATES_PATH = RISK_INPUT_DIR / "replacement_candidates.csv"

QUALITY_SUMMARY_PATH = OUTPUT_DIR / "no_label_quality_summary.csv"
RISK_SUMMARY_PATH = OUTPUT_DIR / "no_label_risk_summary.csv"
PERSISTENCE_PATH = OUTPUT_DIR / "no_label_candidate_persistence.csv"
REPORT_PATH = OUTPUT_DIR / "no_label_evaluation.md"

MAX_DIAGNOSTIC_PLOTS = 20
HIGH_PR_THRESHOLD = 1.20
MIN_IRRADIATION_FOR_ZERO_YIELD = 0.5


def _read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    transformed = pd.read_csv(TRANSFORMED_PATH, parse_dates=["date"])
    risk = pd.read_csv(RISK_PATH, parse_dates=["risk_date", "latest_date"])
    candidates = pd.read_csv(CANDIDATES_PATH, parse_dates=["risk_date", "latest_date"])
    return transformed, risk, candidates


def build_quality_summary(transformed: pd.DataFrame) -> pd.DataFrame:
    df = transformed.copy()
    df["is_logger_row"] = df["device_name"].str.startswith("Inverter(COM", na=False)
    df["valid_pr"] = df["performance_ratio"].notna()
    df["high_pr"] = df["performance_ratio"] > HIGH_PR_THRESHOLD
    df["zero_yield_high_irr"] = (
        (df["yield_kwh"].fillna(0) <= 0)
        & (df["irradiation_kwh_m2"].fillna(0) >= MIN_IRRADIATION_FOR_ZERO_YIELD)
    )

    summary = (
        df.groupby("date")
        .agg(
            rows=("device_name", "count"),
            devices=("device_name", "nunique"),
            logger_rows=("is_logger_row", "sum"),
            missing_capacity=("installed_capacity_kwp", lambda s: int(s.isna().sum())),
            missing_irradiation=("irradiation_kwh_m2", lambda s: int(s.isna().sum())),
            missing_pr=("performance_ratio", lambda s: int(s.isna().sum())),
            valid_pr_rows=("valid_pr", "sum"),
            high_pr_rows=("high_pr", "sum"),
            zero_yield_high_irr_rows=("zero_yield_high_irr", "sum"),
            mean_pr=("performance_ratio", "mean"),
            median_pr=("performance_ratio", "median"),
            mean_irradiation=("irradiation_kwh_m2", "mean"),
        )
        .reset_index()
        .sort_values("date")
    )
    summary["valid_pr_rate"] = summary["valid_pr_rows"] / summary["rows"]
    return summary


def build_risk_summary(risk: pd.DataFrame) -> pd.DataFrame:
    summary = (
        risk.groupby("risk_date")
        .agg(
            scored_inverters=("device_name", "count"),
            candidate_count=("replacement_candidate", "sum"),
            monitor_count=("replacement_window", lambda s: int((s == "monitor").sum())),
            max_risk_score=("risk_score", "max"),
            mean_risk_score=("risk_score", "mean"),
            median_risk_score=("risk_score", "median"),
            p95_risk_score=("risk_score", lambda s: float(s.quantile(0.95))),
        )
        .reset_index()
        .sort_values("risk_date")
    )
    summary["candidate_rate"] = summary["candidate_count"] / summary["scored_inverters"]
    return summary


def build_persistence_summary(risk: pd.DataFrame) -> pd.DataFrame:
    ordered = risk.sort_values(["zone", "device_name", "risk_date"]).copy()
    ordered["candidate_int"] = ordered["replacement_candidate"].astype(int)
    ordered["candidate_days_last_5"] = (
        ordered.groupby(["zone", "device_name"])["candidate_int"]
        .rolling(window=5, min_periods=1)
        .sum()
        .reset_index(level=[0, 1], drop=True)
    )
    ordered["persistent_3_of_5"] = ordered["candidate_days_last_5"] >= 3

    persistence = (
        ordered.groupby(["zone", "device_name"], as_index=False)
        .agg(
            candidate_days=("candidate_int", "sum"),
            persistent_days=("persistent_3_of_5", "sum"),
            max_risk_score=("risk_score", "max"),
            mean_candidate_risk=(
                "risk_score",
                lambda s: float(s[ordered.loc[s.index, "candidate_int"] == 1].mean())
                if int(ordered.loc[s.index, "candidate_int"].sum()) > 0
                else np.nan,
            ),
            first_candidate_date=(
                "risk_date",
                lambda s: s[ordered.loc[s.index, "candidate_int"] == 1].min()
                if int(ordered.loc[s.index, "candidate_int"].sum()) > 0
                else pd.NaT,
            ),
            last_candidate_date=(
                "risk_date",
                lambda s: s[ordered.loc[s.index, "candidate_int"] == 1].max()
                if int(ordered.loc[s.index, "candidate_int"].sum()) > 0
                else pd.NaT,
            ),
        )
    )
    persistence = persistence[persistence["candidate_days"] > 0].copy()
    return persistence.sort_values(
        ["persistent_days", "candidate_days", "max_risk_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def plot_candidate_diagnostics(
    transformed: pd.DataFrame,
    risk: pd.DataFrame,
    persistence: pd.DataFrame,
) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    selected = persistence.head(MAX_DIAGNOSTIC_PLOTS)
    written: list[Path] = []

    for _, row in selected.iterrows():
        zone = int(row["zone"])
        device = row["device_name"]
        tfm_device = transformed[
            (transformed["zone"] == zone) & (transformed["device_name"] == device)
        ].sort_values("date")
        risk_device = risk[
            (risk["zone"] == zone) & (risk["device_name"] == device)
        ].sort_values("risk_date")
        if tfm_device.empty or risk_device.empty:
            continue

        candidate_dates = risk_device.loc[risk_device["replacement_candidate"], "risk_date"]

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"Zone {zone} - {device}", fontsize=13, fontweight="bold")

        axes[0].plot(tfm_device["date"], tfm_device["performance_ratio"], color="#2c7bb6", linewidth=1.3)
        axes[0].axhline(0.4, color="#d73027", linestyle="--", linewidth=0.9, label="PR 0.40")
        axes[0].set_ylabel("PR")
        axes[0].legend(loc="upper right", fontsize=8)

        axes[1].plot(risk_device["risk_date"], risk_device["latest_relative_pr"], color="#1a9641", linewidth=1.3)
        axes[1].axhline(0.7, color="#fdae61", linestyle="--", linewidth=0.9, label="Relative PR 0.70")
        axes[1].axhline(0.5, color="#d73027", linestyle="--", linewidth=0.9, label="Relative PR 0.50")
        axes[1].set_ylabel("Relative PR")
        axes[1].legend(loc="upper right", fontsize=8)

        axes[2].plot(risk_device["risk_date"], risk_device["risk_score"], color="#756bb1", linewidth=1.4)
        axes[2].scatter(
            candidate_dates,
            risk_device.loc[risk_device["replacement_candidate"], "risk_score"],
            color="#d73027",
            s=22,
            label="Candidate day",
            zorder=3,
        )
        axes[2].axhline(65, color="#fdae61", linestyle="--", linewidth=0.9, label="30d threshold")
        axes[2].axhline(80, color="#d73027", linestyle="--", linewidth=0.9, label="14d threshold")
        axes[2].set_ylabel("Risk score")
        axes[2].legend(loc="upper right", fontsize=8)
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axes[2].tick_params(axis="x", rotation=30)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        safe_device = str(device).replace("/", "_").replace("\\", "_").replace(" ", "_")
        out_path = FIGURE_DIR / f"zone{zone}_{safe_device}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)

    return written


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"

    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_datetime64_any_dtype(formatted[col]):
            formatted[col] = formatted[col].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        else:
            formatted[col] = formatted[col].map(lambda value: "" if pd.isna(value) else str(value))

    headers = list(formatted.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in formatted.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def write_report(
    transformed: pd.DataFrame,
    risk: pd.DataFrame,
    candidates: pd.DataFrame,
    quality: pd.DataFrame,
    risk_summary: pd.DataFrame,
    persistence: pd.DataFrame,
    plot_paths: list[Path],
) -> None:
    latest_date = risk["risk_date"].max()
    latest_day = risk[risk["risk_date"] == latest_date]
    latest_top = latest_day.sort_values("risk_score", ascending=False).head(10)

    worst_quality = quality.sort_values(
        ["missing_pr", "zero_yield_high_irr_rows", "high_pr_rows"],
        ascending=False,
    ).head(10)
    top_persistent = persistence.head(10)

    report = [
        "# No-Label Pipeline Evaluation",
        "",
        "This report evaluates the pipeline without ground-truth failure labels. It checks data quality, risk stability, and candidate persistence. It does not measure true prediction accuracy.",
        "",
        "## Dataset Summary",
        "",
        f"- Transformed rows: {len(transformed):,}",
        f"- Risk-score rows: {len(risk):,}",
        f"- Candidate-days: {len(candidates):,}",
        f"- Scored dates: {risk['risk_date'].nunique():,}",
        f"- Scored date range: {risk['risk_date'].min().date()} to {risk['risk_date'].max().date()}",
        f"- Latest scored date: {latest_date.date()}",
        "",
        "## Data-Quality Checks",
        "",
        f"- Dates with any missing PR: {int((quality['missing_pr'] > 0).sum()):,}",
        f"- Dates with high PR rows above {HIGH_PR_THRESHOLD}: {int((quality['high_pr_rows'] > 0).sum()):,}",
        f"- Dates with zero yield during irradiation >= {MIN_IRRADIATION_FOR_ZERO_YIELD}: {int((quality['zero_yield_high_irr_rows'] > 0).sum()):,}",
        "",
        "Worst data-quality dates:",
        "",
        worst_quality[
            [
                "date",
                "rows",
                "missing_irradiation",
                "missing_pr",
                "high_pr_rows",
                "zero_yield_high_irr_rows",
                "valid_pr_rate",
            ]
        ].pipe(_markdown_table),
        "",
        "## Risk Stability",
        "",
        f"- Unique candidate devices: {candidates[['zone', 'device_name']].drop_duplicates().shape[0]:,}",
        f"- Devices with at least one persistent 3-of-5 candidate window: {int((persistence['persistent_days'] > 0).sum()):,}",
        "",
        "Top persistent candidate devices:",
        "",
        top_persistent[
            [
                "zone",
                "device_name",
                "candidate_days",
                "persistent_days",
                "max_risk_score",
                "first_candidate_date",
                "last_candidate_date",
            ]
        ].pipe(_markdown_table),
        "",
        "## Latest Watchlist",
        "",
        latest_top[
            [
                "zone",
                "device_name",
                "risk_score",
                "fleet_relative_risk_score",
                "replacement_window",
                "mean_relative_pr_14d",
                "severe_low_rate_14d",
            ]
        ].pipe(_markdown_table),
        "",
        "## Generated Artifacts",
        "",
        f"- Data-quality summary: `{QUALITY_SUMMARY_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Daily risk summary: `{RISK_SUMMARY_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Candidate persistence summary: `{PERSISTENCE_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Diagnostic plot folder: `{FIGURE_DIR.relative_to(PROJECT_ROOT)}`",
        f"- Diagnostic plots generated: {len(plot_paths)}",
        "",
        "## Interpretation",
        "",
        "Without labels, the strongest evidence is repeated and persistent underperformance relative to same-day fleet peers. A one-day candidate flag should be treated as a review item; repeated 3-of-5 candidate windows deserve higher priority.",
        "",
        "This report should be compared with maintenance notes, alarms, curtailment records, and raw inverter behavior when those records become available.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    transformed, risk, candidates = _read_inputs()
    quality = build_quality_summary(transformed)
    risk_summary = build_risk_summary(risk)
    persistence = build_persistence_summary(risk)

    quality.to_csv(QUALITY_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    risk_summary.to_csv(RISK_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    persistence.to_csv(PERSISTENCE_PATH, index=False, encoding="utf-8-sig")

    plot_paths = plot_candidate_diagnostics(transformed, risk, persistence)
    write_report(transformed, risk, candidates, quality, risk_summary, persistence, plot_paths)

    print(f"Quality summary -> {QUALITY_SUMMARY_PATH}")
    print(f"Risk summary -> {RISK_SUMMARY_PATH}")
    print(f"Candidate persistence -> {PERSISTENCE_PATH}")
    print(f"No-label report -> {REPORT_PATH}")
    print(f"Diagnostic plots -> {FIGURE_DIR}")
    print(f"Diagnostic plots generated: {len(plot_paths)}")


if __name__ == "__main__":
    main()
