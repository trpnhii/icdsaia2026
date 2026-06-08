"""
Run the full weekly decision pipeline and export a PDF report.

Pipeline:
  risk score -> forecast -> reorder signal / optimal N -> financial summary
  -> weekly_report.pdf with one real-week appendix screenshot.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output_data" / "weekly_report"
REPORT_PATH = OUTPUT_DIR / "weekly_report.pdf"
APPENDIX_SCREENSHOT_PATH = OUTPUT_DIR / "appendix_week_screenshot.png"

RISK_PATH = PROJECT_ROOT / "input_data" / "risk" / "risk_scores.csv"
CANDIDATES_PATH = PROJECT_ROOT / "input_data" / "risk" / "replacement_candidates.csv"
WATCHLIST_PATH = PROJECT_ROOT / "input_data" / "risk" / "risk_watchlist.csv"
FORECAST_PATH = PROJECT_ROOT / "input_data" / "forecast" / "forecast_table.csv"
INVENTORY_PATH = PROJECT_ROOT / "input_data" / "inventory" / "spare_inventory_cost_summary.csv"
FINANCIAL_PATH = PROJECT_ROOT / "output_data" / "financial" / "npv_irr_comparison.csv"
TIMELINE_PATH = PROJECT_ROOT / "output_data" / "financial" / "npv_timeline.csv"
MISSING_INPUTS_PATH = PROJECT_ROOT / "output_data" / "financial" / "missing_financial_inputs.csv"
NPV_FIGURE_PATH = PROJECT_ROOT / "output_data" / "financial" / "npv_timeline.png"
FORECAST_FIGURE_PATH = PROJECT_ROOT / "output_data" / "figures" / "forecast.png"
STL_FIGURE_PATH = PROJECT_ROOT / "output_data" / "figures" / "stl_decomposition.png"
INVENTORY_FIGURE_PATH = PROJECT_ROOT / "output_data" / "figures" / "inventory" / "spare_inventory_cost_curve.png"


@dataclass(frozen=True)
class PipelineResult:
    stage: str
    returncode: int
    stdout_tail: str
    stderr_tail: str


def run_stage(name: str, args: list[str]) -> PipelineResult:
    command = [sys.executable, *args]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout_tail = "\n".join(completed.stdout.splitlines()[-12:])
    stderr_tail = "\n".join(completed.stderr.splitlines()[-12:])
    if completed.returncode != 0:
        raise RuntimeError(
            f"Stage failed: {name}\nCommand: {' '.join(command)}\n"
            f"stdout:\n{stdout_tail}\nstderr:\n{stderr_tail}"
        )
    return PipelineResult(name, completed.returncode, stdout_tail, stderr_tail)


def run_pipeline(capex_ai_usd: float, forecast_horizon: int) -> list[PipelineResult]:
    stages = [
        ("risk score", ["preprocess/risk_daily.py"]),
        ("forecast", ["preprocess/stl_forecast_daily.py", "--horizon", str(forecast_horizon)]),
        ("reorder signal / optimal N", ["preprocess/spare_inventory_cost.py"]),
        (
            "financial summary",
            ["preprocess/financial_npv_irr.py", "--capex-ai-usd", str(capex_ai_usd)],
        ),
    ]
    results = []
    for name, command in stages:
        print(f"Running stage: {name}")
        results.append(run_stage(name, command))
    return results


def fmt_number(value: object, digits: int = 0) -> str:
    if pd.isna(value):
        return "n/a"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def latest_week_window(
    risk: pd.DataFrame,
    candidates: pd.DataFrame,
    end_date: str | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    risk_dates = pd.to_datetime(risk["risk_date"])
    if end_date:
        end = pd.Timestamp(end_date)
    elif not candidates.empty:
        end = pd.to_datetime(candidates["risk_date"]).max()
    else:
        end = risk_dates.max()
    start = end - pd.Timedelta(days=6)
    return start.normalize(), end.normalize()


def read_outputs(report_end_date: str | None) -> dict[str, object]:
    risk = pd.read_csv(RISK_PATH, parse_dates=["risk_date"])
    candidates = pd.read_csv(CANDIDATES_PATH, parse_dates=["risk_date"])
    watchlist = pd.read_csv(WATCHLIST_PATH)
    forecast = pd.read_csv(FORECAST_PATH, parse_dates=["forecast_date"])
    inventory = pd.read_csv(INVENTORY_PATH)
    financial = pd.read_csv(FINANCIAL_PATH)
    timeline = pd.read_csv(TIMELINE_PATH)
    missing = pd.read_csv(MISSING_INPUTS_PATH)

    week_start, week_end = latest_week_window(risk, candidates, report_end_date)
    week_risk = risk[risk["risk_date"].between(week_start, week_end)].copy()
    week_candidates = candidates[candidates["risk_date"].between(week_start, week_end)].copy()
    optimal = inventory.loc[inventory["total_cost_vnd"].idxmin()].copy()
    ai_financial = financial[financial["scenario"] == "With AI"].iloc[0].copy()

    return {
        "risk": risk,
        "candidates": candidates,
        "watchlist": watchlist,
        "forecast": forecast,
        "inventory": inventory,
        "financial": financial,
        "timeline": timeline,
        "missing": missing,
        "week_start": week_start,
        "week_end": week_end,
        "week_risk": week_risk,
        "week_candidates": week_candidates,
        "optimal": optimal,
        "ai_financial": ai_financial,
    }


def add_title_page(pdf: PdfPages, data: dict[str, object], stages: list[PipelineResult]) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.06, 0.92, "Weekly AI O&M Decision Report", fontsize=22, weight="bold")
    fig.text(
        0.06,
        0.86,
        f"Real-week appendix window: {data['week_start'].date()} to {data['week_end'].date()}",
        fontsize=12,
    )

    financial = data["ai_financial"]
    optimal = data["optimal"]
    lines = [
        f"Replacement candidate-days in week: {len(data['week_candidates']):,}",
        f"Scored inverter-days in week: {len(data['week_risk']):,}",
        f"Optimal spare inventory N: {int(optimal['stock_n'])}",
        f"AI NPV: {fmt_number(financial['npv_vnd'])} VND",
        f"Monthly IRR: {float(financial['irr_monthly']):.2%}",
        f"AI CAPEX included: {fmt_number(financial['capex_ai_included_vnd'])} VND",
    ]
    fig.text(0.06, 0.76, "\n".join(lines), fontsize=13, linespacing=1.6)

    stage_lines = [f"{idx}. {stage.stage}: OK" for idx, stage in enumerate(stages, start=1)]
    fig.text(0.06, 0.40, "Pipeline run status", fontsize=14, weight="bold")
    fig.text(0.06, 0.35, "\n".join(stage_lines), fontsize=11, linespacing=1.5)

    fig.text(
        0.06,
        0.08,
        "Note: operational savings are derived from input_data. Financial assumptions are read from Financial_Assumptions.xlsx.",
        fontsize=9,
        color="#555555",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_table_page(pdf: PdfPages, title: str, df: pd.DataFrame, max_rows: int = 12) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=16, weight="bold", pad=18)
    preview = df.head(max_rows).copy()
    for col in preview.columns:
        if pd.api.types.is_datetime64_any_dtype(preview[col]):
            preview[col] = preview[col].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_float_dtype(preview[col]):
            preview[col] = preview[col].map(lambda x: f"{x:,.2f}")
    table = ax.table(
        cellText=preview.astype(str).values,
        colLabels=preview.columns,
        loc="upper left",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.25)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_image_page(pdf: PdfPages, title: str, path: Path) -> None:
    if not path.exists():
        return
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=16, weight="bold", pad=12)
    image = plt.imread(path)
    ax.imshow(image)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_appendix_screenshot(data: dict[str, object], output_path: Path) -> None:
    weekly = (
        data["week_candidates"]
        .sort_values(["risk_date", "risk_score"], ascending=[True, False])
        [[
            "risk_date",
            "zone",
            "device_name",
            "risk_score",
            "fleet_relative_risk_score",
            "replacement_window",
            "mean_relative_pr_14d",
            "severe_low_rate_14d",
        ]]
        .head(20)
        .copy()
    )
    if weekly.empty:
        weekly = pd.DataFrame(
            [{"message": "No replacement candidates in selected one-week window."}]
        )
    if "risk_date" in weekly.columns:
        weekly["risk_date"] = pd.to_datetime(weekly["risk_date"]).dt.strftime("%Y-%m-%d")
    for col in weekly.columns:
        if pd.api.types.is_float_dtype(weekly[col]):
            weekly[col] = weekly[col].map(lambda x: f"{x:,.2f}")

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")
    ax.set_title(
        f"Appendix Screenshot - Real Week {data['week_start'].date()} to {data['week_end'].date()}",
        loc="left",
        fontsize=16,
        weight="bold",
        pad=18,
    )
    table = ax.table(
        cellText=weekly.astype(str).values,
        colLabels=weekly.columns,
        loc="upper left",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_report(data: dict[str, object], stages: list[PipelineResult], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    build_appendix_screenshot(data, APPENDIX_SCREENSHOT_PATH)

    forecast = data["forecast"][
        ["forecast_date", "expected_new_failures", "cumulative_failures", "risk_level"]
    ].head(14)
    inventory = data["inventory"][
        ["stock_n", "expected_generation_loss_vnd", "holding_cost_vnd", "total_cost_vnd"]
    ]
    financial = data["financial"]
    missing = data["missing"]

    with PdfPages(report_path) as pdf:
        add_title_page(pdf, data, stages)
        add_table_page(pdf, "14-Day Failure Forecast", forecast)
        add_table_page(pdf, "Reorder / Spare Inventory Cost Curve", inventory)
        add_table_page(pdf, "Financial Summary: NPV and IRR", financial)
        if not missing.empty:
            add_table_page(pdf, "Open Inputs / Caveats", missing)
        add_image_page(pdf, "Forecast Chart", FORECAST_FIGURE_PATH)
        add_image_page(pdf, "STL Decomposition", STL_FIGURE_PATH)
        add_image_page(pdf, "Optimal N Cost Curve", INVENTORY_FIGURE_PATH)
        add_image_page(pdf, "NPV Timeline", NPV_FIGURE_PATH)
        add_image_page(pdf, "Appendix: One-Week Screenshot", APPENDIX_SCREENSHOT_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full weekly AI O&M pipeline and export weekly_report.pdf.")
    parser.add_argument("--capex-ai-usd", type=float, default=3000.0)
    parser.add_argument("--forecast-horizon", type=int, default=30, choices=[14, 30])
    parser.add_argument("--report-end-date", default=None, help="YYYY-MM-DD. Defaults to latest scored date.")
    parser.add_argument("--skip-run", action="store_true", help="Build the report from existing outputs only.")
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stages = [] if args.skip_run else run_pipeline(args.capex_ai_usd, args.forecast_horizon)
    data = read_outputs(args.report_end_date)
    build_report(data, stages, args.output)
    print(f"Weekly report -> {args.output}")
    print(f"Appendix screenshot -> {APPENDIX_SCREENSHOT_PATH}")


if __name__ == "__main__":
    main()
