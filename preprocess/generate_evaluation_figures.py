"""
Generate additional label-free evaluation figures for the 2-year pipeline.

Outputs are written to:
  output_data/figures/evaluation/
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
EVAL_INPUT_DIR = PROJECT_ROOT / "input_data" / "evaluation"
OUTPUT_DIR = PROJECT_ROOT / "output_data" / "figures" / "evaluation"

RISK_PATH = RISK_INPUT_DIR / "risk_scores.csv"
TRANSFORMED_PATH = BASE_INPUT_DIR / "transformed.csv"
QUALITY_PATH = EVAL_INPUT_DIR / "no_label_quality_summary.csv"
RISK_SUMMARY_PATH = EVAL_INPUT_DIR / "no_label_risk_summary.csv"
PERSISTENCE_PATH = EVAL_INPUT_DIR / "no_label_candidate_persistence.csv"

STYLE = "seaborn-v0_8-whitegrid"


def _save(fig: plt.Figure, name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")
    return out_path


def plot_daily_candidate_count(risk_summary: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(risk_summary["risk_date"], risk_summary["candidate_count"], color="#2c7fb8", alpha=0.8, width=0.9)
    ax.set_title("Daily Replacement-Candidate Count", fontweight="bold")
    ax.set_ylabel("Candidate inverters")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, "daily_candidate_count.png")


def plot_risk_score_distribution(risk: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(risk["risk_score"], bins=50, color="#756bb1", alpha=0.85)
    axes[0].set_title("Risk Score Distribution", fontweight="bold")
    axes[0].set_xlabel("Risk score")
    axes[0].set_ylabel("Inverter-days")
    axes[0].axvline(45, color="#fdae61", linestyle="--", linewidth=1, label="Monitor")
    axes[0].axvline(65, color="#f46d43", linestyle="--", linewidth=1, label="30-day candidate")
    axes[0].axvline(80, color="#d73027", linestyle="--", linewidth=1, label="14-day candidate")
    axes[0].legend(fontsize=8)

    axes[1].boxplot(
        [
            risk.loc[~risk["replacement_candidate"], "risk_score"],
            risk.loc[risk["replacement_candidate"], "risk_score"],
        ],
        labels=["Non-candidate", "Candidate"],
        patch_artist=True,
        boxprops=dict(facecolor="#a6bddb", color="#444444"),
        medianprops=dict(color="#d73027", linewidth=1.4),
    )
    axes[1].set_title("Candidate vs Non-Candidate Scores", fontweight="bold")
    axes[1].set_ylabel("Risk score")
    fig.tight_layout()
    return _save(fig, "risk_score_distribution.png")


def plot_candidate_persistence(persistence: pd.DataFrame) -> Path:
    top = persistence.sort_values(["candidate_days", "persistent_days"], ascending=False).head(15).copy()
    top["label"] = "Z" + top["zone"].astype(str) + " - " + top["device_name"]
    top = top.sort_values("candidate_days", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(top["label"], top["candidate_days"], color="#2c7fb8", alpha=0.85, label="Candidate days")
    ax.barh(top["label"], top["persistent_days"], color="#d73027", alpha=0.75, label="Persistent 3-of-5 days")
    ax.set_title("Top Candidate Persistence by Inverter", fontweight="bold")
    ax.set_xlabel("Days")
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, "candidate_persistence.png")


def plot_data_quality_timeline(quality: pd.DataFrame) -> Path:
    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax1.bar(
        quality["date"],
        quality["missing_irradiation"],
        color="#9ecae1",
        alpha=0.65,
        width=0.9,
        label="Missing irradiation rows",
    )
    ax1.bar(
        quality["date"],
        quality["missing_pr"],
        color="#fc9272",
        alpha=0.55,
        width=0.9,
        label="Missing PR rows",
    )
    ax1.set_ylabel("Rows")
    ax1.set_xlabel("Date")

    ax2 = ax1.twinx()
    ax2.plot(quality["date"], quality["valid_pr_rate"], color="#238b45", linewidth=1.4, label="Valid PR rate")
    ax2.set_ylabel("Valid PR rate")
    ax2.set_ylim(-0.05, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    ax1.set_title("Data Quality Timeline", fontweight="bold")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.tick_params(axis="x", rotation=30)
    return _save(fig, "data_quality_timeline.png")


def plot_fleet_pr_and_risk(transformed: pd.DataFrame, risk_summary: pd.DataFrame) -> Path:
    daily_pr = (
        transformed[transformed["performance_ratio"].notna()]
        .groupby("date", as_index=False)
        .agg(mean_pr=("performance_ratio", "mean"), median_pr=("performance_ratio", "median"))
    )

    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax1.plot(daily_pr["date"], daily_pr["mean_pr"], color="#2c7bb6", linewidth=1.2, label="Mean PR")
    ax1.plot(daily_pr["date"], daily_pr["median_pr"], color="#1a9641", linewidth=1.2, label="Median PR")
    ax1.set_ylabel("Performance ratio")
    ax1.set_xlabel("Date")

    ax2 = ax1.twinx()
    ax2.plot(risk_summary["risk_date"], risk_summary["max_risk_score"], color="#d73027", linewidth=1.2, label="Max risk score")
    ax2.plot(risk_summary["risk_date"], risk_summary["mean_risk_score"], color="#756bb1", linewidth=1.1, label="Mean risk score")
    ax2.set_ylabel("Risk score")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    ax1.set_title("Fleet PR and Risk Over Time", fontweight="bold")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.tick_params(axis="x", rotation=30)
    return _save(fig, "fleet_pr_and_risk_over_time.png")


def plot_risk_vs_relative_pr(risk: pd.DataFrame) -> Path:
    sample = risk.sample(n=min(20000, len(risk)), random_state=42)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = np.where(sample["replacement_candidate"], "#d73027", "#2c7fb8")
    ax.scatter(
        sample["latest_relative_pr"],
        sample["risk_score"],
        c=colors,
        s=8,
        alpha=0.35,
        linewidths=0,
    )
    ax.axvline(0.7, color="#fdae61", linestyle="--", linewidth=1, label="Relative PR 0.70")
    ax.axvline(0.5, color="#d73027", linestyle="--", linewidth=1, label="Relative PR 0.50")
    ax.axhline(65, color="#f46d43", linestyle="--", linewidth=1, label="Risk 65")
    ax.axhline(80, color="#d73027", linestyle=":", linewidth=1, label="Risk 80")
    ax.set_title("Risk Score vs Latest Relative PR", fontweight="bold")
    ax.set_xlabel("Latest relative PR")
    ax.set_ylabel("Risk score")
    ax.set_xlim(0, 2)
    ax.legend(fontsize=8)
    return _save(fig, "risk_vs_relative_pr.png")


def plot_zone_candidate_comparison(risk: pd.DataFrame) -> Path:
    daily_zone = (
        risk.groupby(["risk_date", "zone"], as_index=False)
        .agg(candidate_count=("replacement_candidate", "sum"))
    )
    pivot = daily_zone.pivot(index="risk_date", columns="zone", values="candidate_count").fillna(0)

    fig, ax = plt.subplots(figsize=(13, 5))
    bottom = np.zeros(len(pivot))
    colors = ["#2c7fb8", "#f03b20", "#31a354", "#756bb1"]
    for idx, zone in enumerate(pivot.columns):
        ax.bar(pivot.index, pivot[zone], bottom=bottom, width=0.9, alpha=0.8, color=colors[idx % len(colors)], label=f"Zone {zone}")
        bottom += pivot[zone].to_numpy()
    ax.set_title("Daily Candidate Count by Zone", fontweight="bold")
    ax.set_ylabel("Candidate inverters")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, "zone_candidate_comparison.png")


def plot_risk_heatmap(risk: pd.DataFrame, persistence: pd.DataFrame) -> Path:
    top_devices = (
        persistence.sort_values(["candidate_days", "max_risk_score"], ascending=False)
        .head(30)[["zone", "device_name"]]
    )
    labels = ["Z" + top_devices["zone"].astype(str) + " - " + top_devices["device_name"]]
    selected = risk.merge(top_devices, on=["zone", "device_name"], how="inner").copy()
    selected["device_label"] = "Z" + selected["zone"].astype(str) + " - " + selected["device_name"]
    pivot = selected.pivot_table(index="device_label", columns="risk_date", values="risk_score", aggfunc="mean")
    device_order = labels[0].tolist()
    pivot = pivot.reindex(device_order)

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", interpolation="nearest", cmap="YlOrRd", vmin=0, vmax=max(80, np.nanmax(pivot.to_numpy())))
    ax.set_title("Risk Score Heatmap for Top Candidate Devices", fontweight="bold")
    ax.set_ylabel("Device")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    dates = pd.to_datetime(pivot.columns)
    tick_idx = np.linspace(0, len(dates) - 1, min(8, len(dates))).astype(int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([dates[i].strftime("%Y-%m") for i in tick_idx], rotation=30, ha="right")
    ax.set_xlabel("Risk date")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Risk score")
    return _save(fig, "risk_heatmap_top_candidates.png")


def main() -> None:
    plt.style.use(STYLE)
    risk = pd.read_csv(RISK_PATH, parse_dates=["risk_date", "latest_date"])
    transformed = pd.read_csv(TRANSFORMED_PATH, parse_dates=["date"])
    quality = pd.read_csv(QUALITY_PATH, parse_dates=["date"])
    risk_summary = pd.read_csv(RISK_SUMMARY_PATH, parse_dates=["risk_date"])
    persistence = pd.read_csv(PERSISTENCE_PATH, parse_dates=["first_candidate_date", "last_candidate_date"])

    paths = [
        plot_daily_candidate_count(risk_summary),
        plot_risk_score_distribution(risk),
        plot_candidate_persistence(persistence),
        plot_data_quality_timeline(quality),
        plot_fleet_pr_and_risk(transformed, risk_summary),
        plot_risk_vs_relative_pr(risk),
        plot_zone_candidate_comparison(risk),
        plot_risk_heatmap(risk, persistence),
    ]
    print(f"Generated figures: {len(paths)}")


if __name__ == "__main__":
    main()
