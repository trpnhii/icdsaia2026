"""
Run extended Table 11 experiments:
  1) Supervised Gradient-Boosting baselines (XGBoost / LightGBM if available)
  2) Synthetic fault injection to increase labeled events (default >= 100)
  3) Cost-sensitive evaluation (Expected Maintenance Cost, USD)

Outputs:
  output_data/experimental_results/table11_extended_comparison.csv
  output_data/experimental_results/table11_extended_comparison.md
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSFORMED_PATH = PROJECT_ROOT / "input_data" / "base" / "transformed.csv"
OUTPUT_DIR = PROJECT_ROOT / "output_data" / "experimental_results"
OUTPUT_CSV = OUTPUT_DIR / "table11_extended_comparison.csv"
OUTPUT_MD = OUTPUT_DIR / "table11_extended_comparison.md"


@dataclass(frozen=True)
class MethodMetrics:
    method: str
    precision: float
    recall: float
    f1: float
    avg_early_days: float | None
    med_early_days: float | None
    detected_events: int
    total_events: int
    alert_episodes: int
    false_alert_episodes: int
    expected_maintenance_cost_usd: float
    zero_shot_ready: str


def normalize_device_name(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    text = " ".join(text.split())
    return text


def safe_ratio(n: float, d: float) -> float:
    return 0.0 if d == 0 else n / d


def collapse_alert_episodes(alerts: pd.DataFrame, gap_days: int = 7) -> pd.DataFrame:
    if alerts.empty:
        return alerts
    alerts = alerts.sort_values(["device_key", "alert_date"]).copy()
    rows: list[dict[str, object]] = []
    for device_key, grp in alerts.groupby("device_key", sort=False):
        last_kept = None
        zone = grp["zone"].iloc[0] if "zone" in grp.columns else None
        for alert_date in pd.to_datetime(grp["alert_date"]).sort_values():
            if last_kept is None or (alert_date - last_kept).days > gap_days:
                rows.append(
                    {"zone": zone, "device_key": device_key, "alert_date": pd.Timestamp(alert_date)}
                )
                last_kept = pd.Timestamp(alert_date)
    return pd.DataFrame(rows)


def evaluate_event_alerts(
    method: str,
    alerts: pd.DataFrame,
    events: pd.DataFrame,
    lead_window_days: int,
    fp_inspection_cost_usd: float,
    fn_miss_cost_usd: float,
) -> MethodMetrics:
    if events.empty:
        return MethodMetrics(method, 0, 0, 0, None, None, 0, 0, 0, 0, 0.0, "No")

    alerts = alerts.sort_values(["device_key", "alert_date"]).copy()
    events = events.sort_values(["device_key", "replacement_date"]).copy()
    alerts["matched"] = False
    matched_events = 0
    matched_leads: list[int] = []

    for _, event in events.iterrows():
        candidates = alerts[
            (alerts["device_key"] == event["device_key"])
            & (~alerts["matched"])
            & (alerts["alert_date"] <= event["replacement_date"])
        ].copy()
        if candidates.empty:
            continue
        candidates["lead_days"] = (event["replacement_date"] - candidates["alert_date"]).dt.days
        candidates = candidates[
            (candidates["lead_days"] >= 0) & (candidates["lead_days"] <= lead_window_days)
        ].sort_values(["lead_days", "alert_date"], ascending=[False, True])
        if candidates.empty:
            continue
        pick = candidates.index[0]
        alerts.loc[pick, "matched"] = True
        matched_events += 1
        matched_leads.append(int(candidates.loc[pick, "lead_days"]))

    tp = matched_events
    fp = int((~alerts["matched"]).sum())
    fn = int(len(events) - matched_events)
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    f1 = safe_ratio(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    expected_cost = fp * fp_inspection_cost_usd + fn * fn_miss_cost_usd
    return MethodMetrics(
        method=method,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        avg_early_days=float(np.mean(matched_leads)) if matched_leads else None,
        med_early_days=float(np.median(matched_leads)) if matched_leads else None,
        detected_events=int(tp),
        total_events=int(len(events)),
        alert_episodes=int(len(alerts)),
        false_alert_episodes=int(fp),
        expected_maintenance_cost_usd=float(expected_cost),
        zero_shot_ready="No",
    )


def build_base_panel(transformed: pd.DataFrame) -> pd.DataFrame:
    df = transformed.copy()
    df["device_key"] = df["device_name"].map(normalize_device_name)
    df = df[
        df["performance_ratio"].notna()
        & df["irradiation_kwh_m2"].notna()
        & (df["irradiation_kwh_m2"] >= 0.5)
        & df["installed_capacity_kwp"].notna()
    ].copy()
    date_median = df.groupby("date")["performance_ratio"].transform("median")
    df["relative_pr"] = (df["performance_ratio"] / date_median.replace(0, np.nan)).clip(0, 2)
    df["peer_deficit"] = (1 - df["relative_pr"]).clip(0, 1)
    df = df.sort_values(["zone", "device_key", "date"]).reset_index(drop=True)

    g = df.groupby(["zone", "device_key"], sort=False)
    df["mean_deficit_14d"] = g["peer_deficit"].transform(lambda s: s.rolling(14, min_periods=1).mean())
    df["mean_deficit_30d"] = g["peer_deficit"].transform(lambda s: s.rolling(30, min_periods=1).mean())
    df["severe_low_14d"] = g["relative_pr"].transform(
        lambda s: (s < 0.5).astype(float).rolling(14, min_periods=1).mean()
    )
    df["severe_low_30d"] = g["relative_pr"].transform(
        lambda s: (s < 0.5).astype(float).rolling(30, min_periods=1).mean()
    )
    df["pr_slope_30d"] = g["relative_pr"].transform(
        lambda s: s.rolling(30, min_periods=5).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) >= 5 else 0.0,
            raw=False,
        )
    ).fillna(0.0)
    return df


def inject_synthetic_faults(
    panel: pd.DataFrame,
    target_events: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_seed)
    df = panel.copy()
    df["synthetic_fault"] = 0
    events: list[dict[str, object]] = []

    devices = (
        df[["zone", "device_key"]]
        .drop_duplicates()
        .sample(frac=1.0, random_state=random_seed)
        .reset_index(drop=True)
    )
    max_trials = target_events * 20
    event_count = 0
    trial = 0

    while event_count < target_events and trial < max_trials:
        trial += 1
        dev = devices.iloc[rng.integers(0, len(devices))]
        dev_rows = df[(df["zone"] == dev["zone"]) & (df["device_key"] == dev["device_key"])].copy()
        if len(dev_rows) < 60:
            continue
        start_idx = int(rng.integers(20, len(dev_rows) - 15))
        fault_type = "degradation" if rng.random() < 0.7 else "hard_fault"
        if fault_type == "degradation":
            duration = int(rng.integers(10, 16))
            factors = np.linspace(0.95, 0.45, duration)
        else:
            duration = int(rng.integers(2, 5))
            factors = np.full(duration, 0.08)

        idx = dev_rows.index[start_idx : start_idx + duration]
        if len(idx) < duration:
            continue
        if df.loc[idx, "synthetic_fault"].sum() > 0:
            continue

        df.loc[idx, "performance_ratio"] = (df.loc[idx, "performance_ratio"].to_numpy() * factors).clip(0, 2)
        df.loc[idx, "synthetic_fault"] = 1
        replacement_date = pd.Timestamp(dev_rows.loc[idx[-1], "date"])
        events.append(
            {
                "zone": int(dev["zone"]),
                "device_key": dev["device_key"],
                "replacement_date": replacement_date,
                "fault_type": fault_type,
            }
        )
        event_count += 1

    events_df = pd.DataFrame(events).drop_duplicates(["device_key", "replacement_date"]).reset_index(drop=True)
    # recompute derived features after injection
    df = build_base_panel(df)
    return df, events_df


def build_solar_guard_alerts(work: pd.DataFrame) -> pd.DataFrame:
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
    alerts = work.loc[mask, ["zone", "device_key", "date"]].rename(columns={"date": "alert_date"})
    return alerts


def build_iforest_alerts(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    cols = ["performance_ratio", "relative_pr", "peer_deficit", "mean_deficit_14d", "severe_low_14d", "pr_slope_30d"]
    X_train = train[cols].fillna(0.0).to_numpy()
    X_test = test[cols].fillna(0.0).to_numpy()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = IsolationForest(contamination=0.05, random_state=42, n_estimators=200)
    model.fit(X_train)
    pred = model.predict(X_test) == -1
    alerts = test.loc[pred, ["zone", "device_key", "date"]].rename(columns={"date": "alert_date"})
    return alerts


def _feature_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = ["performance_ratio", "relative_pr", "peer_deficit", "mean_deficit_14d", "severe_low_14d", "pr_slope_30d"]
    return df[cols].fillna(0.0).to_numpy()


def _score_to_proba(model: object, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    scores = np.asarray(model.decision_function(X), dtype=float)
    # map unbounded scores to (0,1) for threshold search
    return 1.0 / (1.0 + np.exp(-scores))


def tune_threshold_on_val_emc(
    model: object,
    val: pd.DataFrame,
    events_val: pd.DataFrame,
    lead_window_days: int,
    episode_gap_days: int,
    fp_inspection_cost_usd: float,
    fn_miss_cost_usd: float,
) -> tuple[float, float]:
    """Pick probability cutoff on validation only by minimizing event-level EMC."""
    if val.empty or events_val.empty:
        return 0.5, float("inf")

    X_val = _feature_matrix(val)
    proba = _score_to_proba(model, X_val)
    thresholds = np.linspace(0.05, 0.95, 37)
    best_t, best_cost = 0.5, float("inf")
    for t in thresholds:
        pred = proba >= t
        alerts = val.loc[pred, ["zone", "device_key", "date"]].rename(columns={"date": "alert_date"})
        collapsed = collapse_alert_episodes(alerts, episode_gap_days)
        metrics = evaluate_event_alerts(
            "val_tune",
            collapsed,
            events_val,
            lead_window_days,
            fp_inspection_cost_usd,
            fn_miss_cost_usd,
        )
        if metrics.expected_maintenance_cost_usd < best_cost:
            best_cost = metrics.expected_maintenance_cost_usd
            best_t = float(t)
    return best_t, best_cost


def train_supervised_and_alerts(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    y_train: np.ndarray,
    events_val: pd.DataFrame,
    lead_window_days: int,
    episode_gap_days: int,
    fp_inspection_cost_usd: float,
    fn_miss_cost_usd: float,
) -> list[tuple[str, pd.DataFrame, float, float]]:
    """Fit on train, tune threshold on val (EMC), alert on test only."""
    X_train = _feature_matrix(train)
    X_test = _feature_matrix(test)
    models: list[tuple[str, object]] = []

    # always available fallback
    models.append(
        ("HistGradientBoosting", HistGradientBoostingClassifier(random_state=42, max_iter=250))
    )
    try:
        import xgboost as xgb  # type: ignore

        models.append(
            (
                "XGBoost",
                xgb.XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                ),
            )
        )
    except Exception:
        pass
    try:
        import lightgbm as lgb  # type: ignore

        models.append(
            (
                "LightGBM",
                lgb.LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=31,
                    random_state=42,
                    verbosity=-1,
                ),
            )
        )
    except Exception:
        pass

    outputs: list[tuple[str, pd.DataFrame, float, float]] = []
    for name, model in models:
        model.fit(X_train, y_train)
        best_t, best_val_emc = tune_threshold_on_val_emc(
            model,
            val,
            events_val,
            lead_window_days,
            episode_gap_days,
            fp_inspection_cost_usd,
            fn_miss_cost_usd,
        )
        test_proba = _score_to_proba(model, X_test)
        test_pred = test_proba >= best_t
        alerts = test.loc[test_pred, ["zone", "device_key", "date"]].rename(columns={"date": "alert_date"})
        outputs.append((name, alerts, best_t, best_val_emc))
    return outputs


def chronological_train_val_test(
    panel: pd.DataFrame,
    train_frac: float,
    val_frac: float,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Date-quantile split: [0, train_frac] train, (train_frac, train_frac+val_frac] val,
    remainder test. Defaults aim for ~55% / 15% / 30% while keeping the historical
    ~30% holdout test mass used in earlier Table 11 runs.
    """
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1.0:
        raise ValueError("Require train_frac>0, val_frac>0, and train_frac+val_frac<1.")
    train_cut = panel["date"].quantile(train_frac)
    val_cut = panel["date"].quantile(train_frac + val_frac)
    train = panel[panel["date"] <= train_cut].copy()
    val = panel[(panel["date"] > train_cut) & (panel["date"] <= val_cut)].copy()
    test = panel[panel["date"] > val_cut].copy()
    return pd.Timestamp(train_cut), pd.Timestamp(val_cut), train, val, test


def make_row_labels_from_events(df: pd.DataFrame, events: pd.DataFrame, lead_window_days: int) -> np.ndarray:
    y = np.zeros(len(df), dtype=int)
    event_map = events.groupby(["zone", "device_key"])["replacement_date"].apply(list).to_dict()
    for i, row in enumerate(df.itertuples(index=False)):
        key = (int(row.zone), row.device_key)
        if key not in event_map:
            continue
        date = pd.Timestamp(row.date)
        for repl in event_map[key]:
            lead = (pd.Timestamp(repl) - date).days
            if 0 <= lead <= lead_window_days:
                y[i] = 1
                break
    return y


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run extended Table 11 with synthetic events and cost-sensitive metrics.")
    p.add_argument("--target-events", type=int, default=300)
    p.add_argument("--lead-window-days", type=int, default=30)
    p.add_argument("--episode-gap-days", type=int, default=7)
    p.add_argument("--fp-inspection-cost-usd", type=float, default=10.0)
    p.add_argument("--fn-miss-cost-usd", type=float, default=1500.0)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument(
        "--train-frac",
        type=float,
        default=0.55,
        help="Chronological quantile end of the train window (exclusive of val/test).",
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Chronological quantile width of the validation window after train.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    transformed = pd.read_csv(TRANSFORMED_PATH, parse_dates=["date"])
    base_panel = build_base_panel(transformed)
    panel, synthetic_events = inject_synthetic_faults(base_panel, args.target_events, args.random_seed)
    if len(synthetic_events) < 100:
        raise RuntimeError(f"Synthetic injection produced only {len(synthetic_events)} events; expected >= 100.")

    train_cut, val_cut, train, val, test = chronological_train_val_test(
        panel, args.train_frac, args.val_frac
    )
    events_train = synthetic_events[synthetic_events["replacement_date"] <= train_cut].copy()
    events_val = synthetic_events[
        (synthetic_events["replacement_date"] > train_cut)
        & (synthetic_events["replacement_date"] <= val_cut)
    ].copy()
    events_test = synthetic_events[synthetic_events["replacement_date"] > val_cut].copy()

    y_train = make_row_labels_from_events(train, synthetic_events, args.lead_window_days)

    results: list[MethodMetrics] = []
    threshold_rows: list[dict[str, object]] = []

    # Fit Isolation Forest on train+val chronology before test (no labels / no threshold tuning).
    pre_test = pd.concat([train, val], ignore_index=True)

    solar_alerts = collapse_alert_episodes(build_solar_guard_alerts(test), args.episode_gap_days)
    solar = evaluate_event_alerts(
        "SolarGuard (Zero-shot)",
        solar_alerts,
        events_test,
        args.lead_window_days,
        args.fp_inspection_cost_usd,
        args.fn_miss_cost_usd,
    )
    results.append(MethodMetrics(**{**solar.__dict__, "zero_shot_ready": "Yes"}))

    iforest_alerts = collapse_alert_episodes(build_iforest_alerts(pre_test, test), args.episode_gap_days)
    iforest = evaluate_event_alerts(
        "Isolation Forest (Zero-shot)",
        iforest_alerts,
        events_test,
        args.lead_window_days,
        args.fp_inspection_cost_usd,
        args.fn_miss_cost_usd,
    )
    results.append(MethodMetrics(**{**iforest.__dict__, "zero_shot_ready": "Yes"}))

    for model_name, alerts, best_t, best_val_emc in train_supervised_and_alerts(
        train,
        val,
        test,
        y_train,
        events_val,
        args.lead_window_days,
        args.episode_gap_days,
        args.fp_inspection_cost_usd,
        args.fn_miss_cost_usd,
    ):
        collapsed = collapse_alert_episodes(alerts, args.episode_gap_days)
        row = evaluate_event_alerts(
            model_name,
            collapsed,
            events_test,
            args.lead_window_days,
            args.fp_inspection_cost_usd,
            args.fn_miss_cost_usd,
        )
        results.append(MethodMetrics(**{**row.__dict__, "zero_shot_ready": "No"}))
        threshold_rows.append(
            {
                "method": model_name,
                "threshold_tuned_on": "validation_EMC",
                "best_threshold": best_t,
                "val_emc_usd": best_val_emc,
                "val_events": len(events_val),
            }
        )

    table = pd.DataFrame([r.__dict__ for r in results]).sort_values(
        ["expected_maintenance_cost_usd", "f1"], ascending=[True, False]
    )
    table.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    thr_path = OUTPUT_DIR / "table11_supervised_thresholds.csv"
    pd.DataFrame(threshold_rows).to_csv(thr_path, index=False, encoding="utf-8-sig")

    lines = [
        "# Table 11 Extended — Supervised Baselines, Synthetic Faults, and Cost-Sensitive Evaluation",
        "",
        f"- Synthetic events injected: {len(synthetic_events)}",
        f"- Chronological split: train ≤ {train_cut.date()} "
        f"(frac={args.train_frac:.2f}), val ≤ {val_cut.date()} "
        f"(+{args.val_frac:.2f}), test after {val_cut.date()}",
        f"- Events: train={len(events_train)}, val={len(events_val)}, test={len(events_test)}",
        f"- Supervised thresholds: fit on train; tuned on **validation EMC only**; reported on test",
        f"- Lead window: {args.lead_window_days} days",
        f"- Cost model: FP=${args.fp_inspection_cost_usd:.2f}, FN=${args.fn_miss_cost_usd:.2f}",
        "",
        "| Method | Zero-shot Ready | Precision | Recall | F1 | Avg early (d) | Med early (d) | Expected Maintenance Cost (USD) | Detected/Total Events | Alert Episodes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in table.iterrows():
        avg_early = "N/A" if pd.isna(row["avg_early_days"]) else f"{row['avg_early_days']:.2f}"
        med_early = "N/A" if pd.isna(row["med_early_days"]) else f"{row['med_early_days']:.2f}"
        lines.append(
            f"| {row['method']} | {row['zero_shot_ready']} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {avg_early} | {med_early} | {row['expected_maintenance_cost_usd']:,.2f} | "
            f"{int(row['detected_events'])}/{int(row['total_events'])} | {int(row['alert_episodes'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Supervised models are fit on the train window; decision thresholds minimize validation EMC and are frozen before test evaluation.",
            "- Zero-shot methods (SolarGuard / Isolation Forest) need no labeled failures for threshold selection.",
            "- Cost-sensitive ranking highlights operational value directly: lower expected maintenance cost is better.",
            "",
            f"Supervised threshold audit: `{thr_path.relative_to(PROJECT_ROOT)}`",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Table 11 CSV -> {OUTPUT_CSV}")
    print(f"Table 11 Markdown -> {OUTPUT_MD}")
    print(f"Threshold audit -> {thr_path}")
    print(
        f"Split dates: train≤{train_cut.date()}, val≤{val_cut.date()}, "
        f"events train/val/test={len(events_train)}/{len(events_val)}/{len(events_test)}"
    )
    print()
    print(table.to_string(index=False))
    if threshold_rows:
        print()
        print(pd.DataFrame(threshold_rows).to_string(index=False))


if __name__ == "__main__":
    main()
