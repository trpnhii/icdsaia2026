# Pipeline Evaluation

This document explains how to evaluate each component of the inverter-risk pipeline and what each metric means. It also separates model-fit checks from true prediction accuracy, because not every component has ground-truth labels available.

## 1. Data Transformation

Script:

```powershell
.\.venv\Scripts\python.exe preprocess\transform_2years.py
```

Inputs:

- `data_2_years/`
- `data_2_years/installed_capacity.xlsx`
- inverter Excel files
- irradiation Excel files

Outputs:

- `input_data/2_years/inverter_daily.csv`
- `input_data/2_years/irradiation_daily.csv`
- `input_data/2_years/transformed.csv`

Purpose:

Convert raw 15-minute inverter and irradiation files into daily inverter-level records. The main calculated field is `performance_ratio`, defined as:

```text
performance_ratio = yield_kwh / (irradiation_kwh_m2 * installed_capacity_kwp)
```

Evaluation checks:

- Row count should match expected inverter-days.
- Date range should match the available inverter files.
- `installed_capacity_kwp` should be missing only for non-replaceable logger rows such as `Inverter(COM...)`.
- `irradiation_kwh_m2` should be present only where irradiation source data exists.
- `performance_ratio` should be missing when capacity or irradiation is missing.
- Very high or very low PR values should be reviewed as possible sensor, data-quality, curtailment, or outage cases.

Current known limitation:

Irradiation data currently ends at `2026-04-30`, so PR-based downstream scoring also ends on `2026-04-30` even though inverter data continues beyond that date.

## 2. Daily Risk Scoring

Script:

```powershell
.\.venv\Scripts\python.exe preprocess\risk_2years.py
```

Inputs:

- `input_data/2_years/transformed.csv`

Aggregate outputs:

- `input_data/2_years/risk_scores.csv`
- `input_data/2_years/replacement_candidates.csv`
- `input_data/2_years/risk_watchlist.csv`

Daily outputs:

- `input_data/2_years/daily/YYYY-MM-DD/risk_scores.csv`
- `input_data/2_years/daily/YYYY-MM-DD/replacement_candidates.csv`
- `input_data/2_years/daily/YYYY-MM-DD/risk_watchlist.csv`

Purpose:

Calculate one risk score per inverter per valid date. Each score uses only information available up to that `risk_date`, not the entire 2-year period.

Main signals:

- recent peer underperformance over 14 days
- recent peer underperformance over 30 days
- severe-low PR frequency
- latest peer deficit
- 30-day deterioration trend

Important columns:

- `risk_date`: date being scored
- `risk_score`: absolute risk score from 0 to 100
- `fleet_relative_risk_score`: percentile rank within the fleet on that date
- `replacement_window`: `within_14_days`, `within_30_days`, `monitor`, or `no_replacement_signal`
- `replacement_candidate`: boolean flag derived from the replacement window
- `mean_relative_pr_14d`, `mean_relative_pr_30d`: recent PR compared with same-day fleet median
- `severe_low_rate_14d`, `severe_low_rate_30d`: fraction of recent days with severe underperformance

Evaluation checks:

- Confirm one folder exists for each scored date.
- Confirm daily `risk_scores.csv` contains only one `risk_date`.
- Confirm daily `replacement_candidates.csv` is empty on normal days and populated only when thresholds are met.
- Inspect candidate rows for very low `mean_relative_pr_14d` or high `severe_low_rate_14d`.
- Check whether repeated candidate-days correspond to known outages, maintenance, curtailment, or device failure.

What this evaluates:

The daily risk scoring evaluates relative operational underperformance. It is a triage score, not a confirmed failure label.

What it does not evaluate:

It does not prove that a device failed. It flags inverters that look persistently weak compared with same-day fleet peers.

## 3. Replacement Candidate Logic

Replacement candidates are selected from daily risk scores using threshold rules.

Current logic:

- `within_14_days` requires high risk and strong short-term evidence.
- `within_30_days` requires elevated risk and broader 30-day evidence.
- `monitor` is used for moderate risk.
- `no_replacement_signal` is used when the score does not cross the alert threshold.

Evaluation checks:

- Review days with `within_14_days` candidates first.
- Confirm whether candidate periods cluster around actual abnormal operation.
- Compare candidate devices against raw yield and PR histories.
- Check if candidate flags disappear after the device returns to normal performance.

Recommended ground-truth validation:

If maintenance or fault records are available, compare candidate dates against:

- confirmed inverter failures
- replacement dates
- repair dates
- downtime logs
- alarm logs

Useful validation metrics with labels:

- precision: how many flagged candidates were true issues
- recall: how many true issues were caught
- F1 score: balance of precision and recall
- lead time: how many days before failure the pipeline flagged the inverter
- false positive rate: how often normal devices were flagged

## 4. STL Decomposition Figure

Script:

```powershell
.\.venv\Scripts\python.exe preprocess\stl_forecast_2years.py
```

Inputs:

- `input_data/2_years/transformed.csv`
- `input_data/2_years/risk_scores.csv`

Outputs:

- `output/figures/2_years/stl_decomposition.png`
- `input_data/2_years/forecast_table.csv`

Purpose:

Decompose daily mean PR into:

- observed PR
- trend
- seasonal component
- residual component

STL is used to understand the shape of the historical PR series and to produce a simple PR forecast for the forecast table.

STL RMSE meaning:

STL RMSE measures how well the STL fitted series matches historical daily mean PR:

```text
actual daily mean PR vs. STL fitted PR
```

It is compared with a naive baseline that always predicts the historical average PR.

What STL RMSE evaluates:

- historical PR curve fit
- whether trend plus seasonality explains the daily mean PR better than a simple average

What STL RMSE does not evaluate:

- Isolation Forest accuracy
- risk-score accuracy
- replacement-candidate correctness
- future failure prediction accuracy

## 5. Forecast Figure

Script:

```powershell
.\.venv\Scripts\python.exe preprocess\stl_forecast_2years.py
```

Outputs:

- `output/figures/2_years/forecast.png`
- `input_data/2_years/forecast_table.csv`

Purpose:

Use the latest daily risk scores to estimate expected future failures over the next 14 or 30 days.

Current model:

The script converts each inverter's latest `risk_score` into a small daily hazard:

```text
daily_hazard = BASE_HAZARD * risk_score / 100
```

Then it sums the expected failure probability across all inverters.

Evaluation checks:

- Confirm the forecast starts the day after the latest scored date.
- Confirm the expected failure count is consistent with the current risk level.
- Confirm high-risk periods in historical data appear as candidate bars.
- Review whether forecast magnitude is plausible for the fleet size.

Important limitation:

This is a risk-based forecast, not a calibrated reliability model. Without true failure labels, the expected failure counts should be interpreted as a planning signal, not a measured probability.

## 6. Isolation Forest Component

Script:

```powershell
.\.venv\Scripts\python.exe preprocess\iforest_risk.py
```

Inputs:

- `input_data/transformed_data.csv`

Outputs:

- `input_data/iforest_risk.csv`
- `input_data/anomaly_list.csv`

Purpose:

Detect unusual inverter-days using an unsupervised Isolation Forest model.

Evaluation checks:

- Inspect top anomalies manually.
- Compare anomaly dates with known operational events.
- Check whether anomalies are caused by data-quality issues, missing irradiation, true low yield, or plant-wide effects.

Important limitation:

Isolation Forest is unsupervised. Without labels, its output cannot be evaluated with standard accuracy metrics. The current STL RMSE is not an Isolation Forest accuracy metric.

Recommended validation if labels are available:

- Build a labeled table with `date`, `device_name`, and `true_failure_or_fault`.
- Join labels to `iforest_risk.csv`.
- Evaluate precision, recall, F1, ROC-AUC, and lead time.

## 7. End-to-End Evaluation Checklist

Use this checklist after every full pipeline run:

- `transform_2years.py` completes without errors.
- `transformed.csv` has expected date range and device count.
- Missing capacity is limited to expected non-inverter logger rows.
- Missing irradiation matches known source-data gaps.
- `risk_2years.py` writes aggregate and daily date-folder outputs.
- Daily folders contain `risk_scores.csv`, `replacement_candidates.csv`, and `risk_watchlist.csv`.
- Latest `risk_watchlist.csv` contains the highest-risk devices for the latest scored date.
- `stl_forecast_2years.py` writes both figures and `forecast_table.csv`.
- Forecast starts after the latest scored date.
- Candidate-days are reviewed against operational logs where available.

## 8. Current Validation Status

Currently validated:

- data transformation consistency checks
- daily rolling risk-score generation
- per-date output generation
- STL historical PR fit
- risk-based forecast generation

Not yet validated:

- true failure prediction accuracy
- replacement recommendation precision/recall
- calibrated probability of failure
- Isolation Forest accuracy against labeled faults

To validate true predictive performance, the pipeline needs ground-truth maintenance, alarm, downtime, or replacement records.
