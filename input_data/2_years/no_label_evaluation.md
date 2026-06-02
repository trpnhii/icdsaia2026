# No-Label Pipeline Evaluation

This report evaluates the pipeline without ground-truth failure labels. It checks data quality, risk stability, and candidate persistence. It does not measure true prediction accuracy.

## Dataset Summary

- Transformed rows: 93,277
- Risk-score rows: 79,902
- Candidate-days: 52
- Scored dates: 397
- Scored date range: 2025-03-01 to 2026-04-30
- Latest scored date: 2026-04-30

## Data-Quality Checks

- Dates with any missing PR: 457
- Dates with high PR rows above 1.2: 4
- Dates with zero yield during irradiation >= 0.5: 33

Worst data-quality dates:

| date | rows | missing_irradiation | missing_pr | high_pr_rows | zero_yield_high_irr_rows | valid_pr_rate |
| --- | --- | --- | --- | --- | --- | --- |
| 2025-09-13 | 219 | 219 | 219 | 0 | 0 | 0.0000 |
| 2025-09-14 | 219 | 219 | 219 | 0 | 0 | 0.0000 |
| 2025-09-15 | 219 | 219 | 219 | 0 | 0 | 0.0000 |
| 2025-09-16 | 219 | 219 | 219 | 0 | 0 | 0.0000 |
| 2025-09-17 | 219 | 219 | 219 | 0 | 0 | 0.0000 |
| 2025-09-18 | 219 | 219 | 219 | 0 | 0 | 0.0000 |
| 2025-09-19 | 219 | 219 | 219 | 0 | 0 | 0.0000 |
| 2025-09-20 | 219 | 219 | 219 | 0 | 0 | 0.0000 |
| 2025-09-21 | 219 | 219 | 219 | 0 | 0 | 0.0000 |
| 2025-09-22 | 219 | 219 | 219 | 0 | 0 | 0.0000 |

## Risk Stability

- Unique candidate devices: 15
- Devices with at least one persistent 3-of-5 candidate window: 8

Top persistent candidate devices:

| zone | device_name | candidate_days | persistent_days | max_risk_score | first_candidate_date | last_candidate_date |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | HF18 Inverter 1 | 9 | 9 | 92.5800 | 2025-06-27 | 2025-07-05 |
| 2 | HF17 Inverter 3 | 7 | 7 | 91.1500 | 2025-07-10 | 2025-07-16 |
| 2 | HF18 Inverter 2 | 6 | 6 | 90.2000 | 2025-06-29 | 2025-07-05 |
| 2 | HF17 Inverter 1 | 5 | 5 | 89.5100 | 2025-07-11 | 2025-07-15 |
| 2 | HF18 Inverter 3 | 5 | 5 | 87.6100 | 2025-06-30 | 2025-07-05 |
| 2 | HF17 Inverter 2 | 4 | 4 | 87.6000 | 2025-07-12 | 2025-07-15 |
| 2 | HF20 Inverter 1 | 3 | 3 | 83.9000 | 2025-05-15 | 2025-05-27 |
| 2 | HF18 Inverter 4 | 3 | 2 | 84.5200 | 2025-07-02 | 2025-07-05 |
| 2 | HF17 Inverter 4 | 2 | 0 | 84.5200 | 2025-07-14 | 2025-07-15 |
| 2 | HF18 Inverter 6 | 2 | 0 | 71.0600 | 2025-07-03 | 2025-07-05 |

## Latest Watchlist

| zone | device_name | risk_score | fleet_relative_risk_score | replacement_window | mean_relative_pr_14d | severe_low_rate_14d |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | HF4 Inverter 9 | 13.0800 | 100.0000 | no_replacement_signal | 0.8519 | 0.0714 |
| 1 | HF2 Inverter 8 | 11.1300 | 99.5400 | no_replacement_signal | 0.8953 | 0.0714 |
| 1 | HF4 Inverter 7 | 10.8200 | 99.0800 | no_replacement_signal | 0.9068 | 0.0714 |
| 1 | HF8 Inverter 7 | 10.4900 | 98.6200 | no_replacement_signal | 0.8928 | 0.0714 |
| 1 | HF8 Inverter 1 | 10.4000 | 98.1600 | no_replacement_signal | 0.8963 | 0.0714 |
| 1 | HF10 Inverter 3 | 10.2700 | 97.7000 | no_replacement_signal | 0.9113 | 0.0714 |
| 1 | HF4 Inverter 8 | 10.0500 | 97.2400 | no_replacement_signal | 0.9269 | 0.0714 |
| 1 | HF10 Inverter 4 | 9.7300 | 96.7700 | no_replacement_signal | 0.9135 | 0.0714 |
| 1 | HF9 Inverter 2 | 9.7100 | 96.3100 | no_replacement_signal | 0.9161 | 0.0714 |
| 1 | HF9 Inverter 4 | 9.6100 | 95.8500 | no_replacement_signal | 0.9189 | 0.0714 |

## Generated Artifacts

- Data-quality summary: `input_data\2_years\no_label_quality_summary.csv`
- Daily risk summary: `input_data\2_years\no_label_risk_summary.csv`
- Candidate persistence summary: `input_data\2_years\no_label_candidate_persistence.csv`
- Diagnostic plot folder: `output\figures\2_years\candidate_diagnostics`
- Diagnostic plots generated: 15

## Interpretation

Without labels, the strongest evidence is repeated and persistent underperformance relative to same-day fleet peers. A one-day candidate flag should be treated as a review item; repeated 3-of-5 candidate windows deserve higher priority.

This report should be compared with maintenance notes, alarms, curtailment records, and raw inverter behavior when those records become available.
