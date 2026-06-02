# Evaluation Protocol

This section describes the evaluation strategy used for the proposed inverter-risk pipeline. Because confirmed inverter failure labels, replacement records, and maintenance annotations are not available in the current dataset, the evaluation is designed as a label-free assessment. The objective is not to measure supervised prediction accuracy, but to examine whether the pipeline produces internally consistent, interpretable, and operationally useful risk signals.

## 1. Evaluation Objective

The pipeline estimates daily inverter-level risk from historical energy production, irradiation, installed capacity, and peer-relative performance. The evaluation has four objectives:

1. Assess the completeness and reliability of the transformed daily dataset.
2. Verify that daily risk scores are generated without future data leakage.
3. Examine the persistence and stability of replacement-candidate signals.
4. Evaluate whether the time-series decomposition and forecast components provide a reasonable summary of historical performance trends.

In the absence of ground-truth fault labels, the evaluation should be interpreted as a consistency and plausibility analysis rather than a measurement of true predictive accuracy.

## 2. Data-Quality Evaluation

The first evaluation stage examines the transformed inverter-day dataset. Data quality is critical because the downstream risk score depends on the daily Performance Ratio (PR), which requires valid yield, irradiation, and installed-capacity values.

For each date, the following quantities are computed:

- total inverter-day rows
- number of unique devices
- missing installed-capacity rows
- missing irradiation rows
- missing PR rows
- valid PR rate
- number of unusually high PR values
- number of zero-yield rows under non-trivial irradiation

The valid PR rate is defined as:

```text
valid_PR_rate = valid_PR_rows / total_rows
```

Rows with missing irradiation or missing capacity are retained in the transformed dataset for traceability but excluded from PR-based risk scoring. Dates with a low valid PR rate are considered unreliable for risk-score interpretation.

This stage is not a model-performance evaluation. It is a prerequisite check to determine whether the input data is sufficient for meaningful scoring.

## 3. Daily Risk-Score Evaluation

Risk scores are evaluated at the inverter-day level. For each scoring date, the model uses only observations available up to that date. This rolling design prevents look-ahead bias and better reflects how the system would operate in deployment.

For each day, the following summary statistics are calculated:

- number of scored inverters
- mean risk score
- median risk score
- 95th percentile risk score
- maximum risk score
- number of replacement candidates
- replacement-candidate rate

The candidate rate is defined as:

```text
candidate_rate = replacement_candidate_count / scored_inverter_count
```

These statistics allow the distribution of risk scores to be examined over time. A stable system should produce low candidate rates on normal days and concentrated candidate signals during periods of sustained inverter-specific underperformance.

## 4. Candidate Persistence Evaluation

Single-day candidate flags can be caused by noise, data-quality issues, transient shading, communication interruptions, curtailment, or short-term operational events. Therefore, candidate persistence is used as a stronger label-free indicator.

For each inverter, the evaluation counts:

- total candidate-days
- first candidate date
- last candidate date
- maximum risk score
- mean risk score on candidate-days
- number of persistent candidate days

A persistent candidate day is defined using a rolling 5-day window:

```text
persistent_candidate = candidate_days_last_5 >= 3
```

This rule identifies inverters that are repeatedly flagged within a short period. In the absence of labels, repeated candidate flags provide stronger evidence than isolated flags because they suggest sustained underperformance relative to the fleet.

The persistence evaluation does not prove hardware failure. It ranks candidates by consistency of abnormal behavior.

## 5. Candidate Diagnostic Evaluation

For each high-priority candidate device, diagnostic plots are generated to support manual inspection. Each plot includes:

- daily PR
- relative PR compared with the fleet median
- daily risk score
- candidate flag dates

The purpose of these plots is to determine whether the candidate signal is coherent. A strong candidate should show low PR, low relative PR, and elevated risk scores over the same period. A weaker candidate may show a short-lived spike or behavior explained by missing data.

The diagnostic plots are especially important in a label-free setting because they allow domain experts to assess whether the algorithmic ranking corresponds to plausible inverter behavior.

## 6. STL Decomposition Evaluation

The pipeline applies Seasonal-Trend decomposition using Loess (STL) to the daily mean PR series. The decomposition separates the observed signal into trend, seasonal, and residual components:

```text
observed_PR = trend + seasonal + residual
```

The fitted STL value is:

```text
fitted_PR = trend + seasonal
```

The STL fit is evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE):

```text
MAE  = mean(abs(observed_PR - fitted_PR))
RMSE = sqrt(mean((observed_PR - fitted_PR)^2))
```

The STL model is compared against a naive baseline that always predicts the historical mean PR. The percentage RMSE improvement is calculated as:

```text
RMSE_improvement = (1 - STL_RMSE / baseline_RMSE) * 100
```

This metric evaluates how well the STL decomposition explains historical daily mean PR. It does not evaluate replacement-candidate accuracy, risk-score accuracy, or failure-prediction accuracy.

## 7. Forecast Evaluation

The forecast component converts each inverter's latest risk score into a daily hazard value:

```text
daily_hazard_i = base_hazard * risk_score_i / 100
```

The expected number of new events on forecast day `t` is:

```text
E[new_events_t] = sum_i daily_hazard_i * (1 - daily_hazard_i)^(t-1)
```

The cumulative expected number of events by day `T` is:

```text
E[cumulative_events_T] = sum_i (1 - (1 - daily_hazard_i)^T)
```

In the current label-free setting, this forecast is evaluated only for internal consistency:

- the forecast must begin after the latest scored date
- the expected event count should increase monotonically in cumulative form
- the magnitude should be consistent with the current distribution of risk scores
- the forecast should be interpreted as a planning signal rather than a calibrated failure probability

Because no confirmed future failure labels are available, forecast accuracy cannot be measured directly.

## 8. Isolation Forest Evaluation

The repository also includes an Isolation Forest component for unsupervised anomaly detection. Isolation Forest produces anomaly scores without requiring labeled faults. However, because the method is unsupervised, standard supervised metrics such as accuracy, precision, recall, and F1 score cannot be computed unless ground-truth labels are added.

In the current evaluation, Isolation Forest output can be assessed only by:

- manual inspection of high-risk anomalies
- comparison with known operational events if available
- checking whether anomaly dates correspond to abnormal yield or PR behavior
- checking whether anomalies are caused by missing or unreliable data

The STL RMSE is not an Isolation Forest accuracy metric. It evaluates the fit of the STL model to the daily mean PR time series only.

## 9. Label-Free Evaluation Outputs

The no-label evaluation script produces the following artifacts:

```text
input_data/2_years/no_label_quality_summary.csv
input_data/2_years/no_label_risk_summary.csv
input_data/2_years/no_label_candidate_persistence.csv
input_data/2_years/no_label_evaluation.md
output/figures/2_years/candidate_diagnostics/
```

These outputs provide an auditable record of:

- data completeness
- missing PR and irradiation periods
- risk-score distribution over time
- candidate frequency
- persistent candidate devices
- device-level diagnostic behavior

## 10. Interpretation of Results

In a label-free setting, the strongest evidence for a meaningful risk signal is not a single high score, but repeated and coherent underperformance. A high-priority candidate should satisfy several conditions:

- low PR relative to the fleet median
- elevated peer deficit
- repeated severe-low days
- persistent candidate flags
- risk score that remains high over multiple days
- behavior localized to the inverter rather than the entire fleet

Conversely, candidate signals should be treated cautiously when they occur during:

- missing irradiation periods
- plant-wide low-performance events
- suspected curtailment
- communication gaps
- very low irradiation days
- isolated one-day spikes

## 11. Current Evaluation Summary

The current label-free evaluation produced the following summary:

- transformed inverter-day rows: 93,277
- risk-score rows: 79,902
- scored dates: 397
- scored date range: 2025-03-01 to 2026-04-30
- candidate-days: 52
- unique candidate devices: 15
- devices with persistent 3-of-5 candidate windows: 8

These results indicate that candidate flags are relatively sparse and concentrated in a small subset of devices. This is desirable for an operational triage system because it reduces the number of devices requiring manual inspection.

## 12. Threats to Validity

The evaluation has several limitations:

- No confirmed failure labels are available.
- Replacement-candidate correctness cannot be measured directly.
- The risk score is not calibrated as a probability of failure.
- The forecast is not validated against future failure events.
- Missing irradiation data limits the range of valid PR-based scoring.
- Data-quality issues may create false candidate signals.
- Fleet-wide operational events may be mistaken for inverter-specific underperformance.

These limitations mean that the current evaluation supports plausibility and consistency claims, but not definitive predictive-performance claims.

## 13. Recommended Supervised Validation

If ground-truth records become available, the evaluation should be extended with supervised validation. Useful labels include:

- confirmed inverter failure dates
- replacement dates
- maintenance records
- alarm logs
- downtime records
- operator-confirmed anomaly periods

With labels, the following metrics can be calculated:

- precision
- recall
- F1 score
- false positive rate
- false negative rate
- lead time before confirmed failure
- ROC-AUC
- PR-AUC
- calibration between risk score and observed failure probability

This would allow the pipeline to be evaluated as a predictive-maintenance model rather than only as a label-free risk-ranking system.
