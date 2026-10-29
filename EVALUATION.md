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

## 11. Current Pipeline Performance

The current pipeline was evaluated on the 2-year dataset using the label-free protocol described above. The transformed dataset contains 93,277 inverter-day rows. PR-based risk scoring produced 79,902 inverter-day risk-score rows across 397 scored dates, covering the period from 2025-03-01 to 2026-04-30.

The scored date range ends on 2026-04-30 because irradiation data is available only up to that date. Although inverter data continues beyond this point, PR-based scoring requires irradiation and therefore cannot be computed reliably for later dates.

### 11.1 Data Completeness

The average valid PR rate across the transformed daily dataset is 0.8842, while the median valid PR rate is 0.9909. This indicates that most dates have nearly complete PR availability, although several periods contain missing irradiation or missing PR values. There are 49 dates with no valid PR rows, mainly corresponding to missing irradiation periods.

The label-free quality summary reported:

- dates with any missing PR: 457
- dates with PR above 1.20: 4
- dates with zero yield under irradiation >= 0.5 kWh/m2: 33

These findings indicate that the dataset is generally usable for daily scoring, but interpretation should exclude or carefully review periods with missing irradiation or abnormal PR values.

Relevant artifact:

```text
output/figures/2_years/evaluation/data_quality_timeline.png
```

### 11.2 Risk-Score Distribution

The daily risk-score distribution is strongly right-skewed. Most inverter-days have very low risk scores:

- mean risk score: 2.41
- median risk score: 1.18
- 95th percentile risk score: 7.97
- maximum risk score: 92.58

This behavior is desirable for a triage-oriented system because the majority of normal inverter-days remain low risk, while only a small number of inverter-days enter the high-risk tail.

The total replacement-candidate rate is:

```text
52 candidate-days / 79,902 scored inverter-days = 0.065%
```

This indicates that the candidate logic is selective rather than over-triggering across the fleet.

Relevant artifact:

```text
output/figures/2_years/evaluation/risk_score_distribution.png
```

### 11.3 Replacement-Candidate Performance

The pipeline generated 52 replacement candidate-days across 15 unique devices. Of these devices, 8 showed persistent candidate behavior under the 3-of-5-day persistence rule.

The top persistent candidates were:

| zone | device_name | candidate_days | persistent_days | max_risk_score | candidate period |
| --- | --- | ---: | ---: | ---: | --- |
| 2 | HF18 Inverter 1 | 9 | 9 | 92.58 | 2025-06-27 to 2025-07-05 |
| 2 | HF17 Inverter 3 | 7 | 7 | 91.15 | 2025-07-10 to 2025-07-16 |
| 2 | HF18 Inverter 2 | 6 | 6 | 90.20 | 2025-06-29 to 2025-07-05 |
| 2 | HF17 Inverter 1 | 5 | 5 | 89.51 | 2025-07-11 to 2025-07-15 |
| 2 | HF18 Inverter 3 | 5 | 5 | 87.61 | 2025-06-30 to 2025-07-05 |

The candidate signals are concentrated in Zone 2 and occur in clustered periods, especially around late June and July 2025. This suggests that the pipeline is not randomly flagging devices across the full fleet; instead, it identifies localized and temporally coherent underperformance episodes.

Relevant artifacts:

```text
output/figures/2_years/evaluation/daily_candidate_count.png
output/figures/2_years/evaluation/candidate_persistence.png
output/figures/2_years/evaluation/zone_candidate_comparison.png
output/figures/2_years/evaluation/risk_heatmap_top_candidates.png
```

### 11.4 Internal Consistency of Risk Signals

The relationship between risk score and latest relative PR was examined using a scatter plot. The expected behavior is that high-risk inverter-days should correspond to low relative PR, since the scoring method is based on peer-relative underperformance.

The generated risk-versus-relative-PR plot supports this internal consistency check: high-risk points are concentrated at lower relative PR values, while most normal inverter-days remain at low risk.

Relevant artifact:

```text
output/figures/2_years/evaluation/risk_vs_relative_pr.png
```

### 11.5 Time-Series Fit and Forecast Output

The STL decomposition was applied to the daily mean PR series. The STL model achieved an RMSE improvement of 48.1% compared with the naive mean baseline. This indicates that the trend and weekly seasonal components explain the historical PR series better than a constant-average model.

The latest scored date is 2026-04-30. The risk-based forecast covers 2026-05-01 to 2026-05-30. Under the current risk distribution, the forecast estimates:

- expected events in 14 days: 0.8
- expected events in 30 days: 1.7
- forecast risk level: Low

These values are consistent with the low latest risk scores observed across the fleet. However, because ground-truth failure labels are unavailable, the forecast should be interpreted as a planning indicator rather than a calibrated failure probability.

Relevant artifacts:

```text
output/figures/2_years/stl_decomposition.png
output/figures/2_years/forecast.png
input_data/2_years/forecast_table.csv
```

### 11.6 Overall Assessment

The current pipeline performs well as a label-free risk-ranking and triage system. It produces sparse candidate flags, identifies persistent underperformance in a small subset of devices, and avoids broad over-flagging across the fleet. The strongest current evidence is the concentration of repeated candidate flags in specific Zone 2 inverters and the coherence between low relative PR and high risk score.

The current results support the claim that the pipeline can identify persistent inverter underperformance without labels. They do not support claims of confirmed failure prediction accuracy, precision, recall, or calibrated probability of failure. Those claims require ground-truth fault, maintenance, or replacement records.

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
