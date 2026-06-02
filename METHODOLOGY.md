# Methodology

This document describes the methodology used to transform inverter monitoring data, calculate daily risk scores, identify replacement candidates, and evaluate the pipeline without ground-truth failure labels. It is written so the content can be adapted into a research paper.

## 1. Data Sources

The pipeline uses three main data sources:

- inverter generation records from the Shundao 1 and Shundao 2 plants
- irradiation measurements from weather station files
- installed inverter capacity from the installed-capacity workbook

The inverter and irradiation records are provided at sub-daily resolution. The final modelling unit is one inverter per day.

## 2. Data Preprocessing

Raw inverter files report `Daily energy(kWh)` as a cumulative counter that resets each day. Therefore, interval-level energy is recovered by differencing consecutive cumulative values within the same inverter and date:

```text
interval_energy_t = cumulative_energy_t - cumulative_energy_(t-1)
```

For the first interval of each inverter-day, the raw cumulative value is retained. Negative differences are clipped to zero to reduce the effect of counter resets, missing records, or logging artifacts:

```text
interval_energy_t = max(interval_energy_t, 0)
```

Daily inverter yield is then calculated as:

```text
yield_kwh = sum(interval_energy_t)
```

Irradiation data is processed differently depending on the source format:

- For instantaneous irradiance in `W/m2`, each 15-minute value is converted to energy:

```text
irradiation_kwh_m2_t = irradiance_w_m2_t * 0.25 / 1000
```

- For cumulative daily irradiation in `kWh/m2`, the same differencing method used for inverter cumulative energy is applied.

Daily irradiation is then aggregated by date. If multiple sensors are available for the same day, the daily value is averaged across sensors.

## 3. Performance Ratio Calculation

The main normalized performance metric is the daily Performance Ratio (PR). For each inverter-day:

```text
PR = yield_kwh / (irradiation_kwh_m2 * installed_capacity_kwp)
```

This normalizes energy production by solar resource and installed DC capacity, allowing inverters of different capacities to be compared.

Rows with missing capacity, missing irradiation, zero irradiation, or undefined yield cannot produce a valid PR. These rows are retained in the transformed dataset for traceability, but they are excluded from PR-based risk scoring.

## 4. Peer-Relative Daily Performance

For each valid scoring date, the fleet median PR is calculated:

```text
median_PR_date = median(PR_i,date)
```

Each inverter's relative PR is then calculated as:

```text
relative_PR_i,date = PR_i,date / median_PR_date
```

The relative PR is clipped to the interval `[0, 2]` to limit the effect of extreme outliers. A peer deficit score is calculated as:

```text
peer_deficit_i,date = max(1 - relative_PR_i,date, 0)
```

This converts peer underperformance into a non-negative signal, where larger values indicate greater underperformance relative to the same-day fleet.

## 5. Daily Rolling Risk Score

The risk score is calculated independently for each valid date. For a given scoring date, only data available up to that date is used. This avoids leakage from future observations.

For each inverter and scoring date, the pipeline calculates rolling features over 14-day and 30-day windows:

- valid days in the window
- mean PR
- mean relative PR
- mean peer deficit
- moderate-low PR rate
- severe-low PR rate

Moderate-low and severe-low flags are defined as:

```text
moderate_low = relative_PR < 0.70 or PR < 0.40
severe_low   = relative_PR < 0.50 or PR < 0.25
```

A 30-day relative-PR trend is estimated using a linear slope:

```text
relative_PR_t = alpha + beta * day_t
```

The deterioration component is:

```text
trend_decline_30d = clip(-beta * 30, 0, 1)
```

The final risk score is a weighted combination of recent underperformance, severe-low frequency, trend deterioration, and latest peer deficit:

```text
risk_score = 100 * (
    0.25 * mean_deficit_14d
  + 0.20 * mean_deficit_30d
  + 0.20 * severe_low_rate_14d
  + 0.15 * severe_low_rate_30d
  + 0.10 * trend_decline_30d
  + 0.10 * latest_peer_deficit
)
```

The score is clipped to `[0, 100]`. A higher score indicates stronger evidence of persistent underperformance.

## 6. Replacement Candidate Classification

Replacement candidates are assigned using threshold rules over the daily risk score and rolling underperformance indicators.

The strongest alert class is `within_14_days`. It requires a high risk score and strong short-term evidence, such as high severe-low frequency, high 14-day peer deficit, or a very low projected relative PR.

The second alert class is `within_30_days`. It requires elevated risk and broader 30-day evidence.

The remaining classes are:

- `monitor`: moderate risk that should be observed
- `no_replacement_signal`: no replacement signal under current thresholds

These classes should be interpreted as triage levels rather than confirmed failure labels.

## 7. Per-Date Output Structure

Because risk is calculated daily, the outputs are separated by scoring date:

```text
input_data/2_years/daily/YYYY-MM-DD/
  risk_scores.csv
  replacement_candidates.csv
  risk_watchlist.csv
```

Aggregate files are also retained for full-period analysis:

```text
input_data/2_years/risk_scores.csv
input_data/2_years/replacement_candidates.csv
input_data/2_years/risk_watchlist.csv
```

This structure supports both day-by-day operational review and long-term retrospective analysis.

## 8. STL Decomposition of Daily Mean PR

The pipeline uses STL decomposition to analyze the historical daily mean PR series. STL separates the signal into:

```text
observed_PR = trend + seasonal + residual
```

The weekly seasonality period is set to 7 days. The fitted STL signal is:

```text
fitted_PR = trend + seasonal
```

The STL fit is evaluated against a naive baseline that always predicts the historical mean PR. RMSE and MAE are calculated for both methods:

```text
RMSE = sqrt(mean((observed_PR - fitted_PR)^2))
MAE  = mean(abs(observed_PR - fitted_PR))
```

The STL RMSE measures how well the STL model fits historical daily mean PR. It does not measure the accuracy of the inverter risk score, replacement recommendation, or Isolation Forest anomaly detection.

## 9. Risk-Based Forecasting

The forecast component uses each inverter's latest risk score to produce a simple planning estimate of future high-risk failure events. The latest score is converted to a daily hazard:

```text
daily_hazard_i = base_hazard * risk_score_i / 100
```

For forecast day `t`, the expected number of new events is:

```text
E[new_events_t] = sum_i daily_hazard_i * (1 - daily_hazard_i)^(t-1)
```

The cumulative expected events by day `T` are:

```text
E[cumulative_events_T] = sum_i (1 - (1 - daily_hazard_i)^T)
```

This forecast is not a calibrated failure-probability model unless ground-truth failure labels are available. Without labels, it should be interpreted as a relative planning signal.

## 10. Label-Free Evaluation

Because confirmed failure or replacement labels are not currently available, the pipeline is evaluated using label-free diagnostics. These diagnostics do not estimate true prediction accuracy; instead, they assess data quality, score stability, and consistency of candidate flags.

The no-label evaluation includes:

- data-quality checks
- missing PR and missing irradiation summaries
- extreme PR checks
- zero-yield under high-irradiation checks
- daily risk summaries
- candidate persistence analysis
- diagnostic plots for candidate devices

Candidate persistence is especially important. A single candidate flag may be noise or a transient operating condition. Repeated candidate flags, such as at least 3 candidate days within a 5-day window, provide stronger evidence of sustained underperformance.

## 11. Candidate Diagnostic Plots

For each high-priority candidate device, the pipeline generates diagnostic plots containing:

- daily PR
- relative PR versus fleet median
- daily risk score
- replacement-candidate dates

These plots support manual review by showing whether the candidate signal is caused by a sustained inverter-specific decline or by short-term noise.

## 12. Interpretation Without Ground-Truth Labels

Without ground-truth labels, the pipeline should be interpreted as an unsupervised or weakly supervised risk-ranking system. It can identify unusual and persistent underperformance, but it cannot prove that a device failed.

The strongest no-label evidence is:

- persistent low relative PR
- repeated severe-low days
- increasing risk score over time
- repeated candidate flags over consecutive or near-consecutive days
- underperformance that affects one inverter rather than the entire fleet

The weakest evidence is:

- one-day candidate flags
- low PR during missing or unreliable irradiation data
- fleet-wide low PR events
- candidate flags during known curtailment or maintenance periods

## 13. Limitations

The main limitations are:

- no confirmed failure labels are currently used
- the risk score is not calibrated as a probability
- the forecast is a planning signal, not a verified reliability model
- irradiation gaps limit PR-based scoring dates
- data-quality issues can create false candidate signals
- plant-wide events may be mistaken for inverter-specific issues if not explicitly filtered

## 14. Recommended Future Validation

If maintenance, alarm, downtime, or replacement records become available, they should be joined to the daily risk outputs. The following supervised validation metrics can then be calculated:

- precision
- recall
- F1 score
- false positive rate
- lead time before confirmed failure
- ROC-AUC or PR-AUC
- calibration between risk score and observed failure probability

This would allow the current risk-ranking pipeline to be converted into a validated predictive-maintenance model.
