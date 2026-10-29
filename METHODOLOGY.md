# Methodology

This document describes the full AI O&M decision pipeline used in this project:

```text
raw data -> daily risk score -> replacement forecast -> reorder signal / optimal N
         -> financial NPV / IRR summary -> weekly_report.pdf
```

The methodology is designed for operational decision support. The model identifies persistent inverter underperformance, estimates future replacement demand, recommends a spare-inverter stock level, and converts the resulting operational benefit into a financial summary.

## 1. Data Sources

The pipeline uses these data sources:

- inverter generation records from Shundao 1 and Shundao 2
- irradiation measurements from weather station files
- installed inverter capacity from `data_2_years/installed_capacity.xlsx`
- financial assumptions from `input_data/Financial_Assumptions.xlsx`
- inventory assumptions from `config/spare_inventory.json`

The modelling unit is one inverter per day.

## 2. Data Preprocessing

Raw inverter files report `Daily energy(kWh)` as a cumulative counter that resets each day. Interval-level energy is recovered by differencing consecutive cumulative values within each inverter-day:

```text
interval_energy_t = cumulative_energy_t - cumulative_energy_(t-1)
```

Negative differences are clipped to zero to reduce the effect of counter resets and logging artifacts:

```text
interval_energy_t = max(interval_energy_t, 0)
```

Daily inverter yield is:

```text
yield_kwh = sum(interval_energy_t)
```

Irradiation data is processed by source format:

- instantaneous irradiance in `W/m2` is converted to 15-minute energy:

```text
irradiation_kwh_m2_t = irradiance_w_m2_t * 0.25 / 1000
```

- cumulative irradiation in `kWh/m2` is differenced using the same method as inverter cumulative energy.

Daily irradiation is aggregated by date. If multiple sensors are available for a day, the daily value is averaged.

The cleaned base dataset is stored at:

```text
input_data/base/transformed.csv
```

## 3. Performance Ratio

The normalized performance metric is daily Performance Ratio:

```text
PR = yield_kwh / (irradiation_kwh_m2 * installed_capacity_kwp)
```

Rows with missing capacity, missing irradiation, zero irradiation, or undefined yield are retained for traceability but excluded from PR-based risk scoring.

## 4. Peer-Relative Performance

For each scoring date, the fleet median PR is calculated:

```text
median_PR_date = median(PR_i,date)
```

Each inverter's relative PR is:

```text
relative_PR_i,date = PR_i,date / median_PR_date
```

Relative PR is clipped to `[0, 2]`. Peer deficit is:

```text
peer_deficit_i,date = max(1 - relative_PR_i,date, 0)
```

This converts underperformance into a non-negative signal, where larger values indicate worse performance versus same-day fleet peers.

## 5. Daily Risk Score

Risk is calculated independently for each valid date using only observations available up to that date.

For each inverter and scoring date, the pipeline calculates 14-day and 30-day rolling features:

- valid days in window
- mean PR
- mean relative PR
- mean peer deficit
- moderate-low PR rate
- severe-low PR rate

The flags are:

```text
moderate_low = relative_PR < 0.70 or PR < 0.40
severe_low   = relative_PR < 0.50 or PR < 0.25
```

A 30-day relative-PR trend is estimated:

```text
relative_PR_t = alpha + beta * day_t
trend_decline_30d = clip(-beta * 30, 0, 1)
```

The daily risk score is:

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

The score is clipped to `[0, 100]`. Higher scores indicate stronger evidence of persistent underperformance.

Risk outputs:

```text
input_data/risk/risk_scores.csv
input_data/risk/replacement_candidates.csv
input_data/risk/risk_watchlist.csv
input_data/risk/daily/YYYY-MM-DD/
  risk_scores.csv
  replacement_candidates.csv
  risk_watchlist.csv
```

## 6. Replacement Candidate Classification

Replacement candidates are assigned using daily risk score and rolling underperformance indicators.

The `within_14_days` class requires high risk and strong short-term evidence such as:

- high severe-low rate in the last 14 days
- high 14-day peer deficit
- very low projected relative PR

The `within_30_days` class captures elevated risk with broader 30-day evidence.

Other classes:

- `monitor`
- `no_replacement_signal`

These classes are triage signals, not confirmed hardware-failure labels.

## 7. Forecasting

The forecast stage uses `preprocess/stl_forecast_daily.py`.

It has two components:

1. STL decomposition of daily mean PR.
2. Risk-score-based expected future replacement events.

The STL model decomposes daily mean PR:

```text
observed_PR = trend + seasonal + residual
```

Weekly seasonality is set to 7 days. The STL fit is compared to a naive historical-mean baseline using MAE and RMSE.

The replacement forecast converts each inverter's latest risk score into a daily hazard:

```text
daily_hazard_i = base_hazard * risk_score_i / 100
```

Expected new events on forecast day `t`:

```text
E[new_events_t] = sum_i daily_hazard_i * (1 - daily_hazard_i)^(t-1)
```

Cumulative expected events by day `T`:

```text
E[cumulative_events_T] = sum_i (1 - (1 - daily_hazard_i)^T)
```

Forecast outputs:

```text
input_data/forecast/forecast_table.csv
output_data/figures/forecast.png
output_data/figures/stl_decomposition.png
```

The forecast is a planning signal. It is not a calibrated failure-probability model unless confirmed failure labels become available.

## 8. Reorder Signal And Optimal Spare Stock

The reorder and inventory-cost stage uses `preprocess/spare_inventory_cost.py`.

The model compares:

1. expected generation loss when replacement candidates are not pre-stocked
2. holding cost for keeping `N` spare inverters

Expected generation-loss cost:

```text
generation_loss_vnd =
    lead_time_days
  * generation_hours_per_day
  * lost_capacity_kw
  * electricity_price_vnd_per_kwh
```

Holding cost:

```text
holding_cost_vnd = N * holding_cost_vnd_per_unit
```

Total cost:

```text
total_cost_vnd = expected_generation_loss_vnd + holding_cost_vnd
```

The optimal spare stock level is:

```text
optimal_N = argmin_N(total_cost_vnd)
```

Current output:

```text
input_data/inventory/spare_inventory_cost_summary.csv
input_data/inventory/spare_inventory_recommendation.md
output_data/figures/inventory/spare_inventory_cost_curve.png
```

In the current run, the optimal stock level is `N = 6`.

## 9. Financial Assumptions

Financial assumptions are read from:

```text
input_data/Financial_Assumptions.xlsx
```

The financial script uses the workbook for assumptions only. Operational savings are derived from `input_data`.

Main financial assumptions:

- `scenario_case`
- `discount_rate_annual`
- `electricity_price_vnd_per_kwh`
- `fx_vnd_usd`
- `actual_data_cutoff_date`
- scenario multipliers
- P90 FM baseline cashflow source references

Operational assumptions retained in the workbook are also used where needed:

- lead time from China
- installation time
- generation hours per day
- inverter unit cost
- current spare stock
- holding-cost rate
- AI recurring monthly cost

The assumption classification output is:

```text
output_data/financial/assumptions_variable_fixed.csv
```

This table should be reviewed by the finance team to confirm which assumptions are fixed and which are variables for sensitivity analysis.

## 10. Financial NPV And IRR

The financial stage uses `preprocess/financial_npv_irr.py`.

The script builds monthly AI savings from:

- replacement candidates from `input_data/risk/daily`
- inverter capacity from `input_data/base/transformed.csv`
- electricity price from the assumption workbook
- holding-cost change from inventory assumptions
- AI recurring monthly cost
- AI CAPEX scenario override or workbook value

Monthly net AI cashflow:

```text
net_ai_cashflow_month =
    avoided_generation_loss_vnd
  + holding_cost_saved_monthly
  - ai_recurring_cost_monthly
  - spare_purchase_cost_vnd
```

Month-zero cashflow:

```text
cashflow_0 = -capex_ai
```

NPV:

```text
NPV = sum_t cashflow_t / (1 + monthly_discount_rate)^t
```

where:

```text
monthly_discount_rate = (1 + discount_rate_annual)^(1/12) - 1
```

IRR is calculated from the cashflow series:

```text
[-capex_ai, month_1_saving, month_2_saving, ...]
```

IRR is only defined when the cashflow series contains at least one negative and one positive value.

Financial outputs:

```text
output_data/financial/monthly_operational_savings.csv
output_data/financial/npv_irr_comparison.csv
output_data/financial/npv_timeline.csv
output_data/financial/npv_timeline.png
output_data/financial/missing_financial_inputs.csv
output_data/financial/ai_financial_effectiveness_summary.md
```

Current scenario uses:

```text
capex_ai = 3,000 USD * 25,500 VND/USD = 76,500,000 VND
```

Current financial result:

```text
NPV = 683,972,080 VND
Monthly IRR = 158.87%
Break-even CAPEX = 760,472,080 VND
```

The annual effective IRR is mathematically reported, but it should not be emphasized because the CAPEX is small and benefits arrive over a short observed period.

## 11. Weekly Report Pipeline

The full report pipeline is orchestrated by:

```text
scripts/full_weekly_pipeline.py
```

The command for a full run is:

```powershell
.\.venv\Scripts\python.exe scripts\full_weekly_pipeline.py --capex-ai-usd 3000
```

The script runs these stages:

1. `preprocess/risk_daily.py`
2. `preprocess/stl_forecast_daily.py`
3. `preprocess/spare_inventory_cost.py`
4. `preprocess/financial_npv_irr.py`
5. PDF report generation
6. appendix screenshot generation

The report can be rebuilt from existing outputs without rerunning the heavy model stages:

```powershell
.\.venv\Scripts\python.exe scripts\full_weekly_pipeline.py --skip-run
```

To force the report to use a specific latest week:

```powershell
.\.venv\Scripts\python.exe scripts\full_weekly_pipeline.py --skip-run --report-end-date 2026-04-30
```

Report outputs:

```text
output_data/weekly_report/weekly_report.pdf
output_data/weekly_report/appendix_week_screenshot.png
```

The PDF includes:

- pipeline run status
- one-week operational summary
- 14-day failure forecast
- reorder / optimal-N table
- NPV and IRR table
- open-input caveats
- forecast chart
- STL decomposition chart
- optimal-N cost curve
- NPV timeline chart
- appendix screenshot

## 12. One-Week Appendix

The appendix screenshot is generated from one real week of risk output.

If `--report-end-date` is provided, the appendix covers:

```text
report_end_date - 6 days  through  report_end_date
```

If no date is provided, the report can use the latest available candidate week or the latest scored week depending on the command settings.

For the latest scored week in the current data:

```text
2026-04-24 to 2026-04-30
```

there are no replacement candidates. The appendix correctly reports that no replacement candidates were found in that selected week.

For a week with actual candidate activity, the appendix uses:

```text
2025-07-11 to 2025-07-17
```

and lists the top replacement-candidate rows.

## 13. Label-Free Evaluation

Confirmed failure or replacement labels are not currently used. Therefore, the pipeline is evaluated using label-free diagnostics rather than supervised accuracy.

Useful diagnostics include:

- data-quality checks
- missing PR and missing irradiation summaries
- extreme PR checks
- zero-yield under high-irradiation checks
- daily risk summaries
- candidate persistence analysis
- diagnostic plots for candidate devices

Candidate persistence is especially important. A single candidate flag may be noise, while repeated candidate flags over consecutive or near-consecutive days provide stronger evidence of sustained underperformance.

## 14. Interpretation

The risk score should be interpreted as a triage and ranking signal, not a confirmed failure diagnosis.

The strongest evidence is:

- persistent low relative PR
- repeated severe-low days
- increasing risk score over time
- repeated candidate flags
- underperformance affecting one inverter rather than the full fleet

The weakest evidence is:

- one-day candidate flags
- low PR during irradiation gaps
- fleet-wide low PR events
- candidate flags during known curtailment or maintenance

The financial output should be interpreted as a scenario-based business case. It is strongest when:

- AI CAPEX is confirmed by finance
- full 24-month operational data is available
- baseline downtime cost is confirmed
- FM baseline cashflows are pasted or linked from the P90 FM

## 15. Current Limitations

The main limitations are:

- no confirmed failure labels are currently used
- risk score is not calibrated as a probability
- forecast is a planning signal, not a verified reliability model
- latest scored week may contain no replacement candidates
- input-data candidate savings currently cover fewer than the requested 24 months
- official AI CAPEX still needs finance confirmation
- baseline downtime cost still needs finance or O&M confirmation
- plant-wide operational events may be mistaken for inverter-specific underperformance if not explicitly filtered

## 16. Recommended Future Validation

If maintenance, alarm, downtime, or replacement records become available, they should be joined to daily risk outputs.

Recommended supervised validation metrics:

- precision
- recall
- F1 score
- false positive rate
- lead time before confirmed failure
- ROC-AUC or PR-AUC
- calibration between risk score and observed failure probability

This would allow the current risk-ranking pipeline to become a validated predictive-maintenance model.
