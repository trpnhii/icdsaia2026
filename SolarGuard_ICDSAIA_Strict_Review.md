# Strict Review of SolarGuard for ICDSAIA 2026

## Target Venue

**Conference:** ICDSAIA 2026  
**Track:** AI Applications for Industry and Society  
**Paper:** *SolarGuard: An End-to-End Decision Support System Linking Unsupervised Anomaly Detection to Inventory and Financial Optimization in PV Operations*

---

# 1. Overall Assessment

SolarGuard is highly relevant to the **AI Applications for Industry and Society** track because it connects artificial intelligence with real industrial operations, spare-part logistics, renewable-energy infrastructure, and financial decision-making.

The paper is stronger as an **applied AI and industrial decision-support contribution** than as a machine-learning methodology paper. Its primary novelty is not a new anomaly-detection algorithm, but the integration of:

1. label-free anomaly detection,
2. inventory decision support, and
3. financial-impact analysis.

The current version is promising but still contains several logical and experimental weaknesses that can reduce reviewer confidence.

### Current estimated score

- **Overall score:** 6.8–7.3/10
- **Recommendation:** Weak Accept
- **Main reason:** Excellent track fit, but the detection → inventory → financial-impact chain is not yet fully supported by consistent evidence.

---

# 2. Summary of the Paper

The paper presents **SolarGuard**, an end-to-end decision-support framework for commercial rooftop photovoltaic operations.

The system contains three main layers:

- **Layer 1 — Anomaly detection:**  
  Uses STL decomposition, peer-relative performance ratios, rolling indicators, and Isolation Forest to identify abnormal inverter behavior without requiring labeled failure data.

- **Layer 2 — Inventory decision support:**  
  Converts anomaly signals into spare-inverter inventory decisions under a procurement lead-time constraint.

- **Layer 3 — Financial evaluation:**  
  Maps avoided downtime and recovered generation into a P90 financial model to estimate changes in NPV and IRR.

The case study uses data from a 26.15 MWp commercial rooftop PV project in Vietnam with:

- 219 inverters,
- 93,277 inverter-days,
- 15-minute telemetry,
- 10 documented real-world failures, and
- synthetic progressive and abrupt fault injections.

The proposed pipeline reports:

- recall: 98.02%,
- F1-score: 0.7046,
- precision: 0.5500,
- median early-warning horizon: 2.51 days,
- 376.65 MWh preserved during the five-month pilot,
- estimated NPV gain: USD 63.5k.

---

# 3. Strong Points

## 3.1 Excellent track fit

The paper is highly aligned with an applied AI track because it addresses:

- industrial predictive maintenance,
- renewable-energy operations,
- supply-chain decision support,
- inventory planning,
- economic impact,
- infrastructure resilience,
- real-world AI deployment under label scarcity.

This is a better venue fit than a machine-learning foundations track.

## 3.2 Strong practical motivation

The paper correctly argues that anomaly detection alone has limited operational value if spare equipment is unavailable.

The anomaly → inventory → finance pipeline gives the paper a clear industrial purpose.

## 3.3 Coherent end-to-end architecture

The formal flow

\[
X_{\mathrm{raw}} \rightarrow P \rightarrow r \rightarrow u \rightarrow \theta
\]

provides a clear connection between raw monitoring data, risk estimates, inventory decisions, and financial outcomes.

## 3.4 Real operational dataset

The dataset is substantial for an applied study:

- 219 inverters,
- 93,277 inverter-days,
- 15 months of data,
- 15-minute monitoring logs.

The preprocessing pipeline is also more transparent than a black-box ML workflow because it includes:

- identifier and timestamp normalization,
- cumulative energy recovery,
- physical validity masking,
- inverter-day aggregation,
- audit flags,
- rolling features.

## 3.5 Cost-sensitive evaluation

The Expected Maintenance Cost metric is better aligned with industrial operations than accuracy alone.

The distinction between:

- False Positive cost: unnecessary inspection,
- False Negative cost: unmitigated generation loss,

is practically meaningful.

## 3.6 Honest novelty positioning

The paper correctly states that the contribution is not a new detector, but the integration of technical, inventory, and financial layers.

This positioning is appropriate for an applied AI conference.

---

# 4. Major Weaknesses and Required Revisions

## 4.1 Early-warning horizon conflicts with procurement lead time

The paper assumes a **14-day procurement lead time**, but reports a median warning horizon of only **2.51 days**.

The current claim that 2.51 days is “enough time to order a spare” is inconsistent with the paper's own assumptions.

### Why this is serious

This directly weakens the core causal chain:

\[
\text{anomaly warning}
\rightarrow
\text{spare procurement}
\rightarrow
\text{avoided downtime}
\]

A 2.51-day warning cannot support new international procurement if delivery requires 14 days.

### Required correction

Clarify that the warning supports:

- reservation of an already-stocked spare,
- allocation among sites,
- crew scheduling,
- maintenance preparation,

rather than procurement from overseas.

### Recommended replacement sentence

> The median 2.51-day warning horizon is insufficient for new international procurement under the assumed 14-day lead time, but it can support spare allocation, reservation, and replacement scheduling when inventory is already held locally.

---

## 4.2 AI-driven inventory does not outperform the current fixed-stock policy

The reported five-month costs are:

| Policy | Total cost |
|---|---:|
| Reactive procurement | USD 31,239.22 |
| Fixed pre-stock: 10 units | USD 1,164.71 |
| AI-driven inventory: 6 units | USD 1,949.02 |

The existing fixed-stock policy is approximately **USD 784 cheaper** than the proposed AI-driven policy during the observed pilot.

### Why this is serious

The paper calls the six-unit stock level “optimal,” but the full reported cost is higher than the ten-unit policy.

A reviewer may ask:

> Why should an operator adopt the proposed inventory layer if the current policy is cheaper?

### Required correction

State the observed result honestly:

> AI-driven inventory substantially outperforms reactive procurement but does not outperform the current fixed pre-stock policy over the five-month pilot.

### Required additional analysis

Add at least one of the following:

- break-even horizon,
- one-year, three-year, and five-year simulation,
- sensitivity to AI recurring cost,
- sensitivity to inverter unit cost,
- sensitivity to holding rate,
- capital tied-up analysis,
- service-level comparison,
- inventory turnover,
- cash-flow or working-capital benefit.

The paper should only claim superiority over fixed pre-stock if such evidence is added.

---

## 4.3 Financial extrapolation from 376.65 MWh to 1,721.52 MWh is unclear

The paper observes 376.65 MWh of preserved generation over five months, but later reports an annualized or long-horizon value of 1,721.52 MWh.

A simple annualization gives:

\[
376.65 \times \frac{12}{5}
\approx 903.96 \text{ MWh}
\]

which is far below 1,721.52 MWh.

### Why this is serious

The NPV gain of USD 63.5k depends on this production estimate.

If the conversion is not transparent, reviewers may question the entire financial layer.

### Required correction

Add an explicit bridge:

\[
376.65
\rightarrow
\text{annualization or projection factor}
\rightarrow
1,721.52
\rightarrow
\Delta \text{Revenue}
\rightarrow
\Delta \text{NPV}
\]

Explain whether 1,721.52 MWh includes:

- annualized event frequency,
- multi-year projection,
- tariff escalation,
- expected number of failures,
- seasonal scaling,
- capacity expansion,
- degradation,
- other modeled factors.

If none applies, recalculate the value.

---

## 4.4 Evaluation-set construction is unclear

The paper reports:

- 10 documented failures,
- 300 synthetic injections,
- 101 target events in the test window.

It is unclear:

- whether the 101 events include the 10 real failures,
- why 300 injections become 101 target events,
- how overlapping events are merged,
- whether injection happens before or after the temporal split,
- whether the same inverter receives multiple injections,
- how leakage is prevented.

### Required clarification

Report:

- number of real events in train/validation/test,
- number of progressive synthetic events,
- number of abrupt synthetic events,
- event deduplication rule,
- overlap handling for 30-day windows,
- exact split dates,
- injection timing,
- whether multiple events can occur for one inverter,
- how target events are defined.

A flow table is recommended:

| Stage | Real events | Synthetic events | Final target events |
|---|---:|---:|---:|
| Raw pool | 10 | 300 | — |
| Training | ... | ... | ... |
| Validation | ... | ... | ... |
| Test | ... | ... | 101 |

---

## 4.5 “Zero-shot” terminology is potentially misleading

Isolation Forest is still fitted on historical operational telemetry, and thresholds are configured.

The intended meaning is likely:

> no labeled failure examples are required.

### Recommended terminology

Use:

- **zero-label deployment**,
- **label-free unsupervised deployment**, or
- define zero-shot explicitly.

### Recommended clarification

> In this work, zero-shot refers to the absence of labeled failure examples, not the absence of unsupervised fitting on historical telemetry.

---

## 4.6 Statistical validation is insufficient

The paper reports single-point values for:

- precision,
- recall,
- F1,
- RMSE,
- EMC,
- NPV,
- generation saved.

There are no:

- confidence intervals,
- bootstrap results,
- multiple random seeds,
- statistical significance tests,
- uncertainty propagation.

### Required improvements

At minimum:

1. Event-level bootstrap 95% confidence intervals for precision, recall, and F1.
2. Multiple random seeds for Isolation Forest.
3. Repeated synthetic injection runs.
4. Sensitivity of EMC to False Positive and False Negative costs.
5. NPV range under multiple tariff, FX, lead-time, and generation scenarios.

Recommended table:

| Metric | Point estimate | 95% CI |
|---|---:|---:|
| Precision | 0.5500 | ... |
| Recall | 0.9802 | ... |
| F1 | 0.7046 | ... |
| EMC | USD 3,810 | ... |

---

## 4.7 Baseline evaluation may be unfair to supervised models

The supervised models are trained mainly on synthetic faults, and thresholds are optimized on the training set.

Potential problems:

- training-set threshold optimization,
- synthetic distribution overfitting,
- no independent validation set,
- unclear calibration,
- unclear class weighting,
- unclear hyperparameter budget,
- unclear event grouping consistency.

### Required correction

Use a temporal split such as:

- training,
- validation,
- test.

Tune thresholds only on validation data.

Report:

- feature parity,
- class weights,
- hyperparameters,
- calibration method,
- tuning budget,
- seeds,
- event-grouping method,
- threshold-selection procedure.

Avoid claiming supervised methods “overfit” unless supported by explicit evidence.

A safer wording is:

> A plausible explanation is that supervised models trained primarily on injected events may learn characteristics specific to the synthetic fault distribution, contributing to reduced precision under heterogeneous test conditions.

---

# 5. Additional Important Corrections

## 5.1 Do not call EMC itself novel

An asymmetric False Positive/False Negative cost is standard in cost-sensitive learning.

The novelty is in its domain-specific application.

Replace:

> a novel asymmetric Expected Maintenance Cost model

with:

> a domain-specific asymmetric Expected Maintenance Cost evaluation

or:

> an application-oriented maintenance cost model.

## 5.2 Fix the unfinished sentence in the conclusion

The conclusion contains an incomplete placeholder:

> reduces alert volume from … to..inherent to early-stage supervised models.

This must be corrected before submission.

Recommended replacement:

> The pipeline reduces alert volume from 512 episodes for the vanilla Isolation Forest baseline to 180 episodes, while improving precision from 0.1953 to 0.5500.

## 5.3 Reduce overly strong language

Recommended changes:

- “strict superiority”  
  → “substantially lower cost than reactive procurement”

- “deployment-ready”  
  → “designed for early-stage deployment”

- “transfers across rooftop and utility scale”  
  → “is designed to transfer with site-specific recalibration”

- “ensures ecological validity”  
  → “improves the realism of the injected scenarios”

- “preserving 376.65 MWh”  
  → “estimated to preserve 376.65 MWh under the evaluated scenario”

## 5.4 Improve figure readability

The preprocessing figure is too small for print.

Recommended actions:

- enlarge the figure,
- simplify box text,
- split preprocessing and framework diagrams,
- increase font size,
- remove unnecessary annotations.

---

# 6. Predicted Reviewer Scores Before Revision

| Reviewer profile | Likely score |
|---|---:|
| Applied AI / Industry | 8/10 — Accept |
| ML / anomaly detection | 6/10 — Weak Reject |
| Energy / operations | 7/10 — Weak Accept |
| Meta-review | 7/10 — Weak Accept |

### Current recommendation

**Weak Accept**

### Current estimated acceptance probability

Approximately **60–75%**, depending on reviewer expertise and conference selectivity.

---

# 7. Expected Evaluation After Fixing All Seven Major Issues

If all seven issues are resolved with actual experiments, transparent calculations, and consistent writing—not merely by rephrasing—the expected quality increases substantially.

## Expected score after complete revision

- **Overall score:** 8.0–8.5/10
- **Recommendation:** Accept
- **Likely reviewer pattern:**  
  - 2 Accept + 1 Weak Accept, or  
  - 2 Accept + 1 Borderline

## Expected acceptance probability

- **General estimate:** 80–90%
- **Applied AI / Industry reviewer mix:** 85–92%
- **If a strict ML reviewer is assigned:** approximately 70–80%

A realistic overall estimate is:

> **Approximately 85% probability of acceptance** for the track AI Applications for Industry and Society, provided the major logical and experimental issues are genuinely fixed.

---

# 8. Most Important Revisions by Priority

## Priority 1 — Critical logic

1. Resolve the contradiction between 2.51-day warning and 14-day procurement.
2. Explain why AI inventory is useful when fixed pre-stock is cheaper in the pilot.
3. Recalculate or transparently explain the 376.65 → 1,721.52 MWh projection.

## Priority 2 — Experimental credibility

4. Explain the 10 real + 300 synthetic → 101 target-event construction.
5. Add bootstrap confidence intervals and repeated runs.
6. Use an independent validation split for threshold tuning.
7. Define “zero-shot” accurately.

## Priority 3 — Presentation

8. Fix the unfinished conclusion sentence.
9. Reduce overclaiming.
10. Improve figure readability.
11. Reframe EMC as domain-specific rather than methodologically novel.

---

# 9. Final Verdict

SolarGuard is a strong fit for the **AI Applications for Industry and Society** track because it demonstrates a practical connection between AI, renewable-energy operations, logistics, and financial decision support.

Its main weakness is not conference relevance or writing quality. The main weakness is reviewer confidence in the full causal chain:

\[
\text{detection}
\rightarrow
\text{inventory action}
\rightarrow
\text{avoided downtime}
\rightarrow
\text{financial gain}
\]

The current version contains three especially important inconsistencies:

- the warning is shorter than procurement lead time,
- the fixed-stock baseline is cheaper during the pilot,
- the production extrapolation is unclear.

After resolving these issues and completing the remaining statistical and experimental revisions, the paper should move from **Weak Accept** to a much clearer **Accept**, with an estimated acceptance probability around **85%**.
