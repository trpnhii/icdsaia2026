# icdsaia2026

## Dashboard (GitHub Pages)

The static dashboard lives at the repository root:

- `index.html`
- `app.js`
- `styles.css`
- `data.js`

To refresh dashboard data after running the pipeline:

```bash
python scripts/build_dashboard_data.py
```

### Host on GitHub Pages

1. Commit and push `index.html`, `app.js`, `styles.css`, and `data.js`.
2. In the repository settings, open **Pages**.
3. Under **Build and deployment**, set **Source** to **Deploy from a branch**.
4. Choose branch `main` (or your default branch) and folder `/ (root)`.
5. Save. GitHub will publish the site at `https://<username>.github.io/<repo>/`.

## Experimental Results

### Ablation Study (Table 4)

The ablation study removes one component at a time from the SolarGuard framework:

- `w/o STL Decomposition`
- `w/o Peer-relative PR`
- `w/o Adaptive Window`
- `Threshold-based (Baseline)`

Run:

```bash
python preprocess/ablation_study.py
```

Outputs:

- `output_data/experimental_results/table4_ablation_study.csv`
- `output_data/experimental_results/table4_ablation_study.md`
- `output_data/experimental_results/figure4_ablation_comparison.png`
- `output_data/experimental_results/figure4_npv_gain.png`

The main interpretation is that SolarGuard should be evaluated as a rare-event
maintenance framework rather than a balanced classifier. The most important
finding is that peer-relative PR is a critical module: removing it sharply
reduces operational value and NPV gain.

### Extended Baselines And Synthetic Fault Injection (Table 11)

To strengthen supervised comparison and increase statistical significance, the
project includes an extended experiment with:

- supervised Gradient Boosting baselines (`XGBoost`, `LightGBM`,
  `HistGradientBoosting`)
- synthetic fault injection to create additional degradation and hard-fault
  events
- cost-sensitive evaluation through `Expected Maintenance Cost (USD)`

Run:

```bash
python preprocess/run_table11_extended.py
```

Useful options:

```bash
python preprocess/run_table11_extended.py \
  --target-events 300 \
  --lead-window-days 30 \
  --fp-inspection-cost-usd 10 \
  --fn-miss-cost-usd 1500 \
  --train-frac 0.55 \
  --val-frac 0.15
```

Chronological protocol:

1. **Train** (first `train-frac` of dates): fit supervised models.
2. **Validation** (next `val-frac`): tune probability cutoffs by minimizing event-level EMC only.
3. **Test** (remainder): report Table 11 metrics; thresholds are frozen.

Outputs:

- `output_data/experimental_results/table11_extended_comparison.csv`
- `output_data/experimental_results/table11_extended_comparison.md`
- `output_data/experimental_results/table11_supervised_thresholds.csv`

Current default run:

- synthetic events injected: `300`
- test-window events: `101`
- lead window: `30` days
- cost model: `FP = $10`, `FN = $1500`

Current headline result:

- `SolarGuard (Zero-shot)` achieves the lowest expected maintenance cost
  (`$3,810`) while maintaining strong event detection
  (`Precision = 0.5500`, `Recall = 0.9802`, `F1 = 0.7046`).

This extended setup is intended for the paper's stronger baseline discussion:
supervised models can be competitive once labels are accumulated, but zero-shot
methods such as SolarGuard and Isolation Forest are deployable immediately
without waiting months for labeled failure data.

### Synthetic Fault Injection Transparency

To avoid concerns that synthetic labels are "hand-drawn" for model advantage,
the injection protocol is explicitly constrained and reproducible:

1. **Injection domain (sunny-day operating regime)**  
   Injection is applied only on rows already filtered for valid PR analysis:
   `irradiation_kwh_m2 >= 0.5`, valid PR, valid capacity, and peer-comparable
   inverter-day rows. This prevents cloud/no-irradiance artifacts from being
   treated as faults.

2. **Peer-relative definition of abnormality**  
   Fault severity is evaluated against same-day peer behavior using:
   - `relative_pr = PR_i / median(PR_peers_same_day)`
   - `peer_deficit = max(1 - relative_pr, 0)`  
   This ensures anomalies are detected as inverter-specific deviations, not
   whole-site weather variation.

3. **Fault templates used in injection (`preprocess/run_table11_extended.py`)**
   - **Degradation fault** (70% of injections): PR multiplier decreases
     linearly from `0.95` to `0.45` over `10-15` days (persistent drift).
   - **Hard fault** (30% of injections): PR multiplier fixed at `0.08` over
     `2-4` days (near-collapse behavior).

4. **Guardrails against unrealistic labeling**
   - Minimum history before injection: `>= 60` rows per inverter.
   - Fault start index sampled after early stabilization period.
   - No overlap with existing synthetic fault windows on the same inverter.
   - All injections tracked as event records with
     `(zone, device_key, replacement_date, fault_type)`.

5. **Reproducibility**
   - Controlled by `--random-seed` (default `42`).
   - Event volume controlled by `--target-events` (default `300`).

This protocol is designed to verify that Isolation Forest and SolarGuard react
to peer-relative performance degradation under comparable irradiance, rather
than reacting to passing-cloud effects.

### Independent Real-World Test Set (Recommended For Paper)

For reviewer confidence, keep at least `1-2` real failure events from the
26.15 MWp site as an **independent holdout test** (not used in synthetic
injection calibration).

Suggested minimal setup:

1. Create `input_data/risk/real_world_test_events.csv` with:
   - `device_name`
   - `replacement_date`
   - `zone` (optional, can be mapped)
   - optional evidence fields: `ticket_id`, `root_cause`, `maintenance_note`
2. Hold these events out from synthetic tuning and threshold search.
3. Report side-by-side:
   - synthetic benchmark (Table 11 extended)
   - real-world holdout benchmark (small-N but high credibility)

Even a 1-2 event holdout materially improves methodological transparency by
showing the model is validated on both simulated and actual operational faults.

## Priority 1 Paper Revisions (Lead Time, Inventory Honesty, MWh Bridge)

ICDSAIA strict-review Priority 1 is addressed by regenerating transparent
artifacts from the inventory cost model:

```bash
python preprocess/run_priority1_revision.py
```

Outputs (under `output_data/experimental_results/`):

- `priority1_revision_summary.md` — start here
- `priority1_lead_time_role.md` — 2.51-day warning vs 14-day procurement
- `priority1_inventory_comparison.md` / `.csv` — Reactive vs Fixed N=10 vs AI N*
- `priority1_inventory_horizons.csv` + `figure_priority1_inventory_horizons.png`
- `priority1_mwh_npv_bridge.md` / `.csv` — preserved MWh → projection → USD benefit

Key paper-facing corrections:

1. Early warning supports **local spare reservation / scheduling**, not overseas
   ordering within ~2.5 days.
2. AI cost-minimizing stock (**N=5** in the current run) beats reactive strongly
   and also beats a fixed overstock (**N=10**) on total expected cost while
   releasing working capital.
3. Replace opaque `376.65 → 1,721.52 MWh` with the explicit bridge in
   `priority1_mwh_npv_bridge.md` (observed preserved energy × horizon / observed
   span). The paper factor itself is documented in
   `mwh_scale_factor_376_to_1721.md`: \(1{,}721.52/376.65 \approx 4.57 = 24/5.25\)
   (24-month P90 horizon over a 5.25-month effective pilot), **not** \(12/5\).

