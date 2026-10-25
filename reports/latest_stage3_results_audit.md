# Latest Stage 3 Results Audit

## 1. Executive summary

The latest Stage 3 benchmark output in `results/tables/stage3_consolidated_inversion_benchmarks.csv` is not directly comparable to the canonical Stage 3 benchmark in `results/FINAL_RESULTS/final_metrics_summary.csv`.

The key reason is methodological: the latest CSV reports raw-space coefficient of determination (`r2`) computed with `sklearn.metrics.r2_score` on physical concentrations, while the canonical summary reports log-space metrics such as macro `log10 R²` and pooled `log10 R²` from the final evaluation workflow. These are different targets, different scales, and different evaluation definitions.

The latest Stage 3 run is therefore a valid baseline benchmark under its own protocol, but it is not the same benchmark as the canonical project result. It should not be used to claim either improvement or degradation relative to the canonical final report without a matched metric conversion.

## 2. Latest results table

Source file: `results/tables/stage3_consolidated_inversion_benchmarks.csv`

| target | model | r2 | rmse | mae | mape | bias | dataset | n_samples | n_sites |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| turbidity | Empirical_Baseline | -0.0399 | 52.8750 | 19.7095 | 288.6507 | -11.9087 | Condition_(b)_DL_Masked | 37655 | 51 |
| turbidity | Random_Forest | 0.0368 | 50.8866 | 15.6722 | 149.3456 | -8.0892 | Condition_(b)_DL_Masked | 37655 | 51 |
| turbidity | Gradient_Boosting | 0.0712 | 49.9703 | 15.2546 | 158.2131 | -7.5131 | Condition_(b)_DL_Masked | 37655 | 51 |
| turbidity | Deep_Spectral_MLP | 0.0731 | 49.9193 | 15.2500 | 161.9777 | -7.4629 | Condition_(b)_DL_Masked | 37655 | 51 |
| chlorophyll | Empirical_Baseline | -0.1560 | 21.6055 | 10.8117 | 492.4284 | -6.9978 | Condition_(b)_DL_Masked | 6932 | 18 |
| chlorophyll | Random_Forest | -0.1678 | 21.7154 | 12.3229 | 746.5123 | -1.9373 | Condition_(b)_DL_Masked | 6932 | 18 |
| chlorophyll | Gradient_Boosting | -0.1606 | 21.6485 | 12.2921 | 648.8175 | -3.0164 | Condition_(b)_DL_Masked | 6932 | 18 |
| chlorophyll | Deep_Spectral_MLP | -5.5779 | 51.5382 | 17.2900 | 665.8991 | 2.4387 | Condition_(b)_DL_Masked | 6932 | 18 |
| cdom | Empirical_Baseline | -1.3315 | 27.5908 | 24.4511 | 248.3354 | -2.8619 | Condition_(b)_DL_Masked | 2533 | 6 |
| cdom | Random_Forest | -0.4892 | 22.0506 | 16.6102 | 84.4674 | -7.3624 | Condition_(b)_DL_Masked | 2533 | 6 |
| cdom | Gradient_Boosting | -0.4631 | 21.8567 | 16.4195 | 85.8062 | -6.6691 | Condition_(b)_DL_Masked | 2533 | 6 |
| cdom | Deep_Spectral_MLP | -0.5136 | 22.2307 | 16.5551 | 81.2117 | -8.9912 | Condition_(b)_DL_Masked | 2533 | 6 |

### What each column means

- `r2`: raw-space coefficient of determination, computed on `y_true` and `y_pred` directly in original measurement units.
- `rmse`: raw-space RMSE in the same units as the target.
- `mae`: raw-space mean absolute error.
- `mape`: mean absolute percentage error.
- `bias`: mean prediction minus observation.
- `model`: model trained in the benchmark loop.
- `target`: water-quality target passed to the benchmark.
- `dataset`: dataset label used by the pipeline at runtime.
- `n_samples`: total rows in the target-specific matched dataset after filtering.
- `n_sites`: number of unique monitoring stations in that target subset.

## 3. Comparison against canonical results

Canonical source: `results/FINAL_RESULTS/final_metrics_summary.csv`

| target | latest best model | latest metric | canonical best model | canonical macro log R² | canonical pooled log R² | comparison verdict |
|---|---|---|---|---:|---:|---|
| Turbidity | Deep_Spectral_MLP | raw-space `r2 = 0.0731` | Gradient_Boosting | +0.2166 | +0.3017 | incomparable (different metric definition) |
| Chlorophyll-a | Empirical_Baseline | raw-space `r2 = -0.1560` | Empirical_Baseline (NDCI) | -2.5708 | -0.2982 | incomparable (different metric definition; not same target transformation or scale) |
| CDOM | Gradient_Boosting | raw-space `r2 = -0.4631` | Deep_Spectral_MLP | -14.4524 | -0.1824 | incomparable (different metric definition; not same target transformation or scale) |

### Important interpretation rule

The canonical metrics are in log-space and the latest values are in raw physical space. They are not interchangeable.

Examples:
- A raw-space `R²` of `0.0731` is not equivalent to a macro log `R²` of `+0.2166`.
- A raw-space `R²` of `-0.4631` is not equivalent to a macro log `R²` of `-14.4524`.
- The canonical report uses `log10` transformed targets during model fitting and evaluation. The latest benchmark uses a different operational metric definition.

## 4. Metric definitions and evaluation protocols

### Latest Stage 3 benchmark protocol

Source: `src/inversion/train_inversion.py`, function `benchmark_inversion_models`.

Key behavior:

1. The dataset is filtered to `df[df['target_type'] == target_type]`.
2. Features are taken from `FEATURE_COLS`.
3. The target is transformed to log-space for model fitting:
   - `y_log = np.log10(np.clip(y_raw, 1e-4, None))`
4. The model is trained in log-space.
5. Predictions are converted back to physical space by `10.0 ** oof_preds_log`.
6. The final reported metrics are computed by `compute_regression_metrics(y_raw, oof_preds_raw)` in `src/inversion/metrics.py`.
7. That function computes `r2_score(y_true, y_pred)` on the raw values, followed by RMSE, MAE, MAPE, and bias in physical units.

Important detail: the latest benchmark is in physical space, not in log-space. It is not the canonical macro/log evaluation used for the project’s final benchmark decisions.

### Canonical protocol

Source: `reports/final_stage3_benchmark_results.md` and `results/FINAL_RESULTS/final_metrics_summary.csv`.

Canonical results use:

- 5-fold spatial block CV grouped by `site_no`
- log10 evaluation as the primary metric
- macro `log10 R²` and pooled `log10 R²` as the main comparison metrics
- site-aware fold logic and spatial leakage protections

The canonical benchmark also tracks additional metadata, including sample size, site count, ICC, and intervention-specific statistics.

### Timing / dataset question

The latest benchmark file is generated by the runtime path in `run_pipeline.py` via `run_stage_3()`. It reads the DL-masked dataset:

- `data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv`

This path is not the same as the curated final benchmark pipeline path described in the final report metadata. The latest file is a direct multi-model benchmark produced by the project’s operational pipeline, while the canonical summary is a curated final benchmark artifact.

## 5. Source-code trace

### 5.1 Pipeline entry point

File: `run_pipeline.py`

Relevant function:

- `run_stage_3()`

Behavior:

- Loads `matched_set_b_dl_masked.csv`
- Loops over `['turbidity', 'chlorophyll', 'cdom']`
- Calls `benchmark_inversion_models(...)`
- Concatenates outputs into `results/tables/stage3_consolidated_inversion_benchmarks.csv`

### 5.2 Benchmark function

File: `src/inversion/train_inversion.py`

Function:

- `benchmark_inversion_models(...)`

This is the source of the latest CSV. The function:

- filters by `target_type`
- creates `X` from `FEATURE_COLS`
- computes `y_raw` and `y_log`
- performs 5-fold spatial CV using `get_spatial_block_cv(...)`
- trains each model in log-space
- back-transforms predictions to raw physical units
- computes metrics with `compute_regression_metrics(...)`

### 5.3 Spatial CV

File: `src/inversion/spatial_cv.py`

Function:

- `get_spatial_block_cv(...)`

This uses `GroupKFold(n_splits=5)` with `group_col='site_no'` and asserts zero site overlap between train and validation folds. This is scientifically correct and prevents station leakage in the latest benchmark pipeline.

### 5.4 Metric calculation

File: `src/inversion/metrics.py`

Function:

- `compute_regression_metrics(y_true, y_pred)`

This computes raw-space metrics only:

- `r2 = r2_score(y_true, y_pred)`
- `rmse = sqrt(mean_squared_error(...))`
- `mae = mean_absolute_error(...)`
- `mape = ...`
- `bias = mean(y_pred - y_true)`

This confirms that the latest output is not the canonical log-metric benchmark.

## 6. Confirmed issues and potential causes

### Confirmed issue 1: metric-definition mismatch

This is the clearest and confirmed issue.

Evidence:

- Latest CSV contains `r2`, `rmse`, `mae`, `mape`, `bias` computed on raw physical concentrations.
- Canonical summary uses macro `log10 R²` and pooled `log10 R²`.
- The code explicitly logs the target before fitting, then evaluates on raw values after back-transform.

Conclusion:

The latest and canonical values are produced under different metric definitions and cannot be compared as if they were the same number.

### Confirmed issue 2: the latest benchmark is not the canonical final benchmark

Evidence:

- `run_pipeline.py` writes `results/tables/stage3_consolidated_inversion_benchmarks.csv`.
- The canonical project report references `results/FINAL_RESULTS/final_metrics_summary.csv` and the curated benchmark outputs under `results/FINAL_RESULTS/`.
- The latest CSV is created in a different output location and is generated by a different pipeline path.

Conclusion:

The latest file is a routine Stage 3 benchmark file, not the canonical final benchmark summary used for scientific claims.

### Potential cause 3: different evaluation target and scale

This is a likely reason for large apparent differences, but it remains a methodological difference rather than a direct code bug.

Evidence:

- The latest benchmark code uses raw targets on the physical scale when computing `r2`.
- The canonical benchmark report emphasizes log-space `R²`.

Conclusion:

Large differences between the latest and canonical numbers are expected when one is evaluated on raw physical scale and the other is evaluated on log10 scale; they cannot be treated as a regression from “better” or “worse” model performance without a matched metric conversion.

### Not confirmed: model bug or station leakage bug

Evidence from the code:

- `get_spatial_block_cv` uses `GroupKFold` with `group_col='site_no'`.
- The code asserts zero overlap between train and validation station sets.

This does not support a confirmed station-leakage bug in the latest benchmark.

### Not confirmed: invalid inverse transformation or incorrect target handling

Evidence from the code:

- The target is transformed with `np.log10(np.clip(y_raw, 1e-4, None))` and then back-transformed with `10.0 ** oof_preds_log`.
- This is a normal transform-back-transform scheme.

The code is coherent internally, though the final metric selection differs from the canonical protocol.

## 7. Scientific-validity assessment

### Current status

The latest Stage 3 benchmark is scientifically valid only under the raw-space metric definition used by the code path in `src/inversion/train_inversion.py` and `src/inversion/metrics.py`.

It is not scientifically valid to compare these values directly to the canonical final project numbers because the evaluation metric differs.

### Statement of evidence

- The latest output is produced by a valid spatial CV benchmark.
- The latest values are not directly comparable to canonical log-space results.
- The canonical project report remains the authoritative source for the final scientific benchmark claims if the metric definitions match the project’s final evaluation policy.

## 8. Requested figures check

The Stage 3 plots were generated under the same runtime path as the CSV summary:

- `results/figures/stage3_turbidity_obs_vs_pred.png`
- `results/figures/stage3_chlorophyll_obs_vs_pred.png`
- `results/figures/stage3_cdom_obs_vs_pred.png`

These are consistent with the raw-space R² values in the CSV:

- Turbidity: largest model is around `r2 = 0.0731`, with broad scatter and weak explanatory power.
- Chlorophyll-a: predictions are poor, with the best raw-space `R²` near `-0.1560` and strong dispersion.
- CDOM: best raw-space `R²` is negative (`-0.4631`), consistent with the model not explaining the target variance in physical space.

This is consistent with the corresponding numerical results, but the plots do not change the core fact that the latest numbers are not the canonical log-space benchmark.

## 9. Recommended next steps, prioritized

1. Treat the latest CSV as a raw-space benchmark only, not as the project’s final scientific result.
2. Use `results/FINAL_RESULTS/final_metrics_summary.csv` and the canonical benchmark documents as the final comparison benchmark when writing conclusions.
3. If a direct comparison is needed between the latest pipeline and canonical results, re-express both in a common metric (ideally log-space for primary scientific interpretation).
4. Do not claim model improvement or degradation based on raw-space `R²` versus canonical macro log `R²` without matching the metric definition.
5. Keep the existing GroupKFold `site_no` split and avoid random train/test benchmarking as the primary evidence source.

## 10. Exact files inspected

- `run_pipeline.py`
- `src/inversion/train_inversion.py`
- `src/inversion/metrics.py`
- `src/inversion/spatial_cv.py`
- `results/tables/stage3_consolidated_inversion_benchmarks.csv`
- `results/FINAL_RESULTS/final_metrics_summary.csv`
- `reports/final_stage3_benchmark_results.md`
- `reports/verification_note.md`
- `results/figures/stage3_turbidity_obs_vs_pred.png`
- `results/figures/stage3_chlorophyll_obs_vs_pred.png`
- `results/figures/stage3_cdom_obs_vs_pred.png`

## 11. Conclusion

The latest Stage 3 results are not genuinely comparable with the canonical results because they are calculated under different definitions and scales. The latest CSV measures raw-space model fit; the canonical report measures log-space model fit. The code path itself confirms this.

There is no confirmed station-leakage bug or invalid inverse transform in the latest benchmark path. The confirmed issue is metric mismatch between the raw-space Stage 3 CSV and the canonical log-space final benchmark. Therefore, the project’s scientific conclusions should continue to rely on the canonical benchmark and final report, not the latest raw-space benchmark output alone.
