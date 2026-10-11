# Site and Basin Generalization Diagnostic Report

## 1. Objective

This diagnostic tests whether poor chlorophyll-a and CDOM station-level generalization is primarily associated with site-specific target distributions, basin optical regimes, target harmonization, limited independent-site diversity, or validation optimism. The existing DL-masked 12-hour benchmark and 17-feature representation were retained.

## 2. Current benchmark

Dataset: `data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv`. The transfer analyses use the existing Gradient Boosting model with log10 targets and site-grouped validation. No U-Net, feature set, or official GroupKFold protocol was changed.

## 3. Target harmonization audit

The harmonization audit is in `results/tables/target_harmonization_audit.csv`. Turbidity is recorded in FNU, chlorophyll in ug/l, and CDOM in ug/l QSE in the current native benchmark. The project code defines site-specific RFU-to-physical conversions for selected chlorophyll and CDOM sites; uncalibrated RFU observations are excluded from the primary native-unit dataset.

The audit shows that parameter codes and measurement protocols are not completely uniform: chlorophyll combines fluorescence, fluorometric, and total-chlorophyll codes, while CDOM uses fDOM/QSE. Harmonization sensitivity must therefore be interpreted as a data-condition comparison rather than proof that all methods are interchangeable.

## 4. Site-level target distributions

Site-level statistics are in `results/tables/site_target_distribution.csv`; the figure is `results/figures/target_distribution_by_site.png`. The site distributions show large differences in target level and range, especially for chlorophyll-a and CDOM.

## 5. Basin-level target distributions

Basin-level statistics are in `results/tables/basin_target_distribution.csv`; the figure is `results/figures/target_distribution_by_basin.png`. Targets do not occur in identical basin/site configurations, so basin transfer is asymmetric.

## 6. Between-site versus within-site variance

| Target | Sites | Total log variance | Between-site fraction | Within-site fraction | ICC |
|---|---:|---:|---:|---:|---:|
| turbidity | 51 | 0.3237 | 67.5% | 32.5% | 0.678 |
| chlorophyll | 18 | 0.3920 | 81.1% | 18.9% | 0.819 |
| cdom | 6 | 0.1780 | 91.3% | 8.7% | 0.926 |

The corrected variance decomposition indicates that between-site variation accounts for 67.5% of turbidity, 81.1% of chlorophyll-a, and 91.3% of CDOM log-target sum of squares under this unbalanced observational decomposition. The corresponding random-effects ICC is also high. These values indicate strong site dependence, but they should not be interpreted as causal proof that every spectral relationship is non-transferable.

## 7. Feature distribution shifts

Site/basin feature shifts, standardized mean differences, and KS diagnostics are in `results/tables/site_feature_shift.csv`. Effect sizes are reported alongside p-values because large sample sizes can make small distribution shifts significant.

## 8. Leave-One-Site-Out results

LOSO results are in `results/tables/leave_one_site_out_results.csv`. Site-level performance is highly heterogeneous: some stations transfer with positive or near-positive log R², while others fail strongly. The CDOM special audit is `results/tables/cdom_site_audit.csv`; the chlorophyll-a special audit is `results/tables/chlorophyll_site_audit.csv`.

## 9. Leave-One-Basin-Out results

| Target | Held-out basin | Sites | Log R² | Log RMSE |
|---|---|---:|---:|---:|
| turbidity | Delaware | 14 | -0.336 | 0.418 |
| turbidity | Illinois | 11 | -1.835 | 0.810 |
| turbidity | Trinity | 7 | -2.699 | 0.479 |
| turbidity | Upper Colorado | 10 | 0.342 | 0.389 |
| turbidity | Willamette | 9 | -3.548 | 0.829 |
| chlorophyll | Delaware | 2 | -4.178 | 0.903 |
| chlorophyll | Illinois | 9 | -8.367 | 1.091 |
| chlorophyll | Trinity | 2 | -1.772 | 0.541 |
| chlorophyll | Willamette | 5 | -5.072 | 1.257 |
| cdom | Delaware | 1 | -14.090 | 0.446 |
| cdom | Illinois | 1 | -33.919 | 0.658 |
| cdom | Willamette | 4 | -14.071 | 0.869 |

For the available data, Upper Colorado was the strongest turbidity transfer destination by log R², while Willamette was the weakest. For chlorophyll-a, Trinity was the least negative basin and Illinois was the hardest. For CDOM, Delaware and Willamette were similarly poor, with Illinois the hardest.

## 10. Random versus grouped validation

| Target | Random log R² | Grouped macro log R² | Random minus grouped |
|---|---:|---:|---:|
| turbidity | 0.926 | -0.109 | 1.035 |
| chlorophyll | 0.949 | -4.828 | 5.777 |
| cdom | 0.986 | -14.257 | 15.244 |

Random row-level splits are dramatically optimistic: the difference is approximately 1.03 log-R² units for turbidity, 5.78 for chlorophyll-a, and 15.24 for CDOM. Random splitting is therefore not scientifically acceptable as the final validation protocol.

## 11. Harmonization sensitivity

The controlled comparison is in `results/tables/harmonization_sensitivity.csv`. In this run the secondary converted datasets did not improve grouped transfer performance: chlorophyll-a and CDOM became worse than the native benchmark. This supports retaining native physical-unit records as the primary benchmark while reporting method heterogeneity explicitly.

## 12. CDOM audit

CDOM has only six sites and large between-site target differences. LOSO failures across Delaware, Illinois, and Willamette show that pooled performance is not representative of a stable cross-site relationship. The strongest evidence supports a combination of insufficient independent-site diversity and strong site-specific calibration; measurement heterogeneity is a secondary concern because the native CDOM unit is consistently QSE in the current benchmark.

## 13. Chlorophyll-a audit

Chlorophyll-a has 18 sites but combines multiple parameter codes and measurement descriptions. Target ranges vary substantially by site. The evidence supports strong between-site structure and protocol heterogeneity as contributors, with no evidence that the current spectral features provide a stable station-independent correction.

## 14. Site-level model performance

The full site-level model table is `results/tables/site_level_model_performance.csv`. Best and worst sites should be read with N and basin together; tiny sites can have unstable R² values. Difficult sites were retained.

## 15. Root-cause interpretation

- **Strongest chlorophyll-a explanation:** multiple factors, led by between-site target distribution shifts and measurement/protocol heterogeneity; random validation substantially hides this failure.
- **Strongest CDOM explanation:** insufficient independent-site diversity plus strong site-specific target levels; the six-site design is too small for reliable transfer conclusions.
- **General conclusion:** CASE G, multiple factors contribute. Site/basin structure and validation optimism are better supported than another feature-engineering hypothesis.

## 16. Limitations

- The variance decomposition is observational and unbalanced across sites.
- CDOM has only six sites, limiting power for basin and site transfer claims.
- Harmonization comparisons depend on the calibration metadata encoded by the existing project.
- LOBO results are supported only for basins containing sufficient observations; no unsupported target/basin result was fabricated.

## 17. Recommended next experiment

Collect or add independent stations and method-matched measurements, especially for CDOM and chlorophyll-a. The next modeling experiment should be a pre-registered site/basin transfer benchmark with harmonization strata reported separately, not another unrestricted feature search.

## Reproducibility

```powershell
cd "C:\Term Paper 2\satellite-water-monitoring-ai"
C:\Python314\python.exe -m src.evaluation.site_basin_generalization
```
