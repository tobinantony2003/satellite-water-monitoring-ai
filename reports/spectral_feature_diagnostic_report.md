# Spectral Feature Diagnostic Report

## 1. Objective

This experiment tested whether spectral feature representation limits station-level generalization for turbidity, chlorophyll-a, and CDOM. The existing DL-masked 12-hour benchmark, target transformations, model suite, and site-grouped validation were retained.

## 2. Current feature set

The current benchmark uses 17 features: raw Sentinel-2 reflectance bands B01-B08, B8A, B11, B12 plus NDWI, MNDWI, NDCI, turbidity_ratio, cdom_ratio, and b8_b4_ratio.

## 3. Feature-quality audit

The complete feature-quality audit is saved in `results/tables/feature_audit_summary.csv`. Candidate features were checked for missingness, variance, quantiles, unique-value counts, and extreme values.

## 4. Target-feature relationships

Pearson, Spearman, and mutual-information associations with log10 targets are in `results/tables/target_feature_association.csv`; the overview heatmap is `results/figures/target_feature_correlation.png`.

## 5. Red-edge analysis

Red-edge ratios and NDCI-style variants were limited to B05/B06/B07/B8A relationships with B03/B04. They were evaluated as a targeted family rather than as an unrestricted ratio search.

## 6. Feature groups

- current: existing 17-feature representation
- raw_bands: raw reflectance bands only
- raw_plus_current_indices: raw bands plus existing indices
- raw_plus_red_edge: raw bands plus targeted red-edge features
- target_specific: physically motivated features selected for the target family

## 7. Ablation results

The complete model comparison is in `results/tables/feature_ablation_model_comparison.csv`. All groups use the same site fold assignment within each target.

## 8. MLP CV correction

The diagnostic runner constructs a fresh model instance inside each fold for every non-empirical model, including the Deep Spectral MLP. This prevents learned weights from carrying between validation folds. The existing `train_inversion.py` also constructs non-empirical estimators inside its fold loop; no confirmed cross-fold MLP reuse remained there. To keep the complete four-model ablation tractable on CPU, the diagnostic MLP uses 10 epochs uniformly across all feature groups; the architecture is unchanged.

## 9. Site/basin distribution analysis

Site- and basin-level target and spectral summaries are in `results/tables/site_feature_distribution_summary.csv`. These summaries are needed to distinguish weak spectral signal from site-specific calibration and limited site diversity.

## 10. Target variance analysis

Per-site and validation-fold target variance summaries are in `results/tables/site_target_variance.csv`. CDOM has only six independent sites, so its held-out-site estimates remain particularly fragile.

## 11. Best feature representation

- **turbidity**: best point estimate was `raw_plus_current_indices` with `Deep_Spectral_MLP`; macro log10 R² = -0.0603, pooled log10 R² = 0.1998. Strongest absolute Spearman associations were B05_center_refl, b2_b4_ratio, b8_b4_ratio.
- **chlorophyll**: best point estimate was `current` with `Empirical_Baseline`; macro log10 R² = -3.5665, pooled log10 R² = -0.3665. Strongest absolute Spearman associations were b3_b4_ratio, turbidity_ratio, mndwi.
- **cdom**: best point estimate was `current` with `Gradient_Boosting`; macro log10 R² = -14.2573, pooled log10 R² = -0.0521. Strongest absolute Spearman associations were b8_b4_ratio, b6_b4_ratio, ndci_b6_b4.

## 12. Target conclusions

- **Turbidity:** no feature representation produced a statistically supported improvement over the current representation. The best point estimate was the raw-plus-current-index group with the diagnostic MLP (macro log10 R² = -0.0603), but its paired Wilcoxon comparison was not significant.
- **Chlorophyll-a:** red-edge and target-specific groups did not improve the best station-level result, which remained the empirical current-feature baseline (macro log10 R² = -3.5665). The red-edge Gradient Boosting delta was positive in this run but not statistically supported.
- **CDOM:** the current representation with Gradient Boosting remained best among the tested combinations (macro log10 R² = -14.2573). Adding red-edge or target-specific ratios worsened Gradient Boosting station-level performance.

The experiment must be interpreted using held-out-site metrics, not training fit. A positive delta is not sufficient evidence of improvement unless it is consistent across folds and supported by the paired comparison.

## 13. Statistical evidence

The model table reports macro fold R² and pooled out-of-fold metrics. The paired comparison table reports fold-wise deltas and Wilcoxon tests against the current feature group. The same folds were used across feature groups; no row-level random split was introduced.

## 14. Limitations

- CDOM has only six independent sites.
- Correlation and mutual information are association diagnostics, not causal evidence.
- Feature groups are targeted scientific hypotheses, not an exhaustive search.
- The U-Net and water-mask outputs were not changed.

## 15. Recommended next experiment

Use the feature group that is stable across site folds rather than the single highest point estimate. If no feature group improves consistently, retain the current representation and focus on site diversity, target harmonization, and basin transferability instead of adding more ratios.

## Reproducibility

```powershell
cd "C:\Term Paper 2\satellite-water-monitoring-ai"
C:\Python314\python.exe -m src.evaluation.spectral_feature_diagnostic
```
