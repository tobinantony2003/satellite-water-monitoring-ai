# Pipeline Execution and Consistency Verification Note

**Project**: Satellite Water Monitoring AI  
**Date**: October 2026  
**Status**: Technical Verification Complete & Audit Confirmed  

---

## 1. Clean Pipeline Execution (Stages 1–4)
All operational stages of `run_pipeline.py` were executed sequentially from a clean environment and confirmed with **Exit Code 0**:

- **Stage 1 (`--stage 1`)**: Water U-Net segmentation evaluated across 31 full scenes and 522 validation patches from `dset-s2`.
  - Metrics: $\text{IoU} = 0.8209$, $\text{Precision} = 0.8369$, $\text{Recall} = 0.9773$, $\text{F1} = 0.9017$, $\text{OA} = 95.82\%$, $\text{Cohen's Kappa} = 0.8754$.
  - Execution Time: 57.7 seconds. **Exit Code: 0**.
- **Stage 2 (`--stage 2`)**: Deep spectral water classification, multi-mask application (Assumed, DL Mask, NDWI Mask), unit harmonization (native physical units, RFU excluded), and basin disagreement analysis across 150,000 observations.
  - Retained: 139,327 assumed water rows, 47,120 DL-masked rows, 24,385 NDWI-masked rows.
  - Execution Time: 80.7 seconds. **Exit Code: 0**.
- **Stage 3 (`--stage 3`)**: 5-Fold Spatial Block Cross-Validation benchmark suite across Turbidity ($N=37,655$), Chlorophyll-a ($N=6,932$), and CDOM ($N=2,533$) across Empirical Baseline, Random Forest, Gradient Boosting, and Deep Spectral MLP.
  - Execution Time: 214.6 seconds. **Exit Code: 0**.
- **Stage 4 (`--stage 4`)**: Core scientific experiments:
  - Experiment A (Masking ablation under dual-space metrics and statistical significance testing).
  - Experiment B (Cross-basin portability: Within-basin vs. LOBO vs. Pooled across 5 river basins).
  - Experiment C (Multi-year temporal robustness: 2015–2021 training vs. 2022–2024 out-of-time evaluation).
  - Execution Time: 33.4 seconds. **Exit Code: 0**.

---

## 2. Turbidity Masking Summary Correction
- In `reports/final_stage3_benchmark_results.md` (Section 4) and `reports/journal_manuscript_report.md` (Abstract, Section 6, Section 7, and Conclusion), the historical single-split claim of an aggregate "+41.8% relative gain" ($0.2047 \to 0.2904$) has been corrected.
- Both reports now cite the validated 5-fold spatial block cross-validation results from `turbidity_multifold_cv_basin_consistency.csv` and `turbidity_multifold_cv_consolidated_summary.csv`:
  - **Aggregate Invariance**: Masking method choice does not produce a statistically significant difference in aggregate ($\text{Micro } \log R^2 = 0.2287 \pm 0.0960$ for Raw DL vs. $0.2585 \pm 0.1148$ for NDWI, paired Wilcoxon $p = 0.625$; $\text{Macro } \log R^2 = -1.4100 \pm 0.7060$ vs. $-1.3925 \pm 0.9646$, $p = 0.813$).
  - **Basin-Dependent Tradeoff**: NDWI masking wins in narrow agricultural and urban streams with riparian edge contamination (winning 80% of folds in Delaware [$\Delta = -0.2440$] and Illinois [$\Delta = -0.3700$]), whereas deep-learning segmentation wins in canopy-occluded or canyon terrain (winning 60% of folds in Willamette [$\Delta = +0.6267$], Trinity, and Upper Colorado [$\Delta = +0.0047$]), recovering up to 61.9% of canyon waters that NDWI misclassifies as land.

---

## 3. Cross-Document Consistency Audit
1. **Target Sample Sizes**: All active deliverables report unit-harmonized native-unit sample counts:
   - Turbidity: $N=37,655$ across 51 monitoring stations.
   - Chlorophyll-a: $N=6,932$ across 18 monitoring stations.
   - CDOM: $N=2,533$ across 6 monitoring stations.
   - Pre-harmonization RFU-mixed counts ($N=10,291$ and $N=3,251$) have been removed from all final summaries.
2. **Confounding & ICC Metrics**:
   - Chlorophyll-a: $\text{ICC} = 0.8181$ (81.8% of variance is between stations); within-site red-edge partial correlation collapses to near-zero ($+0.048$ for Gitelson 2-band, $+0.023$ for Gitelson 3-band).
   - CDOM: $\text{ICC} = 0.9354$ (93.5% of variance is between stations); zero-shot spatial transfer collapses across held-out stations.
3. **No Contradictory Claims**: Grep verification confirms zero occurrences of uncontextualized single-split claims across all reports.
