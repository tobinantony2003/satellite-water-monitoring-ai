# Satellite Water Monitoring AI: Sentinel-2 Water Extraction & Inversion Pipeline

A standardized remote sensing pipeline integrating deep-learning water body segmentation with machine-learning water quality parameter inversion across multi-basin USGS monitoring networks.

---

## 1. System Requirements & Environment Setup

### Software Requirements
- Python 3.10+ (tested on Python 3.11/3.12, Windows 11 and Linux)
- CUDA-compatible GPU optional (CPU inference supported out-of-the-box)

### Installation
Dependencies are defined in [`requirements.txt`](file:///c:/Term%20Paper%202/satellite-water-monitoring-ai/requirements.txt). Install via pip:

```bash
pip install -r requirements.txt
```

Core libraries include:
- `torch`, `torchvision`: Neural network modeling (`WaterUNet`, Deep Spectral Classifier, Deep Spectral MLP)
- `scikit-learn`, `xgboost`: Machine learning inversion algorithms (Random Forest, Gradient Boosting, Ridge)
- `pandas`, `numpy`, `scipy`, `statsmodels`: Data manipulation, mixed-effects modeling, and statistical tests
- `matplotlib`, `seaborn`, `tifffile`: Optical imagery handling and figure generation

---

## 2. Repository Structure

```
satellite-water-monitoring-ai/
├── data/
│   ├── raw/
│   │   ├── dset-s2/                   # Global 95-scene Sentinel-2 water extraction benchmark
│   │   └── matched_wq_s2/             # USGS Sentinel-2 aquatic reflectance matchups (1.7M rows)
│   └── processed/
│       └── matched_wq_cleaned/        # Cleaned and unit-harmonized parallel datasets
│           ├── matched_set_a_ground_truth.csv
│           ├── matched_set_b_dl_masked.csv
│           └── matched_set_c_ndwi_masked.csv
├── src/
│   ├── segmentation/                  # Stage 1: Water U-Net architecture, training, and evaluation
│   │   ├── unet.py                    # 6-channel PyTorch U-Net architecture
│   │   ├── dataset.py                 # Multi-spectral tile loading and spatial augmentation
│   │   ├── losses.py                  # Composite BCE + Soft Dice Loss
│   │   ├── train.py                   # Stage 1 training script
│   │   ├── run_eval.py                # Standalone evaluation on validation scenes
│   │   └── spectral_model.py          # Point-scale Deep Spectral Water Classifier
│   ├── inversion/                     # Stages 2 & 3: Data construction and inversion modeling
│   │   ├── build_matched_sets.py      # Parallel dataset construction & disagreement analysis
│   │   ├── apply_unit_harmonization.py# Native unit harmonization logic
│   │   ├── models.py                  # Model definitions (Empirical, RF, GBDT, MLP)
│   │   ├── train_inversion.py         # 5-fold spatial block cross-validation engine
│   │   ├── spatial_cv.py              # GroupKFold partitioning by station
│   │   ├── metrics.py                 # Dual-space metric calculations (log10 and physical)
│   │   └── eda.py                     # Exploratory data analysis & spectral signatures
│   └── evaluation/                    # Stage 4 & Diagnostic Audits
│       ├── ablation.py                # Experiment A: Masking quality ablation
│       ├── transferability.py         # Experiment B: Cross-basin portability (Within vs LOBO vs Pooled)
│       ├── temporal.py                # Experiment C: Multi-year temporal generalization
│       ├── splits.py                  # Canonical fold loading utilities
│       ├── audit_r2_computation_space.py # Audit of metric spaces
│       └── run_steps_8_to_12.py       # Master script for Steps 8 to 12
├── results/
│   ├── checkpoints/                   # Trained PyTorch models
│   ├── figures/                       # Publication figures and diagnostic plots
│   ├── tables/                        # Complete raw benchmark outputs (CSVs)
│   └── FINAL_RESULTS/                 # Curated canonical deliverables package
├── reports/
│   ├── final_stage3_benchmark_results.md # Final Stage 3 benchmark synthesis
│   ├── journal_manuscript_report.md      # Full manuscript technical synthesis
│   └── verification_note.md              # Pipeline verification note
├── run_pipeline.py                    # Master CLI pipeline runner
├── requirements.txt                   # Environment package dependencies
└── README.md                          # Project documentation
```

---

## 3. How to Reproduce Each Pipeline Stage

The master pipeline is orchestrated via [`run_pipeline.py`](file:///c:/Term%20Paper%202/satellite-water-monitoring-ai/run_pipeline.py):

### Run the Full Pipeline End-to-End
```bash
python run_pipeline.py --stage all
```

### Run Individual Stages

#### Stage 1: Water Extraction Model Evaluation
Evaluates the trained 6-channel Water U-Net checkpoint against 31 held-out Sentinel-2 scenes (522 patches):
```bash
python run_pipeline.py --stage 1
```
*Outputs: `results/tables/stage1_segmentation_metrics.csv` and `results/figures/water_extraction_comparison.png`.*

#### Stage 2: Water Mask Application & Parallel Dataset Construction
Applies the deep spectral classifier and NDWI thresholds to matched USGS gauge observations, performs unit harmonization, and computes basin disagreement rates:
```bash
python run_pipeline.py --stage 2
```
*Outputs: `data/processed/matched_wq_cleaned/matched_set_{a,b,c}*.csv` and `results/tables/stage2_disagreement_analysis.csv`.*

#### Stage 3: Inversion Model Suite with 5-Fold Spatial Block CV
Evaluates Empirical Baseline, Random Forest, Gradient Boosting, and Deep Spectral MLP under strict spatial station grouping:
```bash
python run_pipeline.py --stage 3
```
*Outputs: `results/tables/stage3_consolidated_inversion_benchmarks.csv` and observed-vs-predicted plots.*

#### Stage 4: Core Experiments & Statistical Validation
Executes Experiment A (masking ablation), Experiment B (cross-basin transferability), and Experiment C (multi-year temporal split):
```bash
python run_pipeline.py --stage 4
```
*Outputs: `results/tables/stage4_experiment_{a,b,c}*.csv`.*

#### Steps 8 to 12 Master Script
Executes temporal filtering, Gitelson red-edge audit, OWT clustering, winsorization, and final multi-model spatial CV benchmarks:
```bash
python -m src.evaluation.run_steps_8_to_12
```

---

## 4. Contents of `results/FINAL_RESULTS/`

The `results/FINAL_RESULTS/` directory contains exclusively curated, canonical outputs:

### 1. `final_metrics_summary.csv`
Consolidated reference table recording:
- Target parameter (`Turbidity`, `Chlorophyll-a`, `CDOM`)
- Best model architecture
- Macro $\log_{10} R^2$ and Pooled $\log_{10} R^2$
- Intra-Class Correlation (ICC)
- Sample size ($N$) and station count
- Masking ablation statistical significance ($p$-value)

### 2. `checkpoints/`
- `water_unet_best.pt`: Validated 6-channel PyTorch Water U-Net checkpoint ($\text{IoU} = 0.8209$, $\text{OA} = 95.82\%$).
- `spectral_water_model.pt`: Validated 6-channel PyTorch Deep Spectral Water Classifier for point-scale extraction.

### 3. `datasets/`
Unit-harmonized primary datasets (native physical units only, RFU excluded):
- `matched_set_a_ground_truth.csv`: Assumed water baseline ($N = 139,327$).
- `matched_set_b_dl_masked.csv`: Deep learning segmented mask ($N = 47,120$).
- `matched_set_c_ndwi_masked.csv`: Naive NDWI index mask ($N = 24,385$).

### 4. `stage3_benchmarks/`
- `final_stage3_benchmark_consolidated.csv`: Full 12-row table across all 4 model families and 3 targets.
- `final_stage3_benchmark_turbidity.csv`: Turbidity spatial block CV breakdown.
- `final_stage3_benchmark_chlorophyll.csv`: Chlorophyll-a spatial block CV breakdown.
- `final_stage3_benchmark_cdom.csv`: CDOM spatial block CV breakdown.
- `stage3_consolidated_inversion_benchmarks.csv`: Baseline multi-model outputs.

### 5. `core_experiments/`
Validated 5-fold spatial cross-validation experiment tables:
- `turbidity_multifold_cv_consolidated_summary.csv`: Multifold ablation summary across conditions.
- `turbidity_multifold_cv_basin_consistency.csv`: Basin-level win rates and delta margins for DL vs. NDWI.
- `stage4_experiment_a_ablation_chlorophyll_5fold_summary.csv`: 5-fold chlorophyll masking ablation.
- `stage4_experiment_b_transferability_dual_space.csv`: Within-basin vs. LOBO vs. Pooled transferability across 5 basins.
- `stage4_experiment_c_temporal_dual_space.csv`: Multi-year temporal robustness (2015–2021 vs. 2022–2024).
- `step8_temporal_window_inversion_summary.csv`: Temporal match window sensitivity (12h vs. 3h vs. 1h vs. continuous $\Delta t$).
- `step9_gitelson_inversion_cv.csv`: Gitelson red-edge indices spatial CV benchmark.
- `step10_owt_model_performance.csv`: Optical Water Type clustering model evaluation.
- `step11_winsorization_summary.csv`: Outlier winsorization evaluation.
- `mixed_effects_harmonization_comparison.csv`: ICC and variance partitioning before and after unit harmonization.

### 6. `figures/`
Curated publication-ready visualizations:
- `water_extraction_comparison.png`: U-Net segmentation vs. NDWI vs. RGB composite.
- `stage2_disagreement_rates.png`: DL vs. NDWI disagreement rate by river basin.
- `eda_target_distributions.png`: Raw vs. $\log_{10}$ parameter distributions.
- `eda_spectral_signatures.png`: Sentinel-2 aquatic surface reflectance across water clarity tiers.
- `stage3_{turbidity,chlorophyll,cdom}_obs_vs_pred.png`: Spatial block cross-validation scatter plots.
- `stage4_experiment_b_transferability_{turbidity,chlorophyll}.png`: Transferability degradation bar charts.
- `chlorophyll_harmonized_site_time_overlap.png`: Station temporal coverage and overlap heatmaps.
- `{turbidity,chlorophyll,cdom}_confounding_check.png`: Raw vs. within-site partial correlation diagnostic plots.
