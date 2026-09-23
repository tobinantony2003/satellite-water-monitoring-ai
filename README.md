# Integrated Deep-Learning Water Extraction & Machine-Learning Water Quality Inversion Pipeline (Sentinel-2)

Targeting Publication in *Remote Sensing of Environment* / *International Journal of Remote Sensing* (Q1).

## 1. Overview & Theoretical Framework

This repository implements an end-to-end remote sensing research pipeline investigating how a deep-learning-based surface water extraction front-end impacts downstream satellite water quality inversion and cross-basin transferability.

The theoretical foundation is anchored in the literature review:
> **Zheng, Y., Wang, J., Kondratenko, Y., & Wu, J. (2024).** *"Research progress in surface water quality monitoring based on remote sensing technology."* **International Journal of Remote Sensing**, 45(7), 2337–2373. [DOI: 10.1080/01431161.2024.2327086](https://doi.org/10.1080/01431161.2024.2327086)

### Four Open Challenges Addressed

| Challenge (Zheng et al., 2024) | Pipeline Mechanism / Research Contribution | Status in this Project |
| :--- | :--- | :--- |
| **1. Atmospheric Interference** | L2A Bottom-Of-Atmosphere (BOA) surface reflectance with quality flag screening (`l2flag_center == 0`) and normalized spectral difference indices (NDCI, MNDWI, NDWI) to mitigate residual aerosol and path reflectance variations. | Addressed (optically active regimes) |
| **2. Poor Portability Across Basins** | Systematic cross-basin transferability experiments across 5 distinct US river basins (Delaware, Illinois, Trinity, Upper Colorado, Willamette) comparing within-basin upper bounds against Leave-One-Basin-Out (LOBO) zero-shot cross-transfer and pooled 5-basin joint models. | Addressed & quantified |
| **3. Low Sensor Resolution & Mixed Pixels** | Replacing naive threshold masking (NDWI > 0) with a multi-spectral 6-band Deep Learning U-Net (`WaterUNet`) and deep spectral classifier trained on global scenes (`dset-s2`) to eliminate riparian edge and sub-pixel mixed vegetation/soil contamination. | Core hypothesis & contribution |
| **4. External Confounding Factors** | Target log-transformation ($\log_{10}(y)$) to stabilize variance across orders of magnitude in seasonal runoff, paired with non-linear tree ensembles (Random Forest, Gradient Boosting) and Deep Spectral MLPs. | Addressed for optically active parameters |

> **Scope Note**: This project focuses on **optically active water quality parameters**:
> - **Chlorophyll-a ($Chl\text{-}a$)** ($\mu\text{g/L}$)
> - **Turbidity** ($\text{FNU}$)
> - **Colored Dissolved Organic Matter ($\text{CDOM} / f\text{DOM}$)** ($\mu\text{g/L QSE}$)
>
> Non-optically-active parameters (e.g. Biochemical Oxygen Demand [BOD], Dissolved Oxygen [DO], Total Nitrogen/Phosphorus) lack direct absorption/backscattering spectral features and are not modeled here.

---

## 2. Repository Architecture

```
satellite-water-monitoring-ai/
├── data/
│   ├── raw/
│   │   ├── dset-s2/             # Global 95-scene Sentinel-2 water dataset (Zenodo: 10.5281/zenodo.5205674)
│   │   └── matched_wq_s2/       # USGS matched S2 aquatic reflectance & in-situ WQ (DOI: 10.5066/P1A7T4FV)
│   └── processed/
│       └── matched_wq_cleaned/  # Masked parallel datasets: (a) Assumed, (b) DL Mask, (c) NDWI Mask
├── src/
│   ├── segmentation/
│   │   ├── dataset.py           # Patch extraction (256x256), per-scene standardization, flip/rot augmentation
│   │   ├── unet.py              # 6-channel PyTorch U-Net architecture
│   │   ├── losses.py            # Composite BCE + Soft Dice Loss
│   │   ├── evaluate.py          # IoU, Precision, Recall, F1, OA, Cohen's Kappa
│   │   ├── train.py             # Stage 1 training and validation pipeline
│   │   └── spectral_model.py    # Deep spectral classifier for point-scale masking
│   ├── inversion/
│   │   ├── build_matched_sets.py# Stage 2 masking, 3 parallel datasets, disagreement analysis
│   │   ├── models.py            # Empirical band-ratio, Random Forest, Gradient Boosting, Deep MLP
│   │   ├── spatial_cv.py        # Spatial Block GroupKFold CV by USGS gauge site
│   │   ├── metrics.py           # R², RMSE, MAE, MAPE, Bias in original physical units
│   │   ├── train_inversion.py   # Stage 3 5-fold spatial CV benchmarking
│   │   └── eda.py               # Exploratory data analysis & spectral signatures
│   └── evaluation/
│       ├── ablation.py          # Experiment A: Segmentation-quality ablation & Wilcoxon/t-tests
│       ├── transferability.py   # Experiment B: Cross-basin portability (Within vs LOBO vs Pooled)
│       └── temporal.py          # Experiment C: Temporal robustness (2015-2021 vs 2022-2024)
├── results/
│   ├── checkpoints/             # Saved PyTorch checkpoints (water_unet_best.pt, spectral_water_model.pt)
│   ├── figures/                 # Publication-grade figures (comparison maps, scatter plots, error curves)
│   └── tables/                  # Metrics CSVs and Markdown summaries
├── reports/
│   └── journal_manuscript_report.md # Synthesis report mapped to Zheng et al. (2024)
├── requirements.txt
└── README.md
```

---

## 3. Reproduction Pipeline

### Stage 1: Train Deep Learning Water Extraction Model
```bash
python -u -m src.segmentation.train
```
- Trains `WaterUNet` (6 input channels: B2, B3, B4, B8, B11, B12) on `dset-s2` 64 training scenes.
- Evaluates on 31 validation scenes.
- Outputs: `results/tables/stage1_segmentation_metrics.csv` and `results/figures/water_extraction_comparison.png`.

### Stage 2: Apply Masks & Construct Parallel Matched Sets
```bash
python -u -m src.inversion.build_matched_sets
```
- Trains deep spectral classifier on `dset-s2` pixel spectra.
- Filters and featurizes `Matched_WQ_S2` (1.7M records).
- Generates 3 parallel datasets: `(a) Assumed Ground Truth`, `(b) DL-Masked`, `(c) NDWI-Masked`.
- Quantifies cross-basin disagreement rates: `results/tables/stage2_disagreement_analysis.csv` and `results/figures/stage2_disagreement_rates.png`.

### Stage 3: Train & Benchmark Water Quality Inversion Models
```bash
python -u -m src.inversion.train_inversion
```
- Evaluates 4 architectures (Empirical, Random Forest, Gradient Boosting, Deep Spectral MLP) with 5-fold Spatial Block Cross-Validation across Chlorophyll-a, Turbidity, and CDOM.
- Generates Observed vs. Predicted scatter plots: `results/figures/stage3_*_obs_vs_pred.png`.

### Stage 4: Core Experiments
```bash
# Experiment A: Segmentation-Quality Ablation & Statistical Testing
python -u -m src.evaluation.ablation

# Experiment B: Cross-Basin Portability & Transferability
python -u -m src.evaluation.transferability

# Experiment C: Temporal Robustness
python -u -m src.evaluation.temporal
```

---

## 4. Key Citations

1. **Zheng et al. (2024)**. *Research progress in surface water quality monitoring based on remote sensing technology*. Int. J. Remote Sens. [DOI: 10.1080/01431161.2024.2327086](https://doi.org/10.1080/01431161.2024.2327086)
2. **Luo et al. (2021)**. *WatNet: A deep-learning network for surface water extraction from Sentinel-2 imagery*. [Zenodo: 10.5281/zenodo.5205674](https://zenodo.org/records/5205674)
3. **Ball & Ducar (2025)**. *Aquatic reflectance data from Sentinel-2 satellite imagery paired with continuous water quality measurements across 5 US river basins (2015-2024)*. USGS Data Release. [DOI: 10.5066/P1A7T4FV](https://doi.org/10.5066/P1A7T4FV)
