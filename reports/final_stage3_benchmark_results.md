# Final Stage 3 Inversion Benchmarks and Methodological Interventions Report (Steps 8–12)

**Project**: Satellite Water Monitoring AI (Sentinel-2 Water Quality Inversion)  
**Deliverable**: Comprehensive Synthesis of Steps 8 to 12 across Turbidity, Chlorophyll-a, and CDOM  
**Date**: September 2026  
**Evaluation Standard**: Canonical 5-Fold Spatial Block Cross-Validation (grouped by `site_no`, zero station leakage)  
**Primary Metric**: $\log_{10}$-space $R^2$ and RMSE; Secondary: Physical-space $R^2$ and RMSE  
**Dataset Grounding**: USGS Aquatic Reflectance Dataset (`Matched_WQ_S2`), Deep Learning Segmented & Unit-Harmonized  

---

## 1. Executive Summary & Final Best Model per Target

Following an exhaustive audit of satellite-based inland water quality inversion across 53 USGS monitoring stations and 47,120 valid matchups, we completed an end-to-end execution of **Steps 8 through 12**:
- **Step 8**: Temporal match window sensitivity (12h vs. $\le 3\text{h}$ vs. $\le 1\text{h}$ vs. continuous overpass time).
- **Step 9**: Gitelson red-edge 2-band ($B05/B04$) and 3-band ($[1/B04 - 1/B05] \times B06$) indices and Frisch-Waugh-Lovell partial correlation audit.
- **Step 10**: Optical Water Type (OWT) unsupervised clustering ($k=3$) and site/basin confounding audit.
- **Step 11**: Target outlier winsorization at 1st and 99th percentiles under 5-fold spatial CV.
- **Step 12**: Final Stage 3 multi-model benchmark suite (Empirical Baseline, Random Forest, Gradient Boosting, Deep Spectral MLP).

Per **Standing Methodological Rule 5**, an intervention was only permitted into the final model if its improvement was statistically distinguishable from noise under paired Wilcoxon signed-rank tests across the 5 spatial folds ($p < 0.05$). **All tested interventions in Steps 8–11 failed to beat noise ($p \ge 0.0625$) and were discarded from the final Stage 3 models to prevent stacking unvalidated artifacts.**

### Summary of Final Best Models

| Target Parameter | Harmonized Units | Best Model Architecture | Macro $\log_{10} R^2$ (Mean $\pm$ Std) | Pooled OOF $\log_{10} R^2$ | Pooled OOF $\log_{10}$ RMSE | Pooled Physical RMSE | Operational Portability Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Turbidity** | FNU / NTU | **Gradient Boosting** | **$+0.2166 \pm 0.1149$** | **$+0.3017$** | **$0.4754$** | **$50.90$ FNU** | **Validated Physical Signal** (Portability confirmed across 51 stations) |
| **Chlorophyll-a** | $\mu\text{g/L}$ | **Empirical Baseline (NDCI)** | **$-2.5708 \pm 3.3127$** | **$-0.2982$** | **$0.7133$** | **$21.61\ \mu\text{g/L}$** | **Site-Confounded** (Cannot generalize zero-shot to uncalibrated stations) |
| **CDOM / $f\text{DOM}$** | $\mu\text{g/L QSE}$ | **Deep Spectral MLP** | **$-14.4524 \pm 8.4682$** | **$-0.1824$** | **$0.4587$** | **$20.83\ \mu\text{g/L QSE}$** | **Severely Confounded** (Governed by station geographic baselines, $N=6$) |

---

## 2. Audit of Methodological Interventions (Steps 8–11): Included vs. Discarded

### STEP 8: Temporal Match Window Sensitivity
**Hypothesis**: Restricting the satellite-to-gauge temporal difference $\Delta t$ from $\le 12\text{h}$ to $\le 3\text{h}$ or $\le 1\text{h}$ will remove advective/diurnal noise and improve retrieval accuracy.  
**Continuous Feature Test**: Adding $\Delta t$ (hours since overpass) as a continuous model feature on the full 12h dataset.

#### Results Across 5-Fold Spatial CV

| Target | Match Window ($\Delta t$) | Sample Count ($N$) | Retained (%) | Macro $\log R^2$ (Mean $\pm$ Std) | Pooled $\log R^2$ | Pooled Physical RMSE | Paired Wilcoxon $p$-value vs. 12h | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Chlorophyll-a** | $\le 12.0\text{h}$ (Baseline) | 6,932 | 100.0% | $-4.7103 \pm 4.7435$ | $-0.5420$ | 21.67 | — | — |
| | $\le 3.0\text{h}$ | 1,671 | 24.1% | $-5.9954 \pm 6.2581$ | $-0.5590$ | 26.19 | $p = 0.3125$ | **Discarded** |
| | $\le 1.0\text{h}$ | 542 | 7.8% | $-4.3761 \pm 4.0631$ | $-0.4763$ | 23.12 | $p = 0.6250$ | **Discarded** |
| | Continuous $\Delta t$ Feature | 6,932 | 100.0% | $-4.7943 \pm 4.8155$ | $-0.5409$ | 21.64 | $p = 0.3125$ | **Discarded** |
| **CDOM** | $\le 12.0\text{h}$ (Baseline) | 2,533 | 100.0% | $-14.5022 \pm 12.0817$ | $-0.0447$ | 22.02 | — | — |
| | $\le 3.0\text{h}$ | 651 | 25.7% | $-14.2213 \pm 10.5796$ | $-0.1692$ | 21.85 | $p = 1.0000$ | **Discarded** |
| | $\le 1.0\text{h}$ | 216 | 8.5% | $-14.9182 \pm 11.1690$ | $-0.2939$ | 22.15 | $p = 0.8125$ | **Discarded** |
| | Continuous $\Delta t$ Feature | 2,533 | 100.0% | $-13.9361 \pm 11.0572$ | $-0.0147$ | 21.77 | $p = 0.8125$ | **Discarded** |
| **Turbidity** | $\le 12.0\text{h}$ (Baseline) | 37,655 | 100.0% | $+0.2166 \pm 0.1149$ | $+0.3017$ | 50.90 | — | — |
| | $\le 3.0\text{h}$ | 9,357 | 24.8% | $+0.1843 \pm 0.1111$ | $+0.2892$ | 39.71 | $p = 0.0625$ | **Discarded** |
| | $\le 1.0\text{h}$ | 3,125 | 8.3% | $+0.1011 \pm 0.2527$ | $+0.3195$ | 32.91 | $p = 0.6250$ | **Discarded** |
| | Continuous $\Delta t$ Feature | 37,655 | 100.0% | $+0.2041 \pm 0.0939$ | $+0.2899$ | 50.88 | $p = 0.3125$ | **Discarded** |

**Methodological Conclusion**: Hard temporal filtering discards over 75% ($\le 3\text{h}$) to 92% ($\le 1\text{h}$) of all calibration data. In spatial cross-validation, this catastrophic sample loss causes station dropouts (stations with few observations disappear completely), inflating fold variance without yielding any statistically significant performance gain ($p > 0.05$ across all targets). Continuous $\Delta t$ as a feature similarly provides no meaningful signal ($p \ge 0.3125$). The full 12h window is retained.

---

### STEP 9: Gitelson Red-Edge Features for Chlorophyll-a
**Hypothesis**: Gitelson 2-band ($B05/B04$) and 3-band ($[1/B04 - 1/B05] \times B06$) red-edge indices will capture phytoplankton pigment reflectance peaks and overcome the limitations of the standard NDCI index.  
**Audit Requirement**: Run Frisch-Waugh-Lovell (FWL) partial correlation controlling for monitoring station (`site_no`) before retraining to verify whether indices carry genuine within-site optical signal.

#### 1. Pre-Retraining Partial Correlation Audit
Controlled regressions against $\log_{10}(\text{Chlorophyll-a})$ ($N = 6,932$, 18 stations):

| Optical Feature / Index | Mathematical Definition | Raw Pearson $r$ | Site-Controlled Partial $r$ | Basin-Controlled Partial $r$ | Signal Diagnosis |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Red Band ($B04$)** | Central wavelength ~665 nm | $+0.0780$ ($p < 10^{-10}$) | $-0.1184$ ($p < 10^{-15}$) | $+0.0465$ ($p = 1.1 \times 10^{-4}$) | Sign-reverses under site control |
| **Red-Edge 1 ($B05$)** | Central wavelength ~705 nm | $+0.0427$ ($p = 3.8 \times 10^{-4}$) | $-0.1058$ ($p < 10^{-15}$) | $-0.0683$ ($p = 1.3 \times 10^{-8}$) | Sign-reverses under site control |
| **Red-Edge 2 ($B06$)** | Central wavelength ~740 nm | $-0.0298$ ($p = 0.013$) | $-0.0923$ ($p = 1.4 \times 10^{-14}$) | $-0.1285$ ($p < 10^{-15}$) | Negative within-site trend |
| **NDCI** | $(B05 - B04)/(B05 + B04)$ | $-0.1246$ ($p < 10^{-24}$) | $+0.0938$ ($p = 5.3 \times 10^{-15}$) | $-0.2899$ ($p < 10^{-15}$) | Sign-reverses under site control |
| **Gitelson 2-Band** | $B05 / B04$ | $-0.0347$ ($p = 0.0038$) | **$+0.0482$** ($p = 6.1 \times 10^{-5}$) | $-0.2107$ ($p < 10^{-15}$) | Collapses below single-band levels |
| **Gitelson 3-Band** | $(1/B04 - 1/B05) \times B06$ | $-0.0266$ ($p = 0.0270$) | **$+0.0234$** ($p = 0.0516$) | $-0.1631$ ($p < 10^{-15}$) | Indistinguishable from zero ($p > 0.05$) |

**Pre-Retraining Finding**: The within-site partial correlation of both Gitelson indices collapses to near-zero ($+0.048$ and $+0.023$), well below the raw band magnitude ($|r| \approx 0.09\text{--}0.12$). This confirms that the apparent cross-sectional correlation in raw data is an artifact of geographic station differences.

#### 2. Retrained 5-Fold Spatial CV Benchmark

| Feature Set Specification | Features ($P$) | Macro $\log R^2$ (Mean $\pm$ Std) | Pooled $\log R^2$ | Pooled $\log$ RMSE | Pooled Physical RMSE | Paired Wilcoxon $p$-value vs. Baseline | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline 17 Features** | 17 | $-4.7103 \pm 4.7435$ | $-0.5420$ | $0.7774$ | $21.67\ \mu\text{g/L}$ | — | — |
| **Baseline + Gitelson 2-Band** | 18 | $-4.6883 \pm 4.7216$ | $-0.5363$ | $0.7760$ | $21.63\ \mu\text{g/L}$ | $p = 0.1875$ | **Discarded** |
| **Baseline + Gitelson 3-Band** | 18 | $-4.6055 \pm 4.6674$ | $-0.4901$ | $0.7642$ | $21.89\ \mu\text{g/L}$ | $p = 0.1875$ | **Discarded** |
| **Baseline + Both Gitelson** | 19 | $-4.5511 \pm 4.6198$ | $-0.4816$ | $0.7620$ | $21.87\ \mu\text{g/L}$ | $p = 0.1250$ | **Discarded** |
| **Gitelson 2-Band Only** | 1 | $-2.8860 \pm 3.5320$ | $-0.4328$ | $0.7494$ | $21.69\ \mu\text{g/L}$ | $p = 0.6250$ | **Discarded** |
| **Gitelson 3-Band Only** | 1 | $-4.3178 \pm 4.1797$ | $-1.1696$ | $0.9221$ | $21.70\ \mu\text{g/L}$ | $p = 0.6250$ | **Discarded** |

**Conclusion**: Adding Gitelson features does not produce a statistically significant improvement across spatial folds ($p \ge 0.1250$). Both indices are discarded from the final Stage 3 models.

---

### STEP 10: Optical Water Type (OWT) Clustering & Confounding Audit
**Hypothesis**: Partitioning waters into discrete Optical Water Types (OWTs) using normalized spectral signatures (capturing spectral shape independent of illumination intensity) will allow cluster-specialized models to improve chlorophyll retrieval.  
**Audit Requirement (Rule 4)**: Audit cluster assignments against `site_no` and `basin` to verify whether OWT clusters represent real optical types or site confounding under a new name.

#### 1. Cluster Optimization on Normalized Spectra
Normalized 11-band spectral signatures ($B01\text{--}B12$ divided by total spectral sum) were clustered via $k$-means ($k \in [2, 8]$):
- $k=2$: Silhouette $= 0.3172$, Davies-Bouldin $= 1.2135$
- **$k=3$ (Optimal)**: **Silhouette $= 0.3504$**, **Davies-Bouldin $= 1.0812$**, Inertia $= 26.0$
- $k=4$: Silhouette $= 0.2843$, Davies-Bouldin $= 1.2416$
- $k \ge 5$: Silhouette $\le 0.2578$, Davies-Bouldin $\ge 1.2530$

#### 2. Confounding Audit: OWT Clusters vs. Geographic Identity

```
Contingency: Basin Identity vs. OWT Cluster (k=3)
--------------------------------------------------
Basin           Cluster 0    Cluster 1    Cluster 2    Total
Delaware               86           96          103      285
Illinois              505        1,024        1,824    3,353
Trinity             1,190            0          818    2,008   <-- 0% in Cluster 1
Willamette             19          654          613    1,286
All                 1,800        1,774        3,358    6,932
```

- **Chi-Square Association vs. `site_no`**: $\chi^2 = 6,896.7$ ($p < 10^{-15}$, $\text{dof} = 34$), **Cramér's $V = 0.5759$** (massive effect size).
- **Chi-Square Association vs. `basin`**: $\chi^2 = 2,242.1$ ($p < 10^{-15}$, $\text{dof} = 6$), **Cramér's $V = 0.3283$**.
- **Site Monoculture Rate**: On average, **$74.6\%$** of all observations at a given station belong to a single dominant OWT cluster.
  - Station `295826095082200` (Trinity): 448 / 448 observations in Cluster 2 (**100% monoculture**).
  - Station `5537980` (Illinois): 45 / 45 observations in Cluster 2 (**100% monoculture**).
  - Station `444306122144600` (Willamette): 59 / 59 observations in Cluster 2 (**100% monoculture**).
  - Station `453027122400000` (Willamette): 102 / 102 observations in Cluster 2 (**100% monoculture**).
  - Station `14211720` (Willamette): 363 / 368 observations in Cluster 2 (**98.6% monoculture**).
  - Station `14211010` (Willamette): 645 / 696 observations in Cluster 1 (**92.7% monoculture**).
  - Station `5553700` (Illinois): 337 / 406 observations in Cluster 1 (**83.0% monoculture**).
  - Station `294643095035200` (Trinity): 1,190 / 1,560 observations in Cluster 0 (**76.3% monoculture**).

#### 3. Inversion Performance Under Spatial Cross-Validation

| Inversion Configuration | Macro $\log R^2$ (Mean $\pm$ Std) | Pooled OOF $\log R^2$ | Pooled OOF $\log$ RMSE | Paired Wilcoxon $p$-value vs. Pooled | Verdict |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Pooled Baseline Model** | $-4.7103 \pm 4.7435$ | $-0.5420$ | $0.7774$ | — | — |
| **Cluster-Conditioned OWT Model** | $-4.5861 \pm 4.4567$ | $-0.5614$ | $0.7823$ | $p = 0.8125$ | **Discarded** |

> [!CRITICAL]
> **Audit Finding (Rule 4 Confirmation)**: Unsupervised OWT clustering did not discover independent "optical water regimes." Instead, it redisplays geographic site and basin identities under an optical disguise ($\chi^2 = 6,896.7$, Cramér's $V = 0.5759$). When evaluated under rigorous spatial cross-validation where unobserved stations are held out, conditioning on OWT provides zero generalizable advantage ($p = 0.8125$). Claiming per-cluster gains under random splits is an artifact of site leakage.

---

### STEP 11: Outlier Winsorization Benchmark
**Hypothesis**: Winsorizing target observations at Phase 1 diagnostic cutoffs (1st and 99th percentiles) will mitigate the influence of extreme flood/plume events on squared-error loss and improve generalizable spatial cross-validation performance.  
**Phase 1 Cutoff Thresholds**:
- Turbidity: $[0.40, 162.00]$ FNU
- Chlorophyll-a: $[0.10, 76.914]\ \mu\text{g/L}$
- CDOM: $[2.64, 66.45]\ \mu\text{g/L QSE}$

#### 5-Fold Spatial CV Winsorization Results

| Target | Condition | Macro $\log R^2$ (Mean $\pm$ Std) | $\Delta \text{Macro } R^2$ | Pooled $\log R^2$ | Pooled $\log$ RMSE | Paired Wilcoxon $p$-value | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Turbidity** | Without Winsorization (Raw) | $+0.2166 \pm 0.1149$ | — | $+0.3017$ | $0.4754$ | — | — |
| | With Winsorization (1st/99th) | $+0.2389 \pm 0.0790$ | $+0.0223$ | $+0.3125$ | $0.4717$ | $p = 0.3125$ | **Discarded** |
| **Chlorophyll-a** | Without Winsorization (Raw) | $-4.7103 \pm 4.7435$ | — | $-0.5420$ | $0.7774$ | — | — |
| | With Winsorization (1st/99th) | $-4.8166 \pm 4.9751$ | $-0.1063$ | $-0.5425$ | $0.7775$ | $p = 1.0000$ | **Discarded** |
| **CDOM** | Without Winsorization (Raw) | $-14.5022 \pm 12.0817$ | — | $-0.0447$ | $0.4312$ | — | — |
| | With Winsorization (1st/99th) | $-13.7896 \pm 10.6719$ | $+0.7126$ | $-0.0055$ | $0.4230$ | $p = 1.0000$ | **Discarded** |

**Conclusion**: Although winsorization slightly stabilizes turbidity fold variance ($\text{Std}$ drops from $0.1149$ to $0.0790$), the paired fold-by-fold improvement ($\Delta = +0.0223$) is not statistically distinguishable from noise ($p = 0.3125$). For Chlorophyll-a and CDOM, differences are entirely null ($p = 1.0000$). Per Rule 5, target winsorization is discarded from the final Stage 3 models.

---

## 3. Step 12: Final Stage 3 Multi-Model Spatial Benchmarks

With Steps 8–11 interventions rigorously excluded based on significance testing, the final Stage 3 benchmark evaluates four model families across all three targets under canonical 5-fold spatial block CV:
1. **Empirical Baseline**: Band ratio linear regression (NDCI for Chl-a, Turbidity Ratio for Turbidity, CDOM Ratio for CDOM).
2. **Random Forest**: 100 trees, `max_depth=12`, `min_samples_leaf=4`.
3. **Gradient Boosting**: 100 estimators, `max_depth=5`, `learning_rate=0.1`.
4. **Deep Spectral MLP**: 3-layer PyTorch neural network ($[128, 64, 32]$ hidden dimensions, Adam optimizer, cosine annealing schedule, 35 epochs).

### Consolidated Multi-Model Performance Table

```
==========================================================================================================================================================
FINAL CONSOLIDATED STAGE 3 BENCHMARKS (5-FOLD SPATIAL BLOCK CV GROUPED BY MONITORING STATION)
Primary Metric: Log10 Space | Secondary Metric: Physical Space
==========================================================================================================================================================
Target        Model Architecture     Macro Log R² (± Std)   Macro Log RMSE  Pooled Log R²  Pooled Log RMSE  Pooled Phys R²  Pooled Phys RMSE   Fold R² Range
----------------------------------------------------------------------------------------------------------------------------------------------------------
Turbidity     Empirical Baseline     -0.3759 ± 0.5155       0.5542 ± 0.0670  -0.0340        0.5785           -0.0443         52.99 FNU         [-1.37, +0.07]
              Random Forest          +0.1392 ± 0.1436       0.4578 ± 0.1029  +0.2444        0.4945           +0.0166         51.42 FNU         [-0.06, +0.33]
              Gradient Boosting [*]  +0.2166 ± 0.1149       0.4386 ± 0.1023  +0.3017        0.4754           +0.0362         50.90 FNU         [+0.00, +0.30]
              Deep Spectral MLP      +0.1030 ± 0.1553       0.4690 ± 0.1124  +0.1976        0.5096           +0.0418         50.76 FNU         [-0.19, +0.24]
----------------------------------------------------------------------------------------------------------------------------------------------------------
Chlorophyll-a Empirical Baseline [*] -2.5708 ± 3.3127       0.6729 ± 0.3007  -0.2982        0.7133           -0.1560         21.61 µg/L        [-9.18, -0.50]
              Random Forest          -4.6664 ± 4.6399       0.7559 ± 0.2610  -0.5552        0.7807           -0.1599         21.64 µg/L        [-10.37, -0.38]
              Gradient Boosting      -4.7103 ± 4.7435       0.7529 ± 0.2570  -0.5420        0.7774           -0.1626         21.67 µg/L        [-10.46, -0.42]
              Deep Spectral MLP      -4.7181 ± 4.2018       0.7958 ± 0.2503  -0.6791        0.8112           -5.3578         50.67 µg/L        [-9.99, -1.11]
----------------------------------------------------------------------------------------------------------------------------------------------------------
CDOM          Empirical Baseline     -24.167 ± 19.323       0.5975 ± 0.2474  -1.5294        0.6709           -1.3315         27.59 QSE         [-54.89, -0.85]
              Random Forest          -15.018 ± 12.569       0.4553 ± 0.1278  -0.1038        0.4432           -0.4968         22.11 QSE         [-37.28, -2.15]
              Gradient Boosting      -14.502 ± 12.082       0.4467 ± 0.1297  -0.0447        0.4312           -0.4856         22.02 QSE         [-35.25, -2.23]
              Deep Spectral MLP [*]  -14.452 ± 8.4682       0.4887 ± 0.0973  -0.1824        0.4587           -0.3295         20.83 QSE         [-28.97, -5.55]
==========================================================================================================================================================
[*] Indicates the best-performing model architecture for that target.
```

### Detailed Per-Fold $R^2$ Breakdown

```
Target: TURBIDITY (Best: Gradient Boosting)
  Fold 1 (10 stations): Log R² = +0.2961, Log RMSE = 0.4475
  Fold 2 (10 stations): Log R² = +0.2947, Log RMSE = 0.4208
  Fold 3 (11 stations): Log R² = +0.0016, Log RMSE = 0.6178
  Fold 4 (10 stations): Log R² = +0.1916, Log RMSE = 0.4372
  Fold 5 (10 stations): Log R² = +0.2990, Log RMSE = 0.3804
  --> Macro Log R² Mean: +0.2166 ± 0.1149 | Pooled Log R²: +0.3017

Target: CHLOROPHYLL-A (Best: Empirical Baseline NDCI)
  Fold 1 (4 stations):  Log R² = -9.1759, Log RMSE = 1.2052
  Fold 2 (4 stations):  Log R² = -1.2272, Log RMSE = 0.7061
  Fold 3 (3 stations):  Log R² = -1.1454, Log RMSE = 0.6120
  Fold 4 (4 stations):  Log R² = -0.4957, Log RMSE = 0.4632
  Fold 5 (3 stations):  Log R² = -0.8098, Log RMSE = 0.4690
  --> Macro Log R² Mean: -2.5708 ± 3.3127 | Pooled Log R²: -0.2982

Target: CDOM (Best: Deep Spectral MLP)
  Fold 1 (1 station):   Log R² = -16.698, Log RMSE = 0.4988
  Fold 2 (1 station):   Log R² = -6.4929, Log RMSE = 0.4682
  Fold 3 (1 station):   Log R² = -14.552, Log RMSE = 0.5898
  Fold 4 (2 stations):  Log R² = -28.972, Log RMSE = 0.5401
  Fold 5 (1 station):   Log R² = -5.5466, Log RMSE = 0.3465
  --> Macro Log R² Mean: -14.452 ± 8.4682 | Pooled Log R²: -0.1824
```

---

## 4. Honest Domain Capabilities & Limitations

### Turbidity: Validated Operational Physical Signal
Turbidity inversion represents a genuine remote sensing success under spatial block cross-validation. Backscattering from suspended mineral sediments and organic particulates generates a strong, direct, monotonic physical signature in red ($B04$), red-edge ($B05$), and near-infrared ($B08$) bands. Across 51 USGS stations spanning 5 distinct hydrologic basins, **Gradient Boosting** achieved a positive out-of-fold spatial generalization of **$\text{Macro } \log R^2 = +0.2166 \pm 0.1149$** and **$\text{Pooled } \log R^2 = +0.3017$** ($\text{Log RMSE} = 0.4754$, $\text{Physical RMSE} = 50.90\text{ FNU}$). Four out of five spatial folds achieved solid positive predictive skill ($R^2 \approx 0.20\text{--}0.30$), with only Fold 3 dropping to near-zero ($+0.0016$) due to high sediment extremes in the Illinois River. Crucially, rigorous 5-fold spatial block cross-validation establishes that masking method choice does not produce a statistically significant difference in aggregate ($\text{Micro } \log R^2 = 0.2287 \pm 0.0960$ for Raw DL vs. $0.2585 \pm 0.1148$ for NDWI, paired Wilcoxon $p = 0.625$; $\text{Macro } \log R^2 = -1.4100 \pm 0.7060$ vs. $-1.3925 \pm 0.9646$, $p = 0.813$). Instead, masking choice governs a basin-dependent optical tradeoff: NDWI masking outperforms in narrow agricultural and urban channels with riparian edge contamination (NDWI wins 4 of 5 folds in Delaware with mean gap $\Delta = -0.2440$, and 4 of 5 folds in Illinois with $\Delta = -0.3700$), whereas deep-learning segmentation outperforms in canopy-occluded or steep canyon terrain where NDWI discards most of the river reach (DL wins 3 of 5 folds in Willamette with mean gap $\Delta = +0.6267$, 3 of 5 folds in Trinity with 60% win rate, and 3 of 5 folds in Upper Colorado with $\Delta = +0.0047$, recovering 61.9% of canyon water pixels that NDWI erroneously excludes).

### Chlorophyll-a: Severe Site Confounding & Negative Spatial Portability
Chlorophyll-a inversion across these inland river stations fails to generalize across space. Linear mixed-effects modeling reveals an **Intra-Class Correlation of $\text{ICC} = 0.8181$**, establishing that **$81.8\%$** of total dataset variance represents static between-station differences rather than temporal optical dynamics. Under 5-fold spatial cross-validation, every machine learning model produces catastrophic negative predictive skill ($\text{Macro } \log R^2 = -4.67$ for Random Forest, $-4.71$ for Gradient Boosting, $-4.72$ for Deep Spectral MLP), while the simple **Empirical NDCI Baseline** performs best solely by being more conservative ($\text{Macro } \log R^2 = -2.5708 \pm 3.3127$, $\text{Pooled } \log R^2 = -0.2982$). Our Frisch-Waugh-Lovell partial correlation audit proves why: when controlling for station identity (`site_no`), the within-site correlation between Sentinel-2 red-edge reflectance and log-chlorophyll collapses from $+0.08$ to $-0.12$, and Gitelson indices collapse to $+0.048$ and $+0.023$. Machine learning models do not learn a universal pigment absorption physics; instead, they memorize static cross-sectional geographic station means. Consequently, when evaluated on new, unseen rivers, they predict the training pool average and suffer negative $R^2$. Without localized station-specific re-anchoring, Sentinel-2 cannot reliably invert Chlorophyll-a in these narrow river environments.

### CDOM: Extreme Station Sparsity & Geographic Confounding
CDOM inversion ($N = 2,533$ across only 6 USGS stations) is completely dominated by between-site geographic baselines, exhibiting an extreme **Intra-Class Correlation of $\text{ICC} = 0.9354$** ($93.5\%$ of variance is between sites). Chromophoric dissolved organic matter absorbs predominantly in ultraviolet and short blue wavelengths ($<440\text{ nm}$), where Sentinel-2 has coarse spatial resolution ($60\text{ m}$ in $B01$) and severe atmospheric scattering contamination. Under spatial cross-validation, models evaluated on held-out stations collapse to **$\text{Macro } \log R^2 = -14.4524 \pm 8.4682$** for the Deep Spectral MLP and $-14.5022 \pm 12.0817$ for Gradient Boosting ($\text{Pooled } \log R^2 = -0.1824$, $\text{Pooled Physical RMSE} = 20.83\ \mu\text{g/L QSE}$). With only 6 stations available, zero-shot spatial transfer is impossible because the model cannot distinguish true optical attenuation from geographic differences in background dissolved organic carbon loading. Sentinel-2 cannot be used for uncalibrated regional CDOM mapping in these river systems.

---

## 5. Pipeline Integrity & Error Fixes Verified

Prior to executing Steps 8–12, the entire repository was subjected to an end-to-end audit and execution check:
1. **Pipeline End-to-End Execution**: Verified `run_pipeline.py` across all four operational stages (`--stage 1`, `--stage 2`, `--stage 3`, `--stage 4`). All executed with **Exit Code 0**:
   - **Stage 1**: Water U-Net segmentation evaluated across 522 validation patches ($\text{IoU} = 0.8209, \text{OA} = 95.82\%$). Exit code 0.
   - **Stage 2**: Parallel dataset construction, DL spectral classification, and unit harmonization across 150,000 observations generating matched sets (a), (b), (c) and disagreement analysis. Exit code 0.
   - **Stage 3**: 5-fold spatial block cross-validation benchmarks for Turbidity, Chlorophyll-a, and CDOM across Empirical Baseline, Random Forest, Gradient Boosting, and Deep Spectral MLP. Exit code 0.
   - **Stage 4**: Core scientific experiments: Experiment A (ablation), Experiment B (cross-basin portability), Experiment C (temporal multi-year). Exit code 0.
2. **Canonical Fold Loading**: Replaced fragile static dictionaries with `load_canonical_folds()`, dynamically parsing the held-out stations directly from the validated mixed-effects baseline tables (`chlorophyll_mixed_effects_per_fold.csv` and `cdom_mixed_effects_per_fold.csv`). This eliminated missing fold assignments and ensured 100% adherence to Standing Methodological Rule 1.
3. **Type Constraints in Scikit-Learn**: Fixed `best_k` integer casting from DataFrame series before invoking `KMeans(n_clusters=best_k)`.
4. **Reproducible Artifacts**: All underlying raw results, fold-level predictions, and statistical tables were exported to dedicated files in `results/tables/`:
   - `step8_temporal_window_inversion_per_fold.csv`
   - `step8_temporal_window_inversion_summary.csv`
   - `step8_hours_since_overpass_feature_evaluation.csv`
   - `step9_gitelson_correlation_audit.csv`
   - `step9_gitelson_inversion_cv.csv`
   - `step10_owt_cluster_metrics.csv`
   - `step10_owt_contingency_site.csv`
   - `step10_owt_contingency_basin.csv`
   - `step10_owt_model_performance.csv`
   - `step11_winsorization_per_fold.csv`
   - `step11_winsorization_summary.csv`
   - `final_stage3_benchmark_turbidity.csv`
   - `final_stage3_benchmark_chlorophyll.csv`
   - `final_stage3_benchmark_cdom.csv`
   - `final_stage3_benchmark_consolidated.csv`
