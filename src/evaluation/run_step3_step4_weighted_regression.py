"""
Script: run_step3_step4_weighted_regression.py
Executes Step 3 and Step 4 of the unmixing water-confidence weighting workflow:
1. Derives endmember spectra from training stations ONLY (strictly zero leakage).
2. Computes unmixing water fractions f_water for all DL samples.
3. Trains Stage 3 regression models (Gradient Boosting) using sample_weight=f_water:
   - Full DL Pool (N=30,503)
   - Variant with Gunnison-at-Delta (09152500) excluded
   - Compares weighting schemes: linear f_water vs quadratic f_water^2
4. Evaluates on the held-out test set (N=7,152):
   - Micro Log R^2 (pooled)
   - Macro Log R^2 (unweighted 4-basin average)
   - Per-basin Log R^2 (Trinity, Willamette, Upper Colorado, Illinois)
5. Generates the consolidated Step 4 comparison table across all approaches:
   - Condition (c) NDWI full pool (N=15,447)
   - Condition (b) Raw DL full pool (N=30,503)
   - Condition (b) Raw DL subsampled (matched N=15,447, mean across seeds)
   - Condition (b) Hard-filtered Clean DL (NDVI > 0 exclusion)
   - Condition (b) Unmixing-Weighted DL (full pool N=30,503)
   - Condition (b) Unmixing-Weighted DL (excluding Gunnison-at-Delta)
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inversion.models import get_inversion_models
from src.inversion.train_inversion import FEATURE_COLS

def main():
    print("=" * 85)
    print("STEP 3 & 4: TESTING UNMIXING-DERIVED CONTINUOUS WEIGHTING IN REGRESSION")
    print("=" * 85)

    df_a = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_a_ground_truth.csv', low_memory=False)
    df_b = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv', low_memory=False)
    df_c = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_c_ndwi_masked.csv', low_memory=False)

    target_type = 'turbidity'
    b_turb = df_b[df_b['target_type'] == target_type].copy()
    c_turb = df_c[df_c['target_type'] == target_type].copy()
    a_turb = df_a[df_a['target_type'] == target_type].copy()

    # Canonical held-out test split
    from src.evaluation.splits import get_test_sites
    test_sites = get_test_sites('turbidity')
    train_sites = set(b_turb['site_no']) - test_sites

    test_df = b_turb[b_turb['site_no'].isin(test_sites)].copy()
    train_b_full = b_turb[~b_turb['site_no'].isin(test_sites)].copy()
    train_c_full = c_turb[~c_turb['site_no'].isin(test_sites)].copy()

    print(f"DL Training Pool:   {len(train_b_full):,} samples across {len(train_sites)} sites")
    print(f"NDWI Training Pool: {len(train_c_full):,} samples")
    print(f"Fixed Test Set:     {len(test_df):,} samples across {len(test_sites)} sites")

    unmix_bands = [
        'B02_center_refl', 'B03_center_refl', 'B04_center_refl',
        'B05_center_refl', 'B06_center_refl', 'B07_center_refl',
        'B08_center_refl', 'B8A_center_refl', 'B11_center_refl', 'B12_center_refl'
    ]

    # Matching keys
    b_keys = set(b_turb['site_no'].astype(str) + '_' + b_turb['sample_datetime_UTC'].astype(str))
    c_keys = set(c_turb['site_no'].astype(str) + '_' + c_turb['sample_datetime_UTC'].astype(str))

    # Tag train_b
    train_b_full['key'] = train_b_full['site_no'].astype(str) + '_' + train_b_full['sample_datetime_UTC'].astype(str)
    train_b_full['is_both_retained'] = train_b_full['key'].isin(c_keys)

    # 1. DERIVE ENDMEMBERS (TRAINING STATIONS ONLY - ZERO LEAKAGE)
    both_retained_train = train_b_full[train_b_full['is_both_retained']]
    water_em = both_retained_train[unmix_bands].median().values

    a_train = a_turb[a_turb['site_no'].isin(train_sites)].copy()
    a_train['key'] = a_train['site_no'].astype(str) + '_' + a_train['sample_datetime_UTC'].astype(str)
    a_rejected = a_train[~a_train['key'].isin(b_keys) & ~a_train['key'].isin(c_keys)].copy()
    a_rejected['ndvi'] = (a_rejected['B08_center_refl'] - a_rejected['B04_center_refl']) / (
        a_rejected['B08_center_refl'] + a_rejected['B04_center_refl'] + 1e-4
    )

    veg_candidates = a_rejected[a_rejected['ndvi'] > 0.45]
    veg_em = veg_candidates[unmix_bands].median().values

    soil_candidates = a_rejected[(a_rejected['basin'] == 'Upper Colorado') & (a_rejected['ndvi'] < 0.20) & (a_rejected['B04_center_refl'] > 0.09)]
    soil_em = soil_candidates[unmix_bands].median().values

    M = np.column_stack([water_em, veg_em, soil_em])
    delta = 10.0
    A_aug = np.vstack([M, delta * np.ones((1, 3))])

    # Unmix train_b_full
    print("Unmixing training samples...")
    X_tr_refl = train_b_full[unmix_bands].values
    f_water_tr = np.zeros(len(train_b_full))
    for i in range(len(train_b_full)):
        y = X_tr_refl[i]
        b_aug = np.hstack([y, delta])
        x, _ = nnls(A_aug, b_aug)
        tot = np.sum(x)
        f_water_tr[i] = x[0] / tot if tot > 0 else 0.5

    train_b_full['f_water'] = f_water_tr
    train_b_full['weight_linear'] = np.clip(f_water_tr, 0.05, 1.0)
    train_b_full['weight_quad'] = np.clip(f_water_tr ** 2, 0.01, 1.0)

    # Prepare Test Set
    X_test = test_df[FEATURE_COLS].values
    y_test_log = np.log10(np.clip(test_df['MeasurementValue'].values, 1e-4, None))
    test_basins = sorted(test_df['basin'].dropna().unique())

    def eval_model(model, scaler, name=""):
        p_log = model.predict(scaler.transform(X_test))
        micro_r2 = r2_score(y_test_log, p_log)
        micro_rmse = np.sqrt(mean_squared_error(y_test_log, p_log))
        
        basin_r2 = {}
        for b in test_basins:
            mask = (test_df['basin'] == b).values
            b_r2 = r2_score(y_test_log[mask], p_log[mask]) if np.sum(mask) > 1 else np.nan
            basin_r2[b] = b_r2
            
        macro_r2 = float(np.mean(list(basin_r2.values())))
        res = {
            'Approach': name,
            'Micro_Log_R2': micro_r2,
            'Micro_Log_RMSE': micro_rmse,
            'Macro_Log_R2': macro_r2,
            **{f"{b}_Log_R2": basin_r2[b] for b in test_basins}
        }
        return res

    all_results = []

    # 1. Condition (c) NDWI Full Pool (N=15,447)
    print("\nTraining 1: NDWI Full Pool...")
    sc_c = StandardScaler()
    X_c_s = sc_c.fit_transform(train_c_full[FEATURE_COLS].values)
    y_c_l = np.log10(np.clip(train_c_full['MeasurementValue'].values, 1e-4, None))
    m_c = get_inversion_models(in_features=len(FEATURE_COLS))['Gradient_Boosting']
    m_c.fit(X_c_s, y_c_l)
    all_results.append(eval_model(m_c, sc_c, name="NDWI Full Pool (N=15,447)"))

    # 2. Condition (b) Raw DL Full Pool (N=30,503)
    print("Training 2: Raw DL Full Pool (Unweighted)...")
    sc_b = StandardScaler()
    X_b_s = sc_b.fit_transform(train_b_full[FEATURE_COLS].values)
    y_b_l = np.log10(np.clip(train_b_full['MeasurementValue'].values, 1e-4, None))
    m_b = get_inversion_models(in_features=len(FEATURE_COLS))['Gradient_Boosting']
    m_b.fit(X_b_s, y_b_l)
    all_results.append(eval_model(m_b, sc_b, name="Raw DL Full Pool (N=30,503)"))

    # 3. Condition (b) Raw DL Subsampled (N=15,447) - 5-seed average
    print("Training 3: Raw DL Subsampled (5 seeds)...")
    sub_runs = []
    for sd in [42, 43, 44, 45, 46]:
        sub_df = train_b_full.sample(n=len(train_c_full), random_state=sd)
        sc_sub = StandardScaler()
        X_sub_s = sc_sub.fit_transform(sub_df[FEATURE_COLS].values)
        y_sub_l = np.log10(np.clip(sub_df['MeasurementValue'].values, 1e-4, None))
        m_sub = get_inversion_models(in_features=len(FEATURE_COLS))['Gradient_Boosting']
        m_sub.fit(X_sub_s, y_sub_l)
        sub_runs.append(eval_model(m_sub, sc_sub, name=f"Sub_{sd}"))
    df_sub = pd.DataFrame(sub_runs)
    avg_sub = {
        'Approach': "Raw DL Subsampled (N=15,447, 5-seed mean)",
        'Micro_Log_R2': df_sub['Micro_Log_R2'].mean(),
        'Micro_Log_RMSE': df_sub['Micro_Log_RMSE'].mean(),
        'Macro_Log_R2': df_sub['Macro_Log_R2'].mean(),
        **{f"{b}_Log_R2": df_sub[f"{b}_Log_R2"].mean() for b in test_basins}
    }
    all_results.append(avg_sub)

    # 4. Condition (b) Hard-Filtered Clean DL (NDVI > 0 exclusion)
    print("Training 4: Hard-filtered Clean DL (NDVI > 0 excluded)...")
    train_b_full['ndvi'] = (train_b_full['B08_center_refl'] - train_b_full['B04_center_refl']) / np.maximum(
        train_b_full['B08_center_refl'] + train_b_full['B04_center_refl'], 1e-4
    )
    clean_dl_df = train_b_full[train_b_full['ndvi'] <= 0.0].copy()
    sc_clean = StandardScaler()
    X_cl_s = sc_clean.fit_transform(clean_dl_df[FEATURE_COLS].values)
    y_cl_l = np.log10(np.clip(clean_dl_df['MeasurementValue'].values, 1e-4, None))
    m_clean = get_inversion_models(in_features=len(FEATURE_COLS))['Gradient_Boosting']
    m_clean.fit(X_cl_s, y_cl_l)
    all_results.append(eval_model(m_clean, sc_clean, name=f"Hard-Filtered Clean DL (NDVI<=0, N={len(clean_dl_df):,})"))

    # 5. Condition (b) Unmixing-Weighted DL (Linear weights = f_water, N=30,503)
    print("Training 5: Unmixing-Weighted DL (Linear f_water)...")
    m_w_lin = get_inversion_models(in_features=len(FEATURE_COLS))['Gradient_Boosting']
    m_w_lin.fit(X_b_s, y_b_l, sample_weight=train_b_full['weight_linear'].values)
    all_results.append(eval_model(m_w_lin, sc_b, name="Unmixing-Weighted DL (Linear w=f_water, N=30,503)"))

    # 6. Condition (b) Unmixing-Weighted DL (Quadratic weights = f_water^2, N=30,503)
    print("Training 6: Unmixing-Weighted DL (Quadratic w=f_water^2)...")
    m_w_quad = get_inversion_models(in_features=len(FEATURE_COLS))['Gradient_Boosting']
    m_w_quad.fit(X_b_s, y_b_l, sample_weight=train_b_full['weight_quad'].values)
    all_results.append(eval_model(m_w_quad, sc_b, name="Unmixing-Weighted DL (Quadratic w=f_water^2, N=30,503)"))

    # 7. Robustness Variant: Unmixing-Weighted DL with Gunnison-at-Delta (09152500) Excluded
    print("Training 7: Unmixing-Weighted DL (Excluding Gunnison at Delta)...")
    no_gunnison_df = train_b_full[~train_b_full['site_no'].astype(str).str.contains('9152500')].copy()
    sc_ng = StandardScaler()
    X_ng_s = sc_ng.fit_transform(no_gunnison_df[FEATURE_COLS].values)
    y_ng_l = np.log10(np.clip(no_gunnison_df['MeasurementValue'].values, 1e-4, None))
    m_w_ng = get_inversion_models(in_features=len(FEATURE_COLS))['Gradient_Boosting']
    m_w_ng.fit(X_ng_s, y_ng_l, sample_weight=no_gunnison_df['weight_linear'].values)
    all_results.append(eval_model(m_w_ng, sc_ng, name="Unmixing-Weighted DL (No 09152500, N=30,454)"))

    df_consolidated = pd.DataFrame(all_results)
    print("\n" + "=" * 85)
    print("STEP 4: FINAL CONSOLIDATED BASIN-HONEST COMPARISON TABLE (LOG SPACE)")
    print("=" * 85)
    print(df_consolidated.round(4).to_string(index=False))

    df_consolidated.to_csv('results/tables/consolidated_unmixing_ablation_comparison.csv', index=False)
    print("\nSaved consolidated table to results/tables/consolidated_unmixing_ablation_comparison.csv")

if __name__ == '__main__':
    main()
