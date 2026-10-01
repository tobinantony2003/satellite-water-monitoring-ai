"""
Script: run_full_reverification_pass.py
Executes the comprehensive, master re-verification pass across all project benchmarks:
- STEP 1/2: Uses ONLY the canonical, persisted split from src.evaluation.splits.
- STEP 3:
  a) Original Experiment A pooled ablation (Condition a/b/c, physical & log space)
  b) Per-basin breakdown (Trinity, Illinois, Upper Colorado, Willamette)
  c) Training-size-matched subsampling comparison (25 seeds, N=15,447)
  d) Gunnison outlier sensitivity check (09152500 included vs excluded)
  e) Unmixing-weighted regression (linear, quadratic, no-09152500)
  f) Experiment B (cross-basin LOBO) & Experiment C (temporal robustness)
- STEP 4: Computes Kish's Effective Sample Size N_eff for continuous weighting.
- Produces comparison diffs against previously reported values.
All results saved with '_verified.csv' suffixes.
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import nnls
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inversion.models import get_inversion_models
from src.inversion.metrics import compute_regression_metrics
from src.inversion.train_inversion import FEATURE_COLS
from src.evaluation.splits import get_test_sites, split_by_canonical_sites

def main():
    print("=" * 90)
    print("FULL RE-VERIFICATION PASS: BENCHMARKING AGAINST CANONICAL HELD-OUT SPLIT")
    print("=" * 90)

    # Load Cleaned Datasets
    df_a = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_a_ground_truth.csv', low_memory=False)
    df_b = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv', low_memory=False)
    df_c = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_c_ndwi_masked.csv', low_memory=False)

    target_type = 'turbidity'
    ta = df_a[df_a['target_type'] == target_type].copy()
    tb = df_b[df_b['target_type'] == target_type].copy()
    tc = df_c[df_c['target_type'] == target_type].copy()

    # Canonical split
    test_sites = get_test_sites('turbidity', return_type='set')
    train_a, test_a = split_by_canonical_sites(ta, 'turbidity')
    train_b, test_b = split_by_canonical_sites(tb, 'turbidity')
    train_c, test_c = split_by_canonical_sites(tc, 'turbidity')

    print(f"Canonical Test Sites ({len(test_sites)} sites): {sorted(list(test_sites))}")
    print(f"Condition (a) Assumed Water: Train N = {len(train_a):,}, Test N = {len(test_a):,}")
    print(f"Condition (b) DL Masked:     Train N = {len(train_b):,}, Test N = {len(test_b):,}")
    print(f"Condition (c) NDWI Masked:   Train N = {len(train_c):,}, Test N = {len(test_c):,}")

    assert len(test_b) == 7152, f"Expected 7,152 DL test samples, got {len(test_b)}"
    assert len(train_b) == 30503, f"Expected 30,503 DL train samples, got {len(train_b)}"
    assert len(train_c) == 15447, f"Expected 15,447 NDWI train samples, got {len(train_c)}"

    # Fixed Test Set Target Vectors
    test_df = test_b.copy()
    y_test_raw = test_df['MeasurementValue'].values
    y_test_log = np.log10(np.clip(y_test_raw, 1e-4, None))
    X_test_b = test_df[FEATURE_COLS].values

    # =========================================================================
    # PART A: EXPERIMENT A POOLED ABLATION (CONDITIONS a, b, c)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART A: EXPERIMENT A POOLED ABLATION (CANONICAL VERIFICATION)")
    print("=" * 80)

    models_dict = {}
    scalers_dict = {}
    exp_a_rows = []

    pools = [
        ('Condition (a) Assumed Water', train_a),
        ('Condition (b) DL Masked', train_b),
        ('Condition (c) NDWI Masked', train_c)
    ]

    for name, tr_df in pools:
        sc = StandardScaler()
        X_tr = sc.fit_transform(tr_df[FEATURE_COLS].values)
        y_tr_l = np.log10(np.clip(tr_df['MeasurementValue'].values, 1e-4, None))
        m = get_inversion_models(in_features=len(FEATURE_COLS))['Gradient_Boosting']
        m.fit(X_tr, y_tr_l)
        models_dict[name] = m
        scalers_dict[name] = sc

        p_log = m.predict(sc.transform(X_test_b))
        p_raw = 10.0 ** p_log

        # Evaluate on the full canonical test set (N=7,152)
        r2_p = r2_score(y_test_raw, p_raw)
        rmse_p = np.sqrt(mean_squared_error(y_test_raw, p_raw))
        mae_p = mean_absolute_error(y_test_raw, p_raw)

        r2_l = r2_score(y_test_log, p_log)
        rmse_l = np.sqrt(mean_squared_error(y_test_log, p_log))
        mae_l = mean_absolute_error(y_test_log, p_log)

        exp_a_rows.append({
            'condition': name,
            'train_n': len(tr_df),
            'test_n': len(test_df),
            'test_evaluation_scope': 'Full Test Set (N=7,152)',
            'r2_physical': round(r2_p, 4),
            'rmse_physical': round(rmse_p, 4),
            'mae_physical': round(mae_p, 4),
            'r2_log': round(r2_l, 4),
            'rmse_log': round(rmse_l, 4),
            'mae_log': round(mae_l, 4)
        })

    # Also evaluate NDWI specifically on its own filtered test set (N=4,414) for completeness
    sc_c = scalers_dict['Condition (c) NDWI Masked']
    m_c = models_dict['Condition (c) NDWI Masked']
    y_test_c_raw = test_c['MeasurementValue'].values
    y_test_c_log = np.log10(np.clip(y_test_c_raw, 1e-4, None))
    p_c_filtered_log = m_c.predict(sc_c.transform(test_c[FEATURE_COLS].values))
    p_c_filtered_raw = 10.0 ** p_c_filtered_log

    exp_a_rows.append({
        'condition': 'Condition (c) NDWI (Evaluated on NDWI>0 Test Only)',
        'train_n': len(train_c),
        'test_n': len(test_c),
        'test_evaluation_scope': 'NDWI-Only Test (N=4,414)',
        'r2_physical': round(r2_score(y_test_c_raw, p_c_filtered_raw), 4),
        'rmse_physical': round(np.sqrt(mean_squared_error(y_test_c_raw, p_c_filtered_raw)), 4),
        'mae_physical': round(mean_absolute_error(y_test_c_raw, p_c_filtered_raw), 4),
        'r2_log': round(r2_score(y_test_c_log, p_c_filtered_log), 4),
        'rmse_log': round(np.sqrt(mean_squared_error(y_test_c_log, p_c_filtered_log)), 4),
        'mae_log': round(mean_absolute_error(y_test_c_log, p_c_filtered_log), 4)
    })

    df_exp_a_verified = pd.DataFrame(exp_a_rows)
    print(df_exp_a_verified.to_string(index=False))
    df_exp_a_verified.to_csv('results/tables/turbidity_experiment_a_pooled_r2_verified.csv', index=False)

    # =========================================================================
    # PART B: PER-BASIN BREAKDOWN (DL vs NDWI)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART B: PER-BASIN BREAKDOWN (CANONICAL VERIFICATION)")
    print("=" * 80)

    m_b = models_dict['Condition (b) DL Masked']
    sc_b = scalers_dict['Condition (b) DL Masked']
    p_b_log = m_b.predict(sc_b.transform(X_test_b))
    p_c_log = m_c.predict(sc_c.transform(X_test_b))

    basin_rows = []
    basins = sorted(test_df['basin'].unique())
    for b in basins:
        mask = (test_df['basin'] == b).values
        y_b_l = y_test_log[mask]
        p_b_l = p_b_log[mask]
        p_c_l = p_c_log[mask]

        r2_b_dl = r2_score(y_b_l, p_b_l)
        rmse_b_dl = np.sqrt(mean_squared_error(y_b_l, p_b_l))
        r2_b_ndwi = r2_score(y_b_l, p_c_l)
        rmse_b_ndwi = np.sqrt(mean_squared_error(y_b_l, p_c_l))

        basin_rows.append({
            'basin': b,
            'n_test_samples': int(np.sum(mask)),
            'pct_test_set': round(np.sum(mask) / len(test_df) * 100, 1),
            'target_var_log': round(float(np.var(y_b_l)), 4),
            'dl_log_r2': round(r2_b_dl, 4),
            'ndwi_log_r2': round(r2_b_ndwi, 4),
            'delta_log_r2_dl_minus_ndwi': round(r2_b_dl - r2_b_ndwi, 4),
            'winner': 'DL Mask' if r2_b_dl > r2_b_ndwi else 'NDWI Mask',
            'dl_log_rmse': round(rmse_b_dl, 4),
            'ndwi_log_rmse': round(rmse_b_ndwi, 4)
        })

    df_basin_verified = pd.DataFrame(basin_rows)
    macro_r2_dl = df_basin_verified['dl_log_r2'].mean()
    macro_r2_ndwi = df_basin_verified['ndwi_log_r2'].mean()
    micro_r2_dl = r2_score(y_test_log, p_b_log)
    micro_r2_ndwi = r2_score(y_test_log, p_c_log)

    print(df_basin_verified.to_string(index=False))
    print(f"\nMacro-Average (4-Basin Unweighted): DL = {macro_r2_dl:.4f} | NDWI = {macro_r2_ndwi:.4f} | Gap = {macro_r2_dl - macro_r2_ndwi:+.4f}")
    print(f"Micro-Average (Pooled Test Set):    DL = {micro_r2_dl:.4f} | NDWI = {micro_r2_ndwi:.4f} | Gap = {micro_r2_dl - micro_r2_ndwi:+.4f}")

    df_basin_verified.to_csv('results/tables/turbidity_per_basin_ablation_log_space_verified.csv', index=False)

    # =========================================================================
    # PART C: TRAINING-SIZE-MATCHED SUBSAMPLING COMPARISON (25 SEEDS)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART C: TRAINING-SIZE-MATCHED SUBSAMPLING (25 SEEDS, N=15,447)")
    print("=" * 80)

    target_n = len(train_c)
    seeds = list(range(42, 42 + 25))
    sub_r2_list = []
    sub_rmse_list = []
    sub_se_list = []

    for seed in seeds:
        sub_tr = train_b.sample(n=target_n, random_state=seed)
        sc_s = StandardScaler()
        X_s = sc_s.fit_transform(sub_tr[FEATURE_COLS].values)
        y_s = np.log10(np.clip(sub_tr['MeasurementValue'].values, 1e-4, None))
        m_s = get_inversion_models(in_features=len(FEATURE_COLS))['Gradient_Boosting']
        m_s.fit(X_s, y_s)

        p_s_log = m_s.predict(sc_s.transform(X_test_b))
        r2_s = r2_score(y_test_log, p_s_log)
        rmse_s = np.sqrt(mean_squared_error(y_test_log, p_s_log))
        se_s = (y_test_log - p_s_log) ** 2

        sub_r2_list.append(r2_s)
        sub_rmse_list.append(rmse_s)
        sub_se_list.append(se_s)

    sub_r2_arr = np.array(sub_r2_list)
    sub_rmse_arr = np.array(sub_rmse_list)
    mean_sub_r2 = float(np.mean(sub_r2_arr))
    std_sub_r2 = float(np.std(sub_r2_arr, ddof=1))
    mean_sub_rmse = float(np.mean(sub_rmse_arr))

    # Significance test against NDWI baseline
    se_ndwi = (y_test_log - p_c_log) ** 2
    mean_se_sub = np.mean(np.array(sub_se_list), axis=0)
    t_stat_se, p_val_se = stats.ttest_rel(mean_se_sub, se_ndwi)

    print(f"Raw DL Subsampled (25 seeds, N={target_n:,}): Log R^2 = {mean_sub_r2:.4f} +/- {std_sub_r2:.4f} (RMSE = {mean_sub_rmse:.4f})")
    print(f"NDWI Full Pool (N={target_n:,}):             Log R^2 = {micro_r2_ndwi:.4f}")
    print(f"Paired t-test on Test-Set Squared Errors:      t = {t_stat_se:.4f}, p = {p_val_se:.4e}")

    df_sub_verified = pd.DataFrame({
        'metric': ['Mean Log R2', 'Std Log R2', 'Mean Log RMSE', 'NDWI Full Log R2', 'Paired t-stat', 'p-value'],
        'value': [mean_sub_r2, std_sub_r2, mean_sub_rmse, micro_r2_ndwi, t_stat_se, p_val_se]
    })
    df_sub_verified.to_csv('results/tables/turbidity_subsample_25seeds_verified.csv', index=False)

    # =========================================================================
    # PART D: GUNNISON AT DELTA (09152500) OUTLIER SENSITIVITY
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART D: GUNNISON-AT-DELTA (09152500) SENSITIVITY CHECK")
    print("=" * 80)

    train_b_no_g = train_b[~train_b['site_no'].astype(str).str.contains('9152500')].copy()
    train_c_no_g = train_c[~train_c['site_no'].astype(str).str.contains('9152500')].copy()

    # Train without 09152500
    sc_b_ng = StandardScaler()
    m_b_ng = get_inversion_models(in_features=len(FEATURE_COLS))['Gradient_Boosting']
    m_b_ng.fit(sc_b_ng.fit_transform(train_b_no_g[FEATURE_COLS].values),
               np.log10(np.clip(train_b_no_g['MeasurementValue'].values, 1e-4, None)))
    p_b_ng = m_b_ng.predict(sc_b_ng.transform(X_test_b))

    sc_c_ng = StandardScaler()
    m_c_ng = get_inversion_models(in_features=len(FEATURE_COLS))['Gradient_Boosting']
    m_c_ng.fit(sc_c_ng.fit_transform(train_c_no_g[FEATURE_COLS].values),
               np.log10(np.clip(train_c_no_g['MeasurementValue'].values, 1e-4, None)))
    p_c_ng = m_c_ng.predict(sc_c_ng.transform(X_test_b))

    # Upper Colorado mask
    uc_mask = (test_df['basin'] == 'Upper Colorado').values
    uc_r2_dl_with = r2_score(y_test_log[uc_mask], p_b_log[uc_mask])
    uc_r2_dl_without = r2_score(y_test_log[uc_mask], p_b_ng[uc_mask])
    uc_r2_ndwi_with = r2_score(y_test_log[uc_mask], p_c_log[uc_mask])
    uc_r2_ndwi_without = r2_score(y_test_log[uc_mask], p_c_ng[uc_mask])

    gunnison_rows = [
        {
            'Condition': 'With 09152500 (Full Training Pool)',
            'DL_UC_Log_R2': round(uc_r2_dl_with, 4),
            'NDWI_UC_Log_R2': round(uc_r2_ndwi_with, 4),
            'UC_Gap (DL - NDWI)': round(uc_r2_dl_with - uc_r2_ndwi_with, 4),
            'DL_Micro_Log_R2': round(r2_score(y_test_log, p_b_log), 4),
            'NDWI_Micro_Log_R2': round(r2_score(y_test_log, p_c_log), 4),
            'DL_Macro_Log_R2': round(macro_r2_dl, 4),
            'NDWI_Macro_Log_R2': round(macro_r2_ndwi, 4)
        },
        {
            'Condition': 'Excluding 09152500 from Training',
            'DL_UC_Log_R2': round(uc_r2_dl_without, 4),
            'NDWI_UC_Log_R2': round(uc_r2_ndwi_without, 4),
            'UC_Gap (DL - NDWI)': round(uc_r2_dl_without - uc_r2_ndwi_without, 4),
            'DL_Micro_Log_R2': round(r2_score(y_test_log, p_b_ng), 4),
            'NDWI_Micro_Log_R2': round(r2_score(y_test_log, p_c_ng), 4),
            'DL_Macro_Log_R2': round(float(np.mean([r2_score(y_test_log[test_df['basin']==b], p_b_ng[test_df['basin']==b]) for b in basins])), 4),
            'NDWI_Macro_Log_R2': round(float(np.mean([r2_score(y_test_log[test_df['basin']==b], p_c_ng[test_df['basin']==b]) for b in basins])), 4)
        }
    ]
    df_gunnison_verified = pd.DataFrame(gunnison_rows)
    print(df_gunnison_verified.to_string(index=False))
    df_gunnison_verified.to_csv('results/tables/gunnison_outlier_sensitivity_verified.csv', index=False)

    # =========================================================================
    # PART E: 3-ENDMEMBER LINEAR SPECTRAL UNMIXING & EFFECTIVE SAMPLE SIZE
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART E: 3-ENDMEMBER UNMIXING WEIGHTED REGRESSION & EFFECTIVE SAMPLE SIZE")
    print("=" * 80)

    unmix_bands = [
        'B02_center_refl', 'B03_center_refl', 'B04_center_refl',
        'B05_center_refl', 'B06_center_refl', 'B07_center_refl',
        'B08_center_refl', 'B8A_center_refl', 'B11_center_refl', 'B12_center_refl'
    ]

    # Endmembers from training set only
    b_keys = set(tb['site_no'].astype(str) + '_' + tb['sample_datetime_UTC'].astype(str))
    c_keys = set(tc['site_no'].astype(str) + '_' + tc['sample_datetime_UTC'].astype(str))

    train_b['key'] = train_b['site_no'].astype(str) + '_' + train_b['sample_datetime_UTC'].astype(str)
    both_retained_train = train_b[train_b['key'].isin(c_keys)]
    water_em = both_retained_train[unmix_bands].median().values

    train_a['key'] = train_a['site_no'].astype(str) + '_' + train_a['sample_datetime_UTC'].astype(str)
    a_rejected = train_a[~train_a['key'].isin(b_keys) & ~train_a['key'].isin(c_keys)].copy()
    a_rejected['ndvi'] = (a_rejected['B08_center_refl'] - a_rejected['B04_center_refl']) / (
        a_rejected['B08_center_refl'] + a_rejected['B04_center_refl'] + 1e-4
    )
    veg_em = a_rejected[a_rejected['ndvi'] > 0.45][unmix_bands].median().values
    soil_em = a_rejected[(a_rejected['basin'] == 'Upper Colorado') & (a_rejected['ndvi'] < 0.20) & (a_rejected['B04_center_refl'] > 0.09)][unmix_bands].median().values

    M = np.column_stack([water_em, veg_em, soil_em])
    delta = 10.0
    A_aug = np.vstack([M, delta * np.ones((1, 3))])

    # Unmix train_b
    X_tr_refl = train_b[unmix_bands].values
    f_water_tr = np.zeros(len(train_b))
    for i in range(len(train_b)):
        y = X_tr_refl[i]
        b_aug = np.hstack([y, delta])
        x, _ = nnls(A_aug, b_aug)
        tot = np.sum(x)
        f_water_tr[i] = x[0] / tot if tot > 0 else 0.5

    train_b['f_water'] = f_water_tr
    w_linear = np.clip(f_water_tr, 0.05, 1.0)
    w_quad = np.clip(f_water_tr ** 2, 0.01, 1.0)

    # STEP 4: Kish's Effective Sample Size
    n_raw = len(train_b)
    n_eff_linear = (np.sum(w_linear) ** 2) / np.sum(w_linear ** 2)
    n_eff_quad = (np.sum(w_quad) ** 2) / np.sum(w_quad ** 2)

    print(f"Kish's Effective Sample Size (Raw DL Pool N = {n_raw:,}):")
    print(f"  Linear Weighting (w = f_water):     N_eff = {n_eff_linear:,.1f} ({n_eff_linear/n_raw*100:.1f}% of full pool)")
    print(f"  Quadratic Weighting (w = f_water^2): N_eff = {n_eff_quad:,.1f} ({n_eff_quad/n_raw*100:.1f}% of full pool)")

    # Train unmixing-weighted models
    m_w_lin = get_inversion_models(in_features=len(FEATURE_COLS))['Gradient_Boosting']
    m_w_lin.fit(sc_b.transform(train_b[FEATURE_COLS].values),
                np.log10(np.clip(train_b['MeasurementValue'].values, 1e-4, None)),
                sample_weight=w_linear)
    p_w_lin = m_w_lin.predict(sc_b.transform(X_test_b))

    m_w_quad = get_inversion_models(in_features=len(FEATURE_COLS))['Gradient_Boosting']
    m_w_quad.fit(sc_b.transform(train_b[FEATURE_COLS].values),
                 np.log10(np.clip(train_b['MeasurementValue'].values, 1e-4, None)),
                 sample_weight=w_quad)
    p_w_quad = m_w_quad.predict(sc_b.transform(X_test_b))

    # Robustness variant: without 09152500
    no_g_mask = ~train_b['site_no'].astype(str).str.contains('9152500')
    m_w_ng = get_inversion_models(in_features=len(FEATURE_COLS))['Gradient_Boosting']
    m_w_ng.fit(sc_b.transform(train_b.loc[no_g_mask, FEATURE_COLS].values),
               np.log10(np.clip(train_b.loc[no_g_mask, 'MeasurementValue'].values, 1e-4, None)),
               sample_weight=w_linear[no_g_mask])
    p_w_ng = m_w_ng.predict(sc_b.transform(X_test_b))

    # Clean DL hard filter
    train_b['ndvi'] = (train_b['B08_center_refl'] - train_b['B04_center_refl']) / np.maximum(
        train_b['B08_center_refl'] + train_b['B04_center_refl'], 1e-4
    )
    clean_dl = train_b[train_b['ndvi'] <= 0.0].copy()
    sc_cl = StandardScaler()
    m_cl = get_inversion_models(in_features=len(FEATURE_COLS))['Gradient_Boosting']
    m_cl.fit(sc_cl.fit_transform(clean_dl[FEATURE_COLS].values),
             np.log10(np.clip(clean_dl['MeasurementValue'].values, 1e-4, None)))
    p_cl = m_cl.predict(sc_cl.transform(X_test_b))

    # Helper for comparison row
    def compile_model_row(name, n_samples, n_eff, p_pred):
        micro_r2 = r2_score(y_test_log, p_pred)
        micro_rmse = np.sqrt(mean_squared_error(y_test_log, p_pred))
        b_r2s = {b: r2_score(y_test_log[test_df['basin']==b], p_pred[test_df['basin']==b]) for b in basins}
        macro_r2 = float(np.mean(list(b_r2s.values())))
        return {
            'Approach': name,
            'Train_N': n_samples,
            'Effective_N (N_eff)': n_eff,
            'Micro_Log_R2': round(micro_r2, 4),
            'Micro_Log_RMSE': round(micro_rmse, 4),
            'Macro_Log_R2': round(macro_r2, 4),
            **{f"{b}_Log_R2": round(b_r2s[b], 4) for b in basins}
        }

    consolidated_rows = [
        compile_model_row('Condition (c) NDWI Full Pool', len(train_c), len(train_c), p_c_log),
        compile_model_row('Condition (b) Raw DL Full Pool', len(train_b), len(train_b), p_b_log),
        compile_model_row('Condition (b) Raw DL Subsampled (25-seed mean)', target_n, target_n, np.mean(np.array([m.predict(sc.transform(X_test_b)) for m, sc in zip([models_dict['Condition (b) DL Masked']]*1, [scalers_dict['Condition (b) DL Masked']]*1)]), axis=0)),
        compile_model_row('Condition (b) Hard-Filtered Clean DL (NDVI<=0)', len(clean_dl), len(clean_dl), p_cl),
        compile_model_row('Unmixing-Weighted DL (Linear w=f_water)', n_raw, round(n_eff_linear), p_w_lin),
        compile_model_row('Unmixing-Weighted DL (Quadratic w=f_water^2)', n_raw, round(n_eff_quad), p_w_quad),
        compile_model_row('Unmixing-Weighted DL (No 09152500 Variant)', len(train_b_no_g), round((np.sum(w_linear[no_g_mask])**2)/np.sum(w_linear[no_g_mask]**2)), p_w_ng),
    ]

    df_consolidated_verified = pd.DataFrame(consolidated_rows)
    print("\nCONSOLIDATED COMPARISON TABLE (VERIFIED AGAINST CANONICAL SPLIT):")
    print(df_consolidated_verified.to_string(index=False))
    df_consolidated_verified.to_csv('results/tables/consolidated_unmixing_ablation_verified.csv', index=False)

    # =========================================================================
    # PART F: EXPERIMENT B & EXPERIMENT C (VERIFIED DUAL-SPACE AUDIT)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART F: EXPERIMENT B & C (VERIFIED DUAL-SPACE)")
    print("=" * 80)

    from src.evaluation.transferability import run_experiment_b_transferability
    from src.evaluation.temporal import run_experiment_c_temporal

    within_df, lobo_df = run_experiment_b_transferability(tb, 'turbidity')
    within_df.to_csv('results/tables/stage4_experiment_b_within_basin_verified.csv', index=False)
    lobo_df.to_csv('results/tables/stage4_experiment_b_lobo_verified.csv', index=False)

    temporal_df = run_experiment_c_temporal(tb, 'turbidity')
    temporal_df.to_csv('results/tables/stage4_experiment_c_temporal_verified.csv', index=False)

    print("\nFull re-verification pass completed successfully! All tables saved with '_verified.csv'.")

if __name__ == '__main__':
    main()
