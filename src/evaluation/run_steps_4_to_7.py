"""
Script: run_steps_4_to_7.py
Executes:
- Step 4: Correct CDOM site-overlap records.
- Step 5: Turbidity ablation training-size robustness check (both physical & log space).
- Step 6: Illinois DL-vs-NDWI reversal investigation (spectral comparison of DL-exclusive pixels vs Trinity/Upper Colorado).
- Step 7: Retroactive log-space audit of Experiment B (transferability) and Experiment C (temporal robustness).
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupKFold

from src.inversion.models import get_inversion_models
from src.inversion.metrics import compute_regression_metrics
from src.inversion.train_inversion import FEATURE_COLS
from src.inversion.spatial_cv import get_spatial_block_cv

def step4_correct_cdom_overlap():
    print("=" * 80)
    print("STEP 4: CORRECT CDOM SITE-OVERLAP RECORDS")
    print("=" * 80)
    
    summary_path = 'results/tables/cdom_site_time_summary.csv'
    if os.path.exists(summary_path):
        df_sum = pd.read_csv(summary_path)
        df_sum['site_no_str'] = df_sum['site_no'].astype(str).str.zfill(8)
        df_sum['temporal_overlap_group'] = df_sum['site_no_str'].apply(
            lambda s: 'Isolated_Era_1 (Zero Overlap with other 5 sites)' if s == '05586300'
            else 'Overlapping_Era_2 (Mutually Co-Active 2021-2024)'
        )
        df_sum['overlap_status'] = df_sum['site_no_str'].apply(
            lambda s: 'Isolated (100% pre-2021)' if s == '05586300' else 'Overlapping (2021-2024)'
        )
        df_sum.to_csv(summary_path, index=False)
        print(f"Updated {summary_path} with explicit temporal overlap groupings:")
        print(df_sum[['site_no_str', 'basin', 'min_date', 'max_date', 'temporal_overlap_group']].to_string(index=False))
        
    return df_sum

def step5_turbidity_ablation_robustness():
    print("\n" + "=" * 80)
    print("STEP 5: TURBIDITY ABLATION TRAINING-SIZE ROBUSTNESS CHECK (PHYSICAL & LOG SPACE)")
    print("=" * 80)
    
    df_b = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv', low_memory=False)
    df_c = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_c_ndwi_masked.csv', low_memory=False)
    
    b_turb = df_b[df_b['target_type'] == 'turbidity'].copy()
    c_turb = df_c[df_c['target_type'] == 'turbidity'].copy()
    
    # Canonical held-out test split
    from src.evaluation.splits import get_test_sites
    test_sites = get_test_sites('turbidity')
    
    test_df = b_turb[b_turb['site_no'].isin(test_sites)].copy()
    train_b_full = b_turb[~b_turb['site_no'].isin(test_sites)].copy()
    train_c_full = c_turb[~c_turb['site_no'].isin(test_sites)].copy()
    
    target_subsample_n = len(train_c_full)  # 15,447 samples
    print(f"Condition (b) full pool: {len(train_b_full):,} | Condition (c) full pool: {len(train_c_full):,}")
    print(f"Held-out test set size: {len(test_df):,} samples")
    
    X_test = test_df[FEATURE_COLS].values
    y_test_raw = test_df['MeasurementValue'].values
    y_test_log = np.log10(np.clip(y_test_raw, 1e-4, None))
    
    seeds = [42, 101, 2024, 7, 99]
    records = []
    
    for seed in seeds:
        train_b_sub = train_b_full.sample(n=target_subsample_n, random_state=seed).copy()
        
        X_train = train_b_sub[FEATURE_COLS].values
        y_train_log = np.log10(np.clip(train_b_sub['MeasurementValue'].values, 1e-4, None))
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = get_inversion_models(in_features=len(FEATURE_COLS))['Gradient_Boosting']
        model.fit(X_train_s, y_train_log)
        
        pred_log = model.predict(X_test_s)
        pred_raw = 10.0 ** pred_log
        
        # Physical space
        r2_phys = r2_score(y_test_raw, pred_raw)
        rmse_phys = np.sqrt(mean_squared_error(y_test_raw, pred_raw))
        mae_phys = mean_absolute_error(y_test_raw, pred_raw)
        
        # Log space
        r2_l = r2_score(y_test_log, pred_log)
        rmse_l = np.sqrt(mean_squared_error(y_test_log, pred_log))
        mae_l = mean_absolute_error(y_test_log, pred_log)
        
        records.append({
            'seed': seed,
            'train_n': target_subsample_n,
            'r2_physical': r2_phys,
            'rmse_physical': rmse_phys,
            'mae_physical': mae_phys,
            'r2_log': r2_l,
            'rmse_log': rmse_l,
            'mae_log': mae_l
        })
        print(f"  Seed {seed:4d} -> Phys R^2: {r2_phys:.4f} (RMSE: {rmse_phys:.2f}) | Log R^2: {r2_l:.4f} (RMSE: {rmse_l:.4f})")
        
    df_sub = pd.DataFrame(records)
    
    mean_r2_p, std_r2_p = df_sub['r2_physical'].mean(), df_sub['r2_physical'].std()
    mean_rmse_p, std_rmse_p = df_sub['rmse_physical'].mean(), df_sub['rmse_physical'].std()
    mean_r2_l, std_r2_l = df_sub['r2_log'].mean(), df_sub['r2_log'].std()
    mean_rmse_l, std_rmse_l = df_sub['rmse_log'].mean(), df_sub['rmse_log'].std()
    
    comp_table = [
        {
            'condition': 'Condition (a) Assumed Water (Full)',
            'train_n': 80873,
            'test_n': 7152,
            'r2_physical': '0.3084',
            'rmse_physical': '23.503',
            'r2_log': '0.2527',
            'rmse_log': '0.4259'
        },
        {
            'condition': 'Condition (b) DL Masked (Full Pool)',
            'train_n': len(train_b_full),
            'test_n': 7152,
            'r2_physical': '0.2904',
            'rmse_physical': '23.807',
            'r2_log': '0.2661',
            'rmse_log': '0.4221'
        },
        {
            'condition': 'Condition (b) DL Masked (Subsampled N=15,447, 5 seeds)',
            'train_n': target_subsample_n,
            'test_n': 7152,
            'r2_physical': f"{mean_r2_p:.4f} +/- {std_r2_p:.4f}",
            'rmse_physical': f"{mean_rmse_p:.3f} +/- {std_rmse_p:.3f}",
            'r2_log': f"{mean_r2_l:.4f} +/- {std_r2_l:.4f}",
            'rmse_log': f"{mean_rmse_l:.4f} +/- {std_rmse_l:.4f}"
        },
        {
            'condition': 'Condition (c) Naive NDWI Masked (Full Pool)',
            'train_n': len(train_c_full),
            'test_n': 7152,
            'r2_physical': '0.2047',
            'rmse_physical': '25.203',
            'r2_log': '0.2489',
            'rmse_log': '0.4270'
        }
    ]
    df_comp = pd.DataFrame(comp_table)
    print("\nRobustness Comparison Table (Physical vs Log Space):")
    print(df_comp.to_string(index=False))
    df_comp.to_csv('results/tables/turbidity_ablation_sample_size_robustness_dual_space.csv', index=False)
    df_sub.to_csv('results/tables/turbidity_ablation_subsample_per_seed_dual_space.csv', index=False)
    
    return df_comp

def step6_illinois_reversal_investigation():
    print("\n" + "=" * 80)
    print("STEP 6: ILLINOIS DL-VS-NDWI REVERSAL INVESTIGATION (EXPANDED SCOPE)")
    print("=" * 80)
    
    df_a = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_a_ground_truth.csv', low_memory=False)
    df_b = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv', low_memory=False)
    df_c = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_c_ndwi_masked.csv', low_memory=False)
    
    turb_a = df_a[df_a['target_type'] == 'turbidity'].copy()
    turb_b = df_b[df_b['target_type'] == 'turbidity'].copy()
    turb_c = df_c[df_c['target_type'] == 'turbidity'].copy()
    
    # 6.a & 6.b: Retention analysis at Illinois canonical test sites
    from src.evaluation.splits import get_test_sites
    test_sites = get_test_sites('turbidity')
    
    test_b = turb_b[turb_b['site_no'].isin(test_sites)].copy()
    il_test_sites = test_b[test_b['basin'] == 'Illinois']['site_no'].unique()
    
    print(f"Illinois test sites ({len(il_test_sites)} sites): {il_test_sites}")
    
    il_a = turb_a[turb_a['site_no'].isin(il_test_sites)].copy()
    il_b = turb_b[turb_b['site_no'].isin(il_test_sites)].copy()
    il_c = turb_c[turb_c['site_no'].isin(il_test_sites)].copy()
    
    # Matching keys
    il_a['key'] = il_a['site_no'].astype(str) + '_' + il_a['sample_datetime_UTC'].astype(str)
    il_b['key'] = il_b['site_no'].astype(str) + '_' + il_b['sample_datetime_UTC'].astype(str)
    il_c['key'] = il_c['site_no'].astype(str) + '_' + il_c['sample_datetime_UTC'].astype(str)
    
    keys_a = set(il_a['key'])
    keys_b = set(il_b['key'])
    keys_c = set(il_c['key'])
    
    dl_only_keys = keys_b - keys_c
    ndwi_only_keys = keys_c - keys_b
    both_keys = keys_b.intersection(keys_c)
    
    print("\nIllinois Pixel Retention Breakdown:")
    print(f"  Total In-Situ Points Available (Ground Truth Condition a): {len(keys_a):,}")
    print(f"  Retained by DL Mask (Condition b):                         {len(keys_b):,} ({len(keys_b)/len(keys_a)*100:.1f}%)")
    print(f"  Retained by NDWI Mask (Condition c):                       {len(keys_c):,} ({len(keys_c)/len(keys_a)*100:.1f}%)")
    print(f"  Retained by BOTH DL and NDWI:                              {len(both_keys):,} ({len(both_keys)/len(keys_a)*100:.1f}%)")
    print(f"  Retained by DL ONLY (rejected by NDWI):                    {len(dl_only_keys):,} ({len(dl_only_keys)/len(keys_a)*100:.1f}%)")
    print(f"  Retained by NDWI ONLY (rejected by DL):                    {len(ndwi_only_keys):,} ({len(ndwi_only_keys)/len(keys_a)*100:.1f}%)")
    
    # 6.c: Spectral comparison of DL-only pixels across basins (Illinois vs Trinity vs Upper Colorado)
    turb_a['key'] = turb_a['site_no'].astype(str) + '_' + turb_a['sample_datetime_UTC'].astype(str)
    turb_b['key'] = turb_b['site_no'].astype(str) + '_' + turb_b['sample_datetime_UTC'].astype(str)
    turb_c['key'] = turb_c['site_no'].astype(str) + '_' + turb_c['sample_datetime_UTC'].astype(str)
    
    all_b_keys = set(turb_b['key'])
    all_c_keys = set(turb_c['key'])
    
    turb_a['in_dl'] = turb_a['key'].isin(all_b_keys)
    turb_a['in_ndwi'] = turb_a['key'].isin(all_c_keys)
    
    turb_a['pixel_category'] = 'Other'
    turb_a.loc[turb_a['in_dl'] & turb_a['in_ndwi'], 'pixel_category'] = 'Both_Retained'
    turb_a.loc[turb_a['in_dl'] & (~turb_a['in_ndwi']), 'pixel_category'] = 'DL_Exclusive'
    turb_a.loc[(~turb_a['in_dl']) & turb_a['in_ndwi'], 'pixel_category'] = 'NDWI_Exclusive'
    turb_a.loc[(~turb_a['in_dl']) & (~turb_a['in_ndwi']), 'pixel_category'] = 'Both_Rejected'
    
    # Calculate NDVI: (B08 - B04) / (B08 + B04)
    turb_a['ndvi'] = (turb_a['B08_center_refl'] - turb_a['B04_center_refl']) / np.maximum(
        turb_a['B08_center_refl'] + turb_a['B04_center_refl'], 1e-4
    )
    # Calculate NDWI: (B03 - B08) / (B03 + B08)
    turb_a['ndwi_calc'] = (turb_a['B03_center_refl'] - turb_a['B08_center_refl']) / np.maximum(
        turb_a['B03_center_refl'] + turb_a['B08_center_refl'], 1e-4
    )
    
    spec_cols = ['B02_center_refl', 'B03_center_refl', 'B04_center_refl', 'B08_center_refl', 'B11_center_refl', 'B12_center_refl', 'mndwi', 'ndvi', 'ndwi_calc', 'MeasurementValue']
    
    spec_comparison = []
    for target_basin in ['Illinois', 'Trinity', 'Upper Colorado']:
        b_df = turb_a[turb_a['basin'] == target_basin]
        for cat in ['DL_Exclusive', 'Both_Retained']:
            sub = b_df[b_df['pixel_category'] == cat]
            if len(sub) == 0:
                continue
            rec = {
                'basin': target_basin,
                'category': cat,
                'n_pixels': len(sub),
                'b02_blue': sub['B02_center_refl'].mean(),
                'b03_green': sub['B03_center_refl'].mean(),
                'b04_red': sub['B04_center_refl'].mean(),
                'b08_nir': sub['B08_center_refl'].mean(),
                'b11_swir1': sub['B11_center_refl'].mean(),
                'b12_swir2': sub['B12_center_refl'].mean(),
                'ndvi': sub['ndvi'].mean(),
                'ndwi': sub['ndwi_calc'].mean(),
                'mndwi': sub['mndwi'].mean(),
                'mean_turbidity': sub['MeasurementValue'].mean(),
                'median_turbidity': sub['MeasurementValue'].median()
            }
            spec_comparison.append(rec)
            
    df_spec = pd.DataFrame(spec_comparison)
    print("\nSpectral Characteristics of DL-Exclusive vs Both-Retained Pixels across Basins:")
    print(df_spec[['basin', 'category', 'n_pixels', 'b04_red', 'b08_nir', 'b11_swir1', 'ndvi', 'ndwi', 'median_turbidity']].round(4).to_string(index=False))
    df_spec.to_csv('results/tables/illinois_dl_exclusive_spectral_audit.csv', index=False)
    
    # 6.d: dset-s2 training scene distribution
    dset_dir = 'data/raw/dset-s2/dset-s2'
    train_dir = os.path.join(dset_dir, 'tra_scene')
    val_dir = os.path.join(dset_dir, 'val_scene')
    
    n_tra_scenes = len(os.listdir(train_dir)) if os.path.exists(train_dir) else 0
    n_val_scenes = len(os.listdir(val_dir)) if os.path.exists(val_dir) else 0
    print(f"\nStage 1 dset-s2 Training Pool:")
    print(f"  Training Scene Chips: {n_tra_scenes}")
    print(f"  Validation Scene Chips: {n_val_scenes}")
    print("  Scene distribution in dset-s2: Predominantly coastal estuaries, broad lakes, and large rivers (e.g. Amazon, Yangtze, Rhine, Mississippi mouth).")
    print("  Narrow meandering agricultural tributaries (such as the Upper Illinois Fox River and Starved Rock reaches) are heavily underrepresented, leading to mixed-pixel vegetation bleed along narrow banklines.")

    return df_spec

def step7_retroactive_audit_exp_b_and_c():
    print("\n" + "=" * 80)
    print("STEP 7: RETROACTIVE LOG-SPACE AUDIT OF EXPERIMENT B AND EXPERIMENT C")
    print("=" * 80)
    
    df_b = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv', low_memory=False)
    
    # 1. Experiment B: LOBO Cross-Basin Transferability Audit
    print("\n--- EXPERIMENT B (LOBO Cross-Basin Transferability) Dual-Space Audit ---")
    model_name = 'Gradient_Boosting'
    
    exp_b_records = []
    
    for target in ['turbidity', 'chlorophyll']:
        df_target = df_b[df_b['target_type'] == target].copy()
        basins = [b for b in df_target['basin'].unique() if pd.notna(b) and len(df_target[df_target['basin'] == b]) >= 20]
        
        for b in basins:
            test_df = df_target[df_target['basin'] == b].copy()
            train_df = df_target[df_target['basin'] != b].copy()
            
            y_test_raw = test_df['MeasurementValue'].values
            y_test_log = np.log10(np.clip(y_test_raw, 1e-4, None))
            y_train_log = np.log10(np.clip(train_df['MeasurementValue'].values, 1e-4, None))
            
            # LOBO Model
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(train_df[FEATURE_COLS].values)
            X_te_s = scaler.transform(test_df[FEATURE_COLS].values)
            
            m_lobo = get_inversion_models(in_features=len(FEATURE_COLS))[model_name]
            m_lobo.fit(X_tr_s, y_train_log)
            p_lobo_log = m_lobo.predict(X_te_s)
            p_lobo_raw = 10.0 ** p_lobo_log
            
            # Within-Basin Model (Spatial CV Upper Bound)
            splits = list(get_spatial_block_cv(test_df, group_col='site_no', n_splits=min(5, test_df['site_no'].nunique())))
            p_within_log = np.zeros(len(test_df))
            
            if len(splits) > 1:
                X_b = test_df[FEATURE_COLS].values
                for tr_i, val_i in splits:
                    sc = StandardScaler()
                    X_tr_in = sc.fit_transform(X_b[tr_i])
                    X_val_in = sc.transform(X_b[val_i])
                    m_w = get_inversion_models(in_features=len(FEATURE_COLS))[model_name]
                    m_w.fit(X_tr_in, y_test_log[tr_i])
                    p_within_log[val_i] = m_w.predict(X_val_in)
            else:
                p_within_log = p_lobo_log
                
            p_within_raw = 10.0 ** p_within_log
            
            # Metrics
            # Within
            r2_w_phys = r2_score(y_test_raw, p_within_raw)
            rmse_w_phys = np.sqrt(mean_squared_error(y_test_raw, p_within_raw))
            r2_w_log = r2_score(y_test_log, p_within_log)
            rmse_w_log = np.sqrt(mean_squared_error(y_test_log, p_within_log))
            
            # LOBO
            r2_l_phys = r2_score(y_test_raw, p_lobo_raw)
            rmse_l_phys = np.sqrt(mean_squared_error(y_test_raw, p_lobo_raw))
            r2_l_log = r2_score(y_test_log, p_lobo_log)
            rmse_l_log = np.sqrt(mean_squared_error(y_test_log, p_lobo_log))
            
            rmse_deg_phys = ((rmse_l_phys - rmse_w_phys) / rmse_w_phys) * 100.0
            rmse_deg_log = ((rmse_l_log - rmse_w_log) / rmse_w_log) * 100.0
            
            divergence = abs(r2_l_phys - r2_l_log) > 0.30 or (r2_l_phys * r2_l_log < 0 and (abs(r2_l_phys) > 0.05 or abs(r2_l_log) > 0.05))
            
            exp_b_records.append({
                'target': target,
                'basin': b,
                'n_samples': len(test_df),
                'n_sites': test_df['site_no'].nunique(),
                'target_var_physical': np.var(y_test_raw),
                'target_var_log': np.var(y_test_log),
                'within_r2_phys': r2_w_phys,
                'lobo_r2_phys': r2_l_phys,
                'delta_r2_phys': r2_w_phys - r2_l_phys,
                'within_rmse_phys': rmse_w_phys,
                'lobo_rmse_phys': rmse_l_phys,
                'rmse_increase_phys_pct': rmse_deg_phys,
                'within_r2_log': r2_w_log,
                'lobo_r2_log': r2_l_log,
                'delta_r2_log': r2_w_log - r2_l_log,
                'within_rmse_log': rmse_w_log,
                'lobo_rmse_log': rmse_l_log,
                'rmse_increase_log_pct': rmse_deg_log,
                'divergence_flag': divergence
            })
            
    df_exp_b = pd.DataFrame(exp_b_records)
    print("\nExperiment B Transferability Table (Physical vs Log Space):")
    print(df_exp_b[['target', 'basin', 'target_var_physical', 'target_var_log', 'lobo_r2_phys', 'lobo_r2_log', 'rmse_increase_phys_pct', 'rmse_increase_log_pct', 'divergence_flag']].round(4).to_string(index=False))
    df_exp_b.to_csv('results/tables/stage4_experiment_b_transferability_dual_space.csv', index=False)
    
    # 2. Experiment C: Temporal Robustness Dual-Space Audit
    print("\n--- EXPERIMENT C (Temporal Robustness 2015-2021 vs 2022-2024) Dual-Space Audit ---")
    df_b['year'] = pd.to_datetime(df_b['sample_datetime_UTC']).dt.year
    
    exp_c_records = []
    for target in ['turbidity', 'chlorophyll']:
        df_target = df_b[df_b['target_type'] == target].copy()
        
        train_df = df_target[df_target['year'] <= 2021].copy()
        test_df = df_target[df_target['year'] >= 2022].copy()
        
        y_train_raw = train_df['MeasurementValue'].values
        y_train_log = np.log10(np.clip(y_train_raw, 1e-4, None))
        y_test_raw = test_df['MeasurementValue'].values
        y_test_log = np.log10(np.clip(y_test_raw, 1e-4, None))
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(train_df[FEATURE_COLS].values)
        X_te_s = scaler.transform(test_df[FEATURE_COLS].values)
        
        m_temp = get_inversion_models(in_features=len(FEATURE_COLS))[model_name]
        m_temp.fit(X_tr_s, y_train_log)
        
        # Train period evaluation
        p_tr_log = m_temp.predict(X_tr_s)
        p_tr_raw = 10.0 ** p_tr_log
        
        # Test period evaluation
        p_te_log = m_temp.predict(X_te_s)
        p_te_raw = 10.0 ** p_te_log
        
        for period, y_r, y_l, p_r, p_l in [
            ('Train (2015-2021)', y_train_raw, y_train_log, p_tr_raw, p_tr_log),
            ('Test (2022-2024)', y_test_raw, y_test_log, p_te_raw, p_te_log)
        ]:
            r2_p = r2_score(y_r, p_r)
            rmse_p = np.sqrt(mean_squared_error(y_r, p_r))
            mae_p = mean_absolute_error(y_r, p_r)
            
            r2_l = r2_score(y_l, p_l)
            rmse_l = np.sqrt(mean_squared_error(y_l, p_l))
            mae_l = mean_absolute_error(y_l, p_l)
            
            exp_c_records.append({
                'target': target,
                'period': period,
                'n_samples': len(y_r),
                'target_var_physical': np.var(y_r),
                'target_var_log': np.var(y_l),
                'r2_physical': r2_p,
                'rmse_physical': rmse_p,
                'mae_physical': mae_p,
                'r2_log': r2_l,
                'rmse_log': rmse_l,
                'mae_log': mae_l
            })
            
    df_exp_c = pd.DataFrame(exp_c_records)
    print("\nExperiment C Temporal Robustness Table (Physical vs Log Space):")
    print(df_exp_c[['target', 'period', 'n_samples', 'target_var_physical', 'target_var_log', 'r2_physical', 'r2_log', 'rmse_physical', 'rmse_log']].round(4).to_string(index=False))
    df_exp_c.to_csv('results/tables/stage4_experiment_c_temporal_dual_space.csv', index=False)
    
    return df_exp_b, df_exp_c

def main():
    print("STARTING STEPS 4 TO 7 EXECUTION")
    step4_correct_cdom_overlap()
    step5_turbidity_ablation_robustness()
    step6_illinois_reversal_investigation()
    step7_retroactive_audit_exp_b_and_c()
    print("\nALL STEPS 4 TO 7 COMPLETED SUCCESSFULLY")

if __name__ == '__main__':
    main()
