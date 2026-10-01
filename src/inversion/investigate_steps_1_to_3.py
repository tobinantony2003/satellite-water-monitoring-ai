"""
Script: investigate_steps_1_to_3.py
Executes detailed analysis for Steps 1, 2, and 3:
- Step 1: Detailed Trinity fold analysis (target & spectral features ndci, b5_b4_ratio distribution vs training set; naive mean predictor vs model; per-fold table).
- Step 2: Harmonized Chlorophyll-a site overlap matrix on the 18 retained sites.
- Step 3: Confounding audit on secondary calibrated-conversion datasets (across Condition a, b, c) and 3-way comparison table.
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
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def step1_trinity_investigation():
    print("=" * 80)
    print("STEP 1: TRINITY (294643095035200) POSITIVE-TRANSFER FOLD INVESTIGATION")
    print("=" * 80)

    data_path = 'data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv'
    df = pd.read_csv(data_path, low_memory=False)
    chl_df = df[df['target_type'] == 'chlorophyll'].copy()
    chl_df['site_no_str'] = chl_df['site_no'].astype(str).str.zfill(8)
    chl_df['log_chla'] = np.log10(np.clip(chl_df['MeasurementValue'], 1e-4, None))
    chl_df['b5_b4_ratio'] = chl_df['B05_center_refl'] / np.maximum(chl_df['B04_center_refl'], 1e-4)

    trinity_site = '294643095035200'
    trinity = chl_df[chl_df['site_no_str'] == trinity_site].copy()
    train = chl_df[chl_df['site_no_str'] != trinity_site].copy()

    features_to_compare = ['MeasurementValue', 'log_chla', 'ndci', 'b5_b4_ratio', 'B05_center_refl', 'B04_center_refl']
    
    rows = []
    for feat in features_to_compare:
        t_vals = trinity[feat].dropna().values
        tr_vals = train[feat].dropna().values

        # Kolmogorov-Smirnov test for distribution equivalence
        ks_stat, ks_pval = stats.ks_2samp(t_vals, tr_vals)

        # Overlap coefficient / range overlap
        t_min, t_max = np.min(t_vals), np.max(t_vals)
        tr_min, tr_max = np.min(tr_vals), np.max(tr_vals)
        in_range_pct = np.mean((t_vals >= tr_min) & (t_vals <= tr_max)) * 100.0

        rows.append({
            'feature': feat,
            'trinity_mean': np.mean(t_vals),
            'trinity_std': np.std(t_vals),
            'trinity_var': np.var(t_vals),
            'trinity_median': np.median(t_vals),
            'trinity_iqr': np.percentile(t_vals, 75) - np.percentile(t_vals, 25),
            'trinity_min': t_min,
            'trinity_max': t_max,
            'train_mean': np.mean(tr_vals),
            'train_std': np.std(tr_vals),
            'train_var': np.var(tr_vals),
            'train_median': np.median(tr_vals),
            'train_iqr': np.percentile(tr_vals, 75) - np.percentile(tr_vals, 25),
            'train_min': tr_min,
            'train_max': tr_max,
            'var_ratio_test_vs_train': np.var(t_vals) / np.var(tr_vals),
            'ks_stat': ks_stat,
            'ks_pval': ks_pval,
            'trinity_within_train_range_pct': in_range_pct
        })

    dist_df = pd.DataFrame(rows)
    print("\nFeature & Target Distribution Comparison: Trinity vs Other 17 Sites:")
    print(dist_df[['feature', 'trinity_mean', 'train_mean', 'trinity_std', 'train_std', 'trinity_var', 'train_var', 'var_ratio_test_vs_train']].to_string(index=False))
    dist_df.to_csv('results/tables/trinity_distribution_comparison.csv', index=False)

    # Reference naive mean-predictor
    y_test_raw = trinity['MeasurementValue'].values
    y_test_log = trinity['log_chla'].values
    y_train_raw = train['MeasurementValue'].values
    y_train_log = train['log_chla'].values

    mean_train_raw = np.mean(y_train_raw)
    mean_train_log = np.mean(y_train_log)

    ss_tot_raw = np.sum((y_test_raw - np.mean(y_test_raw))**2)
    ss_res_naive_raw = np.sum((y_test_raw - mean_train_raw)**2)
    naive_r2_raw = 1.0 - (ss_res_naive_raw / ss_tot_raw)
    naive_rmse_raw = np.sqrt(np.mean((y_test_raw - mean_train_raw)**2))
    naive_mae_raw = np.mean(np.abs(y_test_raw - mean_train_raw))

    ss_tot_log = np.sum((y_test_log - np.mean(y_test_log))**2)
    ss_res_naive_log = np.sum((y_test_log - mean_train_log)**2)
    naive_r2_log = 1.0 - (ss_res_naive_log / ss_tot_log)
    naive_rmse_log = np.sqrt(np.mean((y_test_log - mean_train_log)**2))

    # Fold breakdown table with naive baseline added
    per_fold = pd.read_csv('results/tables/chlorophyll_mixed_effects_per_fold.csv')
    per_fold['naive_mean_r2_raw'] = np.nan
    per_fold['naive_mean_rmse_raw'] = np.nan

    # Compute naive predictor for all folds for comparison
    gkf = GroupKFold(n_splits=5)
    for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(chl_df, groups=chl_df['site_no_str'])):
        fold_num = fold_idx + 1
        y_val = chl_df.iloc[val_idx]['MeasurementValue'].values
        y_tr = chl_df.iloc[tr_idx]['MeasurementValue'].values
        m_tr = np.mean(y_tr)
        ss_t = np.sum((y_val - np.mean(y_val))**2)
        ss_r = np.sum((y_val - m_tr)**2)
        r2_n = 1.0 - (ss_r / ss_t) if ss_t > 0 else np.nan
        rmse_n = np.sqrt(np.mean((y_val - m_tr)**2))
        
        per_fold.loc[per_fold['fold'] == fold_num, 'naive_mean_r2_raw'] = round(r2_n, 4)
        per_fold.loc[per_fold['fold'] == fold_num, 'naive_mean_rmse_raw'] = round(rmse_n, 2)

    print("\nUpdated Spatial Block CV Per-Fold Performance with Naive Baseline:")
    print(per_fold.to_string(index=False))
    per_fold.to_csv('results/tables/chlorophyll_per_fold_with_naive_baseline.csv', index=False)

    return dist_df, per_fold

def step2_chlorophyll_overlap():
    print("\n" + "=" * 80)
    print("STEP 2: RECOMPUTE CHLOROPHYLL-A SITE OVERLAP MATRIX (18 HARMONIZED SITES)")
    print("=" * 80)

    from src.inversion.check_site_time_confounding import compute_site_timeline_and_overlap

    data_path = 'data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv'
    df = pd.read_csv(data_path, low_memory=False)
    chl_df = df[df['target_type'] == 'chlorophyll'].copy()

    summary_df, df_simpson, df_jaccard, df_shared_m, df_metrics = compute_site_timeline_and_overlap(chl_df, 'Chlorophyll-a Harmonized (18 Sites)')

    n_sites = len(summary_df)
    total_pairs = (n_sites * (n_sites - 1)) // 2
    upper_idx = np.triu_indices(n_sites, k=1)

    zero_overlap_pairs = int((df_simpson.values[upper_idx] == 0).sum())
    zero_month_pairs = int((df_shared_m.values[upper_idx] == 0).sum())
    mean_overlap_coeff = float(np.mean(df_simpson.values[upper_idx]))
    mean_shared_months = float(np.mean(df_shared_m.values[upper_idx]))

    print(f"\nHarmonized Chlorophyll-a (18 Sites, N={len(chl_df):,}):")
    print(f"  Total distinct station pairs: {total_pairs}")
    print(f"  Pairs with ZERO temporal window overlap: {zero_overlap_pairs} / {total_pairs} ({zero_overlap_pairs / total_pairs * 100:.1f}%)")
    print(f"  Pairs with ZERO shared active calendar months: {zero_month_pairs} / {total_pairs} ({zero_month_pairs / total_pairs * 100:.1f}%)")
    print(f"  Mean Simpson Overlap Coefficient: {mean_overlap_coeff:.3f}")
    print(f"  Mean Shared Active Months: {mean_shared_months:.1f} months")

    overlap_stats = pd.DataFrame([{
        'dataset': 'Chlorophyll-a Harmonized Native',
        'n_sites': n_sites,
        'n_samples': len(chl_df),
        'total_pairs': total_pairs,
        'zero_window_overlap_pairs': zero_overlap_pairs,
        'zero_window_pct': round(zero_overlap_pairs / total_pairs * 100, 2),
        'zero_shared_month_pairs': zero_month_pairs,
        'zero_month_pct': round(zero_month_pairs / total_pairs * 100, 2),
        'mean_overlap_coefficient': round(mean_overlap_coeff, 3),
        'mean_shared_months': round(mean_shared_months, 1)
    }])
    overlap_stats.to_csv('results/tables/chlorophyll_harmonized_overlap_summary_stats.csv', index=False)
    print("\nOverlap Summary Table:")
    print(overlap_stats.to_string(index=False))

    return overlap_stats

def step3_secondary_calibrated_confounding():
    print("\n" + "=" * 80)
    print("STEP 3: CONFOUNDING AUDIT ON SECONDARY CALIBRATED-CONVERSION DATASETS")
    print("=" * 80)

    from src.inversion.rerun_harmonized_diagnostics import audit_target_confounding, evaluate_mixedlm_spatial_cv

    conditions = [
        ('Condition_a_GroundTruth', 'data/processed/matched_wq_cleaned/matched_set_a_ground_truth_harmonized.csv'),
        ('Condition_b_DLMasked', 'data/processed/matched_wq_cleaned/matched_set_b_dl_masked_harmonized.csv'),
        ('Condition_c_NDWIMasked', 'data/processed/matched_wq_cleaned/matched_set_c_ndwi_masked_harmonized.csv')
    ]

    all_cdom_results = []
    all_chl_results = []
    all_mixed_results = []

    cdom_feats = ['ndwi', 'mndwi', 'B04_center_refl', 'cdom_ratio', 'B8A_center_refl', 'B03_center_refl', 'B02_center_refl']
    chl_feats = ['ndci', 'b5_b4_ratio', 'B05_center_refl', 'B07_center_refl', 'B8A_center_refl', 'B04_center_refl', 'B03_center_refl', 'B02_center_refl']

    for cond_name, filepath in conditions:
        if not os.path.exists(filepath):
            print(f"Warning: File {filepath} not found, skipping...")
            continue
        print(f"\nProcessing {cond_name} from {filepath}...")
        df = pd.read_csv(filepath, low_memory=False)
        df['site_no_str'] = df['site_no'].astype(str).str.zfill(8)
        df['log_target'] = np.log10(np.clip(df['MeasurementValue'], 1e-4, None))
        df['b5_b4_ratio'] = df['B05_center_refl'] / np.maximum(df['B04_center_refl'], 1e-4)

        # CDOM
        cdom_df = df[df['target_type'] == 'cdom'].copy()
        if len(cdom_df) > 0:
            res_cdom = audit_target_confounding(cdom_df, f"CDOM_{cond_name}", cdom_feats, 'log_target')
            res_cdom['condition'] = cond_name
            all_cdom_results.append(res_cdom)

            cdom_form = "log_target ~ ndwi + mndwi + B04_center_refl"
            cdom_mixed, _ = evaluate_mixedlm_spatial_cv(cdom_df, cdom_form, 'MeasurementValue', 'log_target', 'QSE')
            all_mixed_results.append({'target': 'CDOM', 'condition': cond_name, 'dataset': 'Secondary_CalibratedConversion', **cdom_mixed})

        # Chlorophyll-a
        chl_df = df[df['target_type'] == 'chlorophyll'].copy()
        if len(chl_df) > 0:
            res_chl = audit_target_confounding(chl_df, f"Chl_{cond_name}", chl_feats, 'log_target')
            res_chl['condition'] = cond_name
            all_chl_results.append(res_chl)

            chl_form = "log_target ~ ndci + B05_center_refl + B04_center_refl"
            chl_mixed, _ = evaluate_mixedlm_spatial_cv(chl_df, chl_form, 'MeasurementValue', 'log_target', 'ug/L')
            all_mixed_results.append({'target': 'Chlorophyll-a', 'condition': cond_name, 'dataset': 'Secondary_CalibratedConversion', **chl_mixed})

    df_cdom_all = pd.concat(all_cdom_results, ignore_index=True) if all_cdom_results else pd.DataFrame()
    df_chl_all = pd.concat(all_chl_results, ignore_index=True) if all_chl_results else pd.DataFrame()
    df_mixed_all = pd.DataFrame(all_mixed_results)

    df_cdom_all.to_csv('results/tables/cdom_secondary_all_conditions.csv', index=False)
    df_chl_all.to_csv('results/tables/chlorophyll_secondary_all_conditions.csv', index=False)
    df_mixed_all.to_csv('results/tables/mixed_effects_secondary_all_conditions.csv', index=False)

    print("\nSecondary Calibrated-Conversion MixedLM Results across Conditions:")
    print(df_mixed_all[['target', 'condition', 'total_samples', 'n_sites', 'icc', 'overall_r2_raw', 'overall_rmse_raw']].to_string(index=False))

    return df_cdom_all, df_chl_all, df_mixed_all

def main():
    print("STARTING STEPS 1-3 INVESTIGATION SCRIPT")
    dist_df, per_fold = step1_trinity_investigation()
    overlap_stats = step2_chlorophyll_overlap()
    df_cdom_all, df_chl_all, df_mixed_all = step3_secondary_calibrated_confounding()
    print("\nALL STEPS 1-3 COMPLETED SUCCESSFULLY")

if __name__ == '__main__':
    main()
