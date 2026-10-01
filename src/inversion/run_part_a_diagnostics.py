"""
Script: run_part_a_diagnostics.py
Executes all 7 diagnostic tasks of Part A:
1. Multicollinearity check (B4/B3, MNDWI, B04, B03 for Turbidity; correlation matrix & VIF).
2. Correct CDOM site-overlap record.
3. Turbidity ablation training-size robustness check (subsample N=15,447 over 5 random seeds).
4. Illinois turbidity failure mode investigation (water-pixel count, turbidity range, variance).
5. Trinity Chlorophyll-a fold investigation (variance, distribution vs training set, naive mean R^2).
6. Chlorophyll-a site-overlap recomputation (18 harmonized sites only).
7. Confounding audit on secondary calibrated-conversion datasets (CDOM & Chl-a).
"""

import os
import sys
from pathlib import Path
import shutil

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

from src.inversion.models import get_inversion_models
from src.inversion.metrics import compute_regression_metrics
from src.inversion.train_inversion import FEATURE_COLS

def part_a1_multicollinearity():
    print("\n" + "=" * 80)
    print("PART A.1: MULTICOLLINEARITY CHECK FOR TURBIDITY")
    print("=" * 80)
    
    data_path = 'data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv'
    df = pd.read_csv(data_path, low_memory=False)
    turb_df = df[df['target_type'] == 'turbidity'].copy()
    
    feats = ['turbidity_ratio', 'mndwi', 'B04_center_refl', 'B03_center_refl']
    sub = turb_df[feats].dropna()
    
    # 1. Pearson correlation matrix
    pearson_corr = sub.corr(method='pearson')
    # 2. Spearman correlation matrix
    spearman_corr = sub.corr(method='spearman')
    
    print("\nPearson Correlation Matrix (Turbidity Features, N=37,655):")
    print(pearson_corr.round(4).to_string())
    
    print("\nSpearman Correlation Matrix (Turbidity Features, N=37,655):")
    print(spearman_corr.round(4).to_string())
    
    # Check if any pair exceeds |r| > 0.80
    high_pairs = []
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            f1, f2 = feats[i], feats[j]
            r_p = pearson_corr.loc[f1, f2]
            r_s = spearman_corr.loc[f1, f2]
            if abs(r_p) >= 0.80 or abs(r_s) >= 0.80:
                high_pairs.append((f1, f2, r_p, r_s))
                
    print(f"\nPairs with |r| >= 0.80 ({len(high_pairs)} pairs):")
    for f1, f2, rp, rs in high_pairs:
        print(f"  {f1} <-> {f2}: Pearson r = {rp:.4f}, Spearman r = {rs:.4f}")
        
    # 3. Variance Inflation Factor (VIF)
    X = sub.copy()
    X = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    print("\nVariance Inflation Factors (VIF):")
    print(vif_data.to_string(index=False))
    
    # Save table
    os.makedirs('results/tables', exist_ok=True)
    pearson_corr.to_csv('results/tables/turbidity_feature_pearson_corr.csv')
    vif_data.to_csv('results/tables/turbidity_feature_vif.csv', index=False)
    
    return pearson_corr, vif_data

def part_a2_cdom_overlap_record():
    print("\n" + "=" * 80)
    print("PART A.2: CORRECT CDOM SITE-OVERLAP RECORD")
    print("=" * 80)
    
    data_path = 'data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv'
    df = pd.read_csv(data_path, low_memory=False)
    cdom_df = df[df['target_type'] == 'cdom'].copy()
    cdom_df['site_no_str'] = cdom_df['site_no'].astype(str).str.zfill(8)
    cdom_df['date'] = pd.to_datetime(cdom_df['sample_datetime_UTC'])
    
    site_recs = []
    for site, grp in cdom_df.groupby('site_no_str'):
        site_recs.append({
            'site_no': site,
            'station_nm': grp['station_nm'].iloc[0],
            'basin': grp['basin'].iloc[0],
            'n_samples': len(grp),
            'min_date': grp['date'].min().strftime('%Y-%m-%d'),
            'max_date': grp['date'].max().strftime('%Y-%m-%d'),
            'active_era': '2015-2020' if grp['date'].max().year <= 2020 else '2021-2024'
        })
    s_df = pd.DataFrame(site_recs).sort_values('min_date').reset_index(drop=True)
    
    # Overlap grouping column
    # Station 05586300 is Era 1 (Isolated, 0 days overlap with rest)
    # The other 5 stations are Era 2 (Mutually overlapping, 2021-2024)
    s_df['temporal_overlap_group'] = s_df['site_no'].apply(
        lambda s: 'Isolated_Era_1 (Zero Overlap with other 5 sites)' if s == '05586300' else 'Overlapping_Era_2 (Mutually Co-Active 2021-2024)'
    )
    
    print("\nCDOM Station Temporal Overlap Grouping:")
    print(s_df[['site_no', 'station_nm', 'basin', 'n_samples', 'min_date', 'max_date', 'temporal_overlap_group']].to_string(index=False))
    s_df.to_csv('results/tables/cdom_temporal_overlap_grouping.csv', index=False)
    return s_df

def part_a3_turbidity_ablation_robustness():
    print("\n" + "=" * 80)
    print("PART A.3: TURBIDITY ABLATION TRAINING-SIZE ROBUSTNESS CHECK")
    print("=" * 80)
    
    df_b = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv', low_memory=False)
    df_c = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_c_ndwi_masked.csv', low_memory=False)
    
    target_type = 'turbidity'
    b_turb = df_b[df_b['target_type'] == target_type].copy()
    c_turb = df_c[df_c['target_type'] == target_type].copy()
    
    # Canonical held-out test set
    from src.evaluation.splits import get_test_sites
    test_sites = get_test_sites('turbidity')
    
    test_df = b_turb[b_turb['site_no'].isin(test_sites)].copy()
    train_b_full = b_turb[~b_turb['site_no'].isin(test_sites)].copy()
    train_c_full = c_turb[~c_turb['site_no'].isin(test_sites)].copy()
    
    target_subsample_n = len(train_c_full)  # 15,447 samples
    print(f"Condition (b) full training pool: {len(train_b_full):,} samples")
    print(f"Condition (c) NDWI training pool: {len(train_c_full):,} samples")
    print(f"Held-out test set size: {len(test_df):,} samples")
    
    X_test = test_df[FEATURE_COLS].values
    y_test_raw = test_df['MeasurementValue'].values
    
    seeds = [42, 101, 2024, 7, 99]
    subsample_results = []
    
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
        
        m = compute_regression_metrics(y_test_raw, pred_raw)
        m['seed'] = seed
        m['train_n'] = target_subsample_n
        m['condition'] = 'Condition_(b)_DL_Subsampled_to_NDWI_Size'
        subsample_results.append(m)
        print(f"  Seed {seed:4d} -> R^2: {m['r2']:.4f} | RMSE: {m['rmse']:.3f} | MAE: {m['mae']:.3f}")
        
    sub_df = pd.DataFrame(subsample_results)
    
    mean_r2 = sub_df['r2'].mean()
    std_r2 = sub_df['r2'].std()
    mean_rmse = sub_df['rmse'].mean()
    std_rmse = sub_df['rmse'].std()
    mean_mae = sub_df['mae'].mean()
    std_mae = sub_df['mae'].std()
    
    print("\nRobustness Check Summary (5 Seeds, Subsampled N=15,447):")
    print(f"  Subsampled DL Masked R^2:  {mean_r2:.4f} +/- {std_r2:.4f}")
    print(f"  Subsampled DL Masked RMSE: {mean_rmse:.3f} +/- {std_rmse:.3f} FNU")
    print(f"  Subsampled DL Masked MAE:  {mean_mae:.3f} +/- {std_mae:.3f} FNU")
    
    # Comparison table
    robustness_comp = [
        {
            'condition': 'Condition (a) Assumed Water (Full)',
            'train_n': 80873,
            'test_n': 7152,
            'r2': '0.3084',
            'rmse': '23.503',
            'mae': '9.517'
        },
        {
            'condition': 'Condition (b) DL Masked (Full Pool)',
            'train_n': len(train_b_full),
            'test_n': 7152,
            'r2': '0.2904',
            'rmse': '23.807',
            'mae': '10.133'
        },
        {
            'condition': 'Condition (b) DL Masked (Subsampled to NDWI N, 5 seeds)',
            'train_n': target_subsample_n,
            'test_n': 7152,
            'r2': f"{mean_r2:.4f} +/- {std_r2:.4f}",
            'rmse': f"{mean_rmse:.3f} +/- {std_rmse:.3f}",
            'mae': f"{mean_mae:.3f} +/- {std_mae:.3f}"
        },
        {
            'condition': 'Condition (c) Naive NDWI Masked (Full Pool)',
            'train_n': len(train_c_full),
            'test_n': 7152,
            'r2': '0.2047',
            'rmse': '25.203',
            'mae': '9.102'
        }
    ]
    rob_df = pd.DataFrame(robustness_comp)
    print("\nAblation Training-Size Disentanglement Table:")
    print(rob_df.to_string(index=False))
    rob_df.to_csv('results/tables/turbidity_ablation_sample_size_robustness.csv', index=False)
    sub_df.to_csv('results/tables/turbidity_ablation_subsample_per_seed.csv', index=False)
    return rob_df

def part_a4_illinois_turbidity_failure():
    print("\n" + "=" * 80)
    print("PART A.4: ILLINOIS TURBIDITY FAILURE MODE INVESTIGATION")
    print("=" * 80)
    
    df_b = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv', low_memory=False)
    df_c = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_c_ndwi_masked.csv', low_memory=False)
    
    turb_b = df_b[df_b['target_type'] == 'turbidity'].copy()
    turb_c = df_c[df_c['target_type'] == 'turbidity'].copy()
    
    # Canonical held-out test sites
    from src.evaluation.splits import get_test_sites
    test_sites = get_test_sites('turbidity')
    
    test_b = turb_b[turb_b['site_no'].isin(test_sites)].copy()
    test_il = test_b[test_b['basin'] == 'Illinois'].copy()
    test_other = test_b[test_b['basin'] != 'Illinois'].copy()
    train_b = turb_b[~turb_b['site_no'].isin(test_sites)].copy()
    
    print(f"Illinois test set: {len(test_il)} samples across {test_il['site_no'].nunique()} sites:")
    for s, grp in test_il.groupby('site_no'):
        print(f"  Site {s} ({grp['station_nm'].iloc[0]}): N={len(grp)} | Turbidity range: [{grp['MeasurementValue'].min():.1f}, {grp['MeasurementValue'].max():.1f}] FNU | Median={grp['MeasurementValue'].median():.1f} | Mean={grp['MeasurementValue'].mean():.1f}")
        
    print(f"\nTurbidity Target Variance Comparison:")
    print(f"  Illinois Test Set:      Variance = {test_il['MeasurementValue'].var():.2f} (std = {test_il['MeasurementValue'].std():.2f}) | Median = {test_il['MeasurementValue'].median():.2f}")
    print(f"  Non-Illinois Test Set:  Variance = {test_other['MeasurementValue'].var():.2f} (std = {test_other['MeasurementValue'].std():.2f}) | Median = {test_other['MeasurementValue'].median():.2f}")
    print(f"  Full Training Set:      Variance = {train_b['MeasurementValue'].var():.2f} (std = {train_b['MeasurementValue'].std():.2f}) | Median = {train_b['MeasurementValue'].median():.2f}")
    
    # Check pixel masking retention at Illinois sites: DL vs NDWI vs Raw
    raw_path = 'data/processed/matched_wq_cleaned/matched_set_a_ground_truth_unitmixed_raw.csv'
    df_raw = pd.read_csv(raw_path, low_memory=False)
    raw_il_test = df_raw[(df_raw['target_type'] == 'turbidity') & (df_raw['site_no'].isin(test_il['site_no'].unique()))]
    
    print(f"\nPixel Retention in Illinois Test Sites:")
    print(f"  Total Raw In-Situ Points Available: {len(raw_il_test):,}")
    print(f"  Retained under DL Mask:              {len(test_il):,} ({len(test_il)/len(raw_il_test)*100:.1f}%)")
    test_c_il = turb_c[turb_c['site_no'].isin(test_il['site_no'].unique())]
    print(f"  Retained under Naive NDWI Mask:      {len(test_c_il):,} ({len(test_c_il)/len(raw_il_test)*100:.1f}%)")
    
    il_summary = {
        'metric': ['N_samples', 'N_sites', 'Turbidity_Mean', 'Turbidity_Median', 'Turbidity_Std', 'Turbidity_Var', 'Turbidity_IQR'],
        'Illinois_Test_Set': [
            len(test_il), test_il['site_no'].nunique(), round(test_il['MeasurementValue'].mean(), 2),
            round(test_il['MeasurementValue'].median(), 2), round(test_il['MeasurementValue'].std(), 2),
            round(test_il['MeasurementValue'].var(), 2),
            round(test_il['MeasurementValue'].quantile(0.75) - test_il['MeasurementValue'].quantile(0.25), 2)
        ],
        'Non_Illinois_Test_Set': [
            len(test_other), test_other['site_no'].nunique(), round(test_other['MeasurementValue'].mean(), 2),
            round(test_other['MeasurementValue'].median(), 2), round(test_other['MeasurementValue'].std(), 2),
            round(test_other['MeasurementValue'].var(), 2),
            round(test_other['MeasurementValue'].quantile(0.75) - test_other['MeasurementValue'].quantile(0.25), 2)
        ],
        'Training_Set_Pool': [
            len(train_b), train_b['site_no'].nunique(), round(train_b['MeasurementValue'].mean(), 2),
            round(train_b['MeasurementValue'].median(), 2), round(train_b['MeasurementValue'].std(), 2),
            round(train_b['MeasurementValue'].var(), 2),
            round(train_b['MeasurementValue'].quantile(0.75) - train_b['MeasurementValue'].quantile(0.25), 2)
        ]
    }
    il_df = pd.DataFrame(il_summary)
    print("\nIllinois vs Global Target Distribution:")
    print(il_df.to_string(index=False))
    il_df.to_csv('results/tables/illinois_turbidity_diagnostic.csv', index=False)
    return il_df

def part_a5_trinity_chla_investigation():
    print("\n" + "=" * 80)
    print("PART A.5: TRINITY CHLOROPHYLL-A FOLD INVESTIGATION")
    print("=" * 80)
    
    data_path = 'data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv'
    df = pd.read_csv(data_path, low_memory=False)
    chl_df = df[df['target_type'] == 'chlorophyll'].copy()
    chl_df['site_no_str'] = chl_df['site_no'].astype(str).str.zfill(8)
    chl_df['log_chla'] = np.log10(np.clip(chl_df['MeasurementValue'], 1e-4, None))
    
    trinity_site = '294643095035200'
    trinity_data = chl_df[chl_df['site_no_str'] == trinity_site].copy()
    train_data = chl_df[chl_df['site_no_str'] != trinity_site].copy()
    
    y_test_raw = trinity_data['MeasurementValue'].values
    y_test_log = trinity_data['log_chla'].values
    
    # 1. Variance and distribution comparison
    print(f"Trinity Station {trinity_site} ({trinity_data['station_nm'].iloc[0]}):")
    print(f"  N = {len(trinity_data):,}")
    print(f"  Physical Raw Chl-a: Mean = {y_test_raw.mean():.3f} ug/L | Median = {np.median(y_test_raw):.3f} | Std = {y_test_raw.std():.3f} | Var = {y_test_raw.var():.3f} | Range = [{y_test_raw.min():.2f}, {y_test_raw.max():.2f}]")
    print(f"  Log10 Chl-a:        Mean = {y_test_log.mean():.3f} | Std = {y_test_log.std():.3f} | Var = {y_test_log.var():.3f}")
    
    print(f"\nTraining Set (Remaining 17 Sites):")
    y_train_raw = train_data['MeasurementValue'].values
    y_train_log = train_data['log_chla'].values
    print(f"  N = {len(train_data):,}")
    print(f"  Physical Raw Chl-a: Mean = {y_train_raw.mean():.3f} ug/L | Median = {np.median(y_train_raw):.3f} | Std = {y_train_raw.std():.3f} | Var = {y_train_raw.var():.3f} | Range = [{y_train_raw.min():.2f}, {y_train_raw.max():.2f}]")
    print(f"  Log10 Chl-a:        Mean = {y_train_log.mean():.3f} | Std = {y_train_log.std():.3f} | Var = {y_train_log.var():.3f}")
    
    # 2. Naive mean-predictor evaluation on Trinity fold
    # Predicting training set mean
    mean_train_pred_raw = np.mean(y_train_raw)
    mean_train_pred_log = np.mean(y_train_log)
    
    ss_tot_raw = np.sum((y_test_raw - np.mean(y_test_raw))**2)
    ss_res_naive_raw = np.sum((y_test_raw - mean_train_pred_raw)**2)
    r2_naive_raw = 1.0 - (ss_res_naive_raw / ss_tot_raw)
    rmse_naive_raw = np.sqrt(np.mean((y_test_raw - mean_train_pred_raw)**2))
    
    ss_tot_log = np.sum((y_test_log - np.mean(y_test_log))**2)
    ss_res_naive_log = np.sum((y_test_log - mean_train_pred_log)**2)
    r2_naive_log = 1.0 - (ss_res_naive_log / ss_tot_log)
    rmse_naive_log = np.sqrt(np.mean((y_test_log - mean_train_pred_log)**2))
    
    print(f"\nNaive Baseline (Predicting Global Training Set Mean) on Trinity Fold:")
    print(f"  Training Set Mean Prediction: {mean_train_pred_raw:.2f} ug/L (vs Trinity actual mean: {y_test_raw.mean():.2f} ug/L)")
    print(f"  Naive R^2 (Physical Space):   {r2_naive_raw:.4f} (RMSE = {rmse_naive_raw:.2f} ug/L)")
    print(f"  Naive R^2 (Log Space):        {r2_naive_log:.4f} (RMSE = {rmse_naive_log:.3f})")
    print(f"  Trained Model R^2 (MixedLM):  +0.0366 (RMSE = 1.91 ug/L)")
    print(f"  Trained Model R^2 (Log Space): -0.0110")
    
    trinity_comp = pd.DataFrame([{
        'fold': 'Fold 2 (Trinity 294643095035200)',
        'target': 'Chlorophyll-a (ug/L)',
        'N_test': len(trinity_data),
        'trinity_target_var': round(float(y_test_raw.var()), 3),
        'training_target_var': round(float(y_train_raw.var()), 3),
        'variance_ratio_test_vs_train': round(float(y_test_raw.var() / y_train_raw.var()), 4),
        'naive_train_mean_prediction': round(float(mean_train_pred_raw), 2),
        'trinity_actual_mean': round(float(y_test_raw.mean()), 2),
        'naive_mean_r2_raw': round(float(r2_naive_raw), 4),
        'naive_mean_rmse_raw': round(float(rmse_naive_raw), 2),
        'mixedlm_model_r2_raw': 0.0366,
        'mixedlm_model_rmse_raw': 1.91
    }])
    trinity_comp.to_csv('results/tables/trinity_chla_fold_diagnostic.csv', index=False)
    print("\nTrinity Fold Diagnostic Table:")
    print(trinity_comp.to_string(index=False))
    return trinity_comp

def part_a6_chlorophyll_harmonized_overlap():
    print("\n" + "=" * 80)
    print("PART A.6: CHLOROPHYLL-A RECOMPUTED SITE OVERLAP MATRIX (18 SITES)")
    print("=" * 80)
    
    from src.inversion.check_site_time_confounding import compute_site_timeline_and_overlap
    
    data_path = 'data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv'
    df = pd.read_csv(data_path, low_memory=False)
    chl_df = df[df['target_type'] == 'chlorophyll'].copy()
    
    summary_df, df_simpson, df_jaccard, df_shared_m, _ = compute_site_timeline_and_overlap(chl_df, 'Chlorophyll-a Harmonized')
    
    n_sites = len(summary_df)
    total_pairs = (n_sites * (n_sites - 1)) // 2
    upper_idx = np.triu_indices(n_sites, k=1)
    zero_window_pairs = int((df_simpson.values[upper_idx] == 0).sum())
    zero_month_pairs = int((df_shared_m.values[upper_idx] == 0).sum())
    
    print(f"\nChlorophyll-a Harmonized (18 Sites) Pairwise Overlap Summary (Total distinct pairs = {total_pairs}):")
    print(f"  Pairs with ZERO window overlap: {zero_window_pairs} / {total_pairs} ({zero_window_pairs/total_pairs*100:.1f}%)")
    print(f"  Pairs with ZERO shared active months: {zero_month_pairs} / {total_pairs} ({zero_month_pairs/total_pairs*100:.1f}%)")
    
    summary_df.to_csv('results/tables/chlorophyll_harmonized_site_time_summary.csv', index=False)
    df_simpson.to_csv('results/tables/chlorophyll_harmonized_pairwise_overlap.csv')
    df_shared_m.to_csv('results/tables/chlorophyll_harmonized_pairwise_shared_months.csv')
    
    # Generate visualization
    plt.figure(figsize=(14, 11))
    sns.heatmap(df_simpson, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1.0,
                linewidths=0.5, linecolor='gray',
                cbar_kws={'label': 'Temporal Overlap Coefficient [overlap / min(span_i, span_j)]'})
    plt.title('Harmonized Chlorophyll-a (18 Sites): Pairwise Temporal Overlap Heatmap', fontsize=12, fontweight='bold')
    plt.xlabel('USGS Gauge Station', fontsize=10, fontweight='bold')
    plt.ylabel('USGS Gauge Station', fontsize=10, fontweight='bold')
    plt.tight_layout()
    
    fig_path = 'results/figures/chlorophyll_harmonized_site_time_overlap.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()
    
    brain_dir = r'C:\Users\Tobin Antony\.gemini\antigravity-ide\brain\730512ff-544c-4445-ac17-b37e3b98b9be'
    shutil.copy(fig_path, os.path.join(brain_dir, 'chlorophyll_harmonized_site_time_overlap.png'))
    print(f"Saved figure to {fig_path} and copied to brain directory.")
    return df_simpson

def part_a7_secondary_calibrated_confounding():
    print("\n" + "=" * 80)
    print("PART A.7: CONFOUNDING AUDIT ON SECONDARY CALIBRATED-CONVERSION DATASETS")
    print("=" * 80)
    
    from src.inversion.rerun_harmonized_diagnostics import audit_target_confounding, evaluate_mixedlm_spatial_cv
    
    sec_path = 'data/processed/matched_wq_cleaned/matched_set_b_dl_masked_harmonized.csv'
    df_sec = pd.read_csv(sec_path, low_memory=False)
    df_sec['site_no_str'] = df_sec['site_no'].astype(str).str.zfill(8)
    df_sec['log_cdom'] = np.log10(np.clip(df_sec['MeasurementValue'], 1e-4, None))
    df_sec['log_chla'] = np.log10(np.clip(df_sec['MeasurementValue'], 1e-4, None))
    df_sec['b5_b4_ratio'] = df_sec['B05_center_refl'] / np.maximum(df_sec['B04_center_refl'], 1e-4)
    
    # 1. CDOM Secondary
    cdom_sec = df_sec[df_sec['target_type'] == 'cdom']
    cdom_feats = ['ndwi', 'mndwi', 'B04_center_refl', 'cdom_ratio', 'B8A_center_refl', 'B03_center_refl', 'B02_center_refl']
    res_cdom_sec = audit_target_confounding(cdom_sec, 'CDOM_SecondaryHarmonized', cdom_feats, 'log_cdom')
    
    cdom_form = "log_cdom ~ ndwi + mndwi + B04_center_refl"
    cdom_sec_mixed, _ = evaluate_mixedlm_spatial_cv(cdom_sec, cdom_form, 'MeasurementValue', 'log_cdom', 'QSE')
    
    # 2. Chlorophyll-a Secondary
    chl_sec = df_sec[df_sec['target_type'] == 'chlorophyll']
    chl_feats = ['ndci', 'b5_b4_ratio', 'B05_center_refl', 'B07_center_refl', 'B8A_center_refl', 'B04_center_refl', 'B03_center_refl', 'B02_center_refl']
    res_chl_sec = audit_target_confounding(chl_sec, 'Chl_SecondaryHarmonized', chl_feats, 'log_chla')
    
    chl_form = "log_chla ~ ndci + B05_center_refl + B04_center_refl"
    chl_sec_mixed, _ = evaluate_mixedlm_spatial_cv(chl_sec, chl_form, 'MeasurementValue', 'log_chla', 'ug/L')
    
    # Load primary and raw for side-by-side 3-way table
    comp_cdom_prev = pd.read_csv('results/tables/cdom_harmonization_comparison.csv')
    comp_cdom_3way = pd.merge(
        comp_cdom_prev,
        res_cdom_sec[['feature', 'n_samples', 'raw_pearson_r', 'site_ctrl_partial_r', 'weighted_within_site_r', 'pct_collapse']].rename(
            columns={c: f"{c}_SecondaryCalibrated" for c in ['n_samples', 'raw_pearson_r', 'site_ctrl_partial_r', 'weighted_within_site_r', 'pct_collapse']}
        ),
        on='feature'
    )
    print("\nCDOM 3-Way Harmonization Comparison Table (Raw vs Primary vs Secondary):")
    print(comp_cdom_3way[['feature', 'raw_pearson_r_RawUnitMixed', 'site_ctrl_partial_r_RawUnitMixed',
                          'raw_pearson_r_HarmonizedNative', 'site_ctrl_partial_r_HarmonizedNative',
                          'raw_pearson_r_SecondaryCalibrated', 'site_ctrl_partial_r_SecondaryCalibrated']].to_string(index=False))
    comp_cdom_3way.to_csv('results/tables/cdom_3way_harmonization_comparison.csv', index=False)
    
    comp_chl_prev = pd.read_csv('results/tables/chlorophyll_harmonization_comparison.csv')
    comp_chl_3way = pd.merge(
        comp_chl_prev,
        res_chl_sec[['feature', 'n_samples', 'raw_pearson_r', 'site_ctrl_partial_r', 'weighted_within_site_r', 'pct_collapse']].rename(
            columns={c: f"{c}_SecondaryCalibrated" for c in ['n_samples', 'raw_pearson_r', 'site_ctrl_partial_r', 'weighted_within_site_r', 'pct_collapse']}
        ),
        on='feature'
    )
    print("\nChlorophyll-a 3-Way Harmonization Comparison Table (Raw vs Primary vs Secondary):")
    print(comp_chl_3way[['feature', 'raw_pearson_r_RawUnitMixed', 'site_ctrl_partial_r_RawUnitMixed',
                         'raw_pearson_r_HarmonizedNative', 'site_ctrl_partial_r_HarmonizedNative',
                         'raw_pearson_r_SecondaryCalibrated', 'site_ctrl_partial_r_SecondaryCalibrated']].to_string(index=False))
    comp_chl_3way.to_csv('results/tables/chlorophyll_3way_harmonization_comparison.csv', index=False)
    
    # 3-way MixedLM comparison
    mixed_prev = pd.read_csv('results/tables/mixed_effects_harmonization_comparison.csv')
    sec_mixed_rows = [
        {'target': 'CDOM', 'dataset': 'Secondary_CalibratedConversion', **cdom_sec_mixed},
        {'target': 'Chlorophyll-a', 'dataset': 'Secondary_CalibratedConversion', **chl_sec_mixed}
    ]
    mixed_3way = pd.concat([mixed_prev, pd.DataFrame(sec_mixed_rows)], ignore_index=True)
    print("\nMixed-Effects Model 3-Way Comparison Table (Raw vs Primary vs Secondary):")
    print(mixed_3way[['target', 'dataset', 'total_samples', 'n_sites', 'icc', 'overall_r2_raw', 'overall_rmse_raw']].to_string(index=False))
    mixed_3way.to_csv('results/tables/mixed_effects_3way_comparison.csv', index=False)
    return comp_cdom_3way, comp_chl_3way, mixed_3way

def main():
    print("=" * 85)
    print("STARTING COMPLETE PART A DIAGNOSTIC TAIL")
    print("=" * 85)
    part_a1_multicollinearity()
    part_a2_cdom_overlap_record()
    part_a3_turbidity_ablation_robustness()
    part_a4_illinois_turbidity_failure()
    part_a5_trinity_chla_investigation()
    part_a6_chlorophyll_harmonized_overlap()
    part_a7_secondary_calibrated_confounding()
    print("\n" + "=" * 85)
    print("PART A DIAGNOSTIC TAIL FULLY COMPLETE")
    print("=" * 85)

if __name__ == '__main__':
    main()
