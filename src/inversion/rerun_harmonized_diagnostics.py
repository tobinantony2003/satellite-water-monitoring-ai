"""
Script: rerun_harmonized_diagnostics.py
Executes Step 3: Reruns confounding audits and mixed-effects models
on both the previous Unit-Mixed (Raw) dataset and the new Harmonized Primary (Native-only) dataset.
Directly compares:
1. CDOM site-controlled partial correlations and within-site r (Raw vs Harmonized)
2. Chlorophyll-a site-controlled partial correlations and within-site r (Raw vs Harmonized)
3. CDOM MixedLM (ICC, Out-of-fold Spatial CV R^2, RMSE) (Raw vs Harmonized)
4. Chlorophyll-a MixedLM (ICC, Out-of-fold Spatial CV R^2, RMSE) (Raw vs Harmonized)
"""

import os
import shutil
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
import seaborn as sns

def compute_partial_corr(x, y, groups):
    """Frisch-Waugh-Lovell partial correlation controlling for categorical groups."""
    df = pd.DataFrame({'x': x, 'y': y, 'group': groups})
    x_mean = df.groupby('group')['x'].transform('mean')
    y_mean = df.groupby('group')['y'].transform('mean')
    x_res = df['x'] - x_mean
    y_res = df['y'] - y_mean
    
    n = len(df)
    n_groups = df['group'].nunique()
    df_resid = max(1, n - n_groups - 1)
    
    r_pearson, _ = stats.pearsonr(x_res, y_res)
    if abs(r_pearson) < 1.0:
        t_stat = r_pearson * np.sqrt(df_resid / (1.0 - r_pearson**2))
        p_pearson = 2 * (1 - stats.t.cdf(abs(t_stat), df=df_resid))
    else:
        p_pearson = 0.0
        
    r_spearman, p_spearman = stats.spearmanr(x_res, y_res)
    return {
        'pearson_r': r_pearson,
        'pearson_p': p_pearson,
        'spearman_r': r_spearman,
        'spearman_p': p_spearman
    }

def audit_target_confounding(df_target, target_name, features, log_col):
    """Computes raw and controlled correlations across specified features."""
    df_target = df_target.copy()
    df_target['site_no_str'] = df_target['site_no'].astype(str).str.zfill(8)
    
    # Calculate within-site correlations
    site_stats = []
    for site, s_df in df_target.groupby('site_no_str'):
        n_s = len(s_df)
        log_std = s_df[log_col].std()
        s_row = {'site_no': site, 'n_samples': n_s, 'log_std': log_std}
        for feat in features:
            f_std = s_df[feat].std()
            if n_s >= 5 and f_std > 1e-6 and log_std > 1e-6:
                r_w, _ = stats.pearsonr(s_df[feat], s_df[log_col])
                s_row[f'{feat}_r'] = r_w
            else:
                s_row[f'{feat}_r'] = np.nan
        site_stats.append(s_row)
    site_stats_df = pd.DataFrame(site_stats)
    
    results = []
    for feat in features:
        valid = df_target[[feat, log_col, 'site_no_str', 'basin']].dropna()
        x = valid[feat].values
        y = valid[log_col].values
        sites = valid['site_no_str'].values
        basins = valid['basin'].values
        
        raw_p_r, raw_p_p = stats.pearsonr(x, y)
        raw_s_r, raw_s_p = stats.spearmanr(x, y)
        site_res = compute_partial_corr(x, y, sites)
        basin_res = compute_partial_corr(x, y, basins)
        
        r_col = f'{feat}_r'
        v_sites = site_stats_df.dropna(subset=[r_col])
        if len(v_sites) > 0 and v_sites['n_samples'].sum() > 0:
            weighted_r = (v_sites[r_col] * v_sites['n_samples']).sum() / v_sites['n_samples'].sum()
            unweighted_r = v_sites[r_col].mean()
        else:
            weighted_r, unweighted_r = np.nan, np.nan
            
        pct_collapse = (
            round(100.0 * (abs(raw_p_r) - abs(site_res['pearson_r'])) / abs(raw_p_r), 1)
            if abs(raw_p_r) > 1e-5 else 0.0
        )
        
        results.append({
            'target': target_name,
            'feature': feat,
            'n_samples': len(valid),
            'n_sites': valid['site_no_str'].nunique(),
            'raw_pearson_r': round(raw_p_r, 4),
            'raw_spearman_r': round(raw_s_r, 4),
            'site_ctrl_partial_r': round(site_res['pearson_r'], 4),
            'site_ctrl_partial_pval': site_res['pearson_p'],
            'weighted_within_site_r': round(weighted_r, 4),
            'pct_collapse': pct_collapse
        })
    return pd.DataFrame(results)

def evaluate_mixedlm_spatial_cv(df_target, formula, target_col, log_col, target_unit_name, n_splits=5):
    """
    Fits MixedLM under Spatial Block GroupKFold CV.
    Returns:
      full_model_icc: Intraclass correlation coefficient
      overall_metrics: dict of out-of-fold metrics
      fold_metrics_df: per-fold performance
    """
    df = df_target.copy()
    df['site_no_str'] = df['site_no'].astype(str).str.zfill(8)
    
    # 1. Full-dataset fit to compute ICC
    full_model = smf.mixedlm(formula, df, groups=df["site_no_str"], re_formula="~1")
    full_fit = full_model.fit(reml=True, method=["lbfgs"])
    
    var_re = float(full_fit.cov_re.iloc[0, 0])
    var_resid = float(full_fit.scale)
    icc = var_re / (var_re + var_resid)
    
    # 2. 5-Fold Spatial CV
    gkf = GroupKFold(n_splits=min(n_splits, df['site_no_str'].nunique()))
    groups = df['site_no_str'].values
    
    fold_records = []
    all_y_true_raw = []
    all_y_pred_raw = []
    all_y_true_log = []
    all_y_pred_log = []
    
    fold_idx = 1
    for train_idx, test_idx in gkf.split(df, groups=groups):
        train_d = df.iloc[train_idx].copy()
        test_d = df.iloc[test_idx].copy()
        
        held_out_sites = test_d['site_no_str'].unique().tolist()
        
        try:
            m = smf.mixedlm(formula, train_d, groups=train_d["site_no_str"], re_formula="~1")
            f = m.fit(reml=True, method=["lbfgs"])
            fe = f.fe_params
        except Exception:
            m_ols = smf.ols(formula, train_d).fit()
            fe = m_ols.params
            
        # Predict using population-level fixed effects
        # Parse predictor names from formula: "y ~ x1 + x2 + ..."
        predictors = [p.strip() for p in formula.split('~')[1].split('+')]
        pred_log = np.full(len(test_d), fe['Intercept'])
        for p in predictors:
            pred_log += fe[p] * test_d[p].values
            
        pred_raw = 10.0 ** pred_log
        true_raw = test_d[target_col].values
        true_log = test_d[log_col].values
        
        # Fold metrics
        ss_res = np.sum((true_raw - pred_raw)**2)
        ss_tot = np.sum((true_raw - np.mean(true_raw))**2)
        fold_r2_raw = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        fold_rmse_raw = np.sqrt(np.mean((true_raw - pred_raw)**2))
        fold_mae_raw = np.mean(np.abs(true_raw - pred_raw))
        
        ss_res_log = np.sum((true_log - pred_log)**2)
        ss_tot_log = np.sum((true_log - np.mean(true_log))**2)
        fold_r2_log = 1.0 - (ss_res_log / ss_tot_log) if ss_tot_log > 0 else 0.0
        
        fold_records.append({
            'fold': fold_idx,
            'held_out_sites': ', '.join(held_out_sites),
            'n_test_samples': len(test_d),
            'r2_raw': round(fold_r2_raw, 4),
            'rmse_raw': round(fold_rmse_raw, 2),
            'mae_raw': round(fold_mae_raw, 2),
            'r2_log': round(fold_r2_log, 4)
        })
        
        all_y_true_raw.extend(true_raw)
        all_y_pred_raw.extend(pred_raw)
        all_y_true_log.extend(true_log)
        all_y_pred_log.extend(pred_log)
        fold_idx += 1
        
    y_true_r = np.array(all_y_true_raw)
    y_pred_r = np.array(all_y_pred_raw)
    y_true_l = np.array(all_y_true_log)
    y_pred_l = np.array(all_y_pred_log)
    
    tot_ss_res = np.sum((y_true_r - y_pred_r)**2)
    tot_ss_tot = np.sum((y_true_r - np.mean(y_true_r))**2)
    overall_r2_raw = 1.0 - (tot_ss_res / tot_ss_tot)
    overall_rmse_raw = np.sqrt(np.mean((y_true_r - y_pred_r)**2))
    overall_mae_raw = np.mean(np.abs(y_true_r - y_pred_r))
    overall_bias_raw = np.mean(y_pred_r - y_true_r)
    
    tot_ss_res_l = np.sum((y_true_l - y_pred_l)**2)
    tot_ss_tot_l = np.sum((y_true_l - np.mean(y_true_l))**2)
    overall_r2_log = 1.0 - (tot_ss_res_l / tot_ss_tot_l)
    
    summary = {
        'icc': round(icc, 4),
        'var_re_site': round(var_re, 4),
        'var_resid': round(var_resid, 4),
        'overall_r2_raw': round(overall_r2_raw, 4),
        'overall_rmse_raw': round(overall_rmse_raw, 2),
        'overall_mae_raw': round(overall_mae_raw, 2),
        'overall_bias_raw': round(overall_bias_raw, 2),
        'overall_r2_log': round(overall_r2_log, 4),
        'total_samples': len(df),
        'n_sites': df['site_no_str'].nunique()
    }
    return summary, pd.DataFrame(fold_records)

def main():
    print("=" * 85)
    print("STEP 3: Comparative Diagnostics: Unit-Mixed (Raw) vs Harmonized Primary (Native Only)")
    print("=" * 85)
    
    raw_path = "data/processed/matched_wq_cleaned/matched_set_b_dl_masked_unitmixed_raw.csv"
    prim_path = "data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv"
    
    df_raw = pd.read_csv(raw_path, low_memory=False)
    df_prim = pd.read_csv(prim_path, low_memory=False)
    
    for d in [df_raw, df_prim]:
        d['site_no_str'] = d['site_no'].astype(str).str.zfill(8)
        d['log_cdom'] = np.log10(np.clip(d['MeasurementValue'], 1e-4, None))
        d['log_chla'] = np.log10(np.clip(d['MeasurementValue'], 1e-4, None))
        d['b5_b4_ratio'] = d['B05_center_refl'] / np.maximum(d['B04_center_refl'], 1e-4)
        
    os.makedirs('results/tables', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 1. CDOM Confounding Audit Comparison
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("1. CDOM CONFOUNDING COMPARISON (Raw vs Harmonized Primary)")
    print("=" * 50)
    cdom_feats = ['ndwi', 'mndwi', 'B04_center_refl', 'cdom_ratio', 'B8A_center_refl', 'B03_center_refl', 'B02_center_refl']
    
    cdom_raw_sub = df_raw[df_raw['target_type'] == 'cdom']
    cdom_prim_sub = df_prim[df_prim['target_type'] == 'cdom']
    
    res_cdom_raw = audit_target_confounding(cdom_raw_sub, 'CDOM_UnitMixed', cdom_feats, 'log_cdom')
    res_cdom_prim = audit_target_confounding(cdom_prim_sub, 'CDOM_HarmonizedPrimary', cdom_feats, 'log_cdom')
    
    # Merge for side-by-side comparison
    comp_cdom = pd.merge(
        res_cdom_raw[['feature', 'n_samples', 'raw_pearson_r', 'site_ctrl_partial_r', 'weighted_within_site_r', 'pct_collapse']],
        res_cdom_prim[['feature', 'n_samples', 'raw_pearson_r', 'site_ctrl_partial_r', 'weighted_within_site_r', 'pct_collapse']],
        on='feature',
        suffixes=('_RawUnitMixed', '_HarmonizedNative')
    )
    print("\nCDOM Side-by-Side Confounding Metrics:")
    print(comp_cdom.to_string(index=False))
    comp_cdom.to_csv("results/tables/cdom_harmonization_comparison.csv", index=False)
    
    # -------------------------------------------------------------------------
    # 2. Chlorophyll-a Confounding Audit Comparison
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("2. CHLOROPHYLL-A CONFOUNDING COMPARISON (Raw vs Harmonized Primary)")
    print("=" * 50)
    chl_feats = [
        'ndci', 'b5_b4_ratio', 'B05_center_refl', 'B07_center_refl', 'B8A_center_refl',
        'B04_center_refl', 'B03_center_refl', 'B02_center_refl'
    ]
    
    chl_raw_sub = df_raw[df_raw['target_type'] == 'chlorophyll']
    chl_prim_sub = df_prim[df_prim['target_type'] == 'chlorophyll']
    
    res_chl_raw = audit_target_confounding(chl_raw_sub, 'Chl_UnitMixed', chl_feats, 'log_chla')
    res_chl_prim = audit_target_confounding(chl_prim_sub, 'Chl_HarmonizedPrimary', chl_feats, 'log_chla')
    
    comp_chl = pd.merge(
        res_chl_raw[['feature', 'n_samples', 'raw_pearson_r', 'site_ctrl_partial_r', 'weighted_within_site_r', 'pct_collapse']],
        res_chl_prim[['feature', 'n_samples', 'raw_pearson_r', 'site_ctrl_partial_r', 'weighted_within_site_r', 'pct_collapse']],
        on='feature',
        suffixes=('_RawUnitMixed', '_HarmonizedNative')
    )
    print("\nChlorophyll-a Side-by-Side Confounding Metrics:")
    print(comp_chl.to_string(index=False))
    comp_chl.to_csv("results/tables/chlorophyll_harmonization_comparison.csv", index=False)
    
    # -------------------------------------------------------------------------
    # 3. CDOM Mixed-Effects Model Comparison
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("3. CDOM Mixed-Effects Model (Spatial CV)")
    print("=" * 50)
    cdom_form = "log_cdom ~ ndwi + mndwi + B04_center_refl"
    
    cdom_raw_mixed, cdom_raw_folds = evaluate_mixedlm_spatial_cv(cdom_raw_sub, cdom_form, 'MeasurementValue', 'log_cdom', 'QSE')
    cdom_prim_mixed, cdom_prim_folds = evaluate_mixedlm_spatial_cv(cdom_prim_sub, cdom_form, 'MeasurementValue', 'log_cdom', 'QSE')
    
    print("\nCDOM MixedLM Summary:")
    print(f"  Raw Unit-Mixed:        N={cdom_raw_mixed['total_samples']:,}, ICC={cdom_raw_mixed['icc']:.4f}, Out-of-fold R^2={cdom_raw_mixed['overall_r2_raw']:.4f}, RMSE={cdom_raw_mixed['overall_rmse_raw']:.2f}")
    print(f"  Harmonized Native-Only: N={cdom_prim_mixed['total_samples']:,}, ICC={cdom_prim_mixed['icc']:.4f}, Out-of-fold R^2={cdom_prim_mixed['overall_r2_raw']:.4f}, RMSE={cdom_prim_mixed['overall_rmse_raw']:.2f}")
    
    # -------------------------------------------------------------------------
    # 4. Chlorophyll-a Mixed-Effects Model Comparison
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("4. Chlorophyll-a Mixed-Effects Model (Spatial CV)")
    print("=" * 50)
    chl_form = "log_chla ~ ndci + B05_center_refl + B04_center_refl"
    
    chl_raw_mixed, chl_raw_folds = evaluate_mixedlm_spatial_cv(chl_raw_sub, chl_form, 'MeasurementValue', 'log_chla', 'ug/L')
    chl_prim_mixed, chl_prim_folds = evaluate_mixedlm_spatial_cv(chl_prim_sub, chl_form, 'MeasurementValue', 'log_chla', 'ug/L')
    
    print("\nChlorophyll-a MixedLM Summary:")
    print(f"  Raw Unit-Mixed:        N={chl_raw_mixed['total_samples']:,} ({chl_raw_mixed['n_sites']} sites), ICC={chl_raw_mixed['icc']:.4f}, Out-of-fold R^2={chl_raw_mixed['overall_r2_raw']:.4f}, RMSE={chl_raw_mixed['overall_rmse_raw']:.2f}")
    print(f"  Harmonized Native-Only: N={chl_prim_mixed['total_samples']:,} ({chl_prim_mixed['n_sites']} sites), ICC={chl_prim_mixed['icc']:.4f}, Out-of-fold R^2={chl_prim_mixed['overall_r2_raw']:.4f}, RMSE={chl_prim_mixed['overall_rmse_raw']:.2f}")
    
    # Save mixed effects comparison table
    mixed_comp = [
        {'target': 'CDOM', 'dataset': 'Raw_UnitMixed', **cdom_raw_mixed},
        {'target': 'CDOM', 'dataset': 'Harmonized_NativeOnly', **cdom_prim_mixed},
        {'target': 'Chlorophyll-a', 'dataset': 'Raw_UnitMixed', **chl_raw_mixed},
        {'target': 'Chlorophyll-a', 'dataset': 'Harmonized_NativeOnly', **chl_prim_mixed}
    ]
    pd.DataFrame(mixed_comp).to_csv("results/tables/mixed_effects_harmonization_comparison.csv", index=False)
    chl_prim_folds.to_csv("results/tables/chlorophyll_mixed_effects_per_fold.csv", index=False)
    
    print("\nChlorophyll-a Harmonized Native-Only Per-Fold Performance:")
    print(chl_prim_folds.to_string(index=False))

if __name__ == '__main__':
    main()
