"""
Phase 2 Diagnostics & Mixed-Effects Evaluation:
Step A: CDOM Sanity-Check (Within-site variances, unweighted vs sample-size-weighted r)
Step B: Chlorophyll-a Confounding Check (Controlling for site & basin fixed effects)
Step C: CDOM Linear Mixed-Effects Model under 5-Fold Spatial Block CV
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
    """
    Frisch-Waugh-Lovell partial correlation controlling for categorical groups.
    """
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
        'spearman_p': p_spearman,
        'x_res': x_res,
        'y_res': y_res
    }

# ==============================================================================
# STEP A: CDOM SANITY CHECK & WEIGHTED WITHIN-SITE CORRELATIONS
# ==============================================================================
def run_step_a_cdom_sanity(df):
    print("=" * 80)
    print("STEP A: CDOM SANITY CHECK (WITHIN-SITE VARIANCES & WEIGHTED CORRELATIONS)")
    print("=" * 80)
    
    cdom_df = df[df['target_type'] == 'cdom'].copy()
    cdom_df = cdom_df[cdom_df['MeasurementValue'] > 0].copy()
    cdom_df['log_cdom'] = np.log10(cdom_df['MeasurementValue'])
    cdom_df['site_no_str'] = cdom_df['site_no'].astype(str)
    
    features = ['ndwi', 'mndwi', 'B04_center_refl', 'cdom_ratio', 'B8A_center_refl', 'B03_center_refl', 'B02_center_refl']
    
    # Check site-by-site variances and within-site r
    site_records = []
    for site, s_df in cdom_df.groupby('site_no_str'):
        n_site = len(s_df)
        log_var = s_df['log_cdom'].var()
        log_std = s_df['log_cdom'].std()
        
        row = {
            'site_no': site,
            'basin': s_df['basin'].iloc[0],
            'n_samples': n_site,
            'log_cdom_std': log_std
        }
        for feat in features:
            f_std = s_df[feat].std()
            row[f'{feat}_std'] = f_std
            if f_std > 1e-6 and log_std > 1e-6 and n_site >= 5:
                r_val, p_val = stats.pearsonr(s_df[feat], s_df['log_cdom'])
                row[f'{feat}_r'] = r_val
            else:
                row[f'{feat}_r'] = np.nan
        site_records.append(row)
        
    site_var_df = pd.DataFrame(site_records).sort_values(by='n_samples', ascending=False)
    site_var_df.to_csv("results/tables/cdom_within_site_breakdown.csv", index=False)
    
    print("\nSite-by-site sample counts, target std, and within-site Pearson r:")
    cols_to_print = ['site_no', 'basin', 'n_samples', 'log_cdom_std', 'ndwi_r', 'mndwi_r', 'B04_center_refl_r']
    print(site_var_df[cols_to_print].to_string(index=False))
    
    # Calculate unweighted vs sample-size weighted within-site r
    summary_list = []
    for feat in features:
        r_col = f'{feat}_r'
        valid_sites = site_var_df.dropna(subset=[r_col])
        n_valid_sites = len(valid_sites)
        
        unweighted_mean = valid_sites[r_col].mean()
        
        # Weighted by n_samples
        total_n = valid_sites['n_samples'].sum()
        weighted_mean = (valid_sites[r_col] * valid_sites['n_samples']).sum() / total_n
        
        # Also raw correlation for comparison
        raw_r, _ = stats.pearsonr(cdom_df[feat], cdom_df['log_cdom'])
        site_ctrl = compute_partial_corr(cdom_df[feat].values, cdom_df['log_cdom'].values, cdom_df['site_no_str'].values)
        
        summary_list.append({
            'feature': feat,
            'raw_pearson_r': round(raw_r, 4),
            'site_ctrl_partial_r': round(site_ctrl['pearson_r'], 4),
            'unweighted_within_site_r': round(unweighted_mean, 4),
            'weighted_within_site_r': round(weighted_mean, 4),
            'n_valid_sites': n_valid_sites,
            'total_samples_represented': total_n
        })
        
    step_a_df = pd.DataFrame(summary_list)
    step_a_df.to_csv("results/tables/cdom_step_a_sanity_check.csv", index=False)
    print("\n--- CDOM Step A: Unweighted vs Sample-Size Weighted Within-Site Pearson r ---")
    print(step_a_df.to_string(index=False))
    return step_a_df, cdom_df

# ==============================================================================
# STEP B: CHLOROPHYLL-A CONFOUNDING CHECK
# ==============================================================================
def run_step_b_chlorophyll_confounding(df):
    print("\n" + "=" * 80)
    print("STEP B: CHLOROPHYLL-A CONFOUNDING CHECK")
    print("=" * 80)
    
    chla_df = df[df['target_type'] == 'chlorophyll'].copy()
    chla_df = chla_df[chla_df['MeasurementValue'] > 0].copy()
    chla_df['log_chla'] = np.log10(chla_df['MeasurementValue'])
    chla_df['site_no_str'] = chla_df['site_no'].astype(str)
    chla_df['b5_b4_ratio'] = chla_df['B05_center_refl'] / np.maximum(chla_df['B04_center_refl'], 1e-4)
    
    n_total = len(chla_df)
    n_sites = chla_df['site_no_str'].nunique()
    n_basins = chla_df['basin'].nunique()
    print(f"Loaded {n_total} Chlorophyll-a records across {n_sites} sites and {n_basins} basins.")
    
    # Feature list specified by user: B05, B07, B8A, B06, NDCI, B5/B4 ratio, plus additional bands
    features = [
        'B05_center_refl', 'B07_center_refl', 'B8A_center_refl', 'B06_center_refl',
        'ndci', 'b5_b4_ratio', 'B04_center_refl', 'B03_center_refl', 'B02_center_refl',
        'B01_center_refl', 'B08_center_refl', 'B11_center_refl', 'B12_center_refl',
        'turbidity_ratio'
    ]
    
    # Site variance and within-site correlations
    site_stats = []
    for site, s_df in chla_df.groupby('site_no_str'):
        n_s = len(s_df)
        log_std = s_df['log_chla'].std()
        s_row = {
            'site_no': site,
            'basin': s_df['basin'].iloc[0],
            'n_samples': n_s,
            'log_chla_std': log_std
        }
        for feat in features:
            f_std = s_df[feat].std()
            s_row[f'{feat}_std'] = f_std
            if n_s >= 10 and f_std > 1e-6 and log_std > 1e-6:
                r_w, _ = stats.pearsonr(s_df[feat], s_df['log_chla'])
                s_row[f'{feat}_r'] = r_w
            else:
                s_row[f'{feat}_r'] = np.nan
        site_stats.append(s_row)
        
    site_stats_df = pd.DataFrame(site_stats).sort_values(by='n_samples', ascending=False)
    site_stats_df.to_csv("results/tables/chlorophyll_site_breakdown.csv", index=False)
    
    results = []
    residuals_store = {}
    
    for feat in features:
        valid_sub = chla_df[[feat, 'log_chla', 'site_no_str', 'basin']].dropna()
        x = valid_sub[feat].values
        y = valid_sub['log_chla'].values
        sites = valid_sub['site_no_str'].values
        basins = valid_sub['basin'].values
        
        # Raw correlations
        raw_p_r, raw_p_p = stats.pearsonr(x, y)
        raw_s_r, raw_s_p = stats.spearmanr(x, y)
        
        # Site-controlled partial correlation
        site_res = compute_partial_corr(x, y, sites)
        residuals_store[feat] = site_res
        
        # Basin-controlled partial correlation
        basin_res = compute_partial_corr(x, y, basins)
        
        # Within-site correlations: unweighted and sample-size weighted
        r_col = f'{feat}_r'
        valid_sites_f = site_stats_df.dropna(subset=[r_col])
        unweighted_mean_r = valid_sites_f[r_col].mean()
        weighted_mean_r = (valid_sites_f[r_col] * valid_sites_f['n_samples']).sum() / valid_sites_f['n_samples'].sum()
        
        # Percent collapse or change
        pct_collapse = (
            round(100.0 * (abs(raw_p_r) - abs(site_res['pearson_r'])) / abs(raw_p_r), 1)
            if abs(raw_p_r) > 1e-5 else 0.0
        )
        
        results.append({
            'feature': feat,
            'raw_pearson_r': round(raw_p_r, 4),
            'raw_spearman_r': round(raw_s_r, 4),
            'basin_ctrl_partial_r': round(basin_res['pearson_r'], 4),
            'site_ctrl_partial_r': round(site_res['pearson_r'], 4),
            'site_ctrl_partial_pval': site_res['pearson_p'],
            'site_ctrl_spearman_r': round(site_res['spearman_r'], 4),
            'unweighted_within_site_r': round(unweighted_mean_r, 4),
            'weighted_within_site_r': round(weighted_mean_r, 4),
            'pct_collapse': pct_collapse,
            'survives_site_control': abs(site_res['pearson_r']) >= 0.15
        })
        
    chla_confounding_df = pd.DataFrame(results).sort_values(by='raw_spearman_r', ascending=False, key=abs)
    chla_confounding_df.to_csv("results/tables/chlorophyll_confounding_check.csv", index=False)
    
    print("\n--- Chlorophyll-a Confounding Check Results (Raw vs Controlled) ---")
    display_cols = [
        'feature', 'raw_pearson_r', 'raw_spearman_r', 'basin_ctrl_partial_r',
        'site_ctrl_partial_r', 'weighted_within_site_r', 'pct_collapse', 'survives_site_control'
    ]
    print(chla_confounding_df[display_cols].to_string(index=False))
    
    # Generate visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel A: Raw vs Site-Controlled vs Basin-Controlled Pearson r
    ax = axes[0, 0]
    bar_width = 0.28
    indices = np.arange(len(chla_confounding_df))
    ax.bar(indices - bar_width, chla_confounding_df['raw_pearson_r'], width=bar_width, label='Raw Pearson r', color='#2b5c8f', alpha=0.85)
    ax.bar(indices, chla_confounding_df['basin_ctrl_partial_r'], width=bar_width, label='Basin-Controlled Partial r', color='#d97706', alpha=0.85)
    ax.bar(indices + bar_width, chla_confounding_df['site_ctrl_partial_r'], width=bar_width, label='Site-Controlled Partial r', color='#10b981', alpha=0.85)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.axhline(0.20, color='red', linestyle=':', linewidth=1.0, label='|r| = 0.20 Threshold')
    ax.set_xticks(indices)
    ax.set_xticklabels(chla_confounding_df['feature'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Correlation with log10(Chlorophyll-a)', fontsize=11)
    ax.set_title('(A) Chlorophyll-a: Raw vs Controlled Partial Correlations\n(Red-Edge Bands & NDCI Robustness)', fontsize=12, fontweight='bold')
    ax.legend(frameon=True)
    ax.grid(True, linestyle=':', alpha=0.5, axis='y')
    
    # Panel B: Raw NDCI vs log10(Chl-a) colored by basin
    ax = axes[0, 1]
    palette = sns.color_palette("Set2", n_colors=chla_df['basin'].nunique())
    sns.scatterplot(
        data=chla_df,
        x='ndci',
        y='log_chla',
        hue='basin',
        alpha=0.35,
        s=15,
        palette=palette,
        ax=ax
    )
    sns.regplot(data=chla_df, x='ndci', y='log_chla', scatter=False, ax=ax, color='black', line_kws={'linestyle': '--', 'linewidth': 1.5, 'label': f"Overall fit (r={chla_confounding_df.loc[chla_confounding_df['feature']=='ndci', 'raw_pearson_r'].values[0]:.2f})"})
    ax.set_title('(B) Raw NDCI vs log10(Chlorophyll-a) by Basin\n(Consistent Positive Trend across Basins)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Sentinel-2 NDCI', fontsize=11)
    ax.set_ylabel('log10(Chlorophyll-a [ug/L])', fontsize=11)
    ax.legend(frameon=True, fontsize=8, title="Basin")
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # Panel C: Site-Demeaned Residual NDCI vs Residual log10(Chl-a)
    ax = axes[1, 0]
    ndci_res = residuals_store['ndci']
    ax.scatter(ndci_res['x_res'], ndci_res['y_res'], alpha=0.25, color='#4f46e5', s=10)
    sns.regplot(x=ndci_res['x_res'], y=ndci_res['y_res'], scatter=False, ax=ax, color='#10b981', line_kws={'linewidth': 2, 'label': f"Site-Controlled Partial fit (r={ndci_res['pearson_r']:.3f}, p={ndci_res['pearson_p']:.1e})"})
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_title('(C) NDCI Residuals vs Chl-a Residuals (Controlling for Site Fixed Effects)\n(Correlation Survives: r=0.203 -> r=0.219)', fontsize=12, fontweight='bold')
    ax.set_xlabel('NDCI Residuals (Site-Demeaned)', fontsize=11)
    ax.set_ylabel('log10(Chl-a) Residuals (Site-Demeaned)', fontsize=11)
    ax.legend(frameon=True)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # Panel D: Site-Demeaned Residual B05 (Red Edge 1) vs Residual log10(Chl-a)
    ax = axes[1, 1]
    b05_res = residuals_store['B05_center_refl']
    ax.scatter(b05_res['x_res'], b05_res['y_res'], alpha=0.25, color='#d97706', s=10)
    sns.regplot(x=b05_res['x_res'], y=b05_res['y_res'], scatter=False, ax=ax, color='#10b981', line_kws={'linewidth': 2, 'label': f"Site-Controlled Partial fit (r={b05_res['pearson_r']:.3f}, p={b05_res['pearson_p']:.1e})"})
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_title('(D) B05 Red Edge Residuals vs Chl-a Residuals (Controlling for Site Fixed Effects)\n(Correlation Survives: r=0.158 -> r=0.181)', fontsize=12, fontweight='bold')
    ax.set_xlabel('B05 (705nm) Residuals (Site-Demeaned)', fontsize=11)
    ax.set_ylabel('log10(Chl-a) Residuals (Site-Demeaned)', fontsize=11)
    ax.legend(frameon=True)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plot_path = "results/figures/diagnostics/chlorophyll_confounding_check.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    brain_dir = "C:/Users/Tobin Antony/.gemini/antigravity-ide/brain/730512ff-544c-4445-ac17-b37e3b98b9be"
    if os.path.exists(brain_dir):
        shutil.copy(plot_path, os.path.join(brain_dir, "chlorophyll_confounding_check.png"))
        print(f"Saved and copied chlorophyll confounding plot to {brain_dir}/chlorophyll_confounding_check.png")
        
    return chla_confounding_df, chla_df

# ==============================================================================
# STEP C: CDOM LINEAR MIXED-EFFECTS MODEL UNDER 5-FOLD SPATIAL BLOCK CV
# ==============================================================================
def run_step_c_cdom_mixed_effects(cdom_df):
    print("\n" + "=" * 80)
    print("STEP C: CDOM LINEAR MIXED-EFFECTS MODEL UNDER 5-FOLD SPATIAL BLOCK CV")
    print("=" * 80)
    
    # Surviving features identified in Step A: ndwi, mndwi, B04_center_refl
    # Formula: log_cdom ~ ndwi + mndwi + B04_center_refl
    # Group: site_no_str
    
    gkf = GroupKFold(n_splits=5)
    groups = cdom_df['site_no_str'].values
    
    fold_metrics = []
    all_y_true_log = []
    all_y_pred_log = []
    all_y_true_raw = []
    all_y_pred_raw = []
    all_sites_held_out = []
    
    print("\nEvaluating statsmodels MixedLM on 5 spatial folds (grouped by site_no)...")
    fold_idx = 1
    
    for train_idx, test_idx in gkf.split(cdom_df, groups=groups):
        train_data = cdom_df.iloc[train_idx].copy()
        test_data = cdom_df.iloc[test_idx].copy()
        
        held_out_sites = test_data['site_no_str'].unique().tolist()
        n_test = len(test_data)
        
        # Fit MixedLM on training folds
        # Fixed effects: Intercept + ndwi + mndwi + B04_center_refl
        # Random intercept: (1 | site_no_str)
        try:
            model = smf.mixedlm(
                "log_cdom ~ ndwi + mndwi + B04_center_refl",
                train_data,
                groups=train_data["site_no_str"],
                re_formula="~1"
            )
            fit_res = model.fit(reml=True, method=["lbfgs"])
            
            fixed_params = fit_res.fe_params
            # Predict log10(cdom) on held-out sites: strictly fixed effects (population level)
            y_pred_log = (
                fixed_params['Intercept'] +
                fixed_params['ndwi'] * test_data['ndwi'].values +
                fixed_params['mndwi'] * test_data['mndwi'].values +
                fixed_params['B04_center_refl'] * test_data['B04_center_refl'].values
            )
            
        except Exception as e:
            print(f"Fold {fold_idx} MixedLM fit warning ({e}), falling back to OLS on training set fixed formula:")
            ols_model = smf.ols("log_cdom ~ ndwi + mndwi + B04_center_refl", train_data).fit()
            y_pred_log = ols_model.predict(test_data[['ndwi', 'mndwi', 'B04_center_refl']]).values
            
        y_true_log = test_data['log_cdom'].values
        
        # Raw scale predictions
        y_true_raw = test_data['MeasurementValue'].values
        y_pred_raw = 10.0 ** y_pred_log
        
        # Compute fold metrics in raw space and log space
        # Raw space R^2
        ss_tot = np.sum((y_true_raw - np.mean(y_true_raw))**2)
        ss_res = np.sum((y_true_raw - y_pred_raw)**2)
        fold_r2_raw = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        fold_rmse_raw = np.sqrt(np.mean((y_true_raw - y_pred_raw)**2))
        fold_mae_raw = np.mean(np.abs(y_true_raw - y_pred_raw))
        fold_bias_raw = np.mean(y_pred_raw - y_true_raw)
        
        # Log space R^2
        ss_tot_log = np.sum((y_true_log - np.mean(y_true_log))**2)
        ss_res_log = np.sum((y_true_log - y_pred_log)**2)
        fold_r2_log = 1.0 - (ss_res_log / ss_tot_log) if ss_tot_log > 0 else np.nan
        fold_rmse_log = np.sqrt(np.mean((y_true_log - y_pred_log)**2))
        
        fold_metrics.append({
            'fold': fold_idx,
            'held_out_sites': ", ".join(held_out_sites),
            'n_test_samples': n_test,
            'pct_test_samples': round(100.0 * n_test / len(cdom_df), 1),
            'test_mean_raw': round(np.mean(y_true_raw), 2),
            'pred_mean_raw': round(np.mean(y_pred_raw), 2),
            'r2_raw': round(fold_r2_raw, 4),
            'rmse_raw': round(fold_rmse_raw, 2),
            'mae_raw': round(fold_mae_raw, 2),
            'bias_raw': round(fold_bias_raw, 2),
            'r2_log': round(fold_r2_log, 4),
            'rmse_log': round(fold_rmse_log, 4)
        })
        
        all_y_true_log.extend(y_true_log)
        all_y_pred_log.extend(y_pred_log)
        all_y_true_raw.extend(y_true_raw)
        all_y_pred_raw.extend(y_pred_raw)
        all_sites_held_out.extend(test_data['site_no_str'].values)
        
        fold_idx += 1
        
    fold_df = pd.DataFrame(fold_metrics)
    
    # Overall pooled metrics
    all_y_true_raw = np.array(all_y_true_raw)
    all_y_pred_raw = np.array(all_y_pred_raw)
    all_y_true_log = np.array(all_y_true_log)
    all_y_pred_log = np.array(all_y_pred_log)
    
    overall_ss_tot = np.sum((all_y_true_raw - np.mean(all_y_true_raw))**2)
    overall_ss_res = np.sum((all_y_true_raw - all_y_pred_raw)**2)
    overall_r2_raw = 1.0 - (overall_ss_res / overall_ss_tot)
    overall_rmse_raw = np.sqrt(np.mean((all_y_true_raw - all_y_pred_raw)**2))
    overall_mae_raw = np.mean(np.abs(all_y_true_raw - all_y_pred_raw))
    overall_bias_raw = np.mean(all_y_pred_raw - all_y_true_raw)
    
    overall_r2_log = 1.0 - (np.sum((all_y_true_log - all_y_pred_log)**2) / np.sum((all_y_true_log - np.mean(all_y_true_log))**2))
    overall_rmse_log = np.sqrt(np.mean((all_y_true_log - all_y_pred_log)**2))
    
    print("\n--- Per-Fold Performance for CDOM Linear Mixed-Effects Model ---")
    print(fold_df[['fold', 'held_out_sites', 'n_test_samples', 'pct_test_samples', 'test_mean_raw', 'pred_mean_raw', 'r2_raw', 'rmse_raw', 'bias_raw']].to_string(index=False))
    
    print("\n--- Overall Pooled Spatial Block CV Performance for CDOM MixedLM ---")
    print(f"Overall Raw R^2:  {overall_r2_raw:.4f}")
    print(f"Overall Raw RMSE: {overall_rmse_raw:.2f} QSE")
    print(f"Overall Raw MAE:  {overall_mae_raw:.2f} QSE")
    print(f"Overall Raw Bias: {overall_bias_raw:.2f} QSE")
    print(f"Overall Log R^2:  {overall_r2_log:.4f}")
    print(f"Overall Log RMSE: {overall_rmse_log:.4f}")
    
    # Save table
    fold_df.to_csv("results/tables/cdom_mixed_effects_per_fold.csv", index=False)
    
    # Also save comparison table: Original Stage 3 (RF/MLP) vs MixedLM
    comparison_summary = [
        {'model': 'Stage 3 Random Forest', 'r2_raw': -0.1982, 'rmse_raw': 18.11, 'bias_raw': -4.82, 'evaluation': '5-Fold Spatial CV'},
        {'model': 'Stage 3 Deep Spectral MLP', 'r2_raw': -0.1518, 'rmse_raw': 17.75, 'bias_raw': -6.45, 'evaluation': '5-Fold Spatial CV'},
        {'model': 'Linear Mixed-Effects (MixedLM Fixed-Only)', 'r2_raw': round(overall_r2_raw, 4), 'rmse_raw': round(overall_rmse_raw, 2), 'bias_raw': round(overall_bias_raw, 2), 'evaluation': '5-Fold Spatial CV'}
    ]
    pd.DataFrame(comparison_summary).to_csv("results/tables/cdom_model_comparison_spatial_cv.csv", index=False)
    
    # Plot predicted vs actual for MixedLM
    plt.figure(figsize=(8, 6))
    plt.scatter(all_y_true_raw, all_y_pred_raw, alpha=0.35, color='#4f46e5', s=20)
    plt.plot([0, 80], [0, 80], 'r--', linewidth=1.5, label='1:1 Ideal Line')
    plt.axhline(np.mean(all_y_pred_raw), color='green', linestyle=':', linewidth=1.5, label=f'Mean Prediction ({np.mean(all_y_pred_raw):.1f} QSE)')
    plt.title(f'CDOM Linear Mixed-Effects Model (Fixed-Effects Out-of-Fold)\n5-Fold Spatial Block CV (R²={overall_r2_raw:.4f}, RMSE={overall_rmse_raw:.2f})', fontsize=11, fontweight='bold')
    plt.xlabel('Observed CDOM (QSE)', fontsize=10)
    plt.ylabel('Predicted CDOM (QSE)', fontsize=10)
    plt.legend(frameon=True)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plot_path = "results/figures/diagnostics/cdom_mixed_effects_spatial_pred.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    brain_dir = "C:/Users/Tobin Antony/.gemini/antigravity-ide/brain/730512ff-544c-4445-ac17-b37e3b98b9be"
    if os.path.exists(brain_dir):
        shutil.copy(plot_path, os.path.join(brain_dir, "cdom_mixed_effects_spatial_pred.png"))
        
    return fold_df, overall_r2_raw, overall_rmse_raw

if __name__ == "__main__":
    data_path = "data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv"
    df = pd.read_csv(data_path)
    
    step_a_df, cdom_df = run_step_a_cdom_sanity(df)
    chla_confounding_df, chla_df = run_step_b_chlorophyll_confounding(df)
    fold_df, overall_r2, overall_rmse = run_step_c_cdom_mixed_effects(cdom_df)
    print("\nALL STEPS A, B, AND C COMPLETE.")
