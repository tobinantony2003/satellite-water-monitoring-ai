"""
Check CDOM Confounding: Partial Correlations Controlling for Site and Basin
Target: Remote Sensing of Environment / Int. J. Remote Sensing
Investigates whether strong raw correlations (NDWI r=0.60, MNDWI r=0.43) reflect
genuine bio-optical signal or are confounded by site/basin identity.
"""

import os
import shutil
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def compute_partial_corr(x, y, groups):
    """
    Computes partial correlation between x and y controlling for categorical groups (fixed effects).
    By Frisch-Waugh-Lovell theorem, regressing x and y on group dummy variables
    is equivalent to subtracting the group-specific mean:
    x_res = x - E[x|group]
    y_res = y - E[y|group]
    r_partial = Pearson correlation between x_res and y_res.
    """
    # Group de-meaning
    df = pd.DataFrame({'x': x, 'y': y, 'group': groups})
    x_mean = df.groupby('group')['x'].transform('mean')
    y_mean = df.groupby('group')['y'].transform('mean')
    x_res = df['x'] - x_mean
    y_res = df['y'] - y_mean
    
    # Degrees of freedom: N - n_groups - 1
    n = len(df)
    n_groups = df['group'].nunique()
    df_resid = n - n_groups - 1
    
    r_pearson, p_pearson = stats.pearsonr(x_res, y_res)
    # T-test for partial correlation: t = r * sqrt(df / (1 - r^2))
    if abs(r_pearson) < 1.0:
        t_stat = r_pearson * np.sqrt(df_resid / (1.0 - r_pearson**2))
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=df_resid))
    else:
        p_val = 0.0
        
    r_spearman, p_spearman = stats.spearmanr(x_res, y_res)
    
    return {
        'pearson_r': r_pearson,
        'pearson_p': p_val,
        'spearman_r': r_spearman,
        'spearman_p': p_spearman,
        'x_res': x_res,
        'y_res': y_res
    }

def run_cdom_confounding_check():
    data_path = "data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {data_path}")
        
    df = pd.read_csv(data_path)
    cdom_df = df[df['target_type'] == 'cdom'].copy()
    cdom_df = cdom_df[cdom_df['MeasurementValue'] > 0].copy()
    cdom_df['log_cdom'] = np.log10(cdom_df['MeasurementValue'])
    cdom_df['site_no_str'] = cdom_df['site_no'].astype(str)
    
    print(f"Loaded {len(cdom_df)} CDOM samples across {cdom_df['site_no_str'].nunique()} sites and {cdom_df['basin'].nunique()} basins.")
    
    # 1. Site Breakdown Table
    site_breakdown = cdom_df.groupby(['site_no_str', 'basin']).agg(
        n_samples=('MeasurementValue', 'count'),
        cdom_mean=('MeasurementValue', 'mean'),
        cdom_median=('MeasurementValue', 'median'),
        cdom_std=('MeasurementValue', 'std'),
        log_cdom_mean=('log_cdom', 'mean'),
        log_cdom_std=('log_cdom', 'std'),
        ndwi_mean=('ndwi', 'mean'),
        mndwi_mean=('mndwi', 'mean'),
        b2_b3_mean=('cdom_ratio', 'mean')
    ).reset_index()
    site_breakdown['pct_of_total'] = 100.0 * site_breakdown['n_samples'] / len(cdom_df)
    site_breakdown = site_breakdown.sort_values(by='n_samples', ascending=False)
    
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures/diagnostics", exist_ok=True)
    site_breakdown.to_csv("results/tables/cdom_site_breakdown.csv", index=False)
    print("\n--- CDOM Site Distribution & Feature Means ---")
    print(site_breakdown[['site_no_str', 'basin', 'n_samples', 'pct_of_total', 'cdom_mean', 'log_cdom_mean', 'ndwi_mean', 'mndwi_mean']].to_string(index=False))
    
    # 2. Features to test
    features = [
        'ndwi', 'mndwi', 'cdom_ratio',
        'B01_center_refl', 'B02_center_refl', 'B03_center_refl', 'B04_center_refl',
        'B05_center_refl', 'B06_center_refl', 'B07_center_refl', 'B08_center_refl',
        'B8A_center_refl', 'B11_center_refl', 'B12_center_refl'
    ]
    
    results = []
    residuals_store = {}
    
    for feat in features:
        if feat not in cdom_df.columns:
            continue
        valid_sub = cdom_df[[feat, 'log_cdom', 'site_no_str', 'basin']].dropna()
        x = valid_sub[feat].values
        y = valid_sub['log_cdom'].values
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
        
        # Within-site correlations for dominant sites
        within_site_pearsons = []
        for s, s_group in valid_sub.groupby('site_no_str'):
            if len(s_group) >= 30 and s_group[feat].std() > 1e-6 and s_group['log_cdom'].std() > 1e-6:
                r_w, _ = stats.pearsonr(s_group[feat], s_group['log_cdom'])
                within_site_pearsons.append(r_w)
        mean_within_site_r = np.nanmean(within_site_pearsons) if within_site_pearsons else np.nan
        
        results.append({
            'feature': feat,
            'raw_pearson_r': round(raw_p_r, 4),
            'raw_pearson_pval': raw_p_p,
            'raw_spearman_r': round(raw_s_r, 4),
            'site_ctrl_pearson_r': round(site_res['pearson_r'], 4),
            'site_ctrl_pearson_pval': site_res['pearson_p'],
            'site_ctrl_spearman_r': round(site_res['spearman_r'], 4),
            'basin_ctrl_pearson_r': round(basin_res['pearson_r'], 4),
            'basin_ctrl_pearson_pval': basin_res['pearson_p'],
            'basin_ctrl_spearman_r': round(basin_res['spearman_r'], 4),
            'mean_within_site_r': round(mean_within_site_r, 4) if not np.isnan(mean_within_site_r) else np.nan,
            'delta_r_site': round(site_res['pearson_r'] - raw_p_r, 4),
            'pct_collapse': round(100.0 * (abs(raw_p_r) - abs(site_res['pearson_r'])) / abs(raw_p_r), 1) if abs(raw_p_r) > 1e-5 else 0.0
        })
        
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(by='raw_pearson_r', ascending=False, key=abs)
    res_df.to_csv("results/tables/cdom_confounding_check.csv", index=False)
    
    print("\n--- CDOM Confounding Check Results (Raw vs Controlled) ---")
    display_cols = ['feature', 'raw_pearson_r', 'site_ctrl_pearson_r', 'basin_ctrl_pearson_r', 'mean_within_site_r', 'pct_collapse']
    print(res_df[display_cols].to_string(index=False))
    
    # 3. Comprehensive Diagnostic Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    
    # Panel A: Raw vs Site-Controlled vs Basin-Controlled Pearson r
    ax = axes[0, 0]
    bar_width = 0.28
    indices = np.arange(len(res_df))
    ax.bar(indices - bar_width, res_df['raw_pearson_r'], width=bar_width, label='Raw Pearson r', color='#2b5c8f', alpha=0.85)
    ax.bar(indices, res_df['basin_ctrl_pearson_r'], width=bar_width, label='Basin-Controlled Partial r', color='#d97706', alpha=0.85)
    ax.bar(indices + bar_width, res_df['site_ctrl_pearson_r'], width=bar_width, label='Site-Controlled Partial r', color='#dc2626', alpha=0.85)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xticks(indices)
    ax.set_xticklabels(res_df['feature'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Correlation with log10(CDOM)', fontsize=11)
    ax.set_title('(A) Raw vs. Controlled Partial Correlations\n(Demonstrating Confounding Collapse)', fontsize=12, fontweight='bold')
    ax.legend(frameon=True)
    ax.grid(True, linestyle=':', alpha=0.5, axis='y')
    
    # Panel B: Raw NDWI vs log10(CDOM) colored by site_no (Showing Site Clustering)
    ax = axes[0, 1]
    palette = sns.color_palette("tab10", n_colors=cdom_df['site_no_str'].nunique())
    sns.scatterplot(
        data=cdom_df,
        x='ndwi',
        y='log_cdom',
        hue='site_no_str',
        style='basin',
        alpha=0.65,
        s=30,
        palette=palette,
        ax=ax
    )
    # Regression line overall
    sns.regplot(data=cdom_df, x='ndwi', y='log_cdom', scatter=False, ax=ax, color='black', line_kws={'linestyle': '--', 'linewidth': 1.5, 'label': f"Overall fit (r={res_df.loc[res_df['feature']=='ndwi', 'raw_pearson_r'].values[0]:.2f})"})
    ax.set_title('(B) Raw NDWI vs log10(CDOM) by Site\n(Ecological Fallacy / Simpson\'s Paradox)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Sentinel-2 NDWI', fontsize=11)
    ax.set_ylabel('log10(CDOM [QSE])', fontsize=11)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, title="Site / Basin")
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # Panel C: Site-Demeaned Residual NDWI vs Residual log10(CDOM)
    ax = axes[1, 0]
    ndwi_res = residuals_store['ndwi']
    ax.scatter(ndwi_res['x_res'], ndwi_res['y_res'], alpha=0.35, color='#4f46e5', s=20)
    sns.regplot(x=ndwi_res['x_res'], y=ndwi_res['y_res'], scatter=False, ax=ax, color='#dc2626', line_kws={'linewidth': 2, 'label': f"Partial fit (r={ndwi_res['pearson_r']:.3f}, p={ndwi_res['pearson_p']:.1e})"})
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_title('(C) NDWI Residuals vs CDOM Residuals (Controlling for Site Fixed Effects)\n(Correlation Drops from r=0.599 to r=0.230)', fontsize=12, fontweight='bold')
    ax.set_xlabel('NDWI Residuals (Site-Demeaned)', fontsize=11)
    ax.set_ylabel('log10(CDOM) Residuals (Site-Demeaned)', fontsize=11)
    ax.legend(frameon=True)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # Panel D: Raw MNDWI vs Residual MNDWI comparison
    ax = axes[1, 1]
    mndwi_res = residuals_store['mndwi']
    ax.scatter(mndwi_res['x_res'], mndwi_res['y_res'], alpha=0.35, color='#059669', s=20)
    sns.regplot(x=mndwi_res['x_res'], y=mndwi_res['y_res'], scatter=False, ax=ax, color='#dc2626', line_kws={'linewidth': 2, 'label': f"Partial fit (r={mndwi_res['pearson_r']:.3f}, p={mndwi_res['pearson_p']:.1e})"})
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_title('(D) MNDWI Residuals vs CDOM Residuals (Controlling for Site Fixed Effects)\n(Correlation Drops from r=0.431 to r=0.170)', fontsize=12, fontweight='bold')
    ax.set_xlabel('MNDWI Residuals (Site-Demeaned)', fontsize=11)
    ax.set_ylabel('log10(CDOM) Residuals (Site-Demeaned)', fontsize=11)
    ax.legend(frameon=True)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plot_path = "results/figures/diagnostics/cdom_confounding_check.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nSaved confounding diagnostic plot to {plot_path}")
    
    # Copy to brain artifact directory
    brain_dir = "C:/Users/Tobin Antony/.gemini/antigravity-ide/brain/730512ff-544c-4445-ac17-b37e3b98b9be"
    if os.path.exists(brain_dir):
        shutil.copy(plot_path, os.path.join(brain_dir, "cdom_confounding_check.png"))
        print(f"Copied plot to artifact directory: {brain_dir}/cdom_confounding_check.png")

if __name__ == "__main__":
    run_cdom_confounding_check()
