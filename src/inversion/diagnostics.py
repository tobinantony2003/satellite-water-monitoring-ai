"""
Phase 1 Diagnostics Suite: Root-Cause Analysis of Chl-a & CDOM Inversion Performance.
Does NOT modify any model code. Analyzes:
1. Temporal matching window audit
2. Target distribution & outlier audit (skewness, kurtosis, Q-Q plots, site dominance)
3. Feature-target correlation audit (Pearson & Spearman)
4. Residual & prediction bias audit on current best models
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

# Suppress joblib loky wmic CPU detection warning on modern Windows 11
if 'LOKY_MAX_CPU_COUNT' not in os.environ:
    os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count() or 4)

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

from src.inversion.models import PyTorchMLPRegressor
from src.inversion.metrics import compute_regression_metrics

sns.set_theme(style='whitegrid', font_scale=1.05)

def run_phase1_diagnostics(
    data_path: str = 'data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv',
    results_dir: str = 'results'
):
    print("=" * 80)
    print("PHASE 1 DIAGNOSTICS: ROOT-CAUSE ANALYSIS OF INVERSION PERFORMANCE")
    print("=" * 80, flush=True)

    fig_dir = os.path.join(results_dir, 'figures', 'diagnostics')
    tab_dir = os.path.join(results_dir, 'tables')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    df = pd.read_csv(data_path, low_memory=False)
    print(f"Loaded {len(df):,} records from {data_path}\n", flush=True)

    # Calculate exact temporal gap in hours
    t_scene = pd.to_datetime(df['scene_datetime_UTC'])
    t_sample = pd.to_datetime(df['sample_datetime_UTC'])
    df['dt_hours'] = np.abs((t_scene - t_sample).dt.total_seconds()) / 3600.0

    # =========================================================================
    # PART 1: TEMPORAL MATCHING AUDIT
    # =========================================================================
    print("-" * 80)
    print("1. TEMPORAL MATCHING AUDIT")
    print("-" * 80, flush=True)

    targets = ['turbidity', 'chlorophyll', 'cdom']
    temp_stats = []

    for t in targets:
        sub = df[df['target_type'] == t]['dt_hours']
        cnt = len(sub)
        temp_stats.append({
            'target': t,
            'count': cnt,
            'min_hours': sub.min(),
            'p25_hours': np.percentile(sub, 25),
            'median_hours': sub.median(),
            'mean_hours': sub.mean(),
            'p75_hours': np.percentile(sub, 75),
            'p90_hours': np.percentile(sub, 90),
            'max_hours': sub.max(),
            'pct_gt_1h': (sub > 1.0).mean() * 100,
            'pct_gt_3h': (sub > 3.0).mean() * 100,
            'pct_gt_6h': (sub > 6.0).mean() * 100,
            'pct_gt_12h': (sub > 12.0).mean() * 100,
            'pct_gt_24h_1day': (sub > 24.0).mean() * 100,
            'pct_gt_3days': (sub > 72.0).mean() * 100,
            'pct_gt_7days': (sub > 168.0).mean() * 100,
        })

    temp_df = pd.DataFrame(temp_stats)
    temp_table_path = os.path.join(tab_dir, 'phase1_temporal_audit.csv')
    temp_df.to_csv(temp_table_path, index=False)
    print(f"Temporal Gap Summary Statistics (saved to {temp_table_path}):")
    print(temp_df[['target', 'count', 'mean_hours', 'median_hours', 'max_hours', 'pct_gt_1h', 'pct_gt_3h', 'pct_gt_6h', 'pct_gt_24h_1day']].to_string(index=False), flush=True)

    # Plot 1: Temporal Distribution Histograms
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=150)
    for i, t in enumerate(targets):
        sub = df[df['target_type'] == t]['dt_hours']
        axes[i].hist(sub, bins=24, color='#1f77b4', edgecolor='black', alpha=0.75)
        axes[i].axvline(sub.median(), color='red', linestyle='--', lw=2, label=f"Median: {sub.median():.2f}h")
        axes[i].axvline(sub.mean(), color='orange', linestyle=':', lw=2, label=f"Mean: {sub.mean():.2f}h")
        axes[i].set_title(f"{t.capitalize()} Temporal Gap (|Δt|)", fontweight='bold')
        axes[i].set_xlabel('Time Difference (Hours)')
        axes[i].set_ylabel('Number of Matchups')
        axes[i].legend()
    plt.suptitle("Temporal Matching Gap Distribution (Overpass vs. In-Situ Timestamp)", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig1_path = os.path.join(fig_dir, 'phase1_temporal_gap_histograms.png')
    plt.savefig(fig1_path, bbox_inches='tight')
    plt.close()
    print(f"Saved temporal audit plot to {fig1_path}\n", flush=True)

    # =========================================================================
    # PART 2: TARGET DISTRIBUTION & SITE DOMINANCE AUDIT
    # =========================================================================
    print("-" * 80)
    print("2. TARGET DISTRIBUTION & OUTLIER AUDIT")
    print("-" * 80, flush=True)

    dist_stats = []
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), dpi=150)

    for i, t in enumerate(targets):
        sub = df[df['target_type'] == t]['MeasurementValue'].dropna()
        sub_pos = sub[sub > 0]
        sub_log = np.log10(sub_pos)

        p1_raw = np.percentile(sub_pos, 1)
        p99_raw = np.percentile(sub_pos, 99)
        p1_log = np.percentile(sub_log, 1)
        p99_log = np.percentile(sub_log, 99)

        dist_stats.append({
            'target': t,
            'n_samples': len(sub_pos),
            'min': sub_pos.min(),
            'max': sub_pos.max(),
            'mean': sub_pos.mean(),
            'median': sub_pos.median(),
            'std': sub_pos.std(),
            'raw_skewness': stats.skew(sub_pos),
            'raw_kurtosis': stats.kurtosis(sub_pos),
            'log_skewness': stats.skew(sub_log),
            'log_kurtosis': stats.kurtosis(sub_log),
            'bottom_1pct_cutoff': p1_raw,
            'top_1pct_cutoff': p99_raw,
        })

        # Col 0: Raw histogram (cut off at p99 for readability)
        axes[i, 0].hist(sub_pos[sub_pos <= p99_raw], bins=40, color='#1f77b4', edgecolor='black', alpha=0.7)
        axes[i, 0].set_title(f"{t.capitalize()} Raw (≤99th pct: {p99_raw:.1f})", fontweight='bold')
        axes[i, 0].set_ylabel('Frequency')

        # Col 1: Log10 histogram
        axes[i, 1].hist(sub_log, bins=40, color='#2ca02c', edgecolor='black', alpha=0.7)
        axes[i, 1].set_title(f"{t.capitalize()} Log10 (Skew: {stats.skew(sub_log):.2f})", fontweight='bold')

        # Col 2: Q-Q Plot against Normal distribution
        stats.probplot(sub_log, dist="norm", plot=axes[i, 2])
        axes[i, 2].set_title(f"{t.capitalize()} Q-Q Plot (Log10 vs Normal)", fontweight='bold')

    plt.suptitle("Target Distribution & Normality Audit (Raw, Log10, and Q-Q Plots)", fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig2_path = os.path.join(fig_dir, 'phase1_target_distributions_qq.png')
    plt.savefig(fig2_path, bbox_inches='tight')
    plt.close()
    print(f"Saved distribution & Q-Q audit plot to {fig2_path}", flush=True)

    dist_df = pd.DataFrame(dist_stats)
    dist_table_path = os.path.join(tab_dir, 'phase1_target_distribution_stats.csv')
    dist_df.to_csv(dist_table_path, index=False)
    print(f"\nDistribution Statistics Table (saved to {dist_table_path}):")
    print(dist_df[['target', 'n_samples', 'min', 'median', 'mean', 'max', 'raw_skewness', 'log_skewness', 'bottom_1pct_cutoff', 'top_1pct_cutoff']].to_string(index=False), flush=True)

    # Basin and Site Dominance Audit
    print("\n--- Site & Basin Dominance Audit ---", flush=True)
    site_stats = []
    for t in targets:
        sub = df[df['target_type'] == t]
        n_basins = sub['basin'].nunique()
        n_sites = sub['site_no'].nunique()
        site_counts = sub['site_no'].value_counts()
        top_site = site_counts.index[0]
        top_site_n = site_counts.iloc[0]
        top_site_pct = (top_site_n / len(sub)) * 100
        top_3_pct = (site_counts.iloc[:3].sum() / len(sub)) * 100

        basin_counts = sub['basin'].value_counts()
        top_basin = basin_counts.index[0]
        top_basin_pct = (basin_counts.iloc[0] / len(sub)) * 100

        site_stats.append({
            'target': t,
            'n_samples': len(sub),
            'n_basins': n_basins,
            'n_sites': n_sites,
            'top_site_id': str(top_site),
            'top_site_pct': round(top_site_pct, 2),
            'top_3_sites_pct': round(top_3_pct, 2),
            'dominant_basin': top_basin,
            'dominant_basin_pct': round(top_basin_pct, 2)
        })

    site_df = pd.DataFrame(site_stats)
    site_table_path = os.path.join(tab_dir, 'phase1_site_dominance.csv')
    site_df.to_csv(site_table_path, index=False)
    print(f"Site & Basin Dominance Table (saved to {site_table_path}):")
    print(site_df.to_string(index=False), flush=True)

    # =========================================================================
    # PART 3: FEATURE-TARGET RELATIONSHIP AUDIT
    # =========================================================================
    print("\n" + "-" * 80)
    print("3. FEATURE-TARGET RELATIONSHIP AUDIT")
    print("-" * 80, flush=True)

    bands = [
        'B01_center_refl', 'B02_center_refl', 'B03_center_refl', 'B04_center_refl',
        'B05_center_refl', 'B06_center_refl', 'B07_center_refl', 'B08_center_refl',
        'B8A_center_refl', 'B11_center_refl', 'B12_center_refl'
    ]

    # Additional standard indices
    df['b5_b4_ratio'] = df['B05_center_refl'] / (df['B04_center_refl'] + 1e-6)

    # Chlorophyll-a Audit
    df_chl = df[df['target_type'] == 'chlorophyll'].copy()
    y_chl_log = np.log10(np.clip(df_chl['MeasurementValue'].values, 1e-4, None))
    chl_features = ['ndci', 'b5_b4_ratio', 'turbidity_ratio'] + bands

    chl_corrs = []
    for f in chl_features:
        vals = df_chl[f].values
        valid = ~np.isnan(vals) & ~np.isnan(y_chl_log)
        p_r, p_pval = stats.pearsonr(vals[valid], y_chl_log[valid])
        s_r, s_pval = stats.spearmanr(vals[valid], y_chl_log[valid])
        chl_corrs.append({
            'target': 'chlorophyll',
            'feature': f,
            'pearson_r': round(p_r, 4),
            'pearson_pval': p_pval,
            'spearman_r': round(s_r, 4),
            'spearman_pval': s_pval,
            'abs_spearman': round(abs(s_r), 4),
            'above_0.2_threshold': abs(s_r) >= 0.20
        })
    chl_corr_df = pd.DataFrame(chl_corrs).sort_values(by='abs_spearman', ascending=False)

    # CDOM Audit
    df_cdom = df[df['target_type'] == 'cdom'].copy()
    y_cdom_log = np.log10(np.clip(df_cdom['MeasurementValue'].values, 1e-4, None))
    cdom_features = ['cdom_ratio', 'ndwi', 'mndwi'] + bands

    cdom_corrs = []
    for f in cdom_features:
        vals = df_cdom[f].values
        valid = ~np.isnan(vals) & ~np.isnan(y_cdom_log)
        p_r, p_pval = stats.pearsonr(vals[valid], y_cdom_log[valid])
        s_r, s_pval = stats.spearmanr(vals[valid], y_cdom_log[valid])
        cdom_corrs.append({
            'target': 'cdom',
            'feature': f,
            'pearson_r': round(p_r, 4),
            'pearson_pval': p_pval,
            'spearman_r': round(s_r, 4),
            'spearman_pval': s_pval,
            'abs_spearman': round(abs(s_r), 4),
            'above_0.2_threshold': abs(s_r) >= 0.20
        })
    cdom_corr_df = pd.DataFrame(cdom_corrs).sort_values(by='abs_spearman', ascending=False)

    corr_table_path = os.path.join(tab_dir, 'phase1_feature_correlations.csv')
    all_corr_df = pd.concat([chl_corr_df, cdom_corr_df], ignore_index=True)
    all_corr_df.to_csv(corr_table_path, index=False)

    print("\nTop Chlorophyll-a Feature Correlations (|Spearman r|):")
    print(chl_corr_df[['feature', 'pearson_r', 'spearman_r', 'above_0.2_threshold']].to_string(index=False), flush=True)

    print("\nTop CDOM Feature Correlations (|Spearman r|):")
    print(cdom_corr_df[['feature', 'pearson_r', 'spearman_r', 'above_0.2_threshold']].to_string(index=False), flush=True)

    # Plot 3: Scatter plots of top features vs. log targets
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=150)
    # Chl-a plots
    top_chl_feats = ['ndci', 'b5_b4_ratio', 'B05_center_refl', 'B04_center_refl']
    for j, f in enumerate(top_chl_feats):
        axes[0, j].scatter(df_chl[f], y_chl_log, alpha=0.25, s=12, color='#2ca02c')
        axes[0, j].set_title(f"Chl-a vs. {f}\n(Spearman r = {df_chl[f].corr(pd.Series(y_chl_log), method='spearman'):+.3f})", fontweight='bold')
        axes[0, j].set_xlabel(f)
        axes[0, j].set_ylabel('log10(Chl-a μg/L)')

    # CDOM plots
    top_cdom_feats = ['cdom_ratio', 'B02_center_refl', 'B03_center_refl', 'ndwi']
    for j, f in enumerate(top_cdom_feats):
        axes[1, j].scatter(df_cdom[f], y_cdom_log, alpha=0.35, s=16, color='#9467bd')
        axes[1, j].set_title(f"CDOM vs. {f}\n(Spearman r = {df_cdom[f].corr(pd.Series(y_cdom_log), method='spearman'):+.3f})", fontweight='bold')
        axes[1, j].set_xlabel(f)
        axes[1, j].set_ylabel('log10(CDOM μg/L QSE)')

    plt.suptitle("Feature-Target Bivariate Relationships (Chlorophyll-a & CDOM)", fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig3_path = os.path.join(fig_dir, 'phase1_feature_target_correlations.png')
    plt.savefig(fig3_path, bbox_inches='tight')
    plt.close()
    print(f"Saved feature correlation plot to {fig3_path}\n", flush=True)

    # =========================================================================
    # PART 4: RESIDUAL AUDIT ON CURRENT BEST MODELS
    # =========================================================================
    print("-" * 80)
    print("4. RESIDUAL AUDIT ON CURRENT BEST MODELS")
    print("-" * 80, flush=True)

    feature_cols = [
        'B01_center_refl', 'B02_center_refl', 'B03_center_refl', 'B04_center_refl',
        'B05_center_refl', 'B06_center_refl', 'B07_center_refl', 'B08_center_refl',
        'B8A_center_refl', 'B11_center_refl', 'B12_center_refl',
        'ndwi', 'mndwi', 'ndci', 'turbidity_ratio', 'cdom_ratio', 'b8_b4_ratio'
    ]

    # Best Chl-a: Random Forest
    print("Fitting 5-fold Spatial Block CV Random Forest for Chlorophyll-a...", flush=True)
    gkf_chl = GroupKFold(n_splits=5)
    X_chl = df_chl[feature_cols].values
    y_chl_raw = df_chl['MeasurementValue'].values
    groups_chl = df_chl['site_no'].values
    oof_chl_rf = np.zeros(len(df_chl))

    for tr, val in gkf_chl.split(X_chl, y_chl_log, groups_chl):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_chl[tr])
        X_val = scaler.transform(X_chl[val])
        rf = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=3, n_jobs=4, random_state=42)
        rf.fit(X_tr, y_chl_log[tr])
        oof_chl_rf[val] = rf.predict(X_val)

    pred_chl_raw = 10.0 ** np.clip(oof_chl_rf, -3.0, 4.0)
    res_chl = pred_chl_raw - y_chl_raw

    # Best CDOM: Deep Spectral MLP
    print("Fitting 5-fold Spatial Block CV Deep Spectral MLP for CDOM...", flush=True)
    gkf_cdom = GroupKFold(n_splits=min(5, df_cdom['site_no'].nunique()))
    X_cdom = df_cdom[feature_cols].values
    y_cdom_raw = df_cdom['MeasurementValue'].values
    groups_cdom = df_cdom['site_no'].values
    oof_cdom_mlp = np.zeros(len(df_cdom))

    for tr, val in gkf_cdom.split(X_cdom, y_cdom_log, groups_cdom):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_cdom[tr])
        X_val = scaler.transform(X_cdom[val])
        mlp = PyTorchMLPRegressor(in_features=len(feature_cols), epochs=30, lr=1e-3)
        mlp.fit(X_tr, y_cdom_log[tr])
        oof_cdom_mlp[val] = mlp.predict(X_val)

    pred_cdom_raw = 10.0 ** np.clip(oof_cdom_mlp, -3.0, 4.0)
    res_cdom = pred_cdom_raw - y_cdom_raw

    # Report Bias Metrics
    m_chl = compute_regression_metrics(y_chl_raw, pred_chl_raw)
    m_cdom = compute_regression_metrics(y_cdom_raw, pred_cdom_raw)

    print(f"\nChlorophyll-a Random Forest: R^2 = {m_chl['r2']:.4f} | RMSE = {m_chl['rmse']:.2f} | MAE = {m_chl['mae']:.2f} | Bias = {m_chl['bias']:.2f}", flush=True)
    print(f"CDOM Deep Spectral MLP:      R^2 = {m_cdom['r2']:.4f} | RMSE = {m_cdom['rmse']:.2f} | MAE = {m_cdom['mae']:.2f} | Bias = {m_cdom['bias']:.2f}", flush=True)

    # Plot 4: Predicted vs. Actual and Residuals vs. Predicted
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), dpi=150)

    # Chl-a Pred vs Actual
    axes[0, 0].scatter(y_chl_raw, pred_chl_raw, alpha=0.3, color='#2ca02c', s=16)
    lim_chl = [max(0.1, min(y_chl_raw.min(), pred_chl_raw.min())), max(y_chl_raw.max(), pred_chl_raw.max())]
    axes[0, 0].plot(lim_chl, lim_chl, 'r--', lw=2, label='1:1 Line')
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_title(f"Chlorophyll-a: Predicted vs. Observed\n(RF R^2 = {m_chl['r2']:.4f}, RMSE = {m_chl['rmse']:.2f} μg/L)", fontweight='bold')
    axes[0, 0].set_xlabel('Observed Chl-a (μg/L)')
    axes[0, 0].set_ylabel('Predicted Chl-a (μg/L)')
    axes[0, 0].legend()

    # Chl-a Residuals vs Pred
    axes[0, 1].scatter(pred_chl_raw, res_chl, alpha=0.3, color='#2ca02c', s=16)
    axes[0, 1].axhline(0, color='red', linestyle='--', lw=2)
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_title(f"Chlorophyll-a: Residuals vs. Predicted\n(Mean Bias = {m_chl['bias']:+.2f} μg/L)", fontweight='bold')
    axes[0, 1].set_xlabel('Predicted Chl-a (μg/L)')
    axes[0, 1].set_ylabel('Residual (Predicted - Observed)')

    # CDOM Pred vs Actual
    axes[1, 0].scatter(y_cdom_raw, pred_cdom_raw, alpha=0.4, color='#9467bd', s=18)
    lim_cdom = [max(0.1, min(y_cdom_raw.min(), pred_cdom_raw.min())), max(y_cdom_raw.max(), pred_cdom_raw.max())]
    axes[1, 0].plot(lim_cdom, lim_cdom, 'r--', lw=2, label='1:1 Line')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title(f"CDOM: Predicted vs. Observed\n(MLP R^2 = {m_cdom['r2']:.4f}, RMSE = {m_cdom['rmse']:.2f} μg/L QSE)", fontweight='bold')
    axes[1, 0].set_xlabel('Observed CDOM (μg/L QSE)')
    axes[1, 0].set_ylabel('Predicted CDOM (μg/L QSE)')
    axes[1, 0].legend()

    # CDOM Residuals vs Pred
    axes[1, 1].scatter(pred_cdom_raw, res_cdom, alpha=0.4, color='#9467bd', s=18)
    axes[1, 1].axhline(0, color='red', linestyle='--', lw=2)
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_title(f"CDOM: Residuals vs. Predicted\n(Mean Bias = {m_cdom['bias']:+.2f} μg/L QSE)", fontweight='bold')
    axes[1, 1].set_xlabel('Predicted CDOM (μg/L QSE)')
    axes[1, 1].set_ylabel('Residual (Predicted - Observed)')

    plt.suptitle("Residual & Prediction Bias Audit (Current Best Inversion Models)", fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig4_path = os.path.join(fig_dir, 'phase1_residual_audit.png')
    plt.savefig(fig4_path, bbox_inches='tight')
    plt.close()
    print(f"Saved residual audit plot to {fig4_path}\n", flush=True)

    print("=" * 80)
    print("PHASE 1 DIAGNOSTICS COMPLETE. READY FOR REVIEW.")
    print("=" * 80, flush=True)

if __name__ == '__main__':
    run_phase1_diagnostics()
