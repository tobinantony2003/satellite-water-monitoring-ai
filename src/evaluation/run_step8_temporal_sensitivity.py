"""
Script: run_step8_temporal_sensitivity.py
Implements Step 8 of Phase 2 Modeling:
1. Temporal match window sensitivity audit (dt <= 1h, <= 2h, <= 3h, <= 6h, <= 12h)
   for Turbidity, Chlorophyll-a, and CDOM:
   - Sample size retention and station coverage
   - Correlation retention (raw and within-site)
2. Inversion performance under 5-Fold Spatial Block CV comparing match windows:
   - Full 12h pool vs. <= 3h window vs. <= 1h window
3. Evaluation of hours_since_overpass (absolute and signed) as an explicit inversion feature:
   - Residual error vs dt_hours correlation
   - Model feature importance
   - 5-Fold Spatial Block CV delta R2
"""

import os
import sys
from pathlib import Path

# Suppress joblib loky wmic CPU detection warning on modern Windows 11
if 'LOKY_MAX_CPU_COUNT' not in os.environ:
    os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count() or 4)

import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from src.inversion.models import get_inversion_models
from src.inversion.train_inversion import FEATURE_COLS
from src.inversion.spatial_cv import get_spatial_block_cv

def load_and_preprocess_dataset():
    data_path = PROJECT_ROOT / 'data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv'
    df = pd.read_csv(data_path, low_memory=False)
    
    t_scene = pd.to_datetime(df['scene_datetime_UTC'])
    t_sample = pd.to_datetime(df['sample_datetime_UTC'])
    df['dt_hours'] = (t_sample - t_scene).abs().dt.total_seconds() / 3600.0
    df['dt_hours_signed'] = (t_sample - t_scene).dt.total_seconds() / 3600.0
    
    # Compute extra bio-optical indices if not present
    if 'b5_b4_ratio' not in df.columns:
        df['b5_b4_ratio'] = df['B05_center_refl'] / (df['B04_center_refl'] + 1e-6)
    
    return df

def audit_window_sensitivity(df):
    print("=" * 80)
    print("STEP 8.1: TEMPORAL MATCH WINDOW SENSITIVITY AUDIT")
    print("=" * 80)
    
    windows = [1.0, 2.0, 3.0, 6.0, 12.0]
    records = []
    
    targets = {
        'turbidity': ['turbidity_ratio', 'B04_center_refl', 'ndwi'],
        'chlorophyll': ['ndci', 'b5_b4_ratio', 'B05_center_refl'],
        'cdom': ['ndwi', 'mndwi', 'cdom_ratio']
    }
    
    for target, primary_features in targets.items():
        sub_all = df[df['target_type'] == target].copy()
        n_total = len(sub_all)
        n_sites_total = sub_all['site_no'].nunique()
        y_all_log = np.log10(np.clip(sub_all['MeasurementValue'].values, 1e-4, None))
        
        print(f"\n--- Target: {target.upper()} (Total Pool N = {n_total:,}, Sites = {n_sites_total}) ---")
        
        for w in windows:
            mask = sub_all['dt_hours'] <= w
            sub_w = sub_all[mask].copy()
            n_w = len(sub_w)
            pct_total = 100.0 * n_w / n_total
            n_sites_w = sub_w['site_no'].nunique()
            
            y_w_log = np.log10(np.clip(sub_w['MeasurementValue'].values, 1e-4, None))
            
            row = {
                'target': target,
                'window_hours': w,
                'n_samples': n_w,
                'pct_retained': pct_total,
                'n_sites': n_sites_w,
                'pct_sites': 100.0 * n_sites_w / n_sites_total,
                'target_mean': sub_w['MeasurementValue'].mean(),
                'target_std': sub_w['MeasurementValue'].std(),
                'target_log_std': np.std(y_w_log)
            }
            
            # Correlations for key features
            for feat in primary_features:
                x_vals = sub_w[feat].values
                valid = np.isfinite(x_vals) & np.isfinite(y_w_log)
                if valid.sum() > 10:
                    r_p, _ = stats.pearsonr(x_vals[valid], y_w_log[valid])
                    r_s, _ = stats.spearmanr(x_vals[valid], y_w_log[valid])
                else:
                    r_p, r_s = np.nan, np.nan
                row[f'{feat}_pearson_r'] = r_p
                row[f'{feat}_spearman_r'] = r_s
                
                # Within-site correlation (sample-weighted)
                site_rs = []
                site_ns = []
                for site, group in sub_w.groupby('site_no'):
                    if len(group) >= 5:
                        gx = group[feat].values
                        gy = np.log10(np.clip(group['MeasurementValue'].values, 1e-4, None))
                        if np.std(gx) > 1e-6 and np.std(gy) > 1e-6:
                            sr, _ = stats.pearsonr(gx, gy)
                            if not np.isnan(sr):
                                site_rs.append(sr)
                                site_ns.append(len(group))
                if len(site_rs) > 0:
                    weighted_within_r = np.average(site_rs, weights=site_ns)
                else:
                    weighted_within_r = np.nan
                row[f'{feat}_within_site_r'] = weighted_within_r
                
            records.append(row)
            print(f"  Window <= {w:4.1f}h: N={n_w:5d} ({pct_total:5.1f}%), Sites={n_sites_w:2d}/{n_sites_total:2d} | "
                  f"Primary Feat ({primary_features[0]}): Pearson r = {row[f'{primary_features[0]}_pearson_r']:+.4f}, "
                  f"Within-Site r = {row[f'{primary_features[0]}_within_site_r']:+.4f}")
            
    df_audit = pd.DataFrame(records)
    out_csv = PROJECT_ROOT / 'results/tables/step8_temporal_window_audit.csv'
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_audit.to_csv(out_csv, index=False)
    print(f"\nSaved temporal window sensitivity audit table to: {out_csv}")
    return df_audit

def evaluate_cv_across_windows(df):
    print("\n" + "=" * 80)
    print("STEP 8.2: 5-FOLD SPATIAL BLOCK CV ACROSS MATCH WINDOWS")
    print("=" * 80)
    
    cv_records = []
    windows = [1.0, 3.0, 12.0]
    
    for target in ['turbidity', 'chlorophyll']:
        sub_all = df[df['target_type'] == target].copy()
        print(f"\n--- Inversion Spatial CV for {target.upper()} ---")
        
        for w in windows:
            sub_w = sub_all[sub_all['dt_hours'] <= w].copy().reset_index(drop=True)
            
            # Check station count
            n_sites = sub_w['site_no'].nunique()
            if n_sites < 5:
                print(f"Skipping window <= {w}h: insufficient stations ({n_sites})")
                continue
                
            # Perform 5-fold Spatial Block CV
            splits = list(get_spatial_block_cv(sub_w, group_col='site_no', n_splits=5))
            
            oof_y_log = np.zeros(len(sub_w))
            oof_pred_log = np.zeros(len(sub_w))
            
            y_raw = sub_w['MeasurementValue'].values
            y_log = np.log10(np.clip(y_raw, 1e-4, None))
            X = sub_w[FEATURE_COLS].values
            
            fold_r2_log = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                X_train, y_train_log = X[train_idx], y_log[train_idx]
                X_val, y_val_log = X[val_idx], y_log[val_idx]
                
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_val_s = scaler.transform(X_val)
                
                model = GradientBoostingRegressor(
                    n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
                )
                model.fit(X_train_s, y_train_log)
                
                preds = model.predict(X_val_s)
                oof_pred_log[val_idx] = preds
                oof_y_log[val_idx] = y_val_log
                
                f_r2 = r2_score(y_val_log, preds)
                fold_r2_log.append(f_r2)
                
            micro_r2_log = r2_score(oof_y_log, oof_pred_log)
            micro_rmse_log = np.sqrt(mean_squared_error(oof_y_log, oof_pred_log))
            macro_r2_log = np.mean(fold_r2_log)
            std_r2_log = np.std(fold_r2_log)
            
            oof_pred_raw = 10.0 ** oof_pred_log
            phys_rmse = np.sqrt(mean_squared_error(y_raw, oof_pred_raw))
            phys_mae = mean_absolute_error(y_raw, oof_pred_raw)
            phys_r2 = r2_score(y_raw, oof_pred_raw)
            
            res = {
                'target': target,
                'window_hours': w,
                'n_samples': len(sub_w),
                'n_sites': n_sites,
                'micro_log_r2': micro_r2_log,
                'macro_log_r2': macro_r2_log,
                'macro_log_r2_std': std_r2_log,
                'micro_log_rmse': micro_rmse_log,
                'phys_rmse': phys_rmse,
                'phys_mae': phys_mae,
                'phys_r2': phys_r2
            }
            cv_records.append(res)
            print(f"  Window <= {w:4.1f}h (N={len(sub_w):5d}): Micro Log R2 = {micro_r2_log:+.4f}, "
                  f"Macro Log R2 = {macro_r2_log:+.4f} +/- {std_r2_log:.4f}, Log RMSE = {micro_rmse_log:.4f}")
            
    df_cv = pd.DataFrame(cv_records)
    out_csv = PROJECT_ROOT / 'results/tables/step8_temporal_window_inversion_cv.csv'
    df_cv.to_csv(out_csv, index=False)
    print(f"\nSaved CV comparison across windows to: {out_csv}")
    return df_cv

def evaluate_hours_since_overpass_feature(df):
    print("\n" + "=" * 80)
    print("STEP 8.3: EVALUATE HOURS_SINCE_OVERPASS AS AN EXPLICIT FEATURE")
    print("=" * 80)
    
    feat_records = []
    
    for target in ['turbidity', 'chlorophyll']:
        sub = df[df['target_type'] == target].copy().reset_index(drop=True)
        print(f"\n--- Feature Importance & Residual Correlation for {target.upper()} (N = {len(sub):,}) ---")
        
        y_raw = sub['MeasurementValue'].values
        y_log = np.log10(np.clip(y_raw, 1e-4, None))
        
        # Test 1: Correlation of dt_hours with target and with baseline residuals
        # Train baseline model on standard FEATURE_COLS under 5-fold CV
        splits = list(get_spatial_block_cv(sub, group_col='site_no', n_splits=5))
        
        X_base = sub[FEATURE_COLS].values
        oof_base_log = np.zeros(len(sub))
        
        for train_idx, val_idx in splits:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_base[train_idx])
            X_va = scaler.transform(X_base[val_idx])
            
            m = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
            m.fit(X_tr, y_log[train_idx])
            oof_base_log[val_idx] = m.predict(X_va)
            
        res_log = np.abs(y_log - oof_base_log)
        signed_res = y_log - oof_base_log
        
        r_dt_y, _ = stats.pearsonr(sub['dt_hours'], y_log)
        r_dt_abs_res, _ = stats.pearsonr(sub['dt_hours'], res_log)
        r_dt_signed_res, _ = stats.pearsonr(sub['dt_hours'], signed_res)
        
        print(f"Correlation between dt_hours and Log(Target): r = {r_dt_y:+.4f}")
        print(f"Correlation between dt_hours and Absolute Residual Error: r = {r_dt_abs_res:+.4f}")
        print(f"Correlation between dt_hours and Signed Error: r = {r_dt_signed_res:+.4f}")
        
        # Test 2: Inversion performance WITH dt_hours and dt_hours_signed added
        extended_cols = FEATURE_COLS + ['dt_hours', 'dt_hours_signed']
        X_ext = sub[extended_cols].values
        oof_ext_log = np.zeros(len(sub))
        fold_r2_ext = []
        feature_importances = []
        
        for train_idx, val_idx in splits:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_ext[train_idx])
            X_va = scaler.transform(X_ext[val_idx])
            
            m_ext = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
            m_ext.fit(X_tr, y_log[train_idx])
            preds = m_ext.predict(X_va)
            oof_ext_log[val_idx] = preds
            fold_r2_ext.append(r2_score(y_log[val_idx], preds))
            feature_importances.append(m_ext.feature_importances_)
            
        base_micro_r2 = r2_score(y_log, oof_base_log)
        ext_micro_r2 = r2_score(y_log, oof_ext_log)
        base_macro_r2 = np.mean([r2_score(y_log[va], oof_base_log[va]) for _, va in splits])
        ext_macro_r2 = np.mean(fold_r2_ext)
        
        mean_fi = np.mean(feature_importances, axis=0)
        fi_dict = dict(zip(extended_cols, mean_fi))
        
        print(f"Baseline Micro Log R2: {base_micro_r2:+.4f} | With dt_hours: {ext_micro_r2:+.4f} (Delta = {ext_micro_r2 - base_micro_r2:+.4f})")
        print(f"Baseline Macro Log R2: {base_macro_r2:+.4f} | With dt_hours: {ext_macro_r2:+.4f} (Delta = {ext_macro_r2 - base_macro_r2:+.4f})")
        print(f"Feature importance dt_hours: {fi_dict['dt_hours']:.4f} (Rank #{np.argsort(-mean_fi).tolist().index(extended_cols.index('dt_hours'))+1} of {len(extended_cols)})")
        print(f"Feature importance dt_hours_signed: {fi_dict['dt_hours_signed']:.4f} (Rank #{np.argsort(-mean_fi).tolist().index(extended_cols.index('dt_hours_signed'))+1} of {len(extended_cols)})")
        
        feat_records.append({
            'target': target,
            'corr_dt_target': r_dt_y,
            'corr_dt_abs_residual': r_dt_abs_res,
            'corr_dt_signed_residual': r_dt_signed_res,
            'baseline_micro_log_r2': base_micro_r2,
            'extended_micro_log_r2': ext_micro_r2,
            'delta_micro_log_r2': ext_micro_r2 - base_micro_r2,
            'baseline_macro_log_r2': base_macro_r2,
            'extended_macro_log_r2': ext_macro_r2,
            'delta_macro_log_r2': ext_macro_r2 - base_macro_r2,
            'dt_hours_fi': fi_dict['dt_hours'],
            'dt_hours_signed_fi': fi_dict['dt_hours_signed']
        })
        
    df_feat = pd.DataFrame(feat_records)
    out_csv = PROJECT_ROOT / 'results/tables/step8_hours_since_overpass_evaluation.csv'
    df_feat.to_csv(out_csv, index=False)
    print(f"\nSaved hours_since_overpass evaluation to: {out_csv}")
    return df_feat

def plot_step8_diagnostics(df):
    print("\nGenerating Step 8 diagnostic figures...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # 1. dt distribution across targets
    ax1 = axes[0, 0]
    for target, col in zip(['turbidity', 'chlorophyll', 'cdom'], ['#2b5c8f', '#2a9d8f', '#e76f51']):
        sub = df[df['target_type'] == target]
        sns.kdeplot(sub['dt_hours'], ax=ax1, label=f"{target.capitalize()} (N={len(sub):,})", color=col, lw=2)
    ax1.axvline(1.0, color='red', linestyle='--', label='1-Hour Window')
    ax1.axvline(3.0, color='darkorange', linestyle='--', label='3-Hour Baseline')
    ax1.set_title("A. Distribution of Match Time Delta (|t_sample - t_scene|)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Time Difference (Hours)")
    ax1.set_ylabel("Density")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative sample retention
    ax2 = axes[0, 1]
    hours_grid = np.linspace(0.1, 12.0, 120)
    for target, col in zip(['turbidity', 'chlorophyll', 'cdom'], ['#2b5c8f', '#2a9d8f', '#e76f51']):
        sub = df[df['target_type'] == target]
        retention = [100.0 * (sub['dt_hours'] <= h).sum() / len(sub) for h in hours_grid]
        ax2.plot(hours_grid, retention, label=target.capitalize(), color=col, lw=2.5)
    ax2.axvline(1.0, color='red', linestyle='--', alpha=0.7)
    ax2.axvline(3.0, color='darkorange', linestyle='--', alpha=0.7)
    ax2.set_title("B. Cumulative Sample Retention vs Match Window", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Match Window Threshold (Hours)")
    ax2.set_ylabel("Data Retained (%)")
    ax2.set_ylim(0, 105)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Turbidity Correlation stability across match window
    ax3 = axes[1, 0]
    turb_sub = df[df['target_type'] == 'turbidity']
    w_steps = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    r_turb_ratio = []
    r_b04 = []
    r_ndwi = []
    for w in w_steps:
        sub_w = turb_sub[turb_sub['dt_hours'] <= w]
        y_l = np.log10(np.clip(sub_w['MeasurementValue'].values, 1e-4, None))
        r_turb_ratio.append(stats.pearsonr(sub_w['turbidity_ratio'], y_l)[0])
        r_b04.append(stats.pearsonr(sub_w['B04_center_refl'], y_l)[0])
        r_ndwi.append(stats.pearsonr(sub_w['ndwi'], y_l)[0])
    ax3.plot(w_steps, r_turb_ratio, marker='o', label='Turbidity Ratio (B4/B3)', color='#2b5c8f', lw=2)
    ax3.plot(w_steps, r_b04, marker='s', label='B04 (Red)', color='#d90429', lw=2)
    ax3.plot(w_steps, r_ndwi, marker='^', label='NDWI', color='#0077b6', lw=2)
    ax3.set_title("C. Turbidity Optical Feature Pearson Correlation vs Window", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Match Window Threshold (Hours)")
    ax3.set_ylabel("Pearson Correlation r with Log10(Turbidity)")
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Inversion Error vs dt_hours
    ax4 = axes[1, 1]
    # Bin dt_hours into 1-hour bins and compute mean absolute log error
    # Load turbidity data
    y_turb_l = np.log10(np.clip(turb_sub['MeasurementValue'].values, 1e-4, None))
    # fit a quick GB model
    m_turb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    m_turb.fit(turb_sub[FEATURE_COLS].values, y_turb_l)
    p_turb_l = m_turb.predict(turb_sub[FEATURE_COLS].values)
    err = np.abs(y_turb_l - p_turb_l)
    
    dt_bins = pd.cut(turb_sub['dt_hours'], bins=np.arange(0, 13, 1))
    err_binned = pd.DataFrame({'bin': dt_bins, 'abs_err': err}).groupby('bin')['abs_err'].agg(['mean', 'std', 'count']).reset_index()
    bin_centers = np.arange(0.5, 12.5, 1.0)
    ax4.errorbar(bin_centers, err_binned['mean'], yerr=err_binned['std']/np.sqrt(err_binned['count']),
                 fmt='-o', color='#2b5c8f', ecolor='gray', capsize=4, lw=2, label='Turbidity Log Abs Error')
    ax4.set_title("D. Residual Inversion Error by Match Time Lag", fontsize=11, fontweight='bold')
    ax4.set_xlabel("Time Lag Bin (Hours)")
    ax4.set_ylabel("Mean Absolute Log Error (+/- SE)")
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = PROJECT_ROOT / 'results/figures/step8_temporal_sensitivity_diagnostics.png'
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved Step 8 diagnostic figure to: {fig_path}")

def main():
    df = load_and_preprocess_dataset()
    df_audit = audit_window_sensitivity(df)
    df_cv = evaluate_cv_across_windows(df)
    df_feat = evaluate_hours_since_overpass_feature(df)
    plot_step8_diagnostics(df)
    print("\nStep 8 execution complete!")

if __name__ == '__main__':
    main()
