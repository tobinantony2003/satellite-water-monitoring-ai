"""
Integrated Execution of Steps 8 to 12 for Satellite Water Monitoring AI
- Step 8: Temporal Match Window Sensitivity (Chlorophyll-a, CDOM, Turbidity)
- Step 9: Gitelson Red-Edge Features & Site Confounding Audit (Chlorophyll-a)
- Step 10: Optical Water Type (OWT) Clustering & Site Confounding Audit (Chlorophyll-a)
- Step 11: Outlier Winsorization Benchmark (Turbidity, Chlorophyll-a, CDOM)
- Step 12: Final Stage 3 Inversion Benchmarks across full model suite

All comparisons strictly follow the Standing Methodological Rules:
1. Canonical 5-fold spatial block CV grouped by site_no (no random splits)
2. Log10-space R2 / RMSE as primary metrics; physical-space R2 / RMSE as secondary
3. Unit-harmonized datasets for Chlorophyll-a and CDOM
4. OWT clustering checked for site_no / basin confounding before treating as optical signal
5. Per-fold results tracked + paired Wilcoxon signed-rank tests for every intervention claim
6. Clear, distinct CSV filenames
"""

import os
import sys
from pathlib import Path

# Suppress joblib loky CPU detection warnings on Windows 11
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
import torch

from src.inversion.models import get_inversion_models
from src.inversion.train_inversion import FEATURE_COLS
from src.inversion.metrics import compute_regression_metrics

# -----------------------------------------------------------------------------
# Canonical Fold Loaders
# -----------------------------------------------------------------------------
def load_canonical_folds(csv_path: Path):
    """Dynamically parses validated canonical fold assignments from mixed effects tables."""
    df_folds = pd.read_csv(csv_path)
    site_to_fold = {}
    for _, row in df_folds.iterrows():
        f = int(row['fold'])
        sites_str = str(row['held_out_sites'])
        for s in sites_str.replace('"', '').replace("'", "").split(','):
            s_clean = s.strip()
            if s_clean:
                try:
                    site_to_fold[int(s_clean)] = f
                except ValueError:
                    pass
                site_to_fold[s_clean] = f
    return site_to_fold

def get_turbidity_folds(df_turb: pd.DataFrame) -> dict:
    """Deterministic stratified 5-fold split by basin and site for turbidity."""
    site_basin = df_turb.drop_duplicates('site_no')[['site_no', 'basin']].sort_values(['basin', 'site_no'])
    site_to_fold = {}
    for b, grp in site_basin.groupby('basin'):
        sites_in_b = grp['site_no'].tolist()
        for idx, s in enumerate(sites_in_b):
            f = (idx % 5) + 1  # 1-indexed folds (1..5)
            site_to_fold[s] = f
            try:
                site_to_fold[int(s)] = f
            except (ValueError, TypeError):
                pass
            site_to_fold[str(s)] = f
    return site_to_fold

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
        'pearson_r': float(r_pearson),
        'pearson_p': float(p_pearson),
        'spearman_r': float(r_spearman),
        'spearman_p': float(p_spearman)
    }

def run_paired_wilcoxon(diffs):
    """Safely runs paired Wilcoxon signed-rank test across n=5 folds."""
    diffs = np.asarray(diffs)
    if np.all(diffs == 0):
        return 0.0, 1.0
    try:
        res = stats.wilcoxon(diffs, zero_method='wilcox')
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return float('nan'), float('nan')

# -----------------------------------------------------------------------------
# STEP 8: Temporal Match Window Sensitivity
# -----------------------------------------------------------------------------
def execute_step_8(df_dl: pd.DataFrame, fold_maps: dict):
    print("\n" + "=" * 80)
    print("STEP 8: TEMPORAL MATCH WINDOW SENSITIVITY & CONTINUOUS FEATURE AUDIT")
    print("=" * 80, flush=True)
    
    t_sc = pd.to_datetime(df_dl['scene_datetime_UTC'])
    t_sa = pd.to_datetime(df_dl['sample_datetime_UTC'])
    df_dl['dt_hours'] = (t_sa - t_sc).abs().dt.total_seconds() / 3600.0
    df_dl['dt_hours_signed'] = (t_sa - t_sc).dt.total_seconds() / 3600.0
    
    windows = [12.0, 3.0, 1.0]
    
    per_fold_records = []
    summary_records = []
    overpass_records = []
    
    for target in ['chlorophyll', 'cdom', 'turbidity']:
        df_t = df_dl[df_dl['target_type'] == target].copy()
        site_fold_map = fold_maps[target]
        df_t['fold'] = df_t['site_no'].map(site_fold_map)
        
        print(f"\n--- Analyzing {target.upper()} (Total N={len(df_t):,}, Sites={df_t['site_no'].nunique()}) ---", flush=True)
        
        baseline_fold_r2 = {}
        
        for w in windows:
            df_w = df_t[df_t['dt_hours'] <= w].copy().reset_index(drop=True)
            n_w = len(df_w)
            n_sites_w = df_w['site_no'].nunique()
            
            oof_pred_log = np.zeros(n_w)
            oof_y_log = np.zeros(n_w)
            oof_pred_raw = np.zeros(n_w)
            oof_y_raw = df_w['MeasurementValue'].values
            
            y_log = np.log10(np.clip(oof_y_raw, 1e-4, None))
            X = df_w[FEATURE_COLS].values
            
            fold_metrics_list = []
            
            for f in range(1, 6):
                val_mask = (df_w['fold'] == f).values
                train_mask = (~val_mask)
                
                if val_mask.sum() == 0 or train_mask.sum() == 0:
                    continue
                    
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X[train_mask])
                X_va = scaler.transform(X[val_mask])
                
                m = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
                m.fit(X_tr, y_log[train_mask])
                
                preds_val_log = m.predict(X_va)
                preds_val_raw = 10.0 ** np.clip(preds_val_log, -3.0, 4.0)
                
                oof_pred_log[val_mask] = preds_val_log
                oof_y_log[val_mask] = y_log[val_mask]
                oof_pred_raw[val_mask] = preds_val_raw
                
                f_r2_log = r2_score(y_log[val_mask], preds_val_log)
                f_rmse_log = np.sqrt(mean_squared_error(y_log[val_mask], preds_val_log))
                f_r2_phys = r2_score(oof_y_raw[val_mask], preds_val_raw)
                f_rmse_phys = np.sqrt(mean_squared_error(oof_y_raw[val_mask], preds_val_raw))
                
                fold_metrics_list.append({
                    'fold': f,
                    'r2_log': f_r2_log,
                    'rmse_log': f_rmse_log,
                    'r2_phys': f_r2_phys,
                    'rmse_phys': f_rmse_phys,
                    'n_val': val_mask.sum()
                })
                
                per_fold_records.append({
                    'target': target,
                    'condition': f'window_le_{int(w)}h' if w != 12.0 else 'window_12h_baseline',
                    'window_hours': w,
                    'fold': f,
                    'n_train': train_mask.sum(),
                    'n_val': val_mask.sum(),
                    'r2_log': f_r2_log,
                    'rmse_log': f_rmse_log,
                    'r2_phys': f_r2_phys,
                    'rmse_phys': f_rmse_phys
                })
                
                if w == 12.0:
                    baseline_fold_r2[f] = f_r2_log
                    
            macro_log_r2 = np.mean([m['r2_log'] for m in fold_metrics_list])
            macro_log_r2_std = np.std([m['r2_log'] for m in fold_metrics_list])
            pooled_log_r2 = r2_score(oof_y_log, oof_pred_log)
            pooled_log_rmse = np.sqrt(mean_squared_error(oof_y_log, oof_pred_log))
            pooled_phys_r2 = r2_score(oof_y_raw, oof_pred_raw)
            pooled_phys_rmse = np.sqrt(mean_squared_error(oof_y_raw, oof_pred_raw))
            
            if w != 12.0:
                diffs = [m['r2_log'] - baseline_fold_r2[m['fold']] for m in fold_metrics_list if m['fold'] in baseline_fold_r2]
                w_stat, p_val = run_paired_wilcoxon(diffs)
            else:
                w_stat, p_val = float('nan'), float('nan')
                
            summary_records.append({
                'target': target,
                'condition': f'window_le_{int(w)}h' if w != 12.0 else 'window_12h_baseline',
                'window_hours': w,
                'n_samples': n_w,
                'pct_samples_retained': round(100.0 * n_w / len(df_t), 2),
                'n_sites': n_sites_w,
                'macro_log_r2_mean': macro_log_r2,
                'macro_log_r2_std': macro_log_r2_std,
                'pooled_oof_log_r2': pooled_log_r2,
                'pooled_oof_log_rmse': pooled_log_rmse,
                'pooled_oof_phys_r2': pooled_phys_r2,
                'pooled_oof_phys_rmse': pooled_phys_rmse,
                'wilcoxon_stat_vs_baseline': w_stat,
                'wilcoxon_pval_vs_baseline': p_val
            })
            
            print(f"  Window <= {w:4.1f}h: N={n_w:5d} ({100.0*n_w/len(df_t):.1f}%), Sites={n_sites_w:2d} | "
                  f"Macro Log R2: {macro_log_r2:+.4f} +/- {macro_log_r2_std:.4f} | "
                  f"Pooled Log R2: {pooled_log_r2:+.4f} | Wilcoxon p={p_val:.4f}", flush=True)
                  
        # Continuous Feature Test on Full 12h Dataset: hours_since_overpass
        ext_cols = FEATURE_COLS + ['dt_hours', 'dt_hours_signed']
        X_ext = df_t[ext_cols].values
        y_log_full = np.log10(np.clip(df_t['MeasurementValue'].values, 1e-4, None))
        y_raw_full = df_t['MeasurementValue'].values
        
        oof_ext_log = np.zeros(len(df_t))
        oof_ext_raw = np.zeros(len(df_t))
        ext_fold_metrics = []
        feature_importances = []
        
        for f in range(1, 6):
            val_mask = (df_t['fold'] == f).values
            train_mask = (~val_mask)
            
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_ext[train_mask])
            X_va = scaler.transform(X_ext[val_mask])
            
            m_ext = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
            m_ext.fit(X_tr, y_log_full[train_mask])
            
            preds_va_log = m_ext.predict(X_va)
            preds_va_raw = 10.0 ** np.clip(preds_va_log, -3.0, 4.0)
            
            oof_ext_log[val_mask] = preds_va_log
            oof_ext_raw[val_mask] = preds_va_raw
            
            f_r2_log = r2_score(y_log_full[val_mask], preds_va_log)
            f_rmse_log = np.sqrt(mean_squared_error(y_log_full[val_mask], preds_va_log))
            f_r2_phys = r2_score(y_raw_full[val_mask], preds_va_raw)
            f_rmse_phys = np.sqrt(mean_squared_error(y_raw_full[val_mask], preds_va_raw))
            
            ext_fold_metrics.append({
                'fold': f,
                'r2_log': f_r2_log,
                'rmse_log': f_rmse_log,
                'r2_phys': f_r2_phys,
                'rmse_phys': f_rmse_phys
            })
            feature_importances.append(m_ext.feature_importances_)
            
            per_fold_records.append({
                'target': target,
                'condition': 'window_12h_with_dt_feature',
                'window_hours': 12.0,
                'fold': f,
                'n_train': train_mask.sum(),
                'n_val': val_mask.sum(),
                'r2_log': f_r2_log,
                'rmse_log': f_rmse_log,
                'r2_phys': f_r2_phys,
                'rmse_phys': f_rmse_phys
            })
            
        ext_macro_log_r2 = np.mean([m['r2_log'] for m in ext_fold_metrics])
        ext_macro_log_r2_std = np.std([m['r2_log'] for m in ext_fold_metrics])
        ext_pooled_log_r2 = r2_score(y_log_full, oof_ext_log)
        ext_pooled_log_rmse = np.sqrt(mean_squared_error(y_log_full, oof_ext_log))
        ext_pooled_phys_r2 = r2_score(y_raw_full, oof_ext_raw)
        ext_pooled_phys_rmse = np.sqrt(mean_squared_error(y_raw_full, oof_ext_raw))
        
        diffs_ext = [m['r2_log'] - baseline_fold_r2[m['fold']] for m in ext_fold_metrics]
        w_ext, p_ext = run_paired_wilcoxon(diffs_ext)
        
        summary_records.append({
            'target': target,
            'condition': 'window_12h_with_dt_feature',
            'window_hours': 12.0,
            'n_samples': len(df_t),
            'pct_samples_retained': 100.0,
            'n_sites': df_t['site_no'].nunique(),
            'macro_log_r2_mean': ext_macro_log_r2,
            'macro_log_r2_std': ext_macro_log_r2_std,
            'pooled_oof_log_r2': ext_pooled_log_r2,
            'pooled_oof_log_rmse': ext_pooled_log_rmse,
            'pooled_oof_phys_r2': ext_pooled_phys_r2,
            'pooled_oof_phys_rmse': ext_pooled_phys_rmse,
            'wilcoxon_stat_vs_baseline': w_ext,
            'wilcoxon_pval_vs_baseline': p_ext
        })
        
        mean_fi = np.mean(feature_importances, axis=0)
        fi_dict = dict(zip(ext_cols, mean_fi))
        
        res_log = np.abs(y_log_full - oof_ext_log)
        r_dt_y, p_dt_y = stats.pearsonr(df_t['dt_hours'], y_log_full)
        r_dt_res, p_dt_res = stats.pearsonr(df_t['dt_hours'], res_log)
        
        overpass_records.append({
            'target': target,
            'corr_dt_hours_vs_target': r_dt_y,
            'corr_dt_hours_vs_abs_residual': r_dt_res,
            'dt_hours_feature_importance': fi_dict['dt_hours'],
            'dt_hours_signed_feature_importance': fi_dict['dt_hours_signed'],
            'baseline_macro_log_r2': summary_records[-2]['macro_log_r2_mean'],
            'extended_macro_log_r2': ext_macro_log_r2,
            'delta_macro_log_r2': ext_macro_log_r2 - summary_records[-2]['macro_log_r2_mean'],
            'wilcoxon_pval': p_ext
        })
        
        print(f"  With dt_hours feature: Macro Log R2: {ext_macro_log_r2:+.4f} +/- {ext_macro_log_r2_std:.4f} | "
              f"Pooled Log R2: {ext_pooled_log_r2:+.4f} | Wilcoxon p={p_ext:.4f}", flush=True)

    df_per_fold = pd.DataFrame(per_fold_records)
    df_summary = pd.DataFrame(summary_records)
    df_overpass = pd.DataFrame(overpass_records)
    
    tables_dir = PROJECT_ROOT / 'results/tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    df_per_fold.to_csv(tables_dir / 'step8_temporal_window_inversion_per_fold.csv', index=False)
    df_summary.to_csv(tables_dir / 'step8_temporal_window_inversion_summary.csv', index=False)
    df_overpass.to_csv(tables_dir / 'step8_hours_since_overpass_feature_evaluation.csv', index=False)
    
    print("\n[OK] Step 8 Results successfully saved to:")
    print("  - results/tables/step8_temporal_window_inversion_per_fold.csv")
    print("  - results/tables/step8_temporal_window_inversion_summary.csv")
    print("  - results/tables/step8_hours_since_overpass_feature_evaluation.csv")
    return df_summary

# -----------------------------------------------------------------------------
# STEP 9: Gitelson Red-Edge Features for Chlorophyll-a
# -----------------------------------------------------------------------------
def execute_step_9(df_dl: pd.DataFrame, chla_folds: dict):
    print("\n" + "=" * 80)
    print("STEP 9: GITELSON RED-EDGE FEATURES & PARTIAL CORRELATION AUDIT")
    print("=" * 80, flush=True)
    
    df_chla = df_dl[df_dl['target_type'] == 'chlorophyll'].copy().reset_index(drop=True)
    df_chla['fold'] = df_chla['site_no'].map(chla_folds)
    
    df_chla['gitelson_2band'] = df_chla['B05_center_refl'] / (df_chla['B04_center_refl'] + 1e-6)
    df_chla['gitelson_3band'] = (1.0 / (df_chla['B04_center_refl'] + 1e-6) - 1.0 / (df_chla['B05_center_refl'] + 1e-6)) * df_chla['B06_center_refl']
    
    y_raw = df_chla['MeasurementValue'].values
    y_log = np.log10(np.clip(y_raw, 1e-4, None))
    sites = df_chla['site_no'].astype(str).values
    basins = df_chla['basin'].astype(str).values
    
    features_to_audit = [
        'B04_center_refl', 'B05_center_refl', 'B06_center_refl',
        'ndci', 'gitelson_2band', 'gitelson_3band'
    ]
    
    corr_records = []
    print("\nFeature Correlation Audit against Log10(Chlorophyll-a):")
    print(f"{'Feature':20s} | {'Raw Pearson r':14s} | {'Site-Controlled r':18s} | {'Basin-Controlled r':18s}")
    print("-" * 75)
    
    for feat in features_to_audit:
        x = df_chla[feat].values
        r_raw, p_raw = stats.pearsonr(x, y_log)
        site_ctrl = compute_partial_corr(x, y_log, sites)
        basin_ctrl = compute_partial_corr(x, y_log, basins)
        
        corr_records.append({
            'feature': feat,
            'raw_pearson_r': r_raw,
            'raw_pearson_p': p_raw,
            'site_controlled_r': site_ctrl['pearson_r'],
            'site_controlled_p': site_ctrl['pearson_p'],
            'basin_controlled_r': basin_ctrl['pearson_r'],
            'basin_controlled_p': basin_ctrl['pearson_p']
        })
        print(f"{feat:20s} | {r_raw:+.4f} (p={p_raw:.2e}) | {site_ctrl['pearson_r']:+.4f} (p={site_ctrl['pearson_p']:.2e}) | {basin_ctrl['pearson_r']:+.4f} (p={basin_ctrl['pearson_p']:.2e})")
        
    df_corr = pd.DataFrame(corr_records)
    tables_dir = PROJECT_ROOT / 'results/tables'
    df_corr.to_csv(tables_dir / 'step9_gitelson_correlation_audit.csv', index=False)
    
    feature_sets = {
        'Baseline_17_Features': FEATURE_COLS,
        'Baseline_plus_Gitelson_2band': FEATURE_COLS + ['gitelson_2band'],
        'Baseline_plus_Gitelson_3band': FEATURE_COLS + ['gitelson_3band'],
        'Baseline_plus_Both_Gitelson': FEATURE_COLS + ['gitelson_2band', 'gitelson_3band'],
        'Gitelson_2band_Only': ['gitelson_2band'],
        'Gitelson_3band_Only': ['gitelson_3band']
    }
    
    cv_records = []
    baseline_folds = {}
    
    for fset_name, fcols in feature_sets.items():
        X = df_chla[fcols].values
        oof_pred_log = np.zeros(len(df_chla))
        fold_metrics = []
        
        for f in range(1, 6):
            val_mask = (df_chla['fold'] == f).values
            train_mask = (~val_mask)
            
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_mask])
            X_va = scaler.transform(X[val_mask])
            
            if len(fcols) == 1:
                m = Ridge(alpha=1.0)
            else:
                m = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
            m.fit(X_tr, y_log[train_mask])
            
            preds_va = m.predict(X_va)
            oof_pred_log[val_mask] = preds_va
            f_r2 = r2_score(y_log[val_mask], preds_va)
            f_rmse = np.sqrt(mean_squared_error(y_log[val_mask], preds_va))
            fold_metrics.append((f, f_r2, f_rmse))
            
            if fset_name == 'Baseline_17_Features':
                baseline_folds[f] = f_r2
                
        macro_r2 = np.mean([fm[1] for fm in fold_metrics])
        macro_r2_std = np.std([fm[1] for fm in fold_metrics])
        pooled_r2 = r2_score(y_log, oof_pred_log)
        pooled_rmse = np.sqrt(mean_squared_error(y_log, oof_pred_log))
        
        oof_pred_raw = 10.0 ** np.clip(oof_pred_log, -3.0, 4.0)
        pooled_phys_r2 = r2_score(y_raw, oof_pred_raw)
        pooled_phys_rmse = np.sqrt(mean_squared_error(y_raw, oof_pred_raw))
        
        diffs = [fm[1] - baseline_folds[fm[0]] for fm in fold_metrics]
        w_stat, p_val = run_paired_wilcoxon(diffs)
        
        cv_records.append({
            'feature_set': fset_name,
            'num_features': len(fcols),
            'macro_log_r2_mean': macro_r2,
            'macro_log_r2_std': macro_r2_std,
            'pooled_oof_log_r2': pooled_r2,
            'pooled_oof_log_rmse': pooled_rmse,
            'pooled_oof_phys_r2': pooled_phys_r2,
            'pooled_oof_phys_rmse': pooled_phys_rmse,
            'fold1_log_r2': fold_metrics[0][1],
            'fold2_log_r2': fold_metrics[1][1],
            'fold3_log_r2': fold_metrics[2][1],
            'fold4_log_r2': fold_metrics[3][1],
            'fold5_log_r2': fold_metrics[4][1],
            'wilcoxon_stat_vs_baseline': w_stat,
            'wilcoxon_pval_vs_baseline': p_val
        })
        print(f"  {fset_name:28s} -> Macro Log R2: {macro_r2:+.4f} +/- {macro_r2_std:.4f} | "
              f"Pooled Log R2: {pooled_r2:+.4f} | Wilcoxon p={p_val:.4f}", flush=True)
              
    df_cv = pd.DataFrame(cv_records)
    df_cv.to_csv(tables_dir / 'step9_gitelson_inversion_cv.csv', index=False)
    print("\n[OK] Step 9 Results saved to:")
    print("  - results/tables/step9_gitelson_correlation_audit.csv")
    print("  - results/tables/step9_gitelson_inversion_cv.csv")
    return df_corr, df_cv

# -----------------------------------------------------------------------------
# STEP 10: Optical Water Type (OWT) Clustering & Confounding Audit
# -----------------------------------------------------------------------------
def execute_step_10(df_dl: pd.DataFrame, chla_folds: dict):
    print("\n" + "=" * 80)
    print("STEP 10: OPTICAL WATER TYPE (OWT) CLUSTERING & SITE CONFOUNDING AUDIT")
    print("=" * 80, flush=True)
    
    df_chla = df_dl[df_dl['target_type'] == 'chlorophyll'].copy().reset_index(drop=True)
    df_chla['fold'] = df_chla['site_no'].map(chla_folds)
    
    spectral_bands = [
        'B01_center_refl', 'B02_center_refl', 'B03_center_refl', 'B04_center_refl',
        'B05_center_refl', 'B06_center_refl', 'B07_center_refl', 'B08_center_refl',
        'B8A_center_refl', 'B11_center_refl', 'B12_center_refl'
    ]
    
    # Area normalization (integral / sum across bands) to capture spectral shape rather than magnitude
    X_spec = df_chla[spectral_bands].values
    band_sums = X_spec.sum(axis=1, keepdims=True)
    band_sums = np.where(band_sums == 0, 1e-6, band_sums)
    X_norm = X_spec / band_sums
    
    k_range = list(range(2, 9))
    cluster_metrics = []
    
    print("\nEvaluating OWT Cluster Counts (k = 2..8):")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_norm)
        sil = silhouette_score(X_norm, labels, sample_size=min(3000, len(X_norm)), random_state=42)
        db = davies_bouldin_score(X_norm, labels)
        
        cluster_metrics.append({
            'k': k,
            'silhouette_score': sil,
            'davies_bouldin_index': db,
            'inertia': km.inertia_
        })
        print(f"  k = {k}: Silhouette Score = {sil:.4f}, Davies-Bouldin = {db:.4f}, Inertia = {km.inertia_:.1f}")
        
    df_k_metrics = pd.DataFrame(cluster_metrics)
    tables_dir = PROJECT_ROOT / 'results/tables'
    df_k_metrics.to_csv(tables_dir / 'step10_owt_cluster_metrics.csv', index=False)
    
    best_k = int(df_k_metrics.sort_values(by='silhouette_score', ascending=False).iloc[0]['k'])
    print(f"\nOptimal Cluster Count selected: k = {best_k}")
    
    km_opt = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df_chla['owt_cluster'] = km_opt.fit_predict(X_norm)
    
    print("\n--- Confounding Check: OWT Cluster Membership vs Station Identity ---")
    cont_site = pd.crosstab(df_chla['site_no'], df_chla['owt_cluster'], margins=True)
    cont_basin = pd.crosstab(df_chla['basin'], df_chla['owt_cluster'], margins=True)
    
    cont_site.to_csv(tables_dir / 'step10_owt_contingency_site.csv')
    cont_basin.to_csv(tables_dir / 'step10_owt_contingency_basin.csv')
    
    chi2_site, p_site, dof_site, _ = stats.chi2_contingency(pd.crosstab(df_chla['site_no'], df_chla['owt_cluster']))
    n_total = len(df_chla)
    cramers_v_site = np.sqrt(chi2_site / (n_total * (min(cont_site.shape) - 1)))
    
    chi2_basin, p_basin, dof_basin, _ = stats.chi2_contingency(pd.crosstab(df_chla['basin'], df_chla['owt_cluster']))
    cramers_v_basin = np.sqrt(chi2_basin / (n_total * (min(cont_basin.shape) - 1)))
    
    print(f"Cluster vs Site_No:  Chi2 = {chi2_site:.1f} (p = {p_site:.2e}), Cramér's V = {cramers_v_site:.4f}")
    print(f"Cluster vs Basin:    Chi2 = {chi2_basin:.1f} (p = {p_basin:.2e}), Cramér's V = {cramers_v_basin:.4f}")
    
    site_purity = []
    for site, grp in df_chla.groupby('site_no'):
        counts = grp['owt_cluster'].value_counts()
        top_pct = 100.0 * counts.iloc[0] / len(grp)
        site_purity.append(top_pct)
    mean_site_purity = np.mean(site_purity)
    print(f"Average Site Monoculture: {mean_site_purity:.1f}% of observations per site fall into one dominant cluster.")
    
    y_raw = df_chla['MeasurementValue'].values
    y_log = np.log10(np.clip(y_raw, 1e-4, None))
    X_feat = df_chla[FEATURE_COLS].values
    
    oof_pooled_log = np.zeros(len(df_chla))
    pooled_fold_r2 = []
    
    for f in range(1, 6):
        val_mask = (df_chla['fold'] == f).values
        train_mask = (~val_mask)
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_feat[train_mask])
        X_va = scaler.transform(X_feat[val_mask])
        
        m = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        m.fit(X_tr, y_log[train_mask])
        
        preds_va = m.predict(X_va)
        oof_pooled_log[val_mask] = preds_va
        pooled_fold_r2.append(r2_score(y_log[val_mask], preds_va))
        
    df_cluster_dummies = pd.get_dummies(df_chla['owt_cluster'], prefix='owt', drop_first=True)
    X_feat_cluster = np.hstack([X_feat, df_cluster_dummies.values.astype(float)])
    
    oof_cluster_log = np.zeros(len(df_chla))
    cluster_fold_r2 = []
    
    for f in range(1, 6):
        val_mask = (df_chla['fold'] == f).values
        train_mask = (~val_mask)
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_feat_cluster[train_mask])
        X_va = scaler.transform(X_feat_cluster[val_mask])
        
        m_c = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        m_c.fit(X_tr, y_log[train_mask])
        
        preds_va = m_c.predict(X_va)
        oof_cluster_log[val_mask] = preds_va
        cluster_fold_r2.append(r2_score(y_log[val_mask], preds_va))
        
    diffs = [c - p for c, p in zip(cluster_fold_r2, pooled_fold_r2)]
    w_stat, p_val = run_paired_wilcoxon(diffs)
    
    model_records = [
        {
            'model_configuration': 'Pooled_Baseline',
            'macro_log_r2_mean': np.mean(pooled_fold_r2),
            'macro_log_r2_std': np.std(pooled_fold_r2),
            'pooled_oof_log_r2': r2_score(y_log, oof_pooled_log),
            'pooled_oof_log_rmse': np.sqrt(mean_squared_error(y_log, oof_pooled_log)),
            'wilcoxon_pval_vs_pooled': 1.0
        },
        {
            'model_configuration': 'Cluster_Conditioned_OWT',
            'macro_log_r2_mean': np.mean(cluster_fold_r2),
            'macro_log_r2_std': np.std(cluster_fold_r2),
            'pooled_oof_log_r2': r2_score(y_log, oof_cluster_log),
            'pooled_oof_log_rmse': np.sqrt(mean_squared_error(y_log, oof_cluster_log)),
            'wilcoxon_pval_vs_pooled': p_val
        }
    ]
    
    df_models = pd.DataFrame(model_records)
    df_models.to_csv(tables_dir / 'step10_owt_model_performance.csv', index=False)
    
    print(f"\nOWT Model Evaluation:")
    print(f"  Pooled Baseline:    Macro Log R2 = {np.mean(pooled_fold_r2):+.4f} | Pooled Log R2 = {r2_score(y_log, oof_pooled_log):+.4f}")
    print(f"  Cluster-Conditioned: Macro Log R2 = {np.mean(cluster_fold_r2):+.4f} | Pooled Log R2 = {r2_score(y_log, oof_cluster_log):+.4f}")
    print(f"  Paired Wilcoxon test p-value: p = {p_val:.4f}")
    
    return df_k_metrics, df_models

# -----------------------------------------------------------------------------
# STEP 11: Outlier Winsorization Benchmark
# -----------------------------------------------------------------------------
def execute_step_11(df_dl: pd.DataFrame, fold_maps: dict):
    print("\n" + "=" * 80)
    print("STEP 11: OUTLIER WINSORIZATION BENCHMARK (1st & 99th PERCENTILES)")
    print("=" * 80, flush=True)
    
    WINSOR_BOUNDS = {
        'turbidity': (0.4, 162.0),
        'chlorophyll': (0.1, 76.914),
        'cdom': (2.64, 66.45)
    }
    
    per_fold_records = []
    summary_records = []
    
    for target in ['turbidity', 'chlorophyll', 'cdom']:
        df_t = df_dl[df_dl['target_type'] == target].copy().reset_index(drop=True)
        df_t['fold'] = df_t['site_no'].map(fold_maps[target])
        
        low_b, high_b = WINSOR_BOUNDS[target]
        print(f"\n--- Target: {target.upper()} (Bounds: [{low_b:.2f}, {high_b:.2f}]) ---", flush=True)
        
        X = df_t[FEATURE_COLS].values
        y_raw = df_t['MeasurementValue'].values
        y_log_raw = np.log10(np.clip(y_raw, 1e-4, None))
        
        # Condition 1: Without Winsorization (Baseline)
        oof_raw_log = np.zeros(len(df_t))
        raw_fold_metrics = []
        
        for f in range(1, 6):
            val_mask = (df_t['fold'] == f).values
            train_mask = (~val_mask)
            
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_mask])
            X_va = scaler.transform(X[val_mask])
            
            m_raw = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
            m_raw.fit(X_tr, y_log_raw[train_mask])
            
            preds_va = m_raw.predict(X_va)
            oof_raw_log[val_mask] = preds_va
            
            f_r2_log = r2_score(y_log_raw[val_mask], preds_va)
            f_rmse_log = np.sqrt(mean_squared_error(y_log_raw[val_mask], preds_va))
            raw_fold_metrics.append((f, f_r2_log, f_rmse_log))
            
            per_fold_records.append({
                'target': target,
                'condition': 'without_winsorization',
                'fold': f,
                'r2_log': f_r2_log,
                'rmse_log': f_rmse_log
            })
            
        # Condition 2: With Winsorization on Training Target
        y_raw_wins = np.clip(y_raw, low_b, high_b)
        y_log_wins = np.log10(y_raw_wins)
        
        oof_win_log = np.zeros(len(df_t))
        win_fold_metrics = []
        
        for f in range(1, 6):
            val_mask = (df_t['fold'] == f).values
            train_mask = (~val_mask)
            
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_mask])
            X_va = scaler.transform(X[val_mask])
            
            m_win = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
            m_win.fit(X_tr, y_log_wins[train_mask])
            
            preds_va = m_win.predict(X_va)
            oof_win_log[val_mask] = preds_va
            
            f_r2_log = r2_score(y_log_raw[val_mask], preds_va)
            f_rmse_log = np.sqrt(mean_squared_error(y_log_raw[val_mask], preds_va))
            win_fold_metrics.append((f, f_r2_log, f_rmse_log))
            
            per_fold_records.append({
                'target': target,
                'condition': 'with_winsorization',
                'fold': f,
                'r2_log': f_r2_log,
                'rmse_log': f_rmse_log
            })
            
        raw_macro_r2 = np.mean([m[1] for m in raw_fold_metrics])
        raw_macro_std = np.std([m[1] for m in raw_fold_metrics])
        raw_pooled_r2 = r2_score(y_log_raw, oof_raw_log)
        raw_pooled_rmse = np.sqrt(mean_squared_error(y_log_raw, oof_raw_log))
        
        win_macro_r2 = np.mean([m[1] for m in win_fold_metrics])
        win_macro_std = np.std([m[1] for m in win_fold_metrics])
        win_pooled_r2 = r2_score(y_log_raw, oof_win_log)
        win_pooled_rmse = np.sqrt(mean_squared_error(y_log_raw, oof_win_log))
        
        diffs = [w[1] - r[1] for w, r in zip(win_fold_metrics, raw_fold_metrics)]
        w_stat, p_val = run_paired_wilcoxon(diffs)
        
        summary_records.append({
            'target': target,
            'condition': 'without_winsorization',
            'macro_log_r2_mean': raw_macro_r2,
            'macro_log_r2_std': raw_macro_std,
            'pooled_oof_log_r2': raw_pooled_r2,
            'pooled_oof_log_rmse': raw_pooled_rmse,
            'wilcoxon_pval_vs_raw': 1.0
        })
        summary_records.append({
            'target': target,
            'condition': 'with_winsorization',
            'macro_log_r2_mean': win_macro_r2,
            'macro_log_r2_std': win_macro_std,
            'pooled_oof_log_r2': win_pooled_r2,
            'pooled_oof_log_rmse': win_pooled_rmse,
            'delta_macro_log_r2': win_macro_r2 - raw_macro_r2,
            'wilcoxon_stat_vs_raw': w_stat,
            'wilcoxon_pval_vs_raw': p_val
        })
        
        print(f"  Raw:          Macro Log R2 = {raw_macro_r2:+.4f} +/- {raw_macro_std:.4f} | Pooled Log R2 = {raw_pooled_r2:+.4f}")
        print(f"  Winsorized:   Macro Log R2 = {win_macro_r2:+.4f} +/- {win_macro_std:.4f} | Pooled Log R2 = {win_pooled_r2:+.4f}")
        print(f"  Delta Macro R2: {win_macro_r2 - raw_macro_r2:+.4f} | Paired Wilcoxon p-value = {p_val:.4f}", flush=True)

    df_pf = pd.DataFrame(per_fold_records)
    df_sm = pd.DataFrame(summary_records)
    tables_dir = PROJECT_ROOT / 'results/tables'
    df_pf.to_csv(tables_dir / 'step11_winsorization_per_fold.csv', index=False)
    df_sm.to_csv(tables_dir / 'step11_winsorization_summary.csv', index=False)
    
    print("\n[OK] Step 11 Results saved to:")
    print("  - results/tables/step11_winsorization_per_fold.csv")
    print("  - results/tables/step11_winsorization_summary.csv")
    return df_sm

# -----------------------------------------------------------------------------
# STEP 12: Final Stage 3 Benchmark
# -----------------------------------------------------------------------------
def execute_step_12(df_dl: pd.DataFrame, fold_maps: dict, interventions_to_include: dict):
    print("\n" + "=" * 80)
    print("STEP 12: FINAL STAGE 3 BENCHMARK ACROSS FULL MODEL SUITE")
    print("=" * 80, flush=True)
    
    model_names = ['Empirical_Baseline', 'Random_Forest', 'Gradient_Boosting', 'Deep_Spectral_MLP']
    consolidated_benchmarks = []
    
    for target in ['turbidity', 'chlorophyll', 'cdom']:
        df_t = df_dl[df_dl['target_type'] == target].copy().reset_index(drop=True)
        df_t['fold'] = df_t['site_no'].map(fold_maps[target])
        
        print(f"\n=======================================================", flush=True)
        print(f"STAGE 3 BENCHMARK: {target.upper()} (N={len(df_t):,}, Sites={df_t['site_no'].nunique()})", flush=True)
        print(f"=======================================================", flush=True)
        
        include_winsor = interventions_to_include.get(target, {}).get('winsorization', False)
        
        X_base = df_t[FEATURE_COLS].values
        y_raw = df_t['MeasurementValue'].values
        
        if include_winsor:
            low_b, high_b = {
                'turbidity': (0.4, 162.0),
                'chlorophyll': (0.1, 76.914),
                'cdom': (2.64, 66.45)
            }[target]
            y_train_target = np.clip(y_raw, low_b, high_b)
        else:
            y_train_target = y_raw
            
        y_log_train = np.log10(np.clip(y_train_target, 1e-4, None))
        y_log_true = np.log10(np.clip(y_raw, 1e-4, None))
        
        target_results = []
        
        for m_name in model_names:
            print(f"Evaluating {m_name:20s}...", flush=True)
            
            oof_pred_log = np.zeros(len(df_t))
            fold_metrics = []
            
            for f in range(1, 6):
                val_mask = (df_t['fold'] == f).values
                train_mask = (~val_mask)
                
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_base[train_mask])
                X_va = scaler.transform(X_base[val_mask])
                
                if m_name == 'Empirical_Baseline':
                    if target == 'chlorophyll':
                        feat_idx = [FEATURE_COLS.index('ndci')]
                    elif target == 'turbidity':
                        feat_idx = [FEATURE_COLS.index('turbidity_ratio')]
                    else:
                        feat_idx = [FEATURE_COLS.index('cdom_ratio')]
                    clf = get_inversion_models(in_features=1)['Empirical_Baseline']
                    clf.fit(X_base[train_mask][:, feat_idx], y_log_train[train_mask])
                    preds_va = clf.predict(X_base[val_mask][:, feat_idx])
                elif m_name == 'Random_Forest':
                    clf = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=4, n_jobs=-1, random_state=42)
                    clf.fit(X_tr, y_log_train[train_mask])
                    preds_va = clf.predict(X_va)
                elif m_name == 'Gradient_Boosting':
                    clf = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
                    clf.fit(X_tr, y_log_train[train_mask])
                    preds_va = clf.predict(X_va)
                elif m_name == 'Deep_Spectral_MLP':
                    from src.inversion.models import PyTorchMLPRegressor
                    clf = PyTorchMLPRegressor(in_features=len(FEATURE_COLS), epochs=35, batch_size=256)
                    clf.fit(X_tr, y_log_train[train_mask])
                    preds_va = clf.predict(X_va)
                    
                oof_pred_log[val_mask] = preds_va
                
                f_r2 = r2_score(y_log_true[val_mask], preds_va)
                f_rmse = np.sqrt(mean_squared_error(y_log_true[val_mask], preds_va))
                f_mae = mean_absolute_error(y_log_true[val_mask], preds_va)
                fold_metrics.append((f, f_r2, f_rmse, f_mae))
                
            macro_r2 = np.mean([fm[1] for fm in fold_metrics])
            macro_r2_std = np.std([fm[1] for fm in fold_metrics])
            macro_rmse = np.mean([fm[2] for fm in fold_metrics])
            macro_rmse_std = np.std([fm[2] for fm in fold_metrics])
            
            pooled_r2 = r2_score(y_log_true, oof_pred_log)
            pooled_rmse = np.sqrt(mean_squared_error(y_log_true, oof_pred_log))
            
            oof_pred_raw = 10.0 ** np.clip(oof_pred_log, -3.0, 4.0)
            pooled_phys_r2 = r2_score(y_raw, oof_pred_raw)
            pooled_phys_rmse = np.sqrt(mean_squared_error(y_raw, oof_pred_raw))
            pooled_phys_mae = mean_absolute_error(y_raw, oof_pred_raw)
            
            res_row = {
                'target': target,
                'model': m_name,
                'n_samples': len(df_t),
                'n_sites': df_t['site_no'].nunique(),
                'macro_log_r2_mean': macro_r2,
                'macro_log_r2_std': macro_r2_std,
                'macro_log_rmse_mean': macro_rmse,
                'macro_log_rmse_std': macro_rmse_std,
                'pooled_oof_log_r2': pooled_r2,
                'pooled_oof_log_rmse': pooled_rmse,
                'pooled_oof_phys_r2': pooled_phys_r2,
                'pooled_oof_phys_rmse': pooled_phys_rmse,
                'pooled_oof_phys_mae': pooled_phys_mae,
                'fold1_log_r2': fold_metrics[0][1],
                'fold2_log_r2': fold_metrics[1][1],
                'fold3_log_r2': fold_metrics[2][1],
                'fold4_log_r2': fold_metrics[3][1],
                'fold5_log_r2': fold_metrics[4][1]
            }
            target_results.append(res_row)
            consolidated_benchmarks.append(res_row)
            
            print(f"  {m_name:20s} -> Macro Log R2: {macro_r2:+.4f} +/- {macro_r2_std:.4f} | "
                  f"Log RMSE: {macro_rmse:.4f} | Pooled Phys RMSE: {pooled_phys_rmse:.2f}", flush=True)
                  
        df_target_res = pd.DataFrame(target_results)
        best_idx = df_target_res['macro_log_r2_mean'].idxmax()
        df_target_res['is_best_model'] = False
        df_target_res.loc[best_idx, 'is_best_model'] = True
        
        tables_dir = PROJECT_ROOT / 'results/tables'
        df_target_res.to_csv(tables_dir / f'final_stage3_benchmark_{target}.csv', index=False)
        print(f"Best model for {target.upper()}: {df_target_res.loc[best_idx, 'model']} (Macro Log R2: {df_target_res.loc[best_idx, 'macro_log_r2_mean']:+.4f})")
        
    df_cons = pd.DataFrame(consolidated_benchmarks)
    df_cons.to_csv(tables_dir / 'final_stage3_benchmark_consolidated.csv', index=False)
    print("\n[OK] Step 12 Benchmarks saved to results/tables/final_stage3_benchmark_*.csv")
    return df_cons

def main():
    print("=" * 80)
    print("SATELLITE WATER MONITORING AI: EXECUTION OF STEPS 8 - 12")
    print("=" * 80, flush=True)
    
    data_path = PROJECT_ROOT / 'data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv'
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}")
        
    print(f"Loading DL-masked dataset from {data_path}...", flush=True)
    df_dl = pd.read_csv(data_path, low_memory=False)
    print(f"Loaded {len(df_dl):,} records across {df_dl['site_no'].nunique()} stations.", flush=True)
    
    # Load canonical folds for all 3 targets
    chla_folds = load_canonical_folds(PROJECT_ROOT / 'results/tables/chlorophyll_mixed_effects_per_fold.csv')
    cdom_folds = load_canonical_folds(PROJECT_ROOT / 'results/tables/cdom_mixed_effects_per_fold.csv')
    turb_folds = get_turbidity_folds(df_dl[df_dl['target_type'] == 'turbidity'])
    
    fold_maps = {
        'chlorophyll': chla_folds,
        'cdom': cdom_folds,
        'turbidity': turb_folds
    }
    
    # 1. Step 8: Temporal match window sensitivity
    df_s8 = execute_step_8(df_dl, fold_maps)
    
    # 2. Step 9: Gitelson red-edge features
    df_s9_corr, df_s9_cv = execute_step_9(df_dl, chla_folds)
    
    # 3. Step 10: Optical Water Type clustering
    df_s10_k, df_s10_m = execute_step_10(df_dl, chla_folds)
    
    # 4. Step 11: Outlier winsorization
    df_s11 = execute_step_11(df_dl, fold_maps)
    
    # Determine statistically significant interventions to include in Step 12
    interventions = {
        'turbidity': {'winsorization': False},
        'chlorophyll': {'winsorization': False},
        'cdom': {'winsorization': False}
    }
    
    for t in ['turbidity', 'chlorophyll', 'cdom']:
        sub_w = df_s11[(df_s11['target'] == t) & (df_s11['condition'] == 'with_winsorization')]
        if len(sub_w) > 0 and sub_w.iloc[0]['wilcoxon_pval_vs_raw'] < 0.05 and sub_w.iloc[0]['delta_macro_log_r2'] > 0:
            print(f"Including Winsorization for {t} in Step 12 (p={sub_w.iloc[0]['wilcoxon_pval_vs_raw']:.4f})")
            interventions[t]['winsorization'] = True
        else:
            p_val = sub_w.iloc[0]['wilcoxon_pval_vs_raw'] if len(sub_w) > 0 else 1.0
            print(f"Discarding Winsorization for {t} in Step 12 (p={p_val:.4f} >= 0.05 or delta <= 0)")
            
    # 5. Step 12: Final Stage 3 Benchmark
    df_s12 = execute_step_12(df_dl, fold_maps, interventions)
    
    print("\n" + "=" * 80)
    print("ALL STEPS 8 - 12 EXECUTED AND EXPORTED SUCCESSFULLY!")
    print("=" * 80, flush=True)

if __name__ == '__main__':
    main()
