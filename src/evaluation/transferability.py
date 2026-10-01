import os
import sys
from pathlib import Path

# Suppress joblib loky wmic CPU detection warning on modern Windows 11
if 'LOKY_MAX_CPU_COUNT' not in os.environ:
    os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count() or 4)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple

from src.inversion.models import get_inversion_models
from src.inversion.spatial_cv import get_spatial_block_cv
from src.inversion.metrics import compute_regression_metrics
from src.inversion.train_inversion import FEATURE_COLS

def run_experiment_b_transferability(
    df_dl: pd.DataFrame,
    target_type: str = 'turbidity',
    model_name: str = 'Gradient_Boosting',
    results_dir: str = 'results'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Experiment B: Cross-Basin Portability & Transferability.
    1. Within-Basin: Train and test within each basin using spatial block CV (Upper bound).
    2. Leave-One-Basin-Out (LOBO): Train on 4 basins, evaluate zero-shot transfer on the 5th basin.
    3. Pooled: Train on all 5 basins jointly with spatial block CV.
    """
    print(f"\n=======================================================", flush=True)
    print(f"EXPERIMENT B: Cross-Basin Transferability for {target_type.upper()}", flush=True)
    print(f"=======================================================", flush=True)
    
    df_target = df_dl[df_dl['target_type'] == target_type].copy()
    basins = sorted(df_target['basin'].unique())
    print(f"Active basins ({len(basins)}): {basins}", flush=True)
    
    within_results = []
    lobo_results = []
    
    # 1. Within-Basin Train/Test Benchmark
    print("\n--- Phase 1: Within-Basin Benchmark (Upper Bound) ---", flush=True)
    for b in basins:
        df_b = df_target[df_target['basin'] == b].copy()
        if len(df_b) < 30 or df_b['site_no'].nunique() < 2:
            print(f"Skipping basin {b} due to insufficient gauge sites.", flush=True)
            continue
            
        splits = list(get_spatial_block_cv(df_b, group_col='site_no', n_splits=min(5, df_b['site_no'].nunique())))
        oof_preds_log = np.zeros(len(df_b))
        
        X = df_b[FEATURE_COLS].values
        y_log = np.log10(np.clip(df_b['MeasurementValue'].values, 1e-4, None))
        
        for train_idx, val_idx in splits:
            X_tr, y_tr = X[train_idx], y_log[train_idx]
            X_val = X[val_idx]
            
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)
            
            m = get_inversion_models(in_features=len(FEATURE_COLS))[model_name]
            m.fit(X_tr_s, y_tr)
            oof_preds_log[val_idx] = m.predict(X_val_s)
            
        oof_preds_raw = 10.0 ** oof_preds_log
        m_dict = compute_regression_metrics(df_b['MeasurementValue'].values, oof_preds_raw)
        m_dict['basin'] = b
        m_dict['paradigm'] = 'Within-Basin (Upper Bound)'
        m_dict['n_samples'] = len(df_b)
        m_dict['n_sites'] = df_b['site_no'].nunique()
        within_results.append(m_dict)
        print(f"  Within-Basin [{b:15s}] -> R^2: {m_dict['r2']:.4f} | RMSE: {m_dict['rmse']:.3f} | MAE: {m_dict['mae']:.3f}", flush=True)
        
    # 2. Leave-One-Basin-Out (LOBO) Benchmark
    print("\n--- Phase 2: Leave-One-Basin-Out (LOBO) Transfer ---", flush=True)
    for target_basin in basins:
        train_df = df_target[df_target['basin'] != target_basin]
        test_df = df_target[df_target['basin'] == target_basin]
        
        if len(test_df) < 10:
            continue
            
        X_train = train_df[FEATURE_COLS].values
        y_train_log = np.log10(np.clip(train_df['MeasurementValue'].values, 1e-4, None))
        
        X_test = test_df[FEATURE_COLS].values
        y_test_raw = test_df['MeasurementValue'].values
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        m = get_inversion_models(in_features=len(FEATURE_COLS))[model_name]
        m.fit(X_train_s, y_train_log)
        
        pred_log = m.predict(X_test_s)
        pred_raw = 10.0 ** pred_log
        
        m_dict = compute_regression_metrics(y_test_raw, pred_raw)
        m_dict['basin'] = target_basin
        m_dict['paradigm'] = 'Leave-One-Basin-Out (Cross-Transfer)'
        m_dict['n_train'] = len(train_df)
        m_dict['n_test'] = len(test_df)
        lobo_results.append(m_dict)
        print(f"  LOBO Transfer -> [{target_basin:15s}] -> R^2: {m_dict['r2']:.4f} | RMSE: {m_dict['rmse']:.3f} | MAE: {m_dict['mae']:.3f}", flush=True)
        
    # 3. Pooled Joint Benchmark
    print("\n--- Phase 3: Pooled Joint Training across all 5 Basins ---", flush=True)
    pooled_splits = list(get_spatial_block_cv(df_target, group_col='site_no', n_splits=5))
    oof_pooled_log = np.zeros(len(df_target))
    X_p = df_target[FEATURE_COLS].values
    y_p_log = np.log10(np.clip(df_target['MeasurementValue'].values, 1e-4, None))
    
    for tr_idx, v_idx in pooled_splits:
        X_tr, y_tr = X_p[tr_idx], y_p_log[tr_idx]
        X_v = X_p[v_idx]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_v_s = scaler.transform(X_v)
        
        m = get_inversion_models(in_features=len(FEATURE_COLS))[model_name]
        m.fit(X_tr_s, y_tr)
        oof_pooled_log[v_idx] = m.predict(X_v_s)
        
    oof_pooled_raw = 10.0 ** oof_pooled_log
    pooled_metrics = compute_regression_metrics(df_target['MeasurementValue'].values, oof_pooled_raw)
    pooled_metrics['basin'] = 'All_5_Basins_Pooled'
    pooled_metrics['paradigm'] = 'Pooled Joint (Spatial CV)'
    pooled_metrics['n_samples'] = len(df_target)
    pooled_metrics['n_sites'] = df_target['site_no'].nunique()
    print(f"  Pooled Benchmark -> R^2: {pooled_metrics['r2']:.4f} | RMSE: {pooled_metrics['rmse']:.3f} | MAE: {pooled_metrics['mae']:.3f}", flush=True)
    
    df_within = pd.DataFrame(within_results)
    df_lobo = pd.DataFrame(lobo_results)
    
    # Combined summary table
    comparison_rows = []
    for b in basins:
        w_sub = df_within[df_within['basin'] == b]
        l_sub = df_lobo[df_lobo['basin'] == b]
        if len(w_sub) > 0 and len(l_sub) > 0:
            w_r2 = w_sub.iloc[0]['r2']
            w_rmse = w_sub.iloc[0]['rmse']
            l_r2 = l_sub.iloc[0]['r2']
            l_rmse = l_sub.iloc[0]['rmse']
            r2_degradation = w_r2 - l_r2
            rmse_increase_pct = (l_rmse - w_rmse) / w_rmse * 100.0 if w_rmse > 0 else 0.0
            comparison_rows.append({
                'basin': b,
                'within_r2': w_r2,
                'lobo_r2': l_r2,
                'delta_r2': r2_degradation,
                'within_rmse': w_rmse,
                'lobo_rmse': l_rmse,
                'rmse_increase_%': rmse_increase_pct
            })
            
    summary_df = pd.DataFrame(comparison_rows)
    
    # Save tables
    tables_dir = os.path.join(results_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    table_path = os.path.join(tables_dir, f'stage4_experiment_b_transferability_{target_type}.csv')
    summary_df.to_csv(table_path, index=False)
    print(f"Saved Experiment B transferability summary to {table_path}", flush=True)
    
    # Plot Transferability Comparison
    if len(summary_df) > 0:
        plt.figure(figsize=(10, 5), dpi=150)
        x = np.arange(len(summary_df))
        w = 0.35
        
        plt.bar(x - w/2, summary_df['within_r2'], w, label='Within-Basin R² (Upper Bound)', color='#2ca02c', alpha=0.85)
        plt.bar(x + w/2, summary_df['lobo_r2'], w, label='LOBO Cross-Transfer R²', color='#d62728', alpha=0.85)
        plt.axhline(pooled_metrics['r2'], color='blue', linestyle='--', lw=1.5, label=f"Pooled All-Basin R² ({pooled_metrics['r2']:.3f})")
        
        plt.xlabel('River Basin', fontsize=11, fontweight='bold')
        plt.ylabel('R² Score', fontsize=11, fontweight='bold')
        plt.title(f"Experiment B: Within-Basin vs. Cross-Basin Portability ({target_type.capitalize()})", fontsize=12, fontweight='bold')
        plt.xticks(x, summary_df['basin'], rotation=15)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        fig_path = os.path.join(results_dir, 'figures', f'stage4_experiment_b_transferability_{target_type}.png')
        plt.savefig(fig_path)
        plt.close()
        print(f"Saved transferability figure to {fig_path}", flush=True)
        
    return summary_df, pd.DataFrame([pooled_metrics])

if __name__ == '__main__':
    data_path = PROJECT_ROOT / 'data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv'
    df_dl = pd.read_csv(data_path, low_memory=False)
    for target in ['turbidity', 'chlorophyll']:
        run_experiment_b_transferability(
            df_dl,
            target_type=target,
            model_name='Gradient_Boosting',
            results_dir=str(PROJECT_ROOT / 'results')
        )
