import os

# Suppress joblib loky wmic CPU detection warning on modern Windows 11
if 'LOKY_MAX_CPU_COUNT' not in os.environ:
    os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count() or 4)

import warnings
warnings.filterwarnings('ignore', message='.*physical cores.*')
warnings.filterwarnings('ignore', category=UserWarning, module='.*loky.*')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Any

from src.inversion.models import get_inversion_models
from src.inversion.spatial_cv import get_spatial_block_cv
from src.inversion.metrics import compute_regression_metrics

FEATURE_COLS = [
    'B01_center_refl', 'B02_center_refl', 'B03_center_refl', 'B04_center_refl',
    'B05_center_refl', 'B06_center_refl', 'B07_center_refl', 'B08_center_refl',
    'B8A_center_refl', 'B11_center_refl', 'B12_center_refl',
    'ndwi', 'mndwi', 'ndci', 'turbidity_ratio', 'cdom_ratio', 'b8_b4_ratio'
]

def benchmark_inversion_models(
    df: pd.DataFrame,
    target_type: str = 'chlorophyll',
    dataset_name: str = 'DL_Masked',
    results_dir: str = 'results'
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Trains and evaluates 4 inversion architectures using 5-fold Spatial Block CV.
    Target is log10-transformed during model fitting, and back-transformed for metric reporting.
    """
    df_target = df[df['target_type'] == target_type].copy()
    if len(df_target) == 0:
        raise ValueError(f"No records found for target_type: {target_type}")
        
    print(f"\n--- Benchmarking Inversion Models for {target_type.upper()} ({dataset_name}) ---", flush=True)
    print(f"Total observations: {len(df_target):,} across {df_target['site_no'].nunique()} monitoring stations.", flush=True)
    
    X = df_target[FEATURE_COLS].values
    y_raw = df_target['MeasurementValue'].values
    y_log = np.log10(np.clip(y_raw, 1e-4, None))
    
    in_features = len(FEATURE_COLS)
    models_dict = get_inversion_models(in_features=in_features)
    
    cv_results = []
    oof_predictions = {}
    
    for model_name in models_dict.keys():
        print(f"Evaluating {model_name}...", flush=True)
        oof_preds_log = np.zeros_like(y_log)
        
        cv_splits = list(get_spatial_block_cv(df_target, group_col='site_no', n_splits=5))
        
        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            X_train, y_train_log = X[train_idx], y_log[train_idx]
            X_val, y_val_log = X[val_idx], y_log[val_idx]
            
            # Standardize features using training fold parameters only
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Empirical baseline uses specific relevant band ratio
            if model_name == 'Empirical_Baseline':
                if target_type == 'chlorophyll':
                    feat_idx = [FEATURE_COLS.index('ndci')]
                elif target_type == 'turbidity':
                    feat_idx = [FEATURE_COLS.index('turbidity_ratio')]
                else:
                    feat_idx = [FEATURE_COLS.index('cdom_ratio')]
                clf = get_inversion_models(in_features=1)['Empirical_Baseline']
                clf.fit(X_train[:, feat_idx], y_train_log)
                pred_val_log = clf.predict(X_val[:, feat_idx])
            else:
                clf = get_inversion_models(in_features=in_features)[model_name]
                clf.fit(X_train_scaled, y_train_log)
                pred_val_log = clf.predict(X_val_scaled)
                
            oof_preds_log[val_idx] = pred_val_log
            
        # Back-transform out-of-fold log predictions to original physical units
        # Safe clamp to physical bounds [-3.0, 4.0] (0.001 to 10,000) to prevent outlier fold exponential explosion
        oof_preds_log = np.clip(oof_preds_log, -3.0, 4.0)
        oof_preds_raw = 10.0 ** oof_preds_log
        oof_predictions[model_name] = oof_preds_raw
        
        metrics = compute_regression_metrics(y_raw, oof_preds_raw)
        metrics['model'] = model_name
        metrics['target'] = target_type
        metrics['dataset'] = dataset_name
        metrics['n_samples'] = len(y_raw)
        metrics['n_sites'] = df_target['site_no'].nunique()
        cv_results.append(metrics)
        
        print(f"  --> {model_name:20s} | R^2: {metrics['r2']:.4f} | RMSE: {metrics['rmse']:.3f} | "
              f"MAE: {metrics['mae']:.3f} | MAPE: {metrics['mape']:.2f}%", flush=True)
        
    results_df = pd.DataFrame(cv_results)
    
    # Save table
    tables_dir = os.path.join(results_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    table_path = os.path.join(tables_dir, f'stage3_{target_type}_inversion_benchmarks.csv')
    results_df.to_csv(table_path, index=False)
    
    # Generate Observed vs Predicted scatter plot for top model
    best_model_name = results_df.sort_values(by='r2', ascending=False).iloc[0]['model']
    best_preds = oof_predictions[best_model_name]
    
    plot_observed_vs_predicted(
        y_true=y_raw,
        y_pred=best_preds,
        target_name=target_type,
        model_name=best_model_name,
        metrics=results_df[results_df['model'] == best_model_name].iloc[0].to_dict(),
        save_path=os.path.join(results_dir, 'figures', f'stage3_{target_type}_obs_vs_pred.png')
    )
    
    return results_df, oof_predictions

run_spatial_inversion_benchmark = benchmark_inversion_models


def plot_observed_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str,
    model_name: str,
    metrics: Dict[str, Any],
    save_path: str
):
    """Publication-grade Observed vs. Predicted scatter plot with 1:1 line."""
    plt.figure(figsize=(7, 6), dpi=200)
    
    unit_map = {
        'chlorophyll': 'μg/L',
        'turbidity': 'FNU',
        'cdom': 'μg/L QSE'
    }
    unit = unit_map.get(target_name.lower(), '')
    
    # Hexbin / scatter with log scale for wide dynamic range
    mask = (y_true > 0) & (y_pred > 0)
    yt = y_true[mask]
    yp = y_pred[mask]
    
    min_val = max(1e-2, min(np.min(yt), np.min(yp)))
    max_val = max(np.max(yt), np.max(yp))
    
    plt.scatter(yt, yp, alpha=0.3, edgecolors='none', color='#1f77b4', s=18)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Reference')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(min_val * 0.8, max_val * 1.2)
    plt.ylim(min_val * 0.8, max_val * 1.2)
    
    stat_text = (
        f"Model: {model_name}\n"
        f"R² = {metrics['r2']:.3f}\n"
        f"RMSE = {metrics['rmse']:.2f} {unit}\n"
        f"MAE = {metrics['mae']:.2f} {unit}\n"
        f"MAPE = {metrics['mape']:.1f}%"
    )
    plt.text(0.05, 0.95, stat_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='#cccccc'))
             
    plt.xlabel(f"Observed In-Situ {target_name.capitalize()} ({unit})", fontsize=11, fontweight='bold')
    plt.ylabel(f"Satellite Retrieved {target_name.capitalize()} ({unit})", fontsize=11, fontweight='bold')
    plt.title(f"Spatial Block CV Retrieval: {target_name.capitalize()}", fontsize=13, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved observed vs predicted plot to {save_path}", flush=True)

if __name__ == '__main__':
    dl_path = 'data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv'
    print(f"Loading DL-masked dataset from {dl_path}...", flush=True)
    df_dl = pd.read_csv(dl_path, low_memory=False)
    
    all_benchmarks = []
    for target in ['turbidity', 'chlorophyll', 'cdom']:
        bench_df, _ = benchmark_inversion_models(
            df=df_dl,
            target_type=target,
            dataset_name='DL_Masked',
            results_dir='results'
        )
        all_benchmarks.append(bench_df)
        
    consolidated_df = pd.concat(all_benchmarks, ignore_index=True)
    cons_path = 'results/tables/stage3_consolidated_inversion_benchmarks.csv'
    consolidated_df.to_csv(cons_path, index=False)
    print(f"\nAll Stage 3 benchmarks complete! Saved to {cons_path}", flush=True)

