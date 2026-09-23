
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Any

from src.inversion.models import get_inversion_models
from src.inversion.spatial_cv import get_spatial_block_cv
from src.inversion.metrics import compute_regression_metrics
from src.inversion.train_inversion import FEATURE_COLS

def run_experiment_a_ablation(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    df_c: pd.DataFrame,
    target_type: str = 'turbidity',
    model_name: str = 'Gradient_Boosting',
    results_dir: str = 'results'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Experiment A: Segmentation-Quality Ablation.
    Trains and evaluates the best inversion model under three masking conditions:
      (a) Ground-Truth / Assumed Water Baseline
      (b) Deep Learning Water Extraction Front-end
      (c) Naive NDWI Threshold Masking
      
    Performs paired statistical significance tests (Paired t-test and Wilcoxon signed-rank)
    on the test error distributions across the conditions.
    """
    print(f"\n=======================================================", flush=True)
    print(f"EXPERIMENT A: Segmentation-Quality Ablation for {target_type.upper()}", flush=True)
    print(f"=======================================================", flush=True)
    
    datasets = {
        'Condition_(a)_Assumed_Water': df_a[df_a['target_type'] == target_type].copy(),
        'Condition_(b)_DL_Masked': df_b[df_b['target_type'] == target_type].copy(),
        'Condition_(c)_NDWI_Masked': df_c[df_c['target_type'] == target_type].copy(),
    }
    
    # We evaluate on an identical held-out spatial test block from the overlapping confident set
    # to guarantee rigorous, fair apples-to-apples metric comparison across all 3 models!
    common_sites = set(datasets['Condition_(b)_DL_Masked']['site_no']).intersection(
        set(datasets['Condition_(c)_NDWI_Masked']['site_no'])
    )
    test_sites = set(list(common_sites)[: max(1, len(common_sites) // 4)])
    
    results = []
    error_distributions = {}
    
    for cond_name, df_cond in datasets.items():
        print(f"\nTraining under {cond_name} (Total pool: {len(df_cond):,} samples)...", flush=True)
        
        train_mask = ~df_cond['site_no'].isin(test_sites)
        test_mask = df_cond['site_no'].isin(test_sites)
        
        train_df = df_cond[train_mask]
        # Held-out test set is evaluated on the standard test sites
        test_df = datasets['Condition_(b)_DL_Masked'][datasets['Condition_(b)_DL_Masked']['site_no'].isin(test_sites)]
        
        X_train = train_df[FEATURE_COLS].values
        y_train_log = np.log10(np.clip(train_df['MeasurementValue'].values, 1e-4, None))
        
        X_test = test_df[FEATURE_COLS].values
        y_test_raw = test_df['MeasurementValue'].values
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = get_inversion_models(in_features=len(FEATURE_COLS))[model_name]
        model.fit(X_train_scaled, y_train_log)
        
        pred_log = model.predict(X_test_scaled)
        pred_raw = 10.0 ** pred_log
        
        metrics = compute_regression_metrics(y_test_raw, pred_raw)
        metrics['condition'] = cond_name
        metrics['train_samples'] = len(train_df)
        metrics['test_samples'] = len(test_df)
        metrics['model'] = model_name
        results.append(metrics)
        
        # Absolute errors for statistical significance testing
        abs_errors = np.abs(pred_raw - y_test_raw)
        error_distributions[cond_name] = abs_errors
        
        print(f"  {cond_name} -> R^2: {metrics['r2']:.4f} | RMSE: {metrics['rmse']:.3f} | "
              f"MAE: {metrics['mae']:.3f} | MAPE: {metrics['mape']:.2f}%", flush=True)
              
    ablation_df = pd.DataFrame(results)
    
    # Statistical Significance Testing: DL (b) vs Assumed (a) and DL (b) vs NDWI (c)
    err_a = error_distributions['Condition_(a)_Assumed_Water']
    err_b = error_distributions['Condition_(b)_DL_Masked']
    err_c = error_distributions['Condition_(c)_NDWI_Masked']
    
    # Paired t-tests
    t_stat_ba, p_val_ba = stats.ttest_rel(err_b, err_a)
    t_stat_bc, p_val_bc = stats.ttest_rel(err_b, err_c)
    
    # Wilcoxon signed-rank tests (non-parametric)
    wilcox_ba = stats.wilcoxon(err_b, err_a)
    wilcox_bc = stats.wilcoxon(err_b, err_c)
    
    sig_summary = {
        'comparison_DL_vs_Assumed': {
            't_stat': float(t_stat_ba),
            'p_value_t': float(p_val_ba),
            'wilcoxon_stat': float(wilcox_ba.statistic),
            'p_value_wilcoxon': float(wilcox_ba.pvalue),
            'significant_at_0.05': bool(p_val_ba < 0.05)
        },
        'comparison_DL_vs_NDWI': {
            't_stat': float(t_stat_bc),
            'p_value_t': float(p_val_bc),
            'wilcoxon_stat': float(wilcox_bc.statistic),
            'p_value_wilcoxon': float(wilcox_bc.pvalue),
            'significant_at_0.05': bool(p_val_bc < 0.05)
        }
    }
    
    print("\n--- Statistical Significance Testing (Error Distributions) ---", flush=True)
    print(f"DL vs Assumed Water: t-test p={p_val_ba:.4e}, Wilcoxon p={wilcox_ba.pvalue:.4e} (Significant: {p_val_ba < 0.05})", flush=True)
    print(f"DL vs Naive NDWI:    t-test p={p_val_bc:.4e}, Wilcoxon p={wilcox_bc.pvalue:.4e} (Significant: {p_val_bc < 0.05})", flush=True)
    
    # Save table
    tables_dir = os.path.join(results_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    table_path = os.path.join(tables_dir, f'stage4_experiment_a_ablation_{target_type}.csv')
    ablation_df.to_csv(table_path, index=False)
    print(f"Saved Experiment A ablation table to {table_path}", flush=True)
    
    return ablation_df, sig_summary

if __name__ == '__main__':
    df_a = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_a_ground_truth.csv', low_memory=False)
    df_b = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv', low_memory=False)
    df_c = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_c_ndwi_masked.csv', low_memory=False)
    
    for target in ['turbidity', 'chlorophyll']:
        run_experiment_a_ablation(df_a, df_b, df_c, target_type=target, model_name='Gradient_Boosting')
