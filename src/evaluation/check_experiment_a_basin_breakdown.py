"""
Script: check_experiment_a_basin_breakdown.py
Analyzes Experiment A (Turbidity Ablation) per basin:
1. Confirms whether DL-masked, NDWI-masked, and Assumed Water were evaluated on the same test set.
2. Reports R^2, RMSE, MAE improvement per basin.
3. Quantifies whether the R^2 = 0.2047 -> 0.2904 flagship improvement is uniform across basins.
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.inversion.models import get_inversion_models
from src.inversion.metrics import compute_regression_metrics
from src.inversion.train_inversion import FEATURE_COLS

def main():
    print("=" * 80)
    print("EXPERIMENT A (TURBIDITY): Per-Basin Breakdown & Test Set Audit")
    print("=" * 80)
    
    df_a = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_a_ground_truth.csv', low_memory=False)
    df_b = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv', low_memory=False)
    df_c = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_c_ndwi_masked.csv', low_memory=False)
    
    target_type = 'turbidity'
    datasets = {
        'Condition_(a)_Assumed_Water': df_a[df_a['target_type'] == target_type].copy(),
        'Condition_(b)_DL_Masked': df_b[df_b['target_type'] == target_type].copy(),
        'Condition_(c)_NDWI_Masked': df_c[df_c['target_type'] == target_type].copy(),
    }
    
    # Replicate exact test split from ablation.py
    common_sites = set(datasets['Condition_(b)_DL_Masked']['site_no']).intersection(
        set(datasets['Condition_(c)_NDWI_Masked']['site_no'])
    )
    test_sites = set(list(common_sites)[: max(1, len(common_sites) // 4)])
    
    print(f"\nTotal common sites between DL and NDWI: {len(common_sites)}")
    print(f"Held-out test sites ({len(test_sites)} sites): {test_sites}")
    
    test_df = datasets['Condition_(b)_DL_Masked'][datasets['Condition_(b)_DL_Masked']['site_no'].isin(test_sites)].copy()
    print(f"Total held-out test samples: {len(test_df):,} across {test_df['basin'].nunique()} basins.")
    
    basin_counts = test_df.groupby('basin')['site_no'].agg(n_samples='count', n_sites='nunique').reset_index()
    print("\nTest Set Breakdown by Basin:")
    print(basin_counts.to_string(index=False))
    
    # Train each condition and evaluate on test_df
    trained_models = {}
    scalers = {}
    
    for cond_name, df_cond in datasets.items():
        train_mask = ~df_cond['site_no'].isin(test_sites)
        train_df = df_cond[train_mask]
        
        X_train = train_df[FEATURE_COLS].values
        y_train_log = np.log10(np.clip(train_df['MeasurementValue'].values, 1e-4, None))
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        model = get_inversion_models(in_features=len(FEATURE_COLS))['Gradient_Boosting']
        model.fit(X_train_scaled, y_train_log)
        
        trained_models[cond_name] = model
        scalers[cond_name] = scaler
        
    # Overall evaluation
    overall_metrics = []
    y_test_raw = test_df['MeasurementValue'].values
    preds = {}
    
    for cond_name in datasets.keys():
        scaler = scalers[cond_name]
        model = trained_models[cond_name]
        X_test_scaled = scaler.transform(test_df[FEATURE_COLS].values)
        pred_log = model.predict(X_test_scaled)
        pred_raw = 10.0 ** pred_log
        preds[cond_name] = pred_raw
        
        m = compute_regression_metrics(y_test_raw, pred_raw)
        m['condition'] = cond_name
        m['scope'] = 'Overall (All Basins)'
        m['test_samples'] = len(test_df)
        overall_metrics.append(m)
        
    print("\nOverall Performance on Identical Test Set:")
    print(pd.DataFrame(overall_metrics)[['condition', 'r2', 'rmse', 'mae', 'mape', 'bias']].to_string(index=False))
    
    # Per-Basin breakdown
    basin_metrics = []
    for b, b_grp in test_df.groupby('basin'):
        idx = b_grp.index
        y_b = b_grp['MeasurementValue'].values
        
        for cond_name in datasets.keys():
            # subset predictions
            p_b = preds[cond_name][test_df.index.isin(idx)]
            m_b = compute_regression_metrics(y_b, p_b)
            m_b['basin'] = b
            m_b['condition'] = cond_name
            m_b['test_samples'] = len(b_grp)
            m_b['n_sites'] = b_grp['site_no'].nunique()
            basin_metrics.append(m_b)
            
    b_df = pd.DataFrame(basin_metrics)
    print("\nPer-Basin Breakdown on Identical Test Set:")
    pivot_r2 = b_df.pivot(index='basin', columns='condition', values='r2')
    pivot_rmse = b_df.pivot(index='basin', columns='condition', values='rmse')
    pivot_mae = b_df.pivot(index='basin', columns='condition', values='mae')
    
    print("\nR^2 by Basin:")
    print(pivot_r2.to_string())
    print("\nRMSE by Basin:")
    print(pivot_rmse.to_string())
    print("\nMAE by Basin:")
    print(pivot_mae.to_string())
    
    # Compute delta (DL Masked vs NDWI Masked)
    delta_r2 = pivot_r2['Condition_(b)_DL_Masked'] - pivot_r2['Condition_(c)_NDWI_Masked']
    delta_rmse = pivot_rmse['Condition_(b)_DL_Masked'] - pivot_rmse['Condition_(c)_NDWI_Masked']
    
    summary_basin = pd.DataFrame({
        'n_samples': test_df.groupby('basin').size(),
        'n_sites': test_df.groupby('basin')['site_no'].nunique(),
        'R2_NDWI_Mask': pivot_r2['Condition_(c)_NDWI_Masked'],
        'R2_DL_Mask': pivot_r2['Condition_(b)_DL_Masked'],
        'Delta_R2': delta_r2,
        'RMSE_NDWI_Mask': pivot_rmse['Condition_(c)_NDWI_Masked'],
        'RMSE_DL_Mask': pivot_rmse['Condition_(b)_DL_Masked'],
        'Delta_RMSE': delta_rmse
    }).reset_index()
    
    print("\nSummary Table (DL Masked vs Naive NDWI Masked by Basin):")
    print(summary_basin.to_string(index=False))
    
    summary_basin.to_csv('results/tables/stage4_experiment_a_turbidity_per_basin.csv', index=False)
    b_df.to_csv('results/tables/stage4_experiment_a_turbidity_basin_all_metrics.csv', index=False)
    print("\nSaved tables to results/tables/stage4_experiment_a_turbidity_per_basin.csv")

if __name__ == '__main__':
    main()
