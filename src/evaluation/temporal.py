import os

# Suppress joblib loky wmic CPU detection warning on modern Windows 11
if 'LOKY_MAX_CPU_COUNT' not in os.environ:
    os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count() or 4)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple

from src.inversion.models import get_inversion_models
from src.inversion.metrics import compute_regression_metrics
from src.inversion.train_inversion import FEATURE_COLS

def run_experiment_c_temporal(
    df_dl: pd.DataFrame,
    target_type: str = 'turbidity',
    model_name: str = 'Gradient_Boosting',
    results_dir: str = 'results'
) -> pd.DataFrame:
    """
    Experiment C: Temporal Robustness and Multi-Year Generalization.
    Trains on historical period (2015-2021) and tests on subsequent period (2022-2024).
    """
    print(f"\n=======================================================", flush=True)
    print(f"EXPERIMENT C: Temporal Robustness for {target_type.upper()}", flush=True)
    print(f"=======================================================", flush=True)
    
    df_target = df_dl[df_dl['target_type'] == target_type].dropna(subset=['year']).copy()
    
    train_df = df_target[df_target['year'] <= 2021].copy()
    test_df = df_target[df_target['year'] >= 2022].copy()
    
    print(f"Historical Training Set (2015-2021): {len(train_df):,} samples across {train_df['site_no'].nunique()} sites", flush=True)
    print(f"Subsequent Evaluation Set (2022-2024): {len(test_df):,} samples across {test_df['site_no'].nunique()} sites", flush=True)
    
    if len(train_df) < 50 or len(test_df) < 50:
        print("Warning: Insufficient sample counts in temporal splits.", flush=True)
        return pd.DataFrame()
        
    X_train = train_df[FEATURE_COLS].values
    y_train_log = np.log10(np.clip(train_df['MeasurementValue'].values, 1e-4, None))
    
    X_test = test_df[FEATURE_COLS].values
    y_test_raw = test_df['MeasurementValue'].values
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    m = get_inversion_models(in_features=len(FEATURE_COLS))[model_name]
    m.fit(X_train_s, y_train_log)
    
    # In-sample train performance
    pred_train_log = m.predict(X_train_s)
    pred_train_raw = 10.0 ** pred_train_log
    train_metrics = compute_regression_metrics(train_df['MeasurementValue'].values, pred_train_raw)
    train_metrics['period'] = 'Train (2015-2021)'
    train_metrics['n_samples'] = len(train_df)
    
    # Out-of-time test performance
    pred_test_log = m.predict(X_test_s)
    pred_test_raw = 10.0 ** pred_test_log
    test_metrics = compute_regression_metrics(y_test_raw, pred_test_raw)
    test_metrics['period'] = 'Test (2022-2024)'
    test_metrics['n_samples'] = len(test_df)
    
    temporal_df = pd.DataFrame([train_metrics, test_metrics])
    print(f"Temporal Train Metrics: R^2 = {train_metrics['r2']:.4f} | RMSE = {train_metrics['rmse']:.3f}", flush=True)
    print(f"Temporal Test Metrics:  R^2 = {test_metrics['r2']:.4f} | RMSE = {test_metrics['rmse']:.3f}", flush=True)
    
    # Save table
    tables_dir = os.path.join(results_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    table_path = os.path.join(tables_dir, f'stage4_experiment_c_temporal_{target_type}.csv')
    temporal_df.to_csv(table_path, index=False)
    print(f"Saved Experiment C temporal summary to {table_path}", flush=True)
    
    return temporal_df

if __name__ == '__main__':
    df_dl = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv', low_memory=False)
    for target in ['turbidity', 'chlorophyll']:
        run_experiment_c_temporal(df_dl, target_type=target, model_name='Gradient_Boosting')

