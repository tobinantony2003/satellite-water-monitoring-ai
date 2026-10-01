"""
Script: audit_r2_computation_space.py
Performs Step 0 audit:
1. Audits code paths for R² calculation across all previous experiments.
2. Recomputes Turbidity Experiment A per-basin breakdown in BOTH physical space and log space.
3. Recomputes Turbidity Experiment A pooled flagship result in BOTH physical space and log space.
4. Generates side-by-side comparison tables.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.inversion.models import get_inversion_models
from src.inversion.train_inversion import FEATURE_COLS

def main():
    print("=" * 80)
    print("STEP 0: R² COMPUTATION SPACE AUDIT")
    print("=" * 80)

    # 1. Audit Table of previous code paths
    audit_records = [
        {
            'experiment_benchmark': 'Stage 3 Inversion Benchmarks (RF, XGB, MLP, Empirical)',
            'script_source_path': 'src/inversion/train_inversion.py (lines 89-95)',
            'computation_space': 'Physical Units',
            'details': 'oof_preds_log back-transformed via 10**log; compute_regression_metrics(y_raw, oof_preds_raw)'
        },
        {
            'experiment_benchmark': 'Experiment A Pooled Ablation (0.2047 -> 0.2904)',
            'script_source_path': 'src/evaluation/ablation.py (lines 75-78)',
            'computation_space': 'Physical Units',
            'details': 'pred_raw = 10.0 ** pred_log; compute_regression_metrics(y_test_raw, pred_raw)'
        },
        {
            'experiment_benchmark': 'Experiment A Per-Basin Breakdown (Trinity, Illinois, etc.)',
            'script_source_path': 'src/evaluation/check_experiment_a_basin_breakdown.py (lines 81-84, 99-105)',
            'computation_space': 'Physical Units',
            'details': 'pred_raw = 10.0 ** pred_log; compute_regression_metrics(y_b, p_b)'
        },
        {
            'experiment_benchmark': 'CDOM & Chlorophyll-a MixedLM Spatial Block CV',
            'script_source_path': 'src/inversion/rerun_harmonized_diagnostics.py (lines 167-175, 198-207)',
            'computation_space': 'Dual Reported (Both Physical & Log)',
            'details': 'Both overall_r2_raw and overall_r2_log evaluated and exported to mixed_effects_3way_comparison.csv'
        },
        {
            'experiment_benchmark': 'Experiment B (Cross-Basin Transferability)',
            'script_source_path': 'src/evaluation/transferability.py (lines 77-80)',
            'computation_space': 'Physical Units',
            'details': 'pred_raw = 10.0 ** pred_log; compute_regression_metrics(y_test_raw, pred_raw)'
        },
        {
            'experiment_benchmark': 'Experiment C (Temporal Transferability)',
            'script_source_path': 'src/evaluation/temporal.py (lines 75-78)',
            'computation_space': 'Physical Units',
            'details': 'pred_raw = 10.0 ** pred_log; compute_regression_metrics(y_test_raw, pred_raw)'
        }
    ]
    df_audit = pd.DataFrame(audit_records)
    print("\n[Audit 0.1] R² Computation Space Across All Pipeline Scripts:")
    print(df_audit[['experiment_benchmark', 'computation_space', 'script_source_path']].to_string(index=False))
    df_audit.to_csv('results/tables/r2_computation_space_audit.csv', index=False)

    # 2 & 3. Recomputing Turbidity Experiment A in Both Spaces
    print("\n" + "=" * 80)
    print("[Audit 0.2 & 0.3] Recomputing Turbidity Experiment A: Physical Space vs Log Space")
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

    # Canonical held-out test split
    from src.evaluation.splits import get_test_sites
    test_sites = get_test_sites(target_type)

    test_df = datasets['Condition_(b)_DL_Masked'][datasets['Condition_(b)_DL_Masked']['site_no'].isin(test_sites)].copy()

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

    # Evaluate predictions on held-out test set
    y_test_raw = test_df['MeasurementValue'].values
    y_test_log = np.log10(np.clip(y_test_raw, 1e-4, None))

    test_preds_raw = {}
    test_preds_log = {}

    for cond_name in datasets.keys():
        scaler = scalers[cond_name]
        model = trained_models[cond_name]
        X_test_scaled = scaler.transform(test_df[FEATURE_COLS].values)
        p_log = model.predict(X_test_scaled)
        p_raw = 10.0 ** p_log

        test_preds_log[cond_name] = p_log
        test_preds_raw[cond_name] = p_raw

    # Overall / Pooled comparison
    pooled_rows = []
    for cond_name in datasets.keys():
        p_raw = test_preds_raw[cond_name]
        p_log = test_preds_log[cond_name]

        # Physical space metrics
        r2_phys = r2_score(y_test_raw, p_raw)
        rmse_phys = np.sqrt(mean_squared_error(y_test_raw, p_raw))
        mae_phys = mean_absolute_error(y_test_raw, p_raw)

        # Log space metrics
        r2_log = r2_score(y_test_log, p_log)
        rmse_log = np.sqrt(mean_squared_error(y_test_log, p_log))
        mae_log = mean_absolute_error(y_test_log, p_log)

        pooled_rows.append({
            'condition': cond_name,
            'scope': 'Overall (All Basins)',
            'n_test_samples': len(test_df),
            'target_var_phys': np.var(y_test_raw),
            'target_var_log': np.var(y_test_log),
            'r2_physical': r2_phys,
            'r2_log': r2_log,
            'delta_r2_log_minus_phys': r2_log - r2_phys,
            'rmse_physical': rmse_phys,
            'rmse_log': rmse_log,
            'mae_physical': mae_phys,
            'mae_log': mae_log
        })

    df_pooled = pd.DataFrame(pooled_rows)
    print("\n[Audit 0.3] Pooled Turbidity Experiment A: Physical vs Log Space:")
    print(df_pooled[['condition', 'r2_physical', 'r2_log', 'rmse_physical', 'rmse_log']].to_string(index=False))

    # Per-Basin breakdown comparison
    basin_rows = []
    for b, b_grp in test_df.groupby('basin'):
        idx = b_grp.index
        y_b_raw = b_grp['MeasurementValue'].values
        y_b_log = np.log10(np.clip(y_b_raw, 1e-4, None))
        var_b_raw = np.var(y_b_raw)
        var_b_log = np.var(y_b_log)

        for cond_name in datasets.keys():
            # Match subset predictions using index location
            pos = [test_df.index.get_loc(i) for i in idx]
            p_b_raw = test_preds_raw[cond_name][pos]
            p_b_log = test_preds_log[cond_name][pos]

            r2_phys = r2_score(y_b_raw, p_b_raw)
            rmse_phys = np.sqrt(mean_squared_error(y_b_raw, p_b_raw))
            mae_phys = mean_absolute_error(y_b_raw, p_b_raw)

            r2_log = r2_score(y_b_log, p_b_log)
            rmse_log = np.sqrt(mean_squared_error(y_b_log, p_b_log))
            mae_log = mean_absolute_error(y_b_log, p_b_log)

            # Divergence flag: if sign flips or difference > 0.30
            divergence = abs(r2_phys - r2_log) > 0.30 or (r2_phys * r2_log < 0 and (abs(r2_phys) > 0.05 or abs(r2_log) > 0.05))

            basin_rows.append({
                'basin': b,
                'condition': cond_name,
                'n_samples': len(b_grp),
                'n_sites': b_grp['site_no'].nunique(),
                'target_var_physical': round(var_b_raw, 2),
                'target_var_log': round(var_b_log, 4),
                'r2_physical': round(r2_phys, 4),
                'r2_log': round(r2_log, 4),
                'divergence_flag': divergence,
                'rmse_physical': round(rmse_phys, 2),
                'rmse_log': round(rmse_log, 4),
                'mae_physical': round(mae_phys, 2),
                'mae_log': round(mae_log, 4)
            })

    df_basin = pd.DataFrame(basin_rows)
    print("\n[Audit 0.2] Per-Basin Turbidity Breakdown: Physical vs Log Space:")
    print(df_basin[['basin', 'condition', 'n_samples', 'target_var_physical', 'target_var_log', 'r2_physical', 'r2_log', 'divergence_flag']].to_string(index=False))

    # Save output tables
    df_pooled.to_csv('results/tables/turbidity_experiment_a_pooled_r2_audit.csv', index=False)
    df_basin.to_csv('results/tables/turbidity_experiment_a_per_basin_r2_audit.csv', index=False)

    return df_audit, df_pooled, df_basin

if __name__ == '__main__':
    main()
