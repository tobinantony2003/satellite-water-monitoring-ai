"""
Script: run_step1_gunnison_outlier_check.py
Investigates station 09152500 (Gunnison at Delta) in Upper Colorado:
1. Checks distribution, presence in Condition (b) DL pool and Condition (c) NDWI pool.
2. Checks whether 09152500 is in training or test split.
3. Re-runs the Upper Colorado log R^2 evaluation for DL vs NDWI with 09152500 included vs excluded.
4. Analyzes the impact on the Upper Colorado DL-vs-NDWI performance gap.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inversion.models import get_inversion_models
from src.inversion.train_inversion import FEATURE_COLS

def main():
    print("=" * 80)
    print("STEP 1: SENSITIVITY CHECK ON GUNNISON-AT-DELTA (STATION 09152500)")
    print("=" * 80)

    df_b = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv', low_memory=False)
    df_c = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_c_ndwi_masked.csv', low_memory=False)

    target_type = 'turbidity'
    b_turb = df_b[df_b['target_type'] == target_type].copy()
    c_turb = df_c[df_c['target_type'] == target_type].copy()

    # Canonical held-out test split
    from src.evaluation.splits import get_test_sites
    test_sites = get_test_sites(target_type)
    print(f"Total sites in DL pool: {b_turb['site_no'].nunique()}, NDWI pool: {c_turb['site_no'].nunique()}")
    print(f"Canonical Test sites ({len(test_sites)} sites): {sorted(list(test_sites))}")

    gunnison_site = 9152500
    # Also check string representation
    b_gunnison = b_turb[b_turb['site_no'].astype(str).str.contains('9152500')]
    c_gunnison = c_turb[c_turb['site_no'].astype(str).str.contains('9152500')]

    print(f"\nGunnison at Delta (09152500) samples:")
    print(f"  In DL Masked (b):   {len(b_gunnison)}")
    print(f"  In NDWI Masked (c): {len(c_gunnison)}")
    if len(b_gunnison) > 0:
        g_site_val = b_gunnison['site_no'].iloc[0]
        in_test = g_site_val in test_sites
        print(f"  Station Site ID: {g_site_val}")
        print(f"  Is in test set? {in_test}")
        print(f"  Turbidity stats: min={b_gunnison['MeasurementValue'].min():.1f}, "
              f"median={b_gunnison['MeasurementValue'].median():.1f}, "
              f"mean={b_gunnison['MeasurementValue'].mean():.1f}, "
              f"max={b_gunnison['MeasurementValue'].max():.1f}")
        print(f"  Upper Colorado overall turbidity median: {b_turb[b_turb['basin']=='Upper Colorado']['MeasurementValue'].median():.1f}")

    # Let's inspect test split and train split compositions
    test_df_orig = b_turb[b_turb['site_no'].isin(test_sites)].copy()
    train_b_orig = b_turb[~b_turb['site_no'].isin(test_sites)].copy()
    train_c_orig = c_turb[~c_turb['site_no'].isin(test_sites)].copy()

    # Function to train and evaluate on Upper Colorado test set
    def evaluate_models(train_b, train_c, test_df, label=""):
        print(f"\n--- Evaluation: {label} ---")
        uc_test = test_df[test_df['basin'] == 'Upper Colorado'].copy()
        print(f"Test set total: {len(test_df):,}, Upper Colorado test: {len(uc_test):,}")

        X_test_all = test_df[FEATURE_COLS].values
        y_test_all_log = np.log10(np.clip(test_df['MeasurementValue'].values, 1e-4, None))

        X_test_uc = uc_test[FEATURE_COLS].values
        y_test_uc_log = np.log10(np.clip(uc_test['MeasurementValue'].values, 1e-4, None))

        # Model DL
        sc_b = StandardScaler()
        X_tr_b = sc_b.fit_transform(train_b[FEATURE_COLS].values)
        y_tr_b = np.log10(np.clip(train_b['MeasurementValue'].values, 1e-4, None))
        m_b = get_inversion_models(in_features=len(FEATURE_COLS))['Gradient_Boosting']
        m_b.fit(X_tr_b, y_tr_b)

        p_b_all = m_b.predict(sc_b.transform(X_test_all))
        p_b_uc = m_b.predict(sc_b.transform(X_test_uc))

        r2_b_all = r2_score(y_test_all_log, p_b_all)
        r2_b_uc = r2_score(y_test_uc_log, p_b_uc)
        rmse_b_uc = np.sqrt(mean_squared_error(y_test_uc_log, p_b_uc))

        # Model NDWI
        sc_c = StandardScaler()
        X_tr_c = sc_c.fit_transform(train_c[FEATURE_COLS].values)
        y_tr_c = np.log10(np.clip(train_c['MeasurementValue'].values, 1e-4, None))
        m_c = get_inversion_models(in_features=len(FEATURE_COLS))['Gradient_Boosting']
        m_c.fit(X_tr_c, y_tr_c)

        p_c_all = m_c.predict(sc_c.transform(X_test_all))
        p_c_uc = m_c.predict(sc_c.transform(X_test_uc))

        r2_c_all = r2_score(y_test_all_log, p_c_all)
        r2_c_uc = r2_score(y_test_uc_log, p_c_uc)
        rmse_c_uc = np.sqrt(mean_squared_error(y_test_uc_log, p_c_uc))

        # Per-basin breakdown for all 4 basins
        basin_results = {}
        for b in sorted(test_df['basin'].dropna().unique()):
            b_mask = (test_df['basin'] == b).values
            y_b = y_test_all_log[b_mask]
            p_b_b = p_b_all[b_mask]
            p_c_b = p_c_all[b_mask]
            r2_b = r2_score(y_b, p_b_b) if len(y_b) > 1 else np.nan
            r2_c = r2_score(y_b, p_c_b) if len(y_b) > 1 else np.nan
            basin_results[b] = {'DL_log_r2': r2_b, 'NDWI_log_r2': r2_c, 'diff (DL - NDWI)': r2_b - r2_c}

        df_basins = pd.DataFrame(basin_results).T

        macro_b = df_basins['DL_log_r2'].mean()
        macro_c = df_basins['NDWI_log_r2'].mean()

        print(f"Overall (Micro) Log R^2: DL={r2_b_all:.4f}, NDWI={r2_c_all:.4f}, Gap={r2_b_all - r2_c_all:+.4f}")
        print(f"Macro Log R^2:           DL={macro_b:.4f}, NDWI={macro_c:.4f}, Gap={macro_b - macro_c:+.4f}")
        print(f"Upper Colorado Log R^2:  DL={r2_b_uc:.4f}, NDWI={r2_c_uc:.4f}, Gap={r2_b_uc - r2_c_uc:+.4f}")
        print(f"Upper Colorado Log RMSE: DL={rmse_b_uc:.4f}, NDWI={rmse_c_uc:.4f}")
        print(df_basins.round(4).to_string())

        return {
            'label': label,
            'DL_micro_r2': r2_b_all, 'NDWI_micro_r2': r2_c_all,
            'DL_macro_r2': macro_b, 'NDWI_macro_r2': macro_c,
            'DL_uc_r2': r2_b_uc, 'NDWI_uc_r2': r2_c_uc,
            'DL_uc_rmse': rmse_b_uc, 'NDWI_uc_rmse': rmse_c_uc,
            'basin_table': df_basins
        }

    # Baseline: All stations included
    res_with_gunnison = evaluate_models(train_b_orig, train_c_orig, test_df_orig, label="With Station 09152500 (Gunnison at Delta)")

    # Exclude 09152500 from both train and test
    train_b_no_g = train_b_orig[~train_b_orig['site_no'].astype(str).str.contains('9152500')].copy()
    train_c_no_g = train_c_orig[~train_c_orig['site_no'].astype(str).str.contains('9152500')].copy()
    test_df_no_g = test_df_orig[~test_df_orig['site_no'].astype(str).str.contains('9152500')].copy()

    res_without_gunnison = evaluate_models(train_b_no_g, train_c_no_g, test_df_no_g, label="Excluding Station 09152500")

    # Save summary
    rows = [
        {
            'Condition': 'With 09152500 (Original)',
            'DL_UC_Log_R2': res_with_gunnison['DL_uc_r2'],
            'NDWI_UC_Log_R2': res_with_gunnison['NDWI_uc_r2'],
            'UC_Gap_DL_minus_NDWI': res_with_gunnison['DL_uc_r2'] - res_with_gunnison['NDWI_uc_r2'],
            'DL_Macro_Log_R2': res_with_gunnison['DL_macro_r2'],
            'NDWI_Macro_Log_R2': res_with_gunnison['NDWI_macro_r2'],
            'Macro_Gap': res_with_gunnison['DL_macro_r2'] - res_with_gunnison['NDWI_macro_r2'],
            'DL_Micro_Log_R2': res_with_gunnison['DL_micro_r2'],
            'NDWI_Micro_Log_R2': res_with_gunnison['NDWI_micro_r2'],
        },
        {
            'Condition': 'Excluding 09152500',
            'DL_UC_Log_R2': res_without_gunnison['DL_uc_r2'],
            'NDWI_UC_Log_R2': res_without_gunnison['NDWI_uc_r2'],
            'UC_Gap_DL_minus_NDWI': res_without_gunnison['DL_uc_r2'] - res_without_gunnison['NDWI_uc_r2'],
            'DL_Macro_Log_R2': res_without_gunnison['DL_macro_r2'],
            'NDWI_Macro_Log_R2': res_without_gunnison['NDWI_macro_r2'],
            'Macro_Gap': res_without_gunnison['DL_macro_r2'] - res_without_gunnison['NDWI_macro_r2'],
            'DL_Micro_Log_R2': res_without_gunnison['DL_micro_r2'],
            'NDWI_Micro_Log_R2': res_without_gunnison['NDWI_micro_r2'],
        }
    ]
    df_out = pd.DataFrame(rows)
    df_out.to_csv('results/tables/gunnison_outlier_sensitivity_step1.csv', index=False)
    print("\nSaved Step 1 sensitivity summary to results/tables/gunnison_outlier_sensitivity_step1.csv")

if __name__ == '__main__':
    main()
