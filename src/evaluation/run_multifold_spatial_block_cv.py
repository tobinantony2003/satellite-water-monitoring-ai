"""
Script: run_multifold_spatial_block_cv.py
Executes 5-fold Spatial Block Cross-Validation for Turbidity Experiment A:
- Stratified group-kfold (k=5) by site_no across all basins to ensure every fold
  contains held-out stations from all basins.
- For each fold:
  1. Condition (b) Raw DL Full Pool
  2. Condition (c) NDWI Full Pool
  3. Condition (b) Volume-Matched Subsampled DL (matched to NDWI train N)
  4. Condition (b) Hard-Filtered Clean DL (NDVI <= 0)
  5. Condition (b) Linear Unmixing-Weighted DL (endmembers from train fold ONLY)
  6. Condition (b) Quadratic Unmixing-Weighted DL
- Evaluates per-fold:
  - Micro Log R^2 & Log RMSE
  - Per-basin Log R^2 for each basin
  - Macro Log R^2 (unweighted basin average)
- Aggregates Mean +/- Std and runs Wilcoxon signed-rank & paired t-tests across folds.
- Checks per-basin win/loss consistency across folds.
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import nnls
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inversion.models import get_inversion_models
from src.inversion.train_inversion import FEATURE_COLS

def main():
    print("=" * 90)
    print("5-FOLD SPATIAL BLOCK CROSS-VALIDATION FOR TURBIDITY EXPERIMENT A ABLATION")
    print("=" * 90)

    # 1. Load data
    df_a = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_a_ground_truth.csv', low_memory=False)
    df_b = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv', low_memory=False)
    df_c = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_c_ndwi_masked.csv', low_memory=False)

    tb = df_b[df_b['target_type'] == 'turbidity'].copy()
    tc = df_c[df_c['target_type'] == 'turbidity'].copy()
    ta = df_a[df_a['target_type'] == 'turbidity'].copy()

    unmix_bands = [
        'B02_center_refl', 'B03_center_refl', 'B04_center_refl',
        'B05_center_refl', 'B06_center_refl', 'B07_center_refl',
        'B08_center_refl', 'B8A_center_refl', 'B11_center_refl', 'B12_center_refl'
    ]

    # Matching keys for unmixing reference
    tb['key'] = tb['site_no'].astype(str) + '_' + tb['sample_datetime_UTC'].astype(str)
    tc['key'] = tc['site_no'].astype(str) + '_' + tc['sample_datetime_UTC'].astype(str)
    ta['key'] = ta['site_no'].astype(str) + '_' + ta['sample_datetime_UTC'].astype(str)

    b_keys_all = set(tb['key'])
    c_keys_all = set(tc['key'])

    # 2. Stratified 5-Fold Partitioning by Basin and Station
    # Ensure every site belongs to exactly one fold, and every fold has stations from all basins
    site_basin = tb.drop_duplicates('site_no')[['site_no', 'basin']].sort_values(['basin', 'site_no'])
    
    # Deterministic assignment
    site_to_fold = {}
    for b, grp in site_basin.groupby('basin'):
        sites_in_b = grp['site_no'].tolist()
        for idx, s in enumerate(sites_in_b):
            site_to_fold[s] = idx % 5

    tb['fold'] = tb['site_no'].map(site_to_fold)
    tc['fold'] = tc['site_no'].map(site_to_fold)
    ta['fold'] = ta['site_no'].map(site_to_fold)

    print("\nStation Count per Fold and Basin:")
    fold_site_tab = tb.drop_duplicates('site_no').groupby(['fold', 'basin'])['site_no'].count().unstack(fill_value=0)
    print(fold_site_tab)

    print("\nDL Sample Count per Fold and Basin:")
    fold_sample_tab = tb.groupby(['fold', 'basin'])['MeasurementValue'].count().unstack(fill_value=0)
    print(fold_sample_tab)

    # Save fold assignments
    fold_map_df = tb.drop_duplicates('site_no')[['site_no', 'basin', 'fold']].sort_values(['fold', 'basin', 'site_no'])
    fold_map_df.to_csv('results/canonical_splits/turbidity_5fold_spatial_splits.csv', index=False)

    # Containers for results across 5 folds
    all_folds_raw_dl = []
    all_folds_ndwi = []
    all_folds_sub_dl = []
    all_folds_clean_dl = []
    all_folds_unmix_lin = []
    all_folds_unmix_quad = []

    per_basin_fold_records = []

    basins_all = sorted(tb['basin'].unique())

    # Helper function to evaluate predictions on a fold
    def eval_predictions(y_true_log, y_pred_log, test_df, model_name, fold_id):
        micro_r2 = r2_score(y_true_log, y_pred_log)
        micro_rmse = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
        
        b_r2s = {}
        for b in basins_all:
            mask = (test_df['basin'] == b).values
            if np.sum(mask) >= 5 and np.var(y_true_log[mask]) > 1e-4:
                b_r2s[b] = r2_score(y_true_log[mask], y_pred_log[mask])
            else:
                b_r2s[b] = np.nan
        
        valid_b_r2s = [v for v in b_r2s.values() if not np.isnan(v)]
        macro_r2 = float(np.mean(valid_b_r2s)) if len(valid_b_r2s) > 0 else np.nan

        res = {
            'fold': fold_id,
            'model': model_name,
            'micro_log_r2': micro_r2,
            'micro_log_rmse': micro_rmse,
            'macro_log_r2': macro_r2,
            **{f"{b}_log_r2": b_r2s[b] for b in basins_all}
        }
        return res

    print("\n" + "=" * 80)
    print("RUNNING 5-FOLD SPATIAL BLOCK CROSS-VALIDATION")
    print("=" * 80)

    for k in range(5):
        print(f"\n--- FOLD {k+1}/5 ---")
        train_tb = tb[tb['fold'] != k].copy()
        test_tb = tb[tb['fold'] == k].copy()

        train_tc = tc[tc['fold'] != k].copy()
        test_tc = tc[tc['fold'] == k].copy()

        train_ta = ta[ta['fold'] != k].copy()

        y_test_log = np.log10(np.clip(test_tb['MeasurementValue'].values, 1e-4, None))
        X_test = test_tb[FEATURE_COLS].values

        # -------------------------------------------------------------
        # 1. Condition (b) Raw DL Full Pool
        # -------------------------------------------------------------
        sc_b = StandardScaler()
        X_tr_b = sc_b.fit_transform(train_tb[FEATURE_COLS].values)
        y_tr_b = np.log10(np.clip(train_tb['MeasurementValue'].values, 1e-4, None))
        m_b = get_inversion_models(in_features=len(FEATURE_COLS))['Gradient_Boosting']
        m_b.fit(X_tr_b, y_tr_b)
        p_b = m_b.predict(sc_b.transform(X_test))

        res_b = eval_predictions(y_test_log, p_b, test_tb, 'Raw DL Full Pool', k)
        all_folds_raw_dl.append(res_b)

        # -------------------------------------------------------------
        # 2. Condition (c) NDWI Full Pool
        # -------------------------------------------------------------
        sc_c = StandardScaler()
        X_tr_c = sc_c.fit_transform(train_tc[FEATURE_COLS].values)
        y_tr_c = np.log10(np.clip(train_tc['MeasurementValue'].values, 1e-4, None))
        m_c = get_inversion_models(in_features=len(FEATURE_COLS))['Gradient_Boosting']
        m_c.fit(X_tr_c, y_tr_c)
        p_c = m_c.predict(sc_c.transform(X_test))

        res_c = eval_predictions(y_test_log, p_c, test_tb, 'NDWI Full Pool', k)
        all_folds_ndwi.append(res_c)

        print(f"  Fold {k}: DL Micro Log R^2 = {res_b['micro_log_r2']:.4f}, Macro = {res_b['macro_log_r2']:.4f}")
        print(f"  Fold {k}: NDWI Micro Log R^2 = {res_c['micro_log_r2']:.4f}, Macro = {res_c['macro_log_r2']:.4f}")

        # Per-basin comparison on this fold
        for b in basins_all:
            dl_b_r2 = res_b[f"{b}_log_r2"]
            ndwi_b_r2 = res_c[f"{b}_log_r2"]
            if not np.isnan(dl_b_r2) and not np.isnan(ndwi_b_r2):
                per_basin_fold_records.append({
                    'fold': k,
                    'basin': b,
                    'n_test_samples': int(np.sum(test_tb['basin'] == b)),
                    'dl_log_r2': dl_b_r2,
                    'ndwi_log_r2': ndwi_b_r2,
                    'gap_dl_minus_ndwi': dl_b_r2 - ndwi_b_r2,
                    'winner': 'DL Mask' if dl_b_r2 > ndwi_b_r2 else 'NDWI Mask'
                })

        # -------------------------------------------------------------
        # 3. Condition (b) Subsampled DL (Volume-Controlled to match NDWI)
        # -------------------------------------------------------------
        target_n = len(train_tc)
        sub_tb = train_tb.sample(n=target_n, random_state=42 + k)
        sc_sub = StandardScaler()
        m_sub = get_inversion_models(in_features=len(FEATURE_COLS))['Gradient_Boosting']
        m_sub.fit(sc_sub.fit_transform(sub_tb[FEATURE_COLS].values),
                  np.log10(np.clip(sub_tb['MeasurementValue'].values, 1e-4, None)))
        p_sub = m_sub.predict(sc_sub.transform(X_test))
        res_sub = eval_predictions(y_test_log, p_sub, test_tb, 'Subsampled DL', k)
        all_folds_sub_dl.append(res_sub)

        # -------------------------------------------------------------
        # 4. Condition (b) Hard-Filtered Clean DL (NDVI <= 0)
        # -------------------------------------------------------------
        train_tb['ndvi'] = (train_tb['B08_center_refl'] - train_tb['B04_center_refl']) / np.maximum(
            train_tb['B08_center_refl'] + train_tb['B04_center_refl'], 1e-4
        )
        clean_tb = train_tb[train_tb['ndvi'] <= 0.0]
        sc_cl = StandardScaler()
        m_cl = get_inversion_models(in_features=len(FEATURE_COLS))['Gradient_Boosting']
        m_cl.fit(sc_cl.fit_transform(clean_tb[FEATURE_COLS].values),
                 np.log10(np.clip(clean_tb['MeasurementValue'].values, 1e-4, None)))
        p_cl = m_cl.predict(sc_cl.transform(X_test))
        res_cl = eval_predictions(y_test_log, p_cl, test_tb, 'Hard-Filtered Clean DL', k)
        all_folds_clean_dl.append(res_cl)

        # -------------------------------------------------------------
        # 5. Linear and Quadratic Unmixing-Weighted DL
        # Endmembers derived STRICTLY from train_tb (folds != k)
        # -------------------------------------------------------------
        c_keys_fold = set(train_tc['key'])
        b_keys_fold = set(train_tb['key'])
        train_both = train_tb[train_tb['key'].isin(c_keys_fold)]
        water_em = train_both[unmix_bands].median().values

        a_rejected = train_ta[~train_ta['key'].isin(b_keys_fold) & ~train_ta['key'].isin(c_keys_fold)].copy()
        a_rejected['ndvi'] = (a_rejected['B08_center_refl'] - a_rejected['B04_center_refl']) / (
            a_rejected['B08_center_refl'] + a_rejected['B04_center_refl'] + 1e-4
        )
        veg_em = a_rejected[a_rejected['ndvi'] > 0.45][unmix_bands].median().values
        soil_em = a_rejected[(a_rejected['basin'] == 'Upper Colorado') & (a_rejected['ndvi'] < 0.20) & (a_rejected['B04_center_refl'] > 0.09)][unmix_bands].median().values

        M = np.column_stack([water_em, veg_em, soil_em])
        delta = 10.0
        A_aug = np.vstack([M, delta * np.ones((1, 3))])

        X_tr_refl = train_tb[unmix_bands].values
        f_water_fold = np.zeros(len(train_tb))
        for i in range(len(train_tb)):
            y = X_tr_refl[i]
            b_aug = np.hstack([y, delta])
            x, _ = nnls(A_aug, b_aug)
            tot = np.sum(x)
            f_water_fold[i] = x[0] / tot if tot > 0 else 0.5

        w_lin = np.clip(f_water_fold, 0.05, 1.0)
        w_quad = np.clip(f_water_fold ** 2, 0.01, 1.0)

        # Linear unmixing regression
        m_w_lin = get_inversion_models(in_features=len(FEATURE_COLS))['Gradient_Boosting']
        m_w_lin.fit(sc_b.transform(train_tb[FEATURE_COLS].values), y_tr_b, sample_weight=w_lin)
        p_w_lin = m_w_lin.predict(sc_b.transform(X_test))
        res_w_lin = eval_predictions(y_test_log, p_w_lin, test_tb, 'Unmixing-Weighted (Linear)', k)
        all_folds_unmix_lin.append(res_w_lin)

        # Quadratic unmixing regression
        m_w_quad = get_inversion_models(in_features=len(FEATURE_COLS))['Gradient_Boosting']
        m_w_quad.fit(sc_b.transform(train_tb[FEATURE_COLS].values), y_tr_b, sample_weight=w_quad)
        p_w_quad = m_w_quad.predict(sc_b.transform(X_test))
        res_w_quad = eval_predictions(y_test_log, p_w_quad, test_tb, 'Unmixing-Weighted (Quadratic)', k)
        all_folds_unmix_quad.append(res_w_quad)

    # -------------------------------------------------------------
    # STEP 2 & 4: AGGREGATE ACROSS 5 FOLDS
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print("STEP 2 & 4: MULTI-FOLD SPATIAL BLOCK CV RESULTS (MEAN +/- STD)")
    print("=" * 90)

    model_groups = [
        ('Condition (c) NDWI Full Pool', all_folds_ndwi),
        ('Condition (b) Raw DL Full Pool', all_folds_raw_dl),
        ('Condition (b) Subsampled DL (Matched N)', all_folds_sub_dl),
        ('Condition (b) Hard-Filtered Clean DL (NDVI<=0)', all_folds_clean_dl),
        ('Unmixing-Weighted DL (Linear w=f_water)', all_folds_unmix_lin),
        ('Unmixing-Weighted DL (Quadratic w=f_water^2)', all_folds_unmix_quad),
    ]

    summary_rows = []
    for name, fold_list in model_groups:
        df_f = pd.DataFrame(fold_list)
        micro_mean = df_f['micro_log_r2'].mean()
        micro_std = df_f['micro_log_r2'].std()
        rmse_mean = df_f['micro_log_rmse'].mean()
        rmse_std = df_f['micro_log_rmse'].std()
        macro_mean = df_f['macro_log_r2'].mean()
        macro_std = df_f['macro_log_r2'].std()

        row = {
            'Approach': name,
            'Micro_Log_R2 (Mean +/- Std)': f"{micro_mean:.4f} +/- {micro_std:.4f}",
            'Micro_Log_RMSE': f"{rmse_mean:.4f} +/- {rmse_std:.4f}",
            'Macro_Log_R2 (Mean +/- Std)': f"{macro_mean:.4f} +/- {macro_std:.4f}",
        }
        for b in basins_all:
            col_b = f"{b}_log_r2"
            b_vals = df_f[col_b].dropna()
            if len(b_vals) > 0:
                row[f"{b}_Log_R2"] = f"{b_vals.mean():.4f} +/- {b_vals.std():.4f}"
            else:
                row[f"{b}_Log_R2"] = "N/A"

        summary_rows.append(row)

    df_multi_summary = pd.DataFrame(summary_rows)
    print(df_multi_summary.to_string(index=False))
    df_multi_summary.to_csv('results/tables/turbidity_multifold_cv_consolidated_summary.csv', index=False)

    # Statistical significance across the 5 folds (DL vs NDWI)
    df_dl_f = pd.DataFrame(all_folds_raw_dl)
    df_ndwi_f = pd.DataFrame(all_folds_ndwi)

    micro_diffs = df_dl_f['micro_log_r2'] - df_ndwi_f['micro_log_r2']
    macro_diffs = df_dl_f['macro_log_r2'] - df_ndwi_f['macro_log_r2']

    w_micro, p_micro = stats.wilcoxon(micro_diffs)
    t_micro, pt_micro = stats.ttest_rel(df_dl_f['micro_log_r2'], df_ndwi_f['micro_log_r2'])

    w_macro, p_macro = stats.wilcoxon(macro_diffs)
    t_macro, pt_macro = stats.ttest_rel(df_dl_f['macro_log_r2'], df_ndwi_f['macro_log_r2'])

    print("\n" + "=" * 80)
    print("STEP 2: FORMAL STATISTICAL TEST ACROSS 5 FOLDS (DL vs NDWI)")
    print("=" * 80)
    print(f"Per-Fold Micro Log R^2 Differences (DL - NDWI): {[round(x, 4) for x in micro_diffs]}")
    print(f"  Micro Wilcoxon Signed-Rank Test: W = {w_micro}, p = {p_micro:.4f}")
    print(f"  Micro Paired t-test:              t = {t_micro:.4f}, p = {pt_micro:.4f}")

    print(f"\nPer-Fold Macro Log R^2 Differences (DL - NDWI): {[round(x, 4) for x in macro_diffs]}")
    print(f"  Macro Wilcoxon Signed-Rank Test: W = {w_macro}, p = {p_macro:.4f}")
    print(f"  Macro Paired t-test:              t = {t_macro:.4f}, p = {pt_macro:.4f}")

    # -------------------------------------------------------------
    # STEP 3: PER-BASIN CONSISTENCY CHECK
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 3: PER-BASIN CONSISTENCY AUDIT ACROSS FOLDS")
    print("=" * 80)
    df_basin_folds = pd.DataFrame(per_basin_fold_records)
    basin_consistency = []
    for b in basins_all:
        sub = df_basin_folds[df_basin_folds['basin'] == b]
        dl_wins = np.sum(sub['winner'] == 'DL Mask')
        ndwi_wins = np.sum(sub['winner'] == 'NDWI Mask')
        n_folds = len(sub)
        mean_gap = sub['gap_dl_minus_ndwi'].mean()
        std_gap = sub['gap_dl_minus_ndwi'].std()

        basin_consistency.append({
            'basin': b,
            'folds_present': n_folds,
            'dl_wins_count': dl_wins,
            'ndwi_wins_count': ndwi_wins,
            'dl_win_rate_pct': round(dl_wins / n_folds * 100, 1) if n_folds > 0 else 0,
            'mean_gap_dl_minus_ndwi': round(mean_gap, 4),
            'std_gap': round(std_gap, 4),
            'overall_basin_winner': 'DL Mask' if dl_wins > ndwi_wins else ('NDWI Mask' if ndwi_wins > dl_wins else 'Tie')
        })

    df_consistency = pd.DataFrame(basin_consistency)
    print(df_consistency.to_string(index=False))
    df_consistency.to_csv('results/tables/turbidity_multifold_cv_basin_consistency.csv', index=False)

    # Save per-fold raw outputs
    pd.DataFrame(all_folds_raw_dl).to_csv('results/tables/turbidity_5fold_raw_dl.csv', index=False)
    pd.DataFrame(all_folds_ndwi).to_csv('results/tables/turbidity_5fold_ndwi.csv', index=False)
    pd.DataFrame(all_folds_unmix_lin).to_csv('results/tables/turbidity_5fold_unmix_linear.csv', index=False)
    pd.DataFrame(all_folds_unmix_quad).to_csv('results/tables/turbidity_5fold_unmix_quad.csv', index=False)

    print("\n5-Fold Spatial Block CV completed successfully!")

if __name__ == '__main__':
    main()
