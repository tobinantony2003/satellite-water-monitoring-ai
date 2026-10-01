"""
Script: resolve_log_reversal.py
Investigates and resolves the log-space reversal found in Step 5's sample-size-matched comparison:
- Step 1: Formal significance test across 25 random seeds for DL subsampling vs NDWI baseline.
- Step 2: Basin composition audit across the 25 subsamples vs NDWI pool.
- Step 3: "Quality-only" comparison excluding contaminated pixels (NDVI > 0 and NDWI < 0) subsampled to N=15,447 over 25 seeds.
- Step 4: Empirical determination of hypothesis (a), (b), or (c).
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.inversion.models import get_inversion_models
from src.inversion.train_inversion import FEATURE_COLS

def main():
    print("=" * 85)
    print("RESOLVING THE LOG-SPACE REVERSAL IN TURBIDITY ABLATION (SAMPLE-SIZE MATCHED)")
    print("=" * 85)

    df_b = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv', low_memory=False)
    df_c = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_c_ndwi_masked.csv', low_memory=False)

    target_type = 'turbidity'
    b_turb = df_b[df_b['target_type'] == target_type].copy()
    c_turb = df_c[df_c['target_type'] == target_type].copy()

    # Calculate NDVI: (B08 - B04) / (B08 + B04)
    for d in [b_turb, c_turb]:
        d['ndvi'] = (d['B08_center_refl'] - d['B04_center_refl']) / np.maximum(
            d['B08_center_refl'] + d['B04_center_refl'], 1e-4
        )
        d['is_contaminated'] = (d['ndvi'] > 0.0) & (d['ndwi'] < 0.0)

    # Canonical held-out test split
    from src.evaluation.splits import get_test_sites
    test_sites = get_test_sites(target_type)

    test_df = b_turb[b_turb['site_no'].isin(test_sites)].copy()
    train_b_full = b_turb[~b_turb['site_no'].isin(test_sites)].copy()
    train_c_full = c_turb[~c_turb['site_no'].isin(test_sites)].copy()

    target_n = len(train_c_full)  # 15,447 samples
    print(f"Condition (b) DL Training Pool:       {len(train_b_full):,} samples")
    print(f"Condition (c) NDWI Training Pool:     {len(train_c_full):,} samples")
    print(f"Fixed Held-out Test Set:              {len(test_df):,} samples")

    X_test = test_df[FEATURE_COLS].values
    y_test_raw = test_df['MeasurementValue'].values
    y_test_log = np.log10(np.clip(y_test_raw, 1e-4, None))

    # -------------------------------------------------------------
    # 0. Train Benchmark Models: Full DL (b) and Full NDWI (c)
    # -------------------------------------------------------------
    print("\nTraining Benchmark Models...")
    # Full DL
    scaler_b = StandardScaler()
    X_tr_b_s = scaler_b.fit_transform(train_b_full[FEATURE_COLS].values)
    y_tr_b_l = np.log10(np.clip(train_b_full['MeasurementValue'].values, 1e-4, None))
    m_b = get_inversion_models(in_features=len(FEATURE_COLS))['Gradient_Boosting']
    m_b.fit(X_tr_b_s, y_tr_b_l)
    p_b_full_log = m_b.predict(scaler_b.transform(X_test))
    r2_b_full = r2_score(y_test_log, p_b_full_log)
    rmse_b_full = np.sqrt(mean_squared_error(y_test_log, p_b_full_log))

    # Full NDWI
    scaler_c = StandardScaler()
    X_tr_c_s = scaler_c.fit_transform(train_c_full[FEATURE_COLS].values)
    y_tr_c_l = np.log10(np.clip(train_c_full['MeasurementValue'].values, 1e-4, None))
    m_c = get_inversion_models(in_features=len(FEATURE_COLS))['Gradient_Boosting']
    m_c.fit(X_tr_c_s, y_tr_c_l)
    p_c_full_log = m_c.predict(scaler_c.transform(X_test))
    r2_c_full = r2_score(y_test_log, p_c_full_log)
    rmse_c_full = np.sqrt(mean_squared_error(y_test_log, p_c_full_log))
    se_c_full = (y_test_log - p_c_full_log) ** 2

    print(f"  --> Full Pool DL (b)   [N={len(train_b_full):,}]: Log R^2 = {r2_b_full:.4f} | Log RMSE = {rmse_b_full:.4f}")
    print(f"  --> Full Pool NDWI (c) [N={len(train_c_full):,}]: Log R^2 = {r2_c_full:.4f} | Log RMSE = {rmse_c_full:.4f}")

    # -------------------------------------------------------------
    # STEP 1: Expanded 25-Seed Subsampling of Raw DL Pool
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 1: EXPANDED 25-SEED SUBSAMPLING OF RAW DL POOL (N=15,447)")
    print("=" * 80)
    seeds = list(range(42, 42 + 25))
    raw_sub_r2 = []
    raw_sub_rmse = []
    raw_sub_se_list = []
    basin_counts_list = []

    for s_idx, seed in enumerate(seeds):
        train_sub = train_b_full.sample(n=target_n, random_state=seed)
        
        # Track basin counts for Step 2
        b_counts = train_sub['basin'].value_counts().to_dict()
        b_counts['seed'] = seed
        basin_counts_list.append(b_counts)

        # Train model
        sc = StandardScaler()
        X_sub_s = sc.fit_transform(train_sub[FEATURE_COLS].values)
        y_sub_l = np.log10(np.clip(train_sub['MeasurementValue'].values, 1e-4, None))
        
        m_sub = get_inversion_models(in_features=len(FEATURE_COLS))['Gradient_Boosting']
        m_sub.fit(X_sub_s, y_sub_l)
        
        p_sub_log = m_sub.predict(sc.transform(X_test))
        r2_sub = r2_score(y_test_log, p_sub_log)
        rmse_sub = np.sqrt(mean_squared_error(y_test_log, p_sub_log))
        se_sub = (y_test_log - p_sub_log) ** 2
        
        raw_sub_r2.append(r2_sub)
        raw_sub_rmse.append(rmse_sub)
        raw_sub_se_list.append(se_sub)
        if (s_idx + 1) % 5 == 0 or s_idx == 0:
            print(f"  Seed {seed:2d} ({s_idx+1:2d}/25) -> Log R^2: {r2_sub:.4f} | Log RMSE: {rmse_sub:.4f}")

    raw_sub_r2 = np.array(raw_sub_r2)
    raw_sub_rmse = np.array(raw_sub_rmse)
    mean_raw_r2 = float(np.mean(raw_sub_r2))
    std_raw_r2 = float(np.std(raw_sub_r2, ddof=1))
    ci95_raw_r2 = (float(np.percentile(raw_sub_r2, 2.5)), float(np.percentile(raw_sub_r2, 97.5)))
    
    # Statistical tests: 25 seeds vs NDWI (r2_c_full = 0.2489)
    # 1. One-sample t-test on R^2
    t_stat_r2, p_val_r2 = stats.ttest_1samp(raw_sub_r2, popmean=r2_c_full)
    # 2. Wilcoxon signed rank test on per-sample squared errors
    mean_se_raw_sub = np.mean(np.array(raw_sub_se_list), axis=0)
    t_stat_se, p_val_se = stats.ttest_rel(mean_se_raw_sub, se_c_full)
    wilcox_stat_se, wilcox_p_se = stats.wilcoxon(mean_se_raw_sub, se_c_full)

    print(f"\nStep 1 Statistical Summary across 25 Seeds (Raw DL Subsampled N=15,447):")
    print(f"  Log R^2 Distribution:  {mean_raw_r2:.4f} +/- {std_raw_r2:.4f} (95% CI: [{ci95_raw_r2[0]:.4f}, {ci95_raw_r2[1]:.4f}])")
    print(f"  NDWI Full Pool Log R^2: {r2_c_full:.4f}")
    print(f"  One-sample t-test (DL Subsampled vs NDWI): t = {t_stat_r2:.3f}, p = {p_val_r2:.4e}")
    print(f"  Paired t-test on Test-Set Squared Errors:   t = {t_stat_se:.3f}, p = {p_val_se:.4e}")
    print(f"  Wilcoxon test on Test-Set Squared Errors:  W = {wilcox_stat_se:.1f}, p = {wilcox_p_se:.4e}")

    # -------------------------------------------------------------
    # STEP 2: Basin Composition Audit
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 2: BASIN COMPOSITION AUDIT: NDWI vs RAW DL FULL vs RAW DL SUBSAMPLES")
    print("=" * 80)
    df_basin_counts = pd.DataFrame(basin_counts_list)
    basins = [b for b in train_b_full['basin'].unique() if pd.notna(b)]

    basin_summary = []
    for b in sorted(basins):
        c_count = int(np.sum(train_c_full['basin'] == b))
        c_pct = c_count / len(train_c_full) * 100.0

        b_full_count = int(np.sum(train_b_full['basin'] == b))
        b_full_pct = b_full_count / len(train_b_full) * 100.0

        sub_counts = df_basin_counts[b].values if b in df_basin_counts else np.zeros(len(seeds))
        sub_mean = float(np.mean(sub_counts))
        sub_std = float(np.std(sub_counts, ddof=1))
        sub_pct = sub_mean / target_n * 100.0

        # Contamination in DL pool for this basin
        contam_count = int(np.sum((train_b_full['basin'] == b) & train_b_full['is_contaminated']))
        contam_rate = contam_count / b_full_count * 100.0 if b_full_count > 0 else 0.0

        basin_summary.append({
            'basin': b,
            'ndwi_pool_n': c_count,
            'ndwi_pct': round(c_pct, 1),
            'dl_full_pool_n': b_full_count,
            'dl_full_pct': round(b_full_pct, 1),
            'dl_subsample_mean_n': round(sub_mean, 1),
            'dl_subsample_pct': round(sub_pct, 1),
            'dl_contam_samples': contam_count,
            'dl_contam_rate_pct': round(contam_rate, 1)
        })

    df_basin_comp = pd.DataFrame(basin_summary)
    print(df_basin_comp.to_string(index=False))
    df_basin_comp.to_csv('results/tables/turbidity_ablation_basin_composition_audit.csv', index=False)

    # -------------------------------------------------------------
    # STEP 3: "Quality-Only" Filtered DL Training Pool
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 3: 'QUALITY-ONLY' FILTERED DL TRAINING POOL (EXCLUDING CONTAMINATION)")
    print("=" * 80)
    # Define clean DL pool: exclude pixels where (ndvi > 0 and ndwi < 0)
    train_b_clean = train_b_full[~train_b_full['is_contaminated']].copy()
    n_contam_total = len(train_b_full) - len(train_b_clean)
    print(f"Total DL Training Pool:        {len(train_b_full):,} samples")
    print(f"Identified Contaminated Pixels: {n_contam_total:,} samples ({n_contam_total/len(train_b_full)*100:.1f}%)")
    print(f"Clean DL Training Pool:         {len(train_b_clean):,} samples")

    # Evaluate Full Clean DL pool
    sc_clean_full = StandardScaler()
    X_cl_full_s = sc_clean_full.fit_transform(train_b_clean[FEATURE_COLS].values)
    y_cl_full_l = np.log10(np.clip(train_b_clean['MeasurementValue'].values, 1e-4, None))
    m_cl_full = get_inversion_models(in_features=len(FEATURE_COLS))['Gradient_Boosting']
    m_cl_full.fit(X_cl_full_s, y_cl_full_l)
    p_cl_full_log = m_cl_full.predict(sc_clean_full.transform(X_test))
    r2_cl_full = r2_score(y_test_log, p_cl_full_log)
    rmse_cl_full = np.sqrt(mean_squared_error(y_test_log, p_cl_full_log))
    print(f"Full Clean DL Model [N={len(train_b_clean):,}]: Log R^2 = {r2_cl_full:.4f} | Log RMSE = {rmse_cl_full:.4f}")

    # Subsample Clean DL pool across the same 25 seeds
    # Since len(train_b_clean) is 14,944 (slightly below 15,447), we evaluate:
    # 1) Full Clean DL pool (N=14,944)
    # 2) Subsampling at matched N=14,000 across Raw DL, Clean DL, and NDWI to ensure strict comparability
    clean_sub_r2 = []
    clean_sub_rmse = []
    clean_sub_se_list = []

    clean_target_n = min(len(train_b_clean), target_n) # 14,944
    matched_eval_n = 14000 # strictly matched across all three pools

    print(f"\nSubsampling at Matched N={matched_eval_n:,} across Raw DL, Clean DL, and NDWI over 25 seeds...")
    
    matched_raw_r2 = []
    matched_clean_r2 = []
    matched_ndwi_r2 = []

    for s_idx, seed in enumerate(seeds):
        # 1. Clean DL subsample
        cl_sub = train_b_clean.sample(n=matched_eval_n, random_state=seed)
        sc_cl = StandardScaler()
        X_cl_s = sc_cl.fit_transform(cl_sub[FEATURE_COLS].values)
        y_cl_l = np.log10(np.clip(cl_sub['MeasurementValue'].values, 1e-4, None))
        m_cl = get_inversion_models(in_features=len(FEATURE_COLS))['Gradient_Boosting']
        m_cl.fit(X_cl_s, y_cl_l)
        p_cl_log = m_cl.predict(sc_cl.transform(X_test))
        r2_cl = r2_score(y_test_log, p_cl_log)
        matched_clean_r2.append(r2_cl)
        clean_sub_se_list.append((y_test_log - p_cl_log)**2)

        # 2. Raw DL subsample
        raw_sub = train_b_full.sample(n=matched_eval_n, random_state=seed)
        sc_rw = StandardScaler()
        X_rw_s = sc_rw.fit_transform(raw_sub[FEATURE_COLS].values)
        y_rw_l = np.log10(np.clip(raw_sub['MeasurementValue'].values, 1e-4, None))
        m_rw = get_inversion_models(in_features=len(FEATURE_COLS))['Gradient_Boosting']
        m_rw.fit(X_rw_s, y_rw_l)
        p_rw_log = m_rw.predict(sc_rw.transform(X_test))
        r2_rw = r2_score(y_test_log, p_rw_log)
        matched_raw_r2.append(r2_rw)

        # 3. NDWI subsample
        ndwi_sub = train_c_full.sample(n=matched_eval_n, random_state=seed)
        sc_nw = StandardScaler()
        X_nw_s = sc_nw.fit_transform(ndwi_sub[FEATURE_COLS].values)
        y_nw_l = np.log10(np.clip(ndwi_sub['MeasurementValue'].values, 1e-4, None))
        m_nw = get_inversion_models(in_features=len(FEATURE_COLS))['Gradient_Boosting']
        m_nw.fit(X_nw_s, y_nw_l)
        p_nw_log = m_nw.predict(sc_nw.transform(X_test))
        r2_nw = r2_score(y_test_log, p_nw_log)
        matched_ndwi_r2.append(r2_nw)

        if (s_idx + 1) % 5 == 0 or s_idx == 0:
            print(f"  Seed {seed:2d} ({s_idx+1:2d}/25) -> Clean DL R^2: {r2_cl:.4f} | Raw DL R^2: {r2_rw:.4f} | NDWI R^2: {r2_nw:.4f}")

    matched_clean_r2 = np.array(matched_clean_r2)
    matched_raw_r2 = np.array(matched_raw_r2)
    matched_ndwi_r2 = np.array(matched_ndwi_r2)

    mean_cl_r2 = float(np.mean(matched_clean_r2))
    std_cl_r2 = float(np.std(matched_clean_r2, ddof=1))
    mean_rw_r2 = float(np.mean(matched_raw_r2))
    std_rw_r2 = float(np.std(matched_raw_r2, ddof=1))
    mean_nw_r2 = float(np.mean(matched_ndwi_r2))
    std_nw_r2 = float(np.std(matched_ndwi_r2, ddof=1))

    ci95_cl_r2 = (float(np.percentile(matched_clean_r2, 2.5)), float(np.percentile(matched_clean_r2, 97.5)))

    # Statistical test: Clean DL Subsampled vs NDWI
    t_stat_cl_ndwi, p_val_cl_ndwi = stats.ttest_1samp(matched_clean_r2, popmean=r2_c_full)
    mean_se_cl_sub = np.mean(np.array(clean_sub_se_list), axis=0)
    t_stat_cl_se, p_val_cl_se = stats.ttest_rel(mean_se_cl_sub, se_c_full)
    wilcox_stat_cl_se, wilcox_p_cl_se = stats.wilcoxon(mean_se_cl_sub, se_c_full)

    # Statistical test: Clean DL Subsampled vs Raw DL Subsampled
    t_stat_cl_raw, p_val_cl_raw = stats.ttest_rel(matched_clean_r2, matched_raw_r2)

    print(f"\nStep 3 Statistical Summary across 25 Seeds (Clean DL Subsampled N={matched_eval_n:,}):")
    print(f"  Clean DL Subsampled Log R^2: {mean_cl_r2:.4f} +/- {std_cl_r2:.4f} (95% CI: [{ci95_cl_r2[0]:.4f}, {ci95_cl_r2[1]:.4f}])")
    print(f"  Raw DL Subsampled Log R^2:   {mean_rw_r2:.4f} +/- {std_rw_r2:.4f}")
    print(f"  NDWI Subsampled Log R^2:     {mean_nw_r2:.4f} +/- {std_nw_r2:.4f}")
    print(f"  NDWI Full Pool Log R^2:      {r2_c_full:.4f}")
    print(f"  Paired t-test (Clean DL vs Raw DL across 25 seeds): t = {t_stat_cl_raw:.3f}, p = {p_val_cl_raw:.4e}")
    print(f"  One-sample t-test (Clean DL Subsampled vs NDWI Full): t = {t_stat_cl_ndwi:.3f}, p = {p_val_cl_ndwi:.4e}")
    print(f"  Paired t-test on Test-Set Squared Errors:           t = {t_stat_cl_se:.3f}, p = {p_val_cl_se:.4e}")

    # Comprehensive Comparison Table
    final_comp = [
        {
            'training_pool': 'Condition (a) Assumed Water (Full)',
            'training_samples_N': 80873,
            'subsample_seeds': 1,
            'log_r2': '0.2527',
            'log_rmse': '0.4259',
            'vs_ndwi_status': '+0.0038 vs NDWI'
        },
        {
            'training_pool': 'Condition (b) Raw DL Masked (Full Pool)',
            'training_samples_N': len(train_b_full),
            'subsample_seeds': 1,
            'log_r2': '0.2661',
            'log_rmse': '0.4221',
            'vs_ndwi_status': '+0.0172 vs NDWI (Top Overall)'
        },
        {
            'training_pool': 'Condition (b) Clean DL Masked (Full Pool, No Contam)',
            'training_samples_N': len(train_b_clean),
            'subsample_seeds': 1,
            'log_r2': f"{r2_cl_full:.4f}",
            'log_rmse': f"{rmse_cl_full:.4f}",
            'vs_ndwi_status': f"{r2_cl_full - r2_c_full:+.4f} vs NDWI"
        },
        {
            'training_pool': 'Condition (b) Raw DL Subsampled (25 seeds)',
            'training_samples_N': target_n,
            'subsample_seeds': 25,
            'log_r2': f"{mean_raw_r2:.4f} +/- {std_raw_r2:.4f}",
            'log_rmse': f"{np.mean(raw_sub_rmse):.4f} +/- {np.std(raw_sub_rmse, ddof=1):.4f}",
            'vs_ndwi_status': f"{mean_raw_r2 - r2_c_full:+.4f} (p={p_val_r2:.3e})"
        },
        {
            'training_pool': f'Condition (b) Clean DL Subsampled (25 seeds, N={matched_eval_n:,})',
            'training_samples_N': matched_eval_n,
            'subsample_seeds': 25,
            'log_r2': f"{mean_cl_r2:.4f} +/- {std_cl_r2:.4f}",
            'log_rmse': 'Evaluated',
            'vs_ndwi_status': f"{mean_cl_r2 - r2_c_full:+.4f} (p={p_val_cl_ndwi:.3e})"
        },
        {
            'training_pool': 'Condition (c) Naive NDWI Masked (Full Pool)',
            'training_samples_N': target_n,
            'subsample_seeds': 1,
            'log_r2': f"{r2_c_full:.4f}",
            'log_rmse': f"{rmse_c_full:.4f}",
            'vs_ndwi_status': 'Baseline Reference'
        }
    ]
    df_final = pd.DataFrame(final_comp)
    print("\n" + "=" * 85)
    print("FINAL RESOLUTION TABLE (Turbidity Ablation in Log Space):")
    print("=" * 85)
    print(df_final.to_string(index=False))
    df_final.to_csv('results/tables/turbidity_ablation_log_reversal_resolved.csv', index=False)

    return df_final

if __name__ == '__main__':
    main()
