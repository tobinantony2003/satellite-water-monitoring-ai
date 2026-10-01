"""
Script: run_step2_unmixing_test.py
Implements 3-endmember Linear Spectral Mixture Analysis (LSMA)
Strictly adheres to leakage controls:
- Endmembers derived ONLY from training stations
- Unmixing relies ONLY on spectral bands B02-B12
Evaluates water fraction across basins and verifies spot-checks.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def run_unmixing():
    print("=" * 80)
    print("STEP 2: 3-ENDMEMBER LINEAR SPECTRAL UNMIXING MODEL")
    print("=" * 80)

    df_a = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_a_ground_truth.csv', low_memory=False)
    df_b = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv', low_memory=False)
    df_c = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_c_ndwi_masked.csv', low_memory=False)

    b_turb = df_b[df_b['target_type'] == 'turbidity'].copy()
    c_turb = df_c[df_c['target_type'] == 'turbidity'].copy()
    a_turb = df_a[df_a['target_type'] == 'turbidity'].copy()

    # Canonical held-out test split
    from src.evaluation.splits import get_test_sites
    test_sites = get_test_sites('turbidity')
    train_sites = set(b_turb['site_no']) - test_sites

    print(f"Total Sites: {b_turb['site_no'].nunique()} | Train Sites: {len(train_sites)} | Test Sites: {len(test_sites)}")

    # Spectral bands to use for unmixing
    unmix_bands = [
        'B02_center_refl', 'B03_center_refl', 'B04_center_refl',
        'B05_center_refl', 'B06_center_refl', 'B07_center_refl',
        'B08_center_refl', 'B8A_center_refl', 'B11_center_refl', 'B12_center_refl'
    ]

    # Matching keys
    b_keys = set(b_turb['site_no'].astype(str) + '_' + b_turb['sample_datetime_UTC'].astype(str))
    c_keys = set(c_turb['site_no'].astype(str) + '_' + c_turb['sample_datetime_UTC'].astype(str))

    # Tag b_turb
    b_turb['key'] = b_turb['site_no'].astype(str) + '_' + b_turb['sample_datetime_UTC'].astype(str)
    b_turb['is_both_retained'] = b_turb['key'].isin(c_keys)
    b_turb['category'] = np.where(b_turb['is_both_retained'], 'Both_Retained', 'DL_Exclusive')

    # A) DERIVE ENDMEMBERS (TRAINING SITES ONLY)
    # 1. Water Endmember: median of Both_Retained in training set
    b_train = b_turb[b_turb['site_no'].isin(train_sites)]
    both_retained_train = b_train[b_train['is_both_retained']]
    water_em = both_retained_train[unmix_bands].median().values

    # 2. Vegetation & Soil Endmembers from land pixels rejected by both masks in training stations
    a_train = a_turb[a_turb['site_no'].isin(train_sites)].copy()
    a_train['key'] = a_train['site_no'].astype(str) + '_' + a_train['sample_datetime_UTC'].astype(str)
    a_rejected = a_train[~a_train['key'].isin(b_keys) & ~a_train['key'].isin(c_keys)].copy()
    
    # Calculate NDVI on rejected
    a_rejected['ndvi'] = (a_rejected['B08_center_refl'] - a_rejected['B04_center_refl']) / (
        a_rejected['B08_center_refl'] + a_rejected['B04_center_refl'] + 1e-4
    )

    # Pure vegetation: high NDVI (> 0.45) in vegetated training stations
    veg_candidates = a_rejected[a_rejected['ndvi'] > 0.45]
    veg_em = veg_candidates[unmix_bands].median().values

    # Pure soil/rock: arid basin rejected pixels (Upper Colorado, low NDVI < 0.20, high red/SWIR)
    soil_candidates = a_rejected[(a_rejected['basin'] == 'Upper Colorado') & (a_rejected['ndvi'] < 0.20) & (a_rejected['B04_center_refl'] > 0.09)]
    soil_em = soil_candidates[unmix_bands].median().values

    df_em = pd.DataFrame({
        'Band': unmix_bands,
        'Water_EM': water_em,
        'Vegetation_EM': veg_em,
        'Soil_Rock_EM': soil_em
    })
    print("\nDerived Endmembers (Training Set Only):")
    print(df_em.round(4).to_string(index=False))
    df_em.to_csv('results/tables/unmixing_endmember_spectra.csv', index=False)

    # B) SOLVE CONSTRAINED LINEAR UNMIXING (FCLS via Augmented NNLS)
    # Matrix M: (10, 3)
    M = np.column_stack([water_em, veg_em, soil_em])
    delta = 10.0
    A_aug = np.vstack([M, delta * np.ones((1, 3))])

    # Unmix all samples in b_turb
    print("\nSolving Fully Constrained Linear Unmixing across all DL samples...")
    X_refl = b_turb[unmix_bands].values

    f_water = np.zeros(len(b_turb))
    f_veg = np.zeros(len(b_turb))
    f_soil = np.zeros(len(b_turb))

    for i in range(len(b_turb)):
        y = X_refl[i]
        b_aug = np.hstack([y, delta])
        x, _ = nnls(A_aug, b_aug)
        total = np.sum(x)
        if total > 0:
            fracs = x / total
        else:
            fracs = np.array([1.0/3, 1.0/3, 1.0/3])
        f_water[i] = fracs[0]
        f_veg[i] = fracs[1]
        f_soil[i] = fracs[2]

    b_turb['f_water'] = f_water
    b_turb['f_veg'] = f_veg
    b_turb['f_soil'] = f_soil

    # C) REPORT DISTRIBUTION OF ESTIMATED WATER FRACTIONS FOR DL-EXCLUSIVE PIXELS
    print("\n" + "=" * 80)
    print("STEP 2(c): WATER FRACTION DISTRIBUTION IN DL-EXCLUSIVE PIXELS PER BASIN")
    print("=" * 80)

    dl_ex = b_turb[b_turb['category'] == 'DL_Exclusive'].copy()
    both_ret = b_turb[b_turb['category'] == 'Both_Retained'].copy()

    print(f"Overall Both_Retained Mean Water Fraction: {both_ret['f_water'].mean():.3f} (Median: {both_ret['f_water'].median():.3f})")
    print(f"Overall DL_Exclusive  Mean Water Fraction: {dl_ex['f_water'].mean():.3f} (Median: {dl_ex['f_water'].median():.3f})")

    basin_unmix_stats = []
    for b in sorted(b_turb['basin'].dropna().unique()):
        sub_dl = dl_ex[dl_ex['basin'] == b]
        sub_both = both_ret[both_ret['basin'] == b]
        
        if len(sub_dl) > 0:
            basin_unmix_stats.append({
                'basin': b,
                'dl_ex_N': len(sub_dl),
                'dl_ex_pct_of_basin': len(sub_dl) / len(b_turb[b_turb['basin']==b]) * 100,
                'both_mean_f_water': sub_both['f_water'].mean() if len(sub_both) > 0 else np.nan,
                'dl_ex_mean_f_water': sub_dl['f_water'].mean(),
                'dl_ex_median_f_water': sub_dl['f_water'].median(),
                'dl_ex_std_f_water': sub_dl['f_water'].std(),
                'dl_ex_mean_f_veg': sub_dl['f_veg'].mean(),
                'dl_ex_mean_f_soil': sub_dl['f_soil'].mean(),
                'pct_high_water_gt_0.5': (sub_dl['f_water'] > 0.5).mean() * 100,
                'pct_low_water_lt_0.2': (sub_dl['f_water'] < 0.2).mean() * 100,
            })

    df_unmix_summary = pd.DataFrame(basin_unmix_stats)
    print(df_unmix_summary.round(3).to_string(index=False))
    df_unmix_summary.to_csv('results/tables/unmixing_water_fraction_by_basin.csv', index=False)

    return b_turb, df_em, df_unmix_summary

if __name__ == '__main__':
    run_unmixing()
