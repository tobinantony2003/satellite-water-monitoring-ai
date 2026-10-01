"""
Script: diagnose_upper_colorado_failure.py
Step 1: Diagnoses Upper Colorado's DL failure mode.
Compares spectral signatures and target distributions between:
- Both_Retained vs DL_Exclusive in Upper Colorado
- Directly compares Upper Colorado against Illinois, Willamette, and Trinity.
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

def main():
    print("=" * 85)
    print("STEP 1: UPPER COLORADO DL FAILURE MODE DIAGNOSTIC")
    print("=" * 85)

    df_b = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv', low_memory=False)
    df_c = pd.read_csv('data/processed/matched_wq_cleaned/matched_set_c_ndwi_masked.csv', low_memory=False)

    turb_b = df_b[df_b['target_type'] == 'turbidity'].copy()
    turb_c = df_c[df_c['target_type'] == 'turbidity'].copy()

    # Create matching key
    turb_b['key'] = turb_b['site_no'].astype(str) + '_' + turb_b['sample_datetime_UTC'].astype(str)
    c_keys = set(turb_c['site_no'].astype(str) + '_' + turb_c['sample_datetime_UTC'].astype(str))
    turb_b['in_ndwi'] = turb_b['key'].isin(c_keys)

    turb_b['category'] = np.where(turb_b['in_ndwi'], 'Both_Retained', 'DL_Exclusive')

    # Indices
    eps = 1e-4
    turb_b['ndvi'] = (turb_b['B08_center_refl'] - turb_b['B04_center_refl']) / np.maximum(
        turb_b['B08_center_refl'] + turb_b['B04_center_refl'], eps
    )
    turb_b['ndwi_calc'] = (turb_b['B03_center_refl'] - turb_b['B08_center_refl']) / np.maximum(
        turb_b['B03_center_refl'] + turb_b['B08_center_refl'], eps
    )
    turb_b['mndwi_calc'] = (turb_b['B03_center_refl'] - turb_b['B11_center_refl']) / np.maximum(
        turb_b['B03_center_refl'] + turb_b['B11_center_refl'], eps
    )
    turb_b['swir_ratio'] = turb_b['B11_center_refl'] / np.maximum(turb_b['B12_center_refl'], eps)
    turb_b['b8_b4_ratio'] = turb_b['B08_center_refl'] / np.maximum(turb_b['B04_center_refl'], eps)
    turb_b['total_reflectance'] = turb_b['B02_center_refl'] + turb_b['B03_center_refl'] + turb_b['B04_center_refl'] + turb_b['B08_center_refl']

    # Focus on Upper Colorado
    uc = turb_b[turb_b['basin'] == 'Upper Colorado'].copy()
    print(f"Upper Colorado Total Turbidity Samples (DL Mask): {len(uc):,}")
    print(f"  Both_Retained (NDWI > 0):  {np.sum(uc['category'] == 'Both_Retained'):,} ({np.mean(uc['category'] == 'Both_Retained')*100:.1f}%)")
    print(f"  DL_Exclusive (NDWI <= 0):  {np.sum(uc['category'] == 'DL_Exclusive'):,} ({np.mean(uc['category'] == 'DL_Exclusive')*100:.1f}%)")

    # Spectral comparison within Upper Colorado
    spectral_cols = [
        'B02_center_refl', 'B03_center_refl', 'B04_center_refl',
        'B08_center_refl', 'B11_center_refl', 'B12_center_refl',
        'ndvi', 'ndwi_calc', 'mndwi_calc', 'swir_ratio', 'b8_b4_ratio', 'total_reflectance',
        'MeasurementValue'
    ]

    uc_both = uc[uc['category'] == 'Both_Retained']
    uc_dl = uc[uc['category'] == 'DL_Exclusive']

    comp_rows = []
    for col in spectral_cols:
        b_vals = uc_both[col].dropna().values
        dl_vals = uc_dl[col].dropna().values

        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.ks_2samp(b_vals, dl_vals)

        comp_rows.append({
            'variable': col,
            'both_retained_mean': np.mean(b_vals),
            'both_retained_std': np.std(b_vals),
            'both_retained_median': np.median(b_vals),
            'dl_exclusive_mean': np.mean(dl_vals),
            'dl_exclusive_std': np.std(dl_vals),
            'dl_exclusive_median': np.median(dl_vals),
            'ratio_dl_vs_both': np.mean(dl_vals) / np.maximum(np.mean(b_vals), 1e-4),
            'ks_stat': ks_stat,
            'ks_pval': ks_pval
        })

    df_uc_comp = pd.DataFrame(comp_rows)
    print("\nUpper Colorado Spectral & Target Audit: Both_Retained vs DL_Exclusive:")
    print(df_uc_comp[['variable', 'both_retained_mean', 'dl_exclusive_mean', 'ratio_dl_vs_both', 'both_retained_median', 'dl_exclusive_median']].round(4).to_string(index=False))

    # Cross-basin comparison of DL_Exclusive pixels
    cross_basin_records = []
    for b in ['Upper Colorado', 'Illinois', 'Willamette', 'Trinity']:
        sub_b = turb_b[turb_b['basin'] == b]
        dl_ex = sub_b[sub_b['category'] == 'DL_Exclusive']
        both_r = sub_b[sub_b['category'] == 'Both_Retained']
        
        cross_basin_records.append({
            'basin': b,
            'dl_exclusive_N': len(dl_ex),
            'pct_of_basin_dl': round(len(dl_ex) / len(sub_b) * 100, 1),
            'dl_ex_b04_red': dl_ex['B04_center_refl'].mean(),
            'dl_ex_b08_nir': dl_ex['B08_center_refl'].mean(),
            'dl_ex_b11_swir1': dl_ex['B11_center_refl'].mean(),
            'dl_ex_ndvi': dl_ex['ndvi'].mean(),
            'dl_ex_ndwi': dl_ex['ndwi_calc'].mean(),
            'dl_ex_mndwi': dl_ex['mndwi_calc'].mean(),
            'dl_ex_swir_ratio': dl_ex['swir_ratio'].mean(),
            'dl_ex_turbidity_mean': dl_ex['MeasurementValue'].mean(),
            'dl_ex_turbidity_median': dl_ex['MeasurementValue'].median(),
            'both_turbidity_median': both_r['MeasurementValue'].median() if len(both_r) > 0 else np.nan
        })

    df_cross = pd.DataFrame(cross_basin_records)
    print("\nCross-Basin Comparison of DL-Exclusive Pixels:")
    print(df_cross[['basin', 'dl_exclusive_N', 'pct_of_basin_dl', 'dl_ex_ndvi', 'dl_ex_ndwi', 'dl_ex_mndwi', 'dl_ex_swir_ratio', 'dl_ex_turbidity_median', 'both_turbidity_median']].round(4).to_string(index=False))

    # Check station breakdown in Upper Colorado
    site_breakdown = uc.groupby('site_no').agg(
        n_total=('MeasurementValue', 'count'),
        n_both=('category', lambda s: (s == 'Both_Retained').sum()),
        n_dl_ex=('category', lambda s: (s == 'DL_Exclusive').sum()),
        dl_ex_pct=('category', lambda s: (s == 'DL_Exclusive').mean() * 100),
        median_turb=('MeasurementValue', 'median'),
        mean_turb=('MeasurementValue', 'mean')
    ).reset_index().sort_values('n_total', ascending=False)

    print("\nUpper Colorado Station Breakdown (Top Stations):")
    print(site_breakdown.head(10).round(1).to_string(index=False))

    # Save tables
    df_uc_comp.to_csv('results/tables/upper_colorado_dl_spectral_diagnostic.csv', index=False)
    df_cross.to_csv('results/tables/cross_basin_dl_exclusive_comparison.csv', index=False)
    site_breakdown.to_csv('results/tables/upper_colorado_station_breakdown.csv', index=False)

    return df_uc_comp, df_cross, site_breakdown

if __name__ == '__main__':
    main()
