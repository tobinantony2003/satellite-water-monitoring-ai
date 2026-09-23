"""
Module: apply_unit_harmonization.py
Implements physical unit harmonization for CDOM, Chlorophyll-a, and Turbidity:
1. Retains parm_cd and parm_unit explicitly.
2. Assigns harmonization_flag: 'native_ugL', 'converted_from_RFU', or 'excluded_no_calibration'.
3. Generates:
   - Primary datasets (native physical units only: ug/l, ug/l QSE, FNU; drops uncalibrated RFU).
   - Secondary datasets (converts RFU for sites with concurrent calibration ratios).
4. Produces unit_harmonization_sample_loss.csv.
"""

import os
import shutil
import numpy as np
import pandas as pd

# Calibration ratios derived from concurrent timestamps at identical stations
SITE_CALIBRATION_RATIOS = {
    'cdom': {
        '01467200': 3.0000  # ug/l QSE / RFU
    },
    'chlorophyll': {
        '01427510': 3.9554,       # ug/l / RFU
        '444306122144600': 2.8167, # ug/l / RFU
        '453027122400000': 4.1860  # ug/l / RFU
    }
}

def harmonize_dataframe(df_input: pd.DataFrame, dataset_name: str = 'b_dl_masked'):
    """
    Applies unit harmonization to a matched dataframe.
    Returns:
      df_primary: native physical units only
      df_secondary: native + converted RFU from calibrated sites
      sample_loss_df: site-level audit table
    """
    df = df_input.copy()
    df['site_no'] = df['site_no'].astype(str).str.zfill(8)
    
    # Initialize harmonization flag
    df['harmonization_flag'] = 'excluded_no_calibration'
    df['MeasurementValue_harmonized'] = df['MeasurementValue']
    
    # 1. Native physical units
    # Turbidity: FNU (parm_cd 63680)
    turb_mask = (df['target_type'] == 'turbidity') & (df['parm_unit'] == 'FNU')
    df.loc[turb_mask, 'harmonization_flag'] = 'native_ugL'
    
    # CDOM: ug/l QSE (parm_cd 32295)
    cdom_native_mask = (df['target_type'] == 'cdom') & (df['parm_unit'] == 'ug/l QSE')
    df.loc[cdom_native_mask, 'harmonization_flag'] = 'native_ugL'
    
    # Chlorophyll: ug/l (parm_cd 32316, 32318, 62361)
    chl_native_mask = (df['target_type'] == 'chlorophyll') & (df['parm_unit'] == 'ug/l')
    df.loc[chl_native_mask, 'harmonization_flag'] = 'native_ugL'
    
    # 2. Converted RFU for sites with site-specific calibration ratios
    # CDOM
    for site, ratio in SITE_CALIBRATION_RATIOS['cdom'].items():
        rfu_mask = (df['target_type'] == 'cdom') & (df['site_no'] == site) & (df['parm_unit'] == 'RFU')
        df.loc[rfu_mask, 'MeasurementValue_harmonized'] = df.loc[rfu_mask, 'MeasurementValue'] * ratio
        df.loc[rfu_mask, 'harmonization_flag'] = 'converted_from_RFU'
        
    # Chlorophyll
    for site, ratio in SITE_CALIBRATION_RATIOS['chlorophyll'].items():
        rfu_mask = (df['target_type'] == 'chlorophyll') & (df['site_no'] == site) & (df['parm_unit'] == 'RFU')
        df.loc[rfu_mask, 'MeasurementValue_harmonized'] = df.loc[rfu_mask, 'MeasurementValue'] * ratio
        df.loc[rfu_mask, 'harmonization_flag'] = 'converted_from_RFU'
        
    # Primary dataset: keep strictly native physical units
    df_primary = df[df['harmonization_flag'] == 'native_ugL'].copy()
    
    # Secondary dataset: keep native + converted RFU that do not duplicate an existing native record
    converted_rfu = df[df['harmonization_flag'] == 'converted_from_RFU'].copy()
    converted_rfu['MeasurementValue'] = converted_rfu['MeasurementValue_harmonized']
    
    # Existing keys in primary
    dedup_cols = ['scene', 'site_no', 'sample_datetime_UTC', 'target_type']
    valid_cols = [c for c in dedup_cols if c in df.columns]
    
    primary_keys = set(tuple(x) for x in df_primary[valid_cols].values)
    rfu_keys = [tuple(x) for x in converted_rfu[valid_cols].values]
    keep_rfu_mask = [k not in primary_keys for k in rfu_keys]
    
    converted_rfu_unique = converted_rfu[keep_rfu_mask].copy()
    df_secondary = pd.concat([df_primary, converted_rfu_unique], ignore_index=True).sort_index()
    
    # Log sample loss per site and target
    loss_records = []
    for (target, site), grp in df.groupby(['target_type', 'site_no']):
        station_nm = grp['station_nm'].iloc[0] if 'station_nm' in grp.columns else 'N/A'
        basin = grp['basin'].iloc[0] if 'basin' in grp.columns else 'N/A'
        units = '/'.join(sorted(grp['parm_unit'].dropna().unique()))
        codes = '/'.join(sorted([str(c) for c in grp['parm_cd'].dropna().unique()]))
        
        n_before = len(grp)
        n_primary = len(grp[grp['harmonization_flag'] == 'native_ugL'])
        n_converted = len(grp[grp['harmonization_flag'] == 'converted_from_RFU'])
        
        sec_sub = df_secondary[(df_secondary['target_type'] == target) & (df_secondary['site_no'] == site)]
        n_sec_total = len(sec_sub)
        n_excluded = n_before - n_primary
        
        if n_before == n_primary:
            reason = '100% Native Physical Units'
        elif n_converted > 0 and n_primary > 0:
            reason = 'Partial RFU converted via site calibration'
        elif n_converted > 0 and n_primary == 0:
            reason = 'All RFU converted via site calibration'
        else:
            reason = 'RFU-only sensor with no USGS calibration reference'
            
        loss_records.append({
            'dataset': dataset_name,
            'target_type': target,
            'site_no': site,
            'station_nm': station_nm,
            'basin': basin,
            'parm_cd': codes,
            'parm_unit': units,
            'n_before': n_before,
            'n_primary_native': n_primary,
            'n_secondary_converted': n_converted,
            'n_secondary_total': n_sec_total,
            'n_excluded_primary': n_excluded,
            'reason': reason
        })
        
    sample_loss_df = pd.DataFrame(loss_records)
    return df_primary, df_secondary, sample_loss_df

def main():
    print("=" * 80)
    print("STEP 2: Building Harmonized Matched Datasets (Primary & Secondary)")
    print("=" * 80)
    
    clean_dir = 'data/processed/matched_wq_cleaned'
    tables_dir = 'results/tables'
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    
    datasets = {
        'a_ground_truth': 'matched_set_a_ground_truth.csv',
        'b_dl_masked': 'matched_set_b_dl_masked.csv',
        'c_ndwi_masked': 'matched_set_c_ndwi_masked.csv'
    }
    
    all_loss_records = []
    
    for key, filename in datasets.items():
        path = os.path.join(clean_dir, filename)
        if not os.path.exists(path):
            print(f"File {path} not found, skipping.")
            continue
            
        # Backup original raw unit-mixed version if not already backed up
        backup_path = os.path.join(clean_dir, f"matched_set_{key}_unitmixed_raw.csv")
        if not os.path.exists(backup_path):
            shutil.copy(path, backup_path)
            print(f"  Backed up raw unit-mixed dataset to {backup_path}")
            
        print(f"\nProcessing {filename} from {backup_path}...")
        df_raw = pd.read_csv(backup_path, low_memory=False)
            
        df_primary, df_secondary, loss_df = harmonize_dataframe(df_raw, dataset_name=key)
        all_loss_records.append(loss_df)
        
        # Save Primary (native-only) version overwriting the standard path
        df_primary.to_csv(path, index=False)
        print(f"  Saved PRIMARY (Native Only) dataset -> {path} ({len(df_primary):,} rows)")
        
        # Save Secondary (harmonized conversion) version
        sec_path = os.path.join(clean_dir, f"matched_set_{key}_harmonized.csv")
        df_secondary.to_csv(sec_path, index=False)
        print(f"  Saved SECONDARY (Harmonized Converted) dataset -> {sec_path} ({len(df_secondary):,} rows)")
        
        # Target distribution comparison in primary vs secondary vs raw
        for t in ['cdom', 'chlorophyll', 'turbidity']:
            n_raw = len(df_raw[df_raw['target_type'] == t])
            n_prim = len(df_primary[df_primary['target_type'] == t])
            n_sec = len(df_secondary[df_secondary['target_type'] == t])
            print(f"    Target {t.upper():12s} | Raw: {n_raw:,} -> Primary: {n_prim:,} -> Secondary: {n_sec:,}")
            
    # Save consolidated unit harmonization sample loss table
    total_loss_df = pd.concat(all_loss_records, ignore_index=True)
    loss_csv_path = os.path.join(tables_dir, 'unit_harmonization_sample_loss.csv')
    total_loss_df.to_csv(loss_csv_path, index=False)
    print(f"\nSaved sample loss audit table to {loss_csv_path}")
    
    # Print summary for set b (flagship DL-masked set)
    b_loss = total_loss_df[total_loss_df['dataset'] == 'b_dl_masked']
    print("\n=== SET B (DL-MASKED) UNIT HARMONIZATION SUMMARY ===")
    print(b_loss[['target_type', 'site_no', 'station_nm', 'parm_unit', 'n_before', 'n_primary_native', 'n_excluded_primary', 'reason']].to_string(index=False))

if __name__ == '__main__':
    main()
