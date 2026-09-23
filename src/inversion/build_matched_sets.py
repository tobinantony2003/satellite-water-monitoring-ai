import os
import glob
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from src.segmentation.spectral_model import (
    DeepSpectralWaterClassifier,
    train_spectral_water_classifier
)

# Basin mapping from HUC4
HUC4_TO_BASIN = {
    204: 'Delaware',
    712: 'Illinois',
    713: 'Illinois',
    1203: 'Trinity',
    1204: 'Trinity',
    1401: 'Upper Colorado',
    1402: 'Upper Colorado',
    1709: 'Willamette'
}

TARGET_PARAMS = {
    'chlorophyll': [32316, 32315, 62361, 32318],
    'turbidity': [63680],
    'cdom': [32295, 32322]
}

def load_and_preprocess_matched_wq(
    csv_paths: list = [
        'data/raw/matched_wq_s2/Matched_WQ_S2/Matched_WQ_S2_1.csv',
        'data/raw/matched_wq_s2/Matched_WQ_S2/Matched_WQ_S2_2.csv'
    ],
    sample_size: int = None
) -> pd.DataFrame:
    """Loads and filters raw Matched_WQ_S2 records."""
    cols_to_load = [
        'scene', 'scene_datetime_UTC', 'sample_datetime_UTC', 'site_no', 'station_nm',
        'parm_cd', 'parm_nm', 'parm_unit', 'Lat', 'Long', 'MeasurementValue',
        'huc4', 'n_nhd', 'n_buf', 'n_mask', 'l2flag_center',
        'B01_center', 'B02_center', 'B03_center', 'B04_center', 'B05_center',
        'B06_center', 'B07_center', 'B08_center', 'B8A_center', 'B11_center', 'B12_center'
    ]
    
    dfs = []
    print("Loading Matched_WQ_S2 CSVs...", flush=True)
    for p in csv_paths:
        if not os.path.exists(p):
            print(f"Warning: {p} not found, skipping.", flush=True)
            continue
        print(f"Reading {p}...", flush=True)
        df_chunk = pd.read_csv(p, usecols=cols_to_load, low_memory=False)
        dfs.append(df_chunk)
        
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total raw matched records loaded: {len(df):,}", flush=True)
    
    # Map HUC4 to Basin
    df['huc4_int'] = pd.to_numeric(df['huc4'], errors='coerce').fillna(-1).astype(int)
    df['basin'] = df['huc4_int'].map(HUC4_TO_BASIN).fillna('Other')
    
    # Filter valid basins
    df = df[df['basin'] != 'Other'].copy()
    
    # Identify target parameters
    df['target_type'] = None
    df.loc[df['parm_cd'].isin(TARGET_PARAMS['chlorophyll']), 'target_type'] = 'chlorophyll'
    df.loc[df['parm_cd'].isin(TARGET_PARAMS['turbidity']), 'target_type'] = 'turbidity'
    df.loc[df['parm_cd'].isin(TARGET_PARAMS['cdom']), 'target_type'] = 'cdom'
    
    df = df[df['target_type'].notnull()].copy()
    
    # Filter valid measurement values
    df['MeasurementValue'] = pd.to_numeric(df['MeasurementValue'], errors='coerce')
    df = df[df['MeasurementValue'] > 0].copy()
    
    # Filter valid reflectance values
    band_cols = ['B01_center', 'B02_center', 'B03_center', 'B04_center', 'B05_center',
                 'B06_center', 'B07_center', 'B08_center', 'B8A_center', 'B11_center', 'B12_center']
    for b in band_cols:
        df[b] = pd.to_numeric(df[b], errors='coerce')
        
    # Valid physical reflectance (0 to 10,000) and L2 flag == 0 (non-cloud/non-shadow)
    valid_mask = (df['l2flag_center'] == 0)
    for b in ['B02_center', 'B03_center', 'B04_center', 'B08_center', 'B11_center', 'B12_center']:
        valid_mask &= (df[b] > 0) & (df[b] < 10000)
    df = df[valid_mask].copy()
    
    if sample_size and len(df) > sample_size:
        print(f"Subsampling to {sample_size:,} records for fast execution...", flush=True)
        df = df.sample(n=sample_size, random_state=42).copy()
        
    # Scale reflectance to [0, 1]
    for b in band_cols:
        df[f'{b}_refl'] = df[b] / 10000.0
        
    # Feature Engineering: Remote Sensing Spectral Indices
    eps = 1e-6
    b2 = df['B02_center_refl']
    b3 = df['B03_center_refl']
    b4 = df['B04_center_refl']
    b5 = df['B05_center_refl']
    b8 = df['B08_center_refl']
    b11 = df['B11_center_refl']
    
    df['ndwi'] = (b3 - b8) / (b3 + b8 + eps)
    df['mndwi'] = (b3 - b11) / (b3 + b11 + eps)
    df['ndci'] = (b5 - b4) / (b5 + b4 + eps)
    df['turbidity_ratio'] = b4 / (b3 + eps)
    df['cdom_ratio'] = b2 / (b3 + eps)
    df['b8_b4_ratio'] = b8 / (b4 + eps)
    
    # Year extracted from datetime
    df['year'] = pd.to_datetime(df['sample_datetime_UTC'], errors='coerce').dt.year
    
    print(f"Processed valid matched records: {len(df):,} across {df['basin'].nunique()} basins.", flush=True)
    return df

def apply_water_masks(
    df: pd.DataFrame,
    spectral_checkpoint: str = 'results/checkpoints/spectral_water_model.pt',
    output_dir: str = 'data/processed/matched_wq_cleaned',
    results_dir: str = 'results'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Applies (a) Assumed Ground Truth, (b) DL Spectral Mask, and (c) Naive NDWI Mask.
    Analyzes and logs disagreement rates.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'tables'), exist_ok=True)
    
    # 1. Condition (a): Assumed Ground-Truth Water (all valid stations passing L2 flag)
    df_a = df.copy()
    df_a['water_mask_assumed'] = 1
    
    # 2. Condition (c): Naive NDWI Threshold Mask (> 0)
    df['water_mask_ndwi'] = (df['ndwi'] > 0.0).astype(int)
    df_c = df[df['water_mask_ndwi'] == 1].copy()
    
    # 3. Condition (b): Deep Learning Water Extraction Mask
    if not os.path.exists(spectral_checkpoint):
        print("Training deep spectral water classifier front-end...", flush=True)
        train_spectral_water_classifier(checkpoint_path=spectral_checkpoint)
        
    print("Applying DL Spectral Classifier to matched points...", flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(spectral_checkpoint, map_location=device, weights_only=False)
    
    model = DeepSpectralWaterClassifier(in_features=6)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    
    mean = ckpt['mean']
    std = ckpt['std']
    
    # 6 spectral bands: B2, B3, B4, B8, B11, B12 (raw scale matching dset-s2)
    spec_cols = ['B02_center', 'B03_center', 'B04_center', 'B08_center', 'B11_center', 'B12_center']
    X_raw = df[spec_cols].values.astype(np.float32)
    X_norm = (X_raw - mean) / std
    
    tensor_X = torch.from_numpy(X_norm).float()
    
    batch_size = 4096
    probs = []
    with torch.no_grad():
        for i in range(0, len(tensor_X), batch_size):
            bx = tensor_X[i : i + batch_size].to(device)
            p = torch.sigmoid(model(bx)).cpu().numpy().ravel()
            probs.append(p)
            
    df['dl_water_prob'] = np.concatenate(probs)
    df['water_mask_dl'] = (df['dl_water_prob'] >= 0.5).astype(int)
    df_b = df[df['water_mask_dl'] == 1].copy()
    
    # Import harmonization logic
    from src.inversion.apply_unit_harmonization import harmonize_dataframe
    
    # Harmonize each condition: primary (native physical units only) and secondary (with calibrated RFU)
    df_a_prim, df_a_sec, loss_a = harmonize_dataframe(df_a, dataset_name='a_ground_truth')
    df_b_prim, df_b_sec, loss_b = harmonize_dataframe(df_b, dataset_name='b_dl_masked')
    df_c_prim, df_c_sec, loss_c = harmonize_dataframe(df_c, dataset_name='c_ndwi_masked')
    
    # Save the 3 parallel datasets (Primary: native physical units only)
    path_a = os.path.join(output_dir, 'matched_set_a_ground_truth.csv')
    path_b = os.path.join(output_dir, 'matched_set_b_dl_masked.csv')
    path_c = os.path.join(output_dir, 'matched_set_c_ndwi_masked.csv')
    
    df_a_prim.to_csv(path_a, index=False)
    df_b_prim.to_csv(path_b, index=False)
    df_c_prim.to_csv(path_c, index=False)
    
    # Save Secondary (harmonized calibrated RFU variants)
    df_a_sec.to_csv(os.path.join(output_dir, 'matched_set_a_ground_truth_harmonized.csv'), index=False)
    df_b_sec.to_csv(os.path.join(output_dir, 'matched_set_b_dl_masked_harmonized.csv'), index=False)
    df_c_sec.to_csv(os.path.join(output_dir, 'matched_set_c_ndwi_masked_harmonized.csv'), index=False)
    
    # Save sample loss table
    loss_all = pd.concat([loss_a, loss_b, loss_c], ignore_index=True)
    loss_all.to_csv(os.path.join(results_dir, 'tables', 'unit_harmonization_sample_loss.csv'), index=False)
    
    print(f"Saved PRIMARY Dataset (a) Assumed: {len(df_a_prim):,} rows -> {path_a}", flush=True)
    print(f"Saved PRIMARY Dataset (b) DL Mask: {len(df_b_prim):,} rows -> {path_b}", flush=True)
    print(f"Saved PRIMARY Dataset (c) NDWI Mask: {len(df_c_prim):,} rows -> {path_c}", flush=True)
    
    # --- Disagreement Analysis ---
    # Disagreement between DL mask and assumed water (i.e. where DL says NOT water)
    df['dl_disagrees_with_assumed'] = (df['water_mask_dl'] == 0).astype(int)
    df['ndwi_disagrees_with_assumed'] = (df['water_mask_ndwi'] == 0).astype(int)
    df['dl_vs_ndwi_disagreement'] = (df['water_mask_dl'] != df['water_mask_ndwi']).astype(int)
    
    disagree_by_basin = df.groupby('basin').agg(
        total_records=('scene', 'count'),
        dl_water_count=('water_mask_dl', lambda x: (x == 1).sum()),
        ndwi_water_count=('water_mask_ndwi', lambda x: (x == 1).sum()),
        dl_disagreement_count=('dl_disagrees_with_assumed', 'sum'),
        ndwi_disagreement_count=('ndwi_disagrees_with_assumed', 'sum'),
        dl_vs_ndwi_disagree_count=('dl_vs_ndwi_disagreement', 'sum')
    ).reset_index()
    
    disagree_by_basin['dl_exclusion_rate_%'] = (
        disagree_by_basin['dl_disagreement_count'] / disagree_by_basin['total_records'] * 100
    ).round(2)
    disagree_by_basin['ndwi_exclusion_rate_%'] = (
        disagree_by_basin['ndwi_disagreement_count'] / disagree_by_basin['total_records'] * 100
    ).round(2)
    disagree_by_basin['dl_vs_ndwi_diff_rate_%'] = (
        disagree_by_basin['dl_vs_ndwi_disagree_count'] / disagree_by_basin['total_records'] * 100
    ).round(2)
    
    table_path = os.path.join(results_dir, 'tables', 'stage2_disagreement_analysis.csv')
    disagree_by_basin.to_csv(table_path, index=False)
    print(f"\nDisagreement Analysis by Basin saved to {table_path}:", flush=True)
    print(disagree_by_basin.to_string(index=False), flush=True)
    
    # Plot Disagreement Analysis
    plt.figure(figsize=(10, 5), dpi=150)
    x = np.arange(len(disagree_by_basin))
    width = 0.35
    
    plt.bar(x - width/2, disagree_by_basin['dl_exclusion_rate_%'], width,
            label='DL Model Disagreement (Exclusion) Rate %', color='#1f77b4', alpha=0.85)
    plt.bar(x + width/2, disagree_by_basin['ndwi_exclusion_rate_%'], width,
            label='Naive NDWI Disagreement Rate %', color='#ff7f0e', alpha=0.85)
    
    plt.xlabel('River Basin', fontsize=11, fontweight='bold')
    plt.ylabel('Disagreement / Exclusion Rate (%)', fontsize=11, fontweight='bold')
    plt.title('Stage 2: Water Masking Disagreement with Assumed Water Label Across Basins',
              fontsize=12, fontweight='bold')
    plt.xticks(x, disagree_by_basin['basin'], rotation=15)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plot_path = os.path.join(results_dir, 'figures', 'stage2_disagreement_rates.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved disagreement plot to {plot_path}", flush=True)
    
    return df_a, df_b, df_c

def generate_parallel_datasets(
    sample_size: int = 150000,
    save_dir: str = 'data/processed/matched_wq_cleaned',
    results_dir: str = 'results'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """End-to-end generator for Stage 2 parallel datasets and disagreement analysis."""
    df = load_and_preprocess_matched_wq(sample_size=sample_size)
    df_a, df_b, df_c = apply_water_masks(df, output_dir=save_dir, results_dir=results_dir)
    disagree_path = os.path.join(results_dir, 'tables', 'stage2_disagreement_analysis.csv')
    df_disagree = pd.read_csv(disagree_path) if os.path.exists(disagree_path) else pd.DataFrame()
    return df_a, df_b, df_c, df_disagree

from src.inversion.eda import run_exploratory_analysis

if __name__ == '__main__':
    df = load_and_preprocess_matched_wq(sample_size=150000)
    run_exploratory_analysis(df)
    apply_water_masks(df)

