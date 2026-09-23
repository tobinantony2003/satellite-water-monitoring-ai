import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_exploratory_analysis(
    df: pd.DataFrame,
    results_dir: str = 'results'
):
    """Generates publication-quality EDA figures for the research report."""
    fig_dir = os.path.join(results_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    print("\n--- Running Exploratory Data Analysis ---", flush=True)
    
    # 1. Parameter Distribution Plots (Raw vs. Log-Transformed)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), dpi=150)
    targets = [
        ('chlorophyll', 'Chlorophyll-a (μg/L)'),
        ('turbidity', 'Turbidity (FNU)'),
        ('cdom', 'fDOM / CDOM (μg/L QSE)')
    ]
    
    for row_idx, (t_name, label) in enumerate(targets):
        sub = df[df['target_type'] == t_name]['MeasurementValue'].dropna()
        sub_pos = sub[sub > 0]
        
        # Raw distribution (with 99th percentile cutoff for visualization)
        p99 = np.percentile(sub_pos, 99)
        axes[row_idx, 0].hist(sub_pos[sub_pos <= p99], bins=40, color='#1f77b4', edgecolor='black', alpha=0.7)
        axes[row_idx, 0].set_title(f"Raw {label} (Skewed)", fontsize=11, fontweight='bold')
        axes[row_idx, 0].set_ylabel('Frequency')
        axes[row_idx, 0].grid(True, alpha=0.3)
        
        # Log-transformed distribution
        log_vals = np.log10(sub_pos)
        axes[row_idx, 1].hist(log_vals, bins=40, color='#2ca02c', edgecolor='black', alpha=0.7)
        axes[row_idx, 1].set_title(f"Log10 {label} (Normalized)", fontsize=11, fontweight='bold')
        axes[row_idx, 1].set_ylabel('Frequency')
        axes[row_idx, 1].grid(True, alpha=0.3)
        
    plt.tight_layout()
    dist_path = os.path.join(fig_dir, 'eda_target_distributions.png')
    plt.savefig(dist_path)
    plt.close()
    print(f"Saved target distribution figure to {dist_path}", flush=True)
    
    # 2. Spectral Signatures Across Water Quality Tiers
    # Stratify turbidity into low (<5 FNU), moderate (5-20 FNU), high (>20 FNU)
    sub_turb = df[df['target_type'] == 'turbidity'].copy()
    bands = ['B01_center_refl', 'B02_center_refl', 'B03_center_refl', 'B04_center_refl',
             'B05_center_refl', 'B06_center_refl', 'B07_center_refl', 'B08_center_refl',
             'B8A_center_refl', 'B11_center_refl', 'B12_center_refl']
    band_names = ['B1 (Aer)', 'B2 (Blu)', 'B3 (Grn)', 'B4 (Red)', 'B5 (RE1)',
                  'B6 (RE2)', 'B7 (RE3)', 'B8 (NIR)', 'B8A (NNIR)', 'B11 (SW1)', 'B12 (SW2)']
                  
    plt.figure(figsize=(10, 5), dpi=150)
    low_turb = sub_turb[sub_turb['MeasurementValue'] < 5][bands].median()
    med_turb = sub_turb[(sub_turb['MeasurementValue'] >= 5) & (sub_turb['MeasurementValue'] <= 25)][bands].median()
    high_turb = sub_turb[sub_turb['MeasurementValue'] > 25][bands].median()
    
    x = np.arange(len(bands))
    plt.plot(x, low_turb, 'o-', color='#1f77b4', lw=2, label='Clear Water (< 5 FNU)')
    plt.plot(x, med_turb, 's-', color='#ff7f0e', lw=2, label='Moderate Turbidity (5-25 FNU)')
    plt.plot(x, high_turb, '^-', color='#d62728', lw=2, label='High Turbidity (> 25 FNU)')
    
    plt.xticks(x, band_names, rotation=30)
    plt.xlabel('Sentinel-2 Spectral Band', fontsize=11, fontweight='bold')
    plt.ylabel('Surface Reflectance (0-1)', fontsize=11, fontweight='bold')
    plt.title('Sentinel-2 Aquatic Surface Reflectance Signatures across Turbidity Regimes', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    spec_path = os.path.join(fig_dir, 'eda_spectral_signatures.png')
    plt.savefig(spec_path)
    plt.close()
    print(f"Saved spectral signatures figure to {spec_path}", flush=True)
    
    # 3. Basin Sample Count Distribution
    plt.figure(figsize=(9, 4), dpi=150)
    basin_counts = df['basin'].value_counts()
    sns.barplot(x=basin_counts.index, y=basin_counts.values, hue=basin_counts.index, palette='Blues_r', legend=False)
    plt.title('Matched Observation Count across USGS River Basins', fontsize=12, fontweight='bold')
    plt.xlabel('River Basin', fontsize=11, fontweight='bold')
    plt.ylabel('Number of Matched Observations', fontsize=11, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    basin_path = os.path.join(fig_dir, 'eda_basin_observations.png')
    plt.savefig(basin_path)
    plt.close()
    print(f"Saved basin distribution figure to {basin_path}", flush=True)
