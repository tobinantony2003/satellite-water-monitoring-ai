"""
Script: check_site_time_confounding.py
Quantifies site-time confounding severity for CDOM and Chlorophyll-a in Matched_WQ_S2.
Computes active deployment windows, pairwise temporal overlap matrices (days, months, Jaccard),
and identifies sites with zero temporal overlap with dominant stations.
Generates publication-quality timeline and heatmap figures.
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compute_site_timeline_and_overlap(df_target, target_name):
    """
    Computes site timeline metrics and pairwise temporal overlap matrices.
    """
    df_target = df_target.copy()
    df_target['site_no'] = df_target['site_no'].astype(str).str.zfill(8)
    
    # Group by site
    site_records = []
    site_months = {}
    site_dates = {}
    
    for site, grp in df_target.groupby('site_no'):
        site_str = str(site).zfill(8)
        dates = pd.to_datetime(grp['sample_datetime_UTC'])
        min_date = dates.min()
        max_date = dates.max()
        span_days = (max_date - min_date).days + 1
        n_samples = len(grp)
        basin = grp['basin'].iloc[0]
        station_nm = grp['station_nm'].iloc[0]
        parm_cd = grp['parm_cd'].iloc[0] if 'parm_cd' in grp.columns else 'N/A'
        parm_nm = grp['parm_nm'].iloc[0] if 'parm_nm' in grp.columns else 'N/A'
        parm_unit = grp['parm_unit'].iloc[0] if 'parm_unit' in grp.columns else 'N/A'
        
        # Unique year-months (drop tz first to avoid warning)
        dates_naive = dates.dt.tz_localize(None) if dates.dt.tz is not None else dates
        ym_set = set(dates_naive.dt.to_period('M'))
        d_set = set(dates_naive.dt.date)
        
        site_months[site_str] = ym_set
        site_dates[site_str] = d_set
        
        site_records.append({
            'site_no': site_str,
            'station_nm': station_nm,
            'basin': basin,
            'parm_cd': parm_cd,
            'parm_nm': parm_nm,
            'parm_unit': parm_unit,
            'n_samples': n_samples,
            'min_date': min_date,
            'max_date': max_date,
            'span_days': span_days,
            'n_active_months': len(ym_set),
            'n_active_days': len(d_set)
        })
        
    summary_df = pd.DataFrame(site_records).sort_values('n_samples', ascending=False).reset_index(drop=True)
    summary_df['pct_samples'] = (summary_df['n_samples'] / summary_df['n_samples'].sum()) * 100.0
    
    # Pairwise overlap matrices
    sites = summary_df['site_no'].tolist()
    n_sites = len(sites)
    
    # 1. Window overlap (days)
    overlap_days_matrix = np.zeros((n_sites, n_sites))
    # 2. Asymmetric overlap: overlap(i,j) / span(i)
    overlap_fraction_i = np.zeros((n_sites, n_sites))
    # 3. Szymkiewicz-Simpson coefficient: overlap(i,j) / min(span(i), span(j))
    overlap_simpson = np.zeros((n_sites, n_sites))
    # 4. Monthly Jaccard: |M_i \cap M_j| / |M_i \cup M_j|
    monthly_jaccard = np.zeros((n_sites, n_sites))
    # 5. Shared active months count
    shared_months_count = np.zeros((n_sites, n_sites))
    
    for i, s_i in enumerate(sites):
        row_i = summary_df[summary_df['site_no'] == s_i].iloc[0]
        start_i, end_i = row_i['min_date'], row_i['max_date']
        span_i = row_i['span_days']
        m_i = site_months[s_i]
        
        for j, s_j in enumerate(sites):
            row_j = summary_df[summary_df['site_no'] == s_j].iloc[0]
            start_j, end_j = row_j['min_date'], row_j['max_date']
            span_j = row_j['span_days']
            m_j = site_months[s_j]
            
            # Intersection of window
            overlap_start = max(start_i, start_j)
            overlap_end = min(end_i, end_j)
            
            if overlap_end >= overlap_start:
                ov_days = (overlap_end - overlap_start).days + 1
            else:
                ov_days = 0
                
            overlap_days_matrix[i, j] = ov_days
            overlap_fraction_i[i, j] = ov_days / span_i if span_i > 0 else 0
            overlap_simpson[i, j] = ov_days / min(span_i, span_j) if min(span_i, span_j) > 0 else 0
            
            # Month overlap
            shared_m = len(m_i.intersection(m_j))
            union_m = len(m_i.union(m_j))
            shared_months_count[i, j] = shared_m
            monthly_jaccard[i, j] = shared_m / union_m if union_m > 0 else 0
            
    df_simpson = pd.DataFrame(overlap_simpson, index=sites, columns=sites)
    df_jaccard = pd.DataFrame(monthly_jaccard, index=sites, columns=sites)
    df_shared_m = pd.DataFrame(shared_months_count, index=sites, columns=sites)
    
    return summary_df, df_simpson, df_jaccard, df_shared_m, site_months

def main():
    print("=" * 70)
    print("STEP 1: Quantifying Site-Time Confounding (CDOM & Chlorophyll-a)")
    print("=" * 70)
    
    data_path = 'data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv'
    df = pd.read_csv(data_path, low_memory=False)
    df['sample_datetime_UTC'] = pd.to_datetime(df['sample_datetime_UTC'])
    
    os.makedirs('results/tables', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 1. CDOM Analysis (6 sites)
    # -------------------------------------------------------------------------
    print("\n--- Auditing CDOM (6 sites) ---")
    df_cdom = df[df['target_type'] == 'cdom'].copy()
    cdom_summary, cdom_simpson, cdom_jaccard, cdom_shared_m, _ = compute_site_timeline_and_overlap(df_cdom, 'CDOM')
    
    print("\nCDOM Site Summary:")
    print(cdom_summary[['site_no', 'station_nm', 'basin', 'n_samples', 'pct_samples', 'min_date', 'max_date', 'n_active_months']].to_string())
    
    print("\nCDOM Overlap Coefficient (Simpson: overlap_days / min(span_i, span_j)):")
    print(cdom_simpson.round(3).to_string())
    
    print("\nCDOM Shared Active Calendar Months:")
    print(cdom_shared_m.astype(int).to_string())
    
    cdom_summary.to_csv('results/tables/cdom_site_time_summary.csv', index=False)
    cdom_simpson.to_csv('results/tables/cdom_pairwise_overlap_coefficient.csv')
    cdom_shared_m.to_csv('results/tables/cdom_pairwise_shared_months.csv')
    
    # Check zero overlap with dominant sites: 01467200 (Delaware, 44.2%) and 05586300 (Illinois, 26.3%)
    dom_sites_cdom = ['01467200', '05586300']
    print("\nCDOM Zero Overlap Audit:")
    for dom in dom_sites_cdom:
        zero_ov = cdom_simpson.loc[dom][cdom_simpson.loc[dom] == 0].index.tolist()
        zero_m = cdom_shared_m.loc[dom][cdom_shared_m.loc[dom] == 0].index.tolist()
        print(f"  Dominant Site {dom} ({cdom_summary[cdom_summary['site_no']==dom]['station_nm'].iloc[0]}):")
        print(f"    Sites with ZERO window overlap (0 days): {zero_ov}")
        print(f"    Sites with ZERO shared active months: {zero_m}")
        
    # -------------------------------------------------------------------------
    # 2. Chlorophyll-a Analysis (27 sites)
    # -------------------------------------------------------------------------
    print("\n--- Auditing Chlorophyll-a (27 sites) ---")
    df_chl = df[df['target_type'] == 'chlorophyll'].copy()
    chl_summary, chl_simpson, chl_jaccard, chl_shared_m, _ = compute_site_timeline_and_overlap(df_chl, 'Chlorophyll-a')
    
    print(f"\nChlorophyll-a Site Summary (Top 10 of {len(chl_summary)} sites):")
    print(chl_summary[['site_no', 'station_nm', 'basin', 'n_samples', 'pct_samples', 'min_date', 'max_date', 'n_active_months']].head(10).to_string())
    
    chl_summary.to_csv('results/tables/chlorophyll_site_time_summary.csv', index=False)
    chl_simpson.to_csv('results/tables/chlorophyll_pairwise_overlap_coefficient.csv')
    chl_shared_m.to_csv('results/tables/chlorophyll_pairwise_shared_months.csv')
    
    # Dominant Chlorophyll sites (top 3)
    dom_sites_chl = chl_summary['site_no'].head(3).tolist()
    print("\nChlorophyll-a Zero Overlap Audit for Top Sites:")
    for dom in dom_sites_chl:
        zero_ov = chl_simpson.loc[dom][chl_simpson.loc[dom] == 0].index.tolist()
        zero_m = chl_shared_m.loc[dom][chl_shared_m.loc[dom] == 0].index.tolist()
        dom_name = chl_summary[chl_summary['site_no']==dom]['station_nm'].iloc[0]
        dom_n = chl_summary[chl_summary['site_no']==dom]['n_samples'].iloc[0]
        dom_pct = chl_summary[chl_summary['site_no']==dom]['pct_samples'].iloc[0]
        print(f"  Site {dom} ({dom_name}, N={dom_n}, {dom_pct:.1f}%):")
        print(f"    Sites with ZERO window overlap ({len(zero_ov)} sites): {zero_ov}")
        print(f"    Sites with ZERO shared active months ({len(zero_m)} sites): {zero_m}")
        
    # Total pair stats for Chlorophyll
    n_chl_sites = len(chl_summary)
    total_pairs = (n_chl_sites * (n_chl_sites - 1)) // 2
    upper_idx = np.triu_indices(n_chl_sites, k=1)
    zero_window_pairs = int((chl_simpson.values[upper_idx] == 0).sum())
    zero_month_pairs = int((chl_shared_m.values[upper_idx] == 0).sum())
    print(f"\nChlorophyll-a Pairwise Overlap Summary (Total distinct site pairs = {total_pairs}):")
    print(f"  Pairs with ZERO window overlap: {zero_window_pairs} / {total_pairs} ({zero_window_pairs/total_pairs*100:.1f}%)")
    print(f"  Pairs with ZERO shared active months: {zero_month_pairs} / {total_pairs} ({zero_month_pairs/total_pairs*100:.1f}%)")

    # -------------------------------------------------------------------------
    # 3. Plotting Publication Figures
    # -------------------------------------------------------------------------
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Figure 1: CDOM Site Timeline & Overlap Matrix
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1.3, 1]})
    
    # Panel A: Gantt timeline
    ax1 = axes[0]
    basin_colors = {'Delaware': '#2b5c8f', 'Illinois': '#d95f02', 'Willamette': '#2ca02c'}
    
    # Sort sites chronologically or by basin
    cdom_plot_df = cdom_summary.sort_values(['basin', 'min_date'], ascending=[True, True]).reset_index(drop=True)
    y_pos = np.arange(len(cdom_plot_df))
    
    for idx, row in cdom_plot_df.iterrows():
        color = basin_colors.get(row['basin'], '#7570b3')
        ax1.barh(idx, (row['max_date'] - row['min_date']).days, left=row['min_date'], 
                 height=0.55, color=color, alpha=0.85, edgecolor='black', linewidth=1)
        # Text label inside or next to bar
        label_text = f"N={row['n_samples']} ({row['pct_samples']:.1f}%)"
        ax1.text(row['max_date'] + pd.Timedelta(days=45), idx, label_text, va='center', fontsize=9, fontweight='bold')
        
    ax1.axvline(pd.Timestamp('2022-01-25'), color='red', linestyle='--', linewidth=1.5, label='N0400 Switch (2022-01-25)')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"{r['site_no']}\n({r['station_nm'][:22]})" for _, r in cdom_plot_df.iterrows()], fontsize=8.5)
    ax1.set_xlabel("Observation Period", fontsize=11, fontweight='bold')
    ax1.set_title("(A) CDOM Active Deployment Timelines by Site", fontsize=12, fontweight='bold')
    ax1.set_xlim(pd.Timestamp('2015-01-01'), pd.Timestamp('2025-06-01'))
    
    # Legend for basins
    handles = [plt.Rectangle((0,0),1,1, color=c, ec='k') for c in basin_colors.values()]
    handles.append(plt.Line2D([0],[0], color='red', linestyle='--', linewidth=1.5))
    ax1.legend(handles, list(basin_colors.keys()) + ['Baseline N0400'], loc='upper left', frameon=True, fontsize=8.5)
    
    # Panel B: Overlap Coefficient Heatmap
    ax2 = axes[1]
    plot_sites = cdom_plot_df['site_no'].tolist()
    sub_simpson = cdom_simpson.loc[plot_sites, plot_sites]
    
    sns.heatmap(sub_simpson, annot=True, fmt='.2f', cmap='Blues', cbar=True, ax=ax2,
                vmin=0, vmax=1.0, linewidths=0.5, linecolor='gray',
                cbar_kws={'label': 'Temporal Overlap Coeff [overlap / min(span_i, span_j)]'})
    ax2.set_title("(B) Pairwise Temporal Overlap Coefficient", fontsize=12, fontweight='bold')
    ax2.set_xlabel("USGS Station", fontsize=10, fontweight='bold')
    ax2.set_ylabel("USGS Station", fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    cdom_fig_path = 'results/figures/cdom_site_time_confounding.png'
    plt.savefig(cdom_fig_path, dpi=300)
    plt.close()
    print(f"Saved CDOM site-time figure to {cdom_fig_path}")
    
    # Figure 2: Chlorophyll-a Timeline & Overlap Matrix
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [1.3, 1.1]})
    
    # Panel A: Chlorophyll Gantt timeline
    ax1 = axes[0]
    b_pal = sns.color_palette('tab10', len(chl_summary['basin'].unique()))
    basin_col_map = dict(zip(chl_summary['basin'].unique(), b_pal))
    
    chl_plot_df = chl_summary.sort_values(['basin', 'min_date'], ascending=[True, True]).reset_index(drop=True)
    y_pos = np.arange(len(chl_plot_df))
    
    for idx, row in chl_plot_df.iterrows():
        color = basin_col_map.get(row['basin'], 'gray')
        ax1.barh(idx, (row['max_date'] - row['min_date']).days, left=row['min_date'], 
                 height=0.6, color=color, alpha=0.85, edgecolor='black', linewidth=0.6)
        label_text = f"N={row['n_samples']} ({row['pct_samples']:.1f}%)"
        ax1.text(row['max_date'] + pd.Timedelta(days=35), idx, label_text, va='center', fontsize=7)
        
    ax1.axvline(pd.Timestamp('2022-01-25'), color='red', linestyle='--', linewidth=1.5, label='N0400 Switch (2022-01-25)')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"{r['site_no']} [{r['basin'][:3]}] - {r['station_nm'][:20]}" for _, r in chl_plot_df.iterrows()], fontsize=7.5)
    ax1.set_xlabel("Observation Period", fontsize=11, fontweight='bold')
    ax1.set_title(f"(A) Chlorophyll-a Active Deployment Timelines (27 Sites across 5 Basins)", fontsize=12, fontweight='bold')
    ax1.set_xlim(pd.Timestamp('2015-01-01'), pd.Timestamp('2025-06-01'))
    
    b_handles = [plt.Rectangle((0,0),1,1, color=basin_col_map[b], ec='k') for b in basin_col_map]
    b_handles.append(plt.Line2D([0],[0], color='red', linestyle='--', linewidth=1.5))
    ax1.legend(b_handles, list(basin_col_map.keys()) + ['Baseline N0400'], loc='upper left', frameon=True, fontsize=8)
    
    # Panel B: Chlorophyll Overlap Heatmap
    ax2 = axes[1]
    plot_chl_sites = chl_plot_df['site_no'].tolist()
    sub_chl_simpson = chl_simpson.loc[plot_chl_sites, plot_chl_sites]
    
    sns.heatmap(sub_chl_simpson, annot=False, cmap='Blues', cbar=True, ax=ax2,
                vmin=0, vmax=1.0, linewidths=0.2, linecolor='lightgray',
                cbar_kws={'label': 'Temporal Overlap Coeff [overlap / min(span_i, span_j)]'})
    ax2.set_title(f"(B) Chlorophyll-a Pairwise Temporal Overlap Heatmap (27x27)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("USGS Station", fontsize=9, fontweight='bold')
    ax2.set_ylabel("USGS Station", fontsize=9, fontweight='bold')
    ax2.set_xticks(np.arange(len(plot_chl_sites)) + 0.5)
    ax2.set_xticklabels(plot_chl_sites, rotation=90, fontsize=6.5)
    ax2.set_yticks(np.arange(len(plot_chl_sites)) + 0.5)
    ax2.set_yticklabels(plot_chl_sites, rotation=0, fontsize=6.5)
    
    plt.tight_layout()
    chl_fig_path = 'results/figures/chlorophyll_site_time_confounding.png'
    plt.savefig(chl_fig_path, dpi=300)
    plt.close()
    print(f"Saved Chlorophyll-a site-time figure to {chl_fig_path}")

    # Copy to brain dir
    brain_dir = r'C:\Users\Tobin Antony\.gemini\antigravity-ide\brain\730512ff-544c-4445-ac17-b37e3b98b9be'
    shutil.copy(cdom_fig_path, os.path.join(brain_dir, 'cdom_site_time_confounding.png'))
    shutil.copy(chl_fig_path, os.path.join(brain_dir, 'chlorophyll_site_time_confounding.png'))
    print("Copied figures to brain directory.")

if __name__ == '__main__':
    main()
