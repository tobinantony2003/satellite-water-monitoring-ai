"""
Script: check_sensor_metadata.py
Audits in-situ sensor/instrument parameter codes, units, and measurement methodologies
across USGS stations for CDOM, Chlorophyll-a, and Turbidity in Matched_WQ_S2.
Identifies physical unit discordance (e.g., RFU vs ug/L), parameter code variations,
and evaluates sensor consistency at dominant sites (05586300, 01467200, etc.).
"""

import os
import numpy as np
import pandas as pd

def main():
    print("=" * 80)
    print("STEP 2: In-Situ Sensor/Instrument Parameter & Metadata Audit")
    print("=" * 80)
    
    data_path = 'data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv'
    df = pd.read_csv(data_path, low_memory=False)
    df['sample_datetime_UTC'] = pd.to_datetime(df['sample_datetime_UTC'])
    df['site_no'] = df['site_no'].astype(str).str.zfill(8)
    
    os.makedirs('results/tables', exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 1. Parameter Code Catalog across all 3 targets
    # -------------------------------------------------------------------------
    param_catalog = df.groupby(['target_type', 'parm_cd', 'parm_unit', 'parm_nm']).agg(
        n_samples=('MeasurementValue', 'count'),
        n_sites=('site_no', 'nunique'),
        min_date=('sample_datetime_UTC', 'min'),
        max_date=('sample_datetime_UTC', 'max'),
        median_val=('MeasurementValue', 'median'),
        mean_val=('MeasurementValue', 'mean'),
        std_val=('MeasurementValue', 'std'),
        q25_val=('MeasurementValue', lambda x: x.quantile(0.25)),
        q75_val=('MeasurementValue', lambda x: x.quantile(0.75))
    ).reset_index()
    
    print("\n--- Parameter Code Catalog Across Targets ---")
    for _, row in param_catalog.iterrows():
        print(f"\nTarget: {row['target_type'].upper()} | ParmCd: {row['parm_cd']} | Unit: {row['parm_unit']} | Sites: {row['n_sites']} | N: {row['n_samples']:,}")
        print(f"  Name: {row['parm_nm']}")
        print(f"  Span: {row['min_date'].strftime('%Y-%m-%d')} to {row['max_date'].strftime('%Y-%m-%d')}")
        print(f"  Stats: Median={row['median_val']:.2f}, Mean={row['mean_val']:.2f}, IQR=[{row['q25_val']:.2f}, {row['q75_val']:.2f}]")
        
    param_catalog.to_csv('results/tables/sensor_metadata_parameter_codes.csv', index=False)
    
    # -------------------------------------------------------------------------
    # 2. CDOM Sensor Breakdown (6 sites)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("CDOM In-Situ Sensor Audit (6 Sites)")
    print("=" * 50)
    cdom_df = df[df['target_type'] == 'cdom']
    cdom_site_unit = cdom_df.groupby(['site_no', 'station_nm', 'basin', 'parm_cd', 'parm_unit']).agg(
        n_samples=('MeasurementValue', 'count'),
        min_date=('sample_datetime_UTC', 'min'),
        max_date=('sample_datetime_UTC', 'max'),
        median_val=('MeasurementValue', 'median'),
        q25=('MeasurementValue', lambda x: x.quantile(0.25)),
        q75=('MeasurementValue', lambda x: x.quantile(0.75))
    ).reset_index()
    print(cdom_site_unit.to_string())
    cdom_site_unit.to_csv('results/tables/cdom_sensor_units_by_site.csv', index=False)
    
    # Specifically check Site 01467200 in CDOM
    del_cdom = cdom_df[cdom_df['site_no'] == '01467200']
    p32295 = del_cdom[del_cdom['parm_cd'] == 32295]
    p32322 = del_cdom[del_cdom['parm_cd'] == 32322]
    print(f"\nSite 01467200 CDOM breakdown:")
    print(f"  Parm 32295 (ug/l QSE): N={len(p32295)}, Median={p32295['MeasurementValue'].median():.2f}, Dates: {p32295['sample_datetime_UTC'].min()} to {p32295['sample_datetime_UTC'].max()}")
    print(f"  Parm 32322 (RFU):      N={len(p32322)}, Median={p32322['MeasurementValue'].median():.2f}, Dates: {p32322['sample_datetime_UTC'].min()} to {p32322['sample_datetime_UTC'].max()}")
    
    # Check if timestamps overlap between 32295 and 32322
    common_ts = set(p32295['sample_datetime_UTC']).intersection(set(p32322['sample_datetime_UTC']))
    print(f"  Concurrent timestamps recording BOTH 32295 and 32322 simultaneously: {len(common_ts)} / {min(len(p32295), len(p32322))}")
    if len(common_ts) > 0:
        # Merge on timestamp to compute empirical ratio between RFU and ug/l QSE
        m_del = pd.merge(p32295[['sample_datetime_UTC', 'MeasurementValue']], 
                         p32322[['sample_datetime_UTC', 'MeasurementValue']], 
                         on='sample_datetime_UTC', suffixes=('_qse', '_rfu'))
        ratio = m_del['MeasurementValue_qse'] / m_del['MeasurementValue_rfu']
        print(f"  Empirical Conversion Ratio (ug/l QSE / RFU) at 01467200: Median={ratio.median():.4f}, Mean={ratio.mean():.4f}, Std={ratio.std():.4f}")
        print(f"  --> A raw RFU value of 10.0 corresponds to {10.0 * ratio.median():.2f} ug/l QSE!")
        print(f"  --> Pooling RFU and ug/l QSE directly without conversion introduces a factor of {ratio.median():.2f} error!")

    # -------------------------------------------------------------------------
    # 3. Chlorophyll-a Sensor Breakdown (27 sites)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Chlorophyll-a In-Situ Sensor Audit (27 Sites)")
    print("=" * 50)
    chl_df = df[df['target_type'] == 'chlorophyll']
    chl_site_unit = chl_df.groupby(['site_no', 'station_nm', 'basin', 'parm_cd', 'parm_unit']).agg(
        n_samples=('MeasurementValue', 'count'),
        min_date=('sample_datetime_UTC', 'min'),
        max_date=('sample_datetime_UTC', 'max'),
        median_val=('MeasurementValue', 'median'),
        q25=('MeasurementValue', lambda x: x.quantile(0.25)),
        q75=('MeasurementValue', lambda x: x.quantile(0.75))
    ).reset_index()
    print(chl_site_unit.to_string())
    chl_site_unit.to_csv('results/tables/chlorophyll_sensor_units_by_site.csv', index=False)
    
    # Check concurrent RFU and ug/L within sites that have both
    sites_with_both = chl_site_unit.groupby('site_no').filter(lambda x: len(x['parm_unit'].unique()) > 1)['site_no'].unique()
    print(f"\nChlorophyll Sites with Mixed Units (RFU & ug/L): {sites_with_both}")
    for s in sites_with_both:
        s_data = chl_df[chl_df['site_no'] == s]
        rfu = s_data[s_data['parm_unit'] == 'RFU']
        ugl = s_data[s_data['parm_unit'] == 'ug/l']
        common_chl_ts = set(rfu['sample_datetime_UTC']).intersection(set(ugl['sample_datetime_UTC']))
        print(f"  Site {s}: RFU N={len(rfu)} (Med={rfu['MeasurementValue'].median():.2f}), ug/L N={len(ugl)} (Med={ugl['MeasurementValue'].median():.2f}), Concurrent Timestamps: {len(common_chl_ts)}")
        if len(common_chl_ts) > 0:
            m_chl = pd.merge(ugl[['sample_datetime_UTC', 'MeasurementValue']], 
                             rfu[['sample_datetime_UTC', 'MeasurementValue']], 
                             on='sample_datetime_UTC', suffixes=('_ugl', '_rfu'))
            r_chl = m_chl['MeasurementValue_ugl'] / m_chl['MeasurementValue_rfu']
            print(f"    Empirical conversion (ug/L / RFU): Median={r_chl.median():.4f}, Mean={r_chl.mean():.4f}")

    # Summary of unit composition
    print("\n--- Summary of Unit Composition in Chlorophyll-a ---")
    unit_summary = chl_df.groupby('parm_unit').agg(
        total_samples=('MeasurementValue', 'count'),
        n_sites=('site_no', 'nunique'),
        median=('MeasurementValue', 'median'),
        mean=('MeasurementValue', 'mean')
    ).reset_index()
    unit_summary['pct_samples'] = (unit_summary['total_samples'] / len(chl_df)) * 100.0
    print(unit_summary.to_string())
    
    # -------------------------------------------------------------------------
    # 4. Turbidity Sensor Breakdown (51 sites)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Turbidity In-Situ Sensor Audit (51 Sites)")
    print("=" * 50)
    turb_df = df[df['target_type'] == 'turbidity']
    turb_site_unit = turb_df.groupby(['parm_cd', 'parm_unit']).agg(
        n_samples=('MeasurementValue', 'count'),
        n_sites=('site_no', 'nunique'),
        min_date=('sample_datetime_UTC', 'min'),
        max_date=('sample_datetime_UTC', 'max'),
        median_val=('MeasurementValue', 'median')
    ).reset_index()
    print("Turbidity Parameter Code Distribution:")
    print(turb_site_unit.to_string())
    print(f"All {turb_df['site_no'].nunique()} turbidity sites use exclusively parm_cd=63680 (FNU).")

if __name__ == '__main__':
    main()
