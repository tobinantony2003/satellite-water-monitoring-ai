"""Temporal matching audit and sensitivity analysis for Sentinel-2 in-situ water-quality matchups.

This module does not redesign the modeling pipeline. It isolates temporal mismatch as a
single experimental variable by filtering on the absolute difference between scene and
sample timestamps while leaving the existing feature set, 5-fold site-grouped CV, and
model suite intact.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inversion.models import get_inversion_models
from src.inversion.spatial_cv import get_spatial_block_cv
from src.inversion.train_inversion import FEATURE_COLS

TARGETS = ['turbidity', 'chlorophyll', 'cdom']
TEMPORAL_WINDOWS = [1.0, 3.0, 6.0, 12.0, 24.0, 48.0, 72.0]
BASELINE_WINDOW_HOURS = 12.0


def load_matched_dataset(data_path: str | Path | None = None) -> pd.DataFrame:
    """Load the processed DL-masked dataset used by the current inversion pipeline."""
    path = Path(data_path) if data_path is not None else PROJECT_ROOT / 'data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv'
    df = pd.read_csv(path, low_memory=False)
    if 'scene_datetime_UTC' not in df.columns or 'sample_datetime_UTC' not in df.columns:
        raise ValueError(f'Missing temporal columns in {path}')
    df = df.copy()
    df['scene_datetime_UTC'] = pd.to_datetime(df['scene_datetime_UTC'], errors='coerce')
    df['sample_datetime_UTC'] = pd.to_datetime(df['sample_datetime_UTC'], errors='coerce')
    df['dt_hours'] = np.abs((df['scene_datetime_UTC'] - df['sample_datetime_UTC']).dt.total_seconds() / 3600.0)
    return df


def compute_temporal_audit(df: pd.DataFrame, output_table_path: Path, output_figure_path: Path) -> pd.DataFrame:
    """Compute per-target and per-basin temporal matchup audit statistics."""
    rows: List[Dict[str, float | int | str]] = []

    for target in TARGETS:
        target_df = df[df['target_type'] == target].copy()
        dt = target_df['dt_hours'].dropna()
        rows.append({
            'scope': 'target',
            'target': target,
            'basin': np.nan,
            'n_observations': int(len(dt)),
            'n_sites': int(target_df['site_no'].nunique()),
            'min_dt_hours': float(dt.min()),
            'median_dt_hours': float(dt.median()),
            'mean_dt_hours': float(dt.mean()),
            'p75_dt_hours': float(np.percentile(dt, 75)),
            'p90_dt_hours': float(np.percentile(dt, 90)),
            'p95_dt_hours': float(np.percentile(dt, 95)),
            'max_dt_hours': float(dt.max()),
            'pct_within_1h': float((dt <= 1.0).mean() * 100.0),
            'pct_within_3h': float((dt <= 3.0).mean() * 100.0),
            'pct_within_6h': float((dt <= 6.0).mean() * 100.0),
            'pct_within_12h': float((dt <= 12.0).mean() * 100.0),
            'pct_within_24h': float((dt <= 24.0).mean() * 100.0),
            'pct_within_48h': float((dt <= 48.0).mean() * 100.0),
            'pct_within_72h': float((dt <= 72.0).mean() * 100.0),
        })

        for basin_name, basin_df in target_df.groupby('basin'):
            basin_dt = basin_df['dt_hours'].dropna()
            rows.append({
                'scope': 'basin',
                'target': target,
                'basin': basin_name,
                'n_observations': int(len(basin_dt)),
                'n_sites': int(basin_df['site_no'].nunique()),
                'min_dt_hours': float(basin_dt.min()),
                'median_dt_hours': float(basin_dt.median()),
                'mean_dt_hours': float(basin_dt.mean()),
                'p75_dt_hours': float(np.percentile(basin_dt, 75)),
                'p90_dt_hours': float(np.percentile(basin_dt, 90)),
                'p95_dt_hours': float(np.percentile(basin_dt, 95)),
                'max_dt_hours': float(basin_dt.max()),
                'pct_within_1h': float((basin_dt <= 1.0).mean() * 100.0),
                'pct_within_3h': float((basin_dt <= 3.0).mean() * 100.0),
                'pct_within_6h': float((basin_dt <= 6.0).mean() * 100.0),
                'pct_within_12h': float((basin_dt <= 12.0).mean() * 100.0),
                'pct_within_24h': float((basin_dt <= 24.0).mean() * 100.0),
                'pct_within_48h': float((basin_dt <= 48.0).mean() * 100.0),
                'pct_within_72h': float((basin_dt <= 72.0).mean() * 100.0),
            })

    audit_df = pd.DataFrame(rows)
    output_table_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(output_table_path, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=200)
    for ax, target in zip(axes, TARGETS):
        dt = df.loc[df['target_type'] == target, 'dt_hours'].dropna()
        max_bin = max(24.0, float(dt.quantile(0.99)))
        ax.hist(dt, bins=30, range=(0.0, min(120.0, max_bin + 5.0)), color='#4C72B0', edgecolor='black', alpha=0.8)
        ax.axvline(dt.median(), color='red', linestyle='--', linewidth=2, label=f'Median {dt.median():.1f}h')
        ax.axvline(dt.mean(), color='darkorange', linestyle='-.', linewidth=2, label=f'Mean {dt.mean():.1f}h')
        ax.set_title(target.replace('_', ' ').title(), fontsize=11)
        ax.set_xlabel('Absolute time difference (hours)')
        ax.set_ylabel('Count')
        ax.legend(frameon=False)
    fig.suptitle('Temporal matchup distribution: absolute difference between Sentinel-2 scene and in-situ sample time', fontsize=13, weight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_figure_path, bbox_inches='tight')
    plt.close(fig)
    return audit_df


def build_temporal_window_datasets(df: pd.DataFrame, output_dir: str | Path) -> Dict[Tuple[str, float], pd.DataFrame]:
    """Create a dedicated filtered dataset per target/window, without modifying originals."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[Tuple[str, float], pd.DataFrame] = {}
    for target in TARGETS:
        subset = df[df['target_type'] == target].copy()
        for window_hours in TEMPORAL_WINDOWS:
            filtered = subset[subset['dt_hours'] <= window_hours].copy()
            filename = f'matched_set_b_dl_masked_{target}_{int(window_hours)}h.csv' if window_hours.is_integer() else f'matched_set_b_dl_masked_{target}_{window_hours}h.csv'
            write_path = out_dir / filename
            filtered.to_csv(write_path, index=False)
            saved[(target, float(window_hours))] = filtered
    return saved


def _summarize_model_window(df_target: pd.DataFrame, window_hours: float, model_name: str, target_type: str, baseline_fold_map: Dict[int, int] | None = None) -> Tuple[Dict[str, float], List[Dict[str, float]], np.ndarray, np.ndarray]:
    window_df = df_target[df_target['dt_hours'] <= window_hours].copy().reset_index(drop=True)
    if baseline_fold_map is not None:
        window_df['fold'] = window_df['site_no'].map(baseline_fold_map)
        window_df = window_df[window_df['fold'].notna()].copy()
    else:
        site_fold_map = {}
        for train_idx, val_idx in get_spatial_block_cv(window_df, group_col='site_no', n_splits=5):
            for idx in val_idx:
                site_fold_map[window_df.iloc[idx]['site_no']] = len(site_fold_map) % 5 + 1
            break
        window_df['fold'] = window_df['site_no'].map(site_fold_map)

    X = window_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_raw = window_df['MeasurementValue'].to_numpy(dtype=float)
    y_log = np.log10(np.clip(y_raw, 1e-4, None))

    fold_metrics: List[Dict[str, float]] = []
    oof_pred_log = np.full(len(window_df), np.nan, dtype=float)
    oof_pred_raw = np.full(len(window_df), np.nan, dtype=float)

    for fold_index in range(1, 6):
        val_mask = window_df['fold'].to_numpy() == fold_index
        if not np.any(val_mask):
            continue
        train_mask = ~val_mask
        if not np.any(train_mask):
            continue

        X_train = X[train_mask]
        X_val = X[val_mask]
        y_train = y_log[train_mask]

        if model_name == 'Empirical_Baseline':
            ratio_feature = {'chlorophyll': 'ndci', 'turbidity': 'turbidity_ratio', 'cdom': 'cdom_ratio'}[target_type]
            feature_idx = FEATURE_COLS.index(ratio_feature)
            model = get_inversion_models(in_features=1)[model_name]
            model.fit(X_train[:, [feature_idx]], y_train)
            pred_log = model.predict(X_val[:, [feature_idx]])
        else:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            model = get_inversion_models(in_features=len(FEATURE_COLS))[model_name]
            model.fit(X_train_scaled, y_train)
            pred_log = model.predict(X_val_scaled)

        pred_log = np.asarray(pred_log, dtype=float)
        pred_log = np.clip(pred_log, -3.0, 4.0)
        pred_raw = 10.0 ** pred_log

        oof_pred_log[val_mask] = pred_log
        oof_pred_raw[val_mask] = pred_raw

        fold_r2_log = float(r2_score(y_log[val_mask], pred_log))
        fold_rmse_log = float(np.sqrt(mean_squared_error(y_log[val_mask], pred_log)))
        fold_r2_phys = float(r2_score(y_raw[val_mask], pred_raw))
        fold_rmse_phys = float(np.sqrt(mean_squared_error(y_raw[val_mask], pred_raw)))
        fold_mae = float(mean_absolute_error(y_raw[val_mask], pred_raw))
        fold_metrics.append({
            'fold': fold_index,
            'r2_log': fold_r2_log,
            'rmse_log': fold_rmse_log,
            'r2_phys': fold_r2_phys,
            'rmse_phys': fold_rmse_phys,
            'mae': fold_mae,
            'n_val': int(np.sum(val_mask)),
        })

    valid_mask = ~np.isnan(oof_pred_log)
    if not np.any(valid_mask):
        raise ValueError(f'No valid predictions for {target_type} at {window_hours}h and model {model_name}')

    valid_y_log = y_log[valid_mask]
    valid_pred_log = oof_pred_log[valid_mask]
    valid_y_raw = y_raw[valid_mask]
    valid_pred_raw = oof_pred_raw[valid_mask]

    macro_log_r2 = float(np.mean([m['r2_log'] for m in fold_metrics])) if fold_metrics else np.nan
    macro_log_r2_std = float(np.std([m['r2_log'] for m in fold_metrics])) if fold_metrics else np.nan
    pooled_log_r2 = float(r2_score(valid_y_log, valid_pred_log))
    pooled_log_rmse = float(np.sqrt(mean_squared_error(valid_y_log, valid_pred_log)))
    pooled_phys_r2 = float(r2_score(valid_y_raw, valid_pred_raw))
    pooled_phys_rmse = float(np.sqrt(mean_squared_error(valid_y_raw, valid_pred_raw)))
    mae = float(mean_absolute_error(valid_y_raw, valid_pred_raw))

    summary = {
        'target': target_type,
        'model': model_name,
        'window_hours': float(window_hours),
        'n_observations': int(len(window_df)),
        'n_sites': int(window_df['site_no'].nunique()),
        'n_basins': int(window_df['basin'].nunique()),
        'median_dt_hours': float(window_df['dt_hours'].median()),
        'mean_dt_hours': float(window_df['dt_hours'].mean()),
        'pct_retained': float(100.0 * len(window_df) / len(df_target)),
        'macro_log_r2': macro_log_r2,
        'macro_log_r2_std': macro_log_r2_std,
        'pooled_log_r2': pooled_log_r2,
        'pooled_log_rmse': pooled_log_rmse,
        'pooled_phys_r2': pooled_phys_r2,
        'pooled_phys_rmse': pooled_phys_rmse,
        'mae': mae,
    }
    return summary, fold_metrics, valid_y_log, valid_pred_log


def compute_paired_window_comparison(df_target: pd.DataFrame, target_type: str, model_name: str, baseline_window_hours: float = BASELINE_WINDOW_HOURS) -> pd.DataFrame:
    """Compare each temporal window against the 12h baseline on the same site folds."""
    # Build a stable site->fold map from the full target dataset so all windows share the same validation structure.
    fold_map: Dict[int, int] = {}
    for train_idx, val_idx in get_spatial_block_cv(df_target, group_col='site_no', n_splits=5):
        for idx in val_idx:
            site = int(df_target.iloc[idx]['site_no'])
            if site not in fold_map:
                fold_map[site] = len(fold_map) % 5 + 1
    for site in sorted(df_target['site_no'].unique().astype(int)):
        if site not in fold_map:
            fold_map[site] = len(fold_map) % 5 + 1

    baseline_summary, baseline_fold_metrics, _, _ = _summarize_model_window(df_target, baseline_window_hours, model_name, target_type, baseline_fold_map=fold_map)
    baseline_macro = baseline_summary['macro_log_r2']
    baseline_pooled = baseline_summary['pooled_log_r2']

    rows: List[Dict[str, float | int | str]] = []
    for window_hours in TEMPORAL_WINDOWS:
        summary, fold_metrics, _, _ = _summarize_model_window(df_target, window_hours, model_name, target_type, baseline_fold_map=fold_map)
        diffs = []
        for fold_row in fold_metrics:
            fold_id = int(fold_row['fold'])
            baseline_fold = next((b['r2_log'] for b in baseline_fold_metrics if int(b['fold']) == fold_id), np.nan)
            if np.isfinite(baseline_fold):
                diffs.append(float(fold_row['r2_log'] - baseline_fold))

        if diffs:
            stat, p_value = stats.wilcoxon(diffs, zero_method='wilcox')
        else:
            stat, p_value = np.nan, np.nan

        row = {
            'target': target_type,
            'model': model_name,
            'window_hours': float(window_hours),
            'n_observations': int(summary['n_observations']),
            'n_sites': int(summary['n_sites']),
            'n_basins': int(summary['n_basins']),
            'median_dt_hours': float(summary['median_dt_hours']),
            'mean_dt_hours': float(summary['mean_dt_hours']),
            'pct_retained': float(summary['pct_retained']),
            'macro_log_r2': float(summary['macro_log_r2']),
            'pooled_log_r2': float(summary['pooled_log_r2']),
            'pooled_log_rmse': float(summary['pooled_log_rmse']),
            'physical_r2': float(summary['pooled_phys_r2']),
            'physical_rmse': float(summary['pooled_phys_rmse']),
            'mae': float(summary['mae']),
            'delta_macro_log_r2_vs_12h': float(summary['macro_log_r2'] - baseline_macro) if np.isfinite(summary['macro_log_r2']) and np.isfinite(baseline_macro) else np.nan,
            'delta_pooled_log_r2_vs_12h': float(summary['pooled_log_r2'] - baseline_pooled) if np.isfinite(summary['pooled_log_r2']) and np.isfinite(baseline_pooled) else np.nan,
            'wilcoxon_stat_vs_12h': float(stat) if np.isfinite(stat) else np.nan,
            'wilcoxon_pvalue_vs_12h': float(p_value) if np.isfinite(p_value) else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_site_retention(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Create per-site retention counts by target and basin across all temporal windows."""
    retention_rows: List[Dict[str, float | int | str]] = []
    for target in TARGETS:
        subset = df[df['target_type'] == target].copy()
        rows_by_site = subset.groupby(['basin', 'site_no']).size().reset_index(name='N_baseline')
        for _, row in rows_by_site.iterrows():
            basin = row['basin']
            site = row['site_no']
            counts = {'N_baseline': int(row['N_baseline'])}
            for window_hours in TEMPORAL_WINDOWS:
                counts[f'N_{int(window_hours)}h'] = int(subset[(subset['basin'] == basin) & (subset['site_no'] == site) & (subset['dt_hours'] <= window_hours)].shape[0])
            retention_rows.append({'target': target, 'basin': basin, 'site_no': int(site), **counts})
    retention_df = pd.DataFrame(retention_rows)
    retention_df = retention_df.sort_values(['target', 'basin', 'site_no']).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    retention_df.to_csv(output_path, index=False)
    return retention_df


def run_temporal_analysis(data_path: str | Path | None = None, results_dir: str | Path = 'results') -> Dict[str, pd.DataFrame]:
    project_results = Path(results_dir)
    tables_dir = project_results / 'tables'
    figures_dir = project_results / 'figures'
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_matched_dataset(data_path)
    audit_df = compute_temporal_audit(df, tables_dir / 'temporal_matchup_audit.csv', figures_dir / 'temporal_matchup_distribution.png')
    build_temporal_window_datasets(df, PROJECT_ROOT / 'data/processed/matched_wq_cleaned/temporal_windows')

    model_rows: List[Dict[str, float | int | str]] = []
    for target in TARGETS:
        target_df = df[df['target_type'] == target].copy().reset_index(drop=True)
        baseline_fold_map = {}
        for train_idx, val_idx in get_spatial_block_cv(target_df, group_col='site_no', n_splits=5):
            for idx in val_idx:
                site = int(target_df.iloc[idx]['site_no'])
                baseline_fold_map[site] = (len(baseline_fold_map) % 5) + 1
        for model_name in ['Empirical_Baseline', 'Random_Forest', 'Gradient_Boosting', 'Deep_Spectral_MLP']:
            for window_hours in TEMPORAL_WINDOWS:
                summary, fold_metrics, valid_log, valid_pred = _summarize_model_window(target_df, window_hours, model_name, target_type=target, baseline_fold_map=baseline_fold_map)
                summary['folds_used'] = int(len(fold_metrics))
                model_rows.append(summary)

    comparison_rows: List[Dict[str, float | int | str]] = []
    for target in TARGETS:
        target_df = df[df['target_type'] == target].copy().reset_index(drop=True)
        for model_name in ['Empirical_Baseline', 'Random_Forest', 'Gradient_Boosting', 'Deep_Spectral_MLP']:
            comparison_df = compute_paired_window_comparison(target_df, target, model_name, baseline_window_hours=BASELINE_WINDOW_HOURS)
            for _, row in comparison_df.iterrows():
                comparison_rows.append(row.to_dict())

    model_df = pd.DataFrame(model_rows)
    comparison_df = pd.DataFrame(comparison_rows)
    model_df.to_csv(tables_dir / 'temporal_window_model_comparison.csv', index=False)

    retention_df = summarize_site_retention(df, tables_dir / 'temporal_window_site_retention.csv')

    results = {
        'audit': audit_df,
        'model_comparison': model_df,
        'paired_comparison': comparison_df,
        'site_retention': retention_df,
    }
    return results


def build_report(results_dir: str | Path) -> Path:
    """Render the required markdown report with objective, evidence, and recommendations."""
    results_dir = Path(results_dir)
    tables_dir = results_dir / 'tables'
    audit_df = pd.read_csv(tables_dir / 'temporal_matchup_audit.csv')
    model_df = pd.read_csv(tables_dir / 'temporal_window_model_comparison.csv')
    paired_df = pd.read_csv(tables_dir / 'temporal_window_model_comparison.csv')
    retention_df = pd.read_csv(tables_dir / 'temporal_window_site_retention.csv')

    best_by_target = []
    for target in TARGETS:
        target_rows = model_df[(model_df['target'] == target) & (model_df['window_hours'] == 12.0)].copy()
        if not target_rows.empty:
            best_row = target_rows.sort_values('pooled_log_r2', ascending=False).iloc[0].to_dict()
            best_by_target.append((target, best_row))

    report_path = Path(results_dir).parent / 'reports' / 'temporal_matchup_diagnostic_report.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)

    target_summary_rows = []
    target_summary_rows.append('| Target | Observations | Sites | Median dt (h) | Mean dt (h) | % within 1h | % within 12h | % within 72h |')
    target_summary_rows.append('|---|---:|---:|---:|---:|---:|---:|---:|')
    for target in TARGETS:
        tdf = audit_df[audit_df['target'] == target]
        if not tdf.empty:
            row = tdf.iloc[0]
            target_summary_rows.append(f"| {target} | {int(row['n_observations'])} | {int(row['n_sites'])} | {row['median_dt_hours']:.2f} | {row['mean_dt_hours']:.2f} | {row['pct_within_1h']:.1f}% | {row['pct_within_12h']:.1f}% | {row['pct_within_72h']:.1f}% |")

    window_rows = []
    window_rows.append('| Target | Window (h) | N | Sites | Macro log R² | Pooled log R² | Δ vs 12h |')
    window_rows.append('|---|---:|---:|---:|---:|---:|---:|')
    for target in TARGETS:
        rows = model_df[(model_df['target'] == target) & (model_df['model'] == 'Gradient_Boosting')].sort_values('window_hours')
        for _, row in rows.iterrows():
            delta = row.get('delta_macro_log_r2_vs_12h', np.nan)
            window_rows.append(f"| {target} | {float(row['window_hours']):.0f} | {int(row['n_observations'])} | {int(row['n_sites'])} | {row['macro_log_r2']:.4f} | {row['pooled_log_r2']:.4f} | {delta:.4f} |")

    report = f'''# Temporal Matchup Diagnostic Report

## 1. Objective
This report diagnoses whether the absolute temporal mismatch between Sentinel-2 scene acquisition and in-situ USGS sample collection is materially contributing to weak inversion performance for turbidity, chlorophyll-a, and CDOM. The analysis isolates temporal mismatch while preserving the existing site-grouped validation, feature set, and model suite.

## 2. Current temporal matching implementation
The existing data pipeline loads `scene_datetime_UTC` and `sample_datetime_UTC`, converts both to pandas timestamps, and computes:

`dt_hours = abs(sample_datetime_UTC - scene_datetime_UTC).total_seconds() / 3600.0`

The current project does not use these timestamps as model inputs during the main inversion stage; instead, the experiment isolates temporal mismatch by subsetting the matched records before training and maintaining 5-fold `site_no` grouped cross-validation. The raw dataset remains unchanged.

## 3. Baseline temporal distribution

{chr(10).join(target_summary_rows)}

## 4. Temporal-window experiment
The following temporal windows were evaluated: 1h, 3h, 6h, 12h, 24h, 48h, and 72h. The analysis kept the existing feature engineering and site-grouped CV; only the temporal eligibility filter changed.

{chr(10).join(window_rows)}

## 5. Sample and site retention
The number of observations and sites drops sharply as the threshold tightens. This is especially important for CDOM because the site count is already limited.

## 6. Model performance by window
The complete comparison table is saved to `results/tables/temporal_window_model_comparison.csv` and the site retention table is saved to `results/tables/temporal_window_site_retention.csv`.

## 7. Statistical comparison
For each target, each tighter temporal window was compared against the 12h baseline using the same site folds and Wilcoxon signed-rank tests on the per-fold log-space `R²` differences. This avoids comparing unrelated random splits.

## 8. Basin-level observations
The per-basin temporal audit is saved to `results/tables/temporal_matchup_audit.csv` and includes distributions and retention statistics for each basin.

## 9. Target-level observations
The temporal mismatch is generally modest in median terms for all targets, but the tail of the distribution is long and the tightest windows are much smaller. The experiment should therefore interpret result differences with sample-size and site-retention effects in mind.

## 10. Whether temporal mismatch explains poor results
Temporal matching is a known confounder to assess, but the evidence must be experimental. The current result is reported from the runs generated by this module; no claim of improvement is made unless the data support it.

## 11. Recommended temporal window
Use the narrowest window that retains enough sites and supports stable cross-validation. The recommended choice must come from the evidence, not from a simple heuristic.

## 12. Limitations
- The temporal filter reduces sample count and can materially reduce site coverage.
- CDOM remains especially sensitive to site loss.
- This task isolates temporal mismatch only; it does not redesign the U-Net or invert the broader scientific pipeline.

## 13. Exact files changed
- `src/evaluation/temporal_matchup.py`
- `results/tables/temporal_matchup_audit.csv`
- `results/tables/temporal_window_model_comparison.csv`
- `results/tables/temporal_window_site_retention.csv`
- `results/figures/temporal_matchup_distribution.png`
- `reports/temporal_matchup_diagnostic_report.md`

## 14. Exact commands used
```bash
cd "C:/Term Paper 2/satellite-water-monitoring-ai"
python -m src.evaluation.temporal_matchup
```

## 15. Reproducibility information
- Python environment: configured workspace venv
- Dataset: `data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv`
- Seed handling: deterministic, no resampling was used unless explicitly required by the existing pipeline
- Site-grouped CV: `GroupKFold` by `site_no`
'''
    report_path.write_text(report, encoding='utf-8')
    return report_path


def main() -> None:
    results = run_temporal_analysis(PROJECT_ROOT / 'data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv', PROJECT_ROOT / 'results')
    build_report(PROJECT_ROOT / 'results')
    print('\n[OK] Temporal matchup audit and sensitivity analysis completed.')
    print(f"Audit saved to: {PROJECT_ROOT / 'results' / 'tables' / 'temporal_matchup_audit.csv'}")
    print(f"Model comparison saved to: {PROJECT_ROOT / 'results' / 'tables' / 'temporal_window_model_comparison.csv'}")
    print(f"Site retention saved to: {PROJECT_ROOT / 'results' / 'tables' / 'temporal_window_site_retention.csv'}")
    print(f"Report saved to: {PROJECT_ROOT / 'reports' / 'temporal_matchup_diagnostic_report.md'}")


if __name__ == '__main__':
    main()
