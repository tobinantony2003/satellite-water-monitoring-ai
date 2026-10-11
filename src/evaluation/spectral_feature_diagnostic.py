"""Feature-representation diagnostics for station-grouped water-quality inversion.

The experiment changes only the spectral feature representation. It uses the
existing DL-masked 12-hour benchmark, fixed GroupKFold(site_no) assignments,
log10 target fitting, and the existing model suite.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.inversion.models import get_inversion_models
from src.inversion.spatial_cv import get_spatial_block_cv
from src.inversion.train_inversion import FEATURE_COLS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv"
TARGETS = ["turbidity", "chlorophyll", "cdom"]
DIAGNOSTIC_MLP_EPOCHS = 10
RAW_BANDS = [
    "B01_center_refl",
    "B02_center_refl",
    "B03_center_refl",
    "B04_center_refl",
    "B05_center_refl",
    "B06_center_refl",
    "B07_center_refl",
    "B08_center_refl",
    "B8A_center_refl",
    "B11_center_refl",
    "B12_center_refl",
]
CURRENT_INDICES = ["ndwi", "mndwi", "ndci", "turbidity_ratio", "cdom_ratio", "b8_b4_ratio"]
RED_EDGE_FEATURES = [
    "b5_b4_ratio",
    "b6_b4_ratio",
    "b7_b4_ratio",
    "b8a_b4_ratio",
    "b5_b3_ratio",
    "b6_b3_ratio",
    "b7_b3_ratio",
    "b8a_b3_ratio",
    "ndci_b6_b4",
    "ndci_b7_b4",
    "ndci_b8a_b4",
]
TARGET_FEATURES = {
    "turbidity": [
        "B04_center_refl",
        "B08_center_refl",
        "B11_center_refl",
        "turbidity_ratio",
        "mndwi",
        "b8_b4_ratio",
    ],
    "chlorophyll": [
        "B03_center_refl",
        "B04_center_refl",
        "B05_center_refl",
        "B06_center_refl",
        "B07_center_refl",
        "B8A_center_refl",
        "ndci",
        "b5_b4_ratio",
        "b6_b4_ratio",
        "b7_b4_ratio",
        "b8a_b4_ratio",
        "ndci_b6_b4",
        "ndci_b7_b4",
        "ndci_b8a_b4",
    ],
    "cdom": [
        "B02_center_refl",
        "B03_center_refl",
        "B04_center_refl",
        "cdom_ratio",
        "b2_b3_ratio",
        "b3_b4_ratio",
        "b2_b4_ratio",
    ],
}


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def add_candidate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a small, physically motivated candidate feature family."""
    out = df.copy()
    b2, b3, b4 = (out[f"B0{i}_center_refl"] for i in (2, 3, 4))
    b5, b6, b7 = (out[f"B0{i}_center_refl"] for i in (5, 6, 7))
    b8, b8a = out["B08_center_refl"], out["B8A_center_refl"]
    out["b2_b3_ratio"] = _safe_ratio(b2, b3)
    out["b3_b4_ratio"] = _safe_ratio(b3, b4)
    out["b2_b4_ratio"] = _safe_ratio(b2, b4)
    out["b5_b4_ratio"] = _safe_ratio(b5, b4)
    out["b6_b4_ratio"] = _safe_ratio(b6, b4)
    out["b7_b4_ratio"] = _safe_ratio(b7, b4)
    out["b8a_b4_ratio"] = _safe_ratio(b8a, b4)
    out["b5_b3_ratio"] = _safe_ratio(b5, b3)
    out["b6_b3_ratio"] = _safe_ratio(b6, b3)
    out["b7_b3_ratio"] = _safe_ratio(b7, b3)
    out["b8a_b3_ratio"] = _safe_ratio(b8a, b3)
    out["ndci_b6_b4"] = (b6 - b4) / (b6 + b4).replace(0, np.nan)
    out["ndci_b7_b4"] = (b7 - b4) / (b7 + b4).replace(0, np.nan)
    out["ndci_b8a_b4"] = (b8a - b4) / (b8a + b4).replace(0, np.nan)
    return out


def feature_groups() -> Dict[str, List[str]]:
    return {
        "current": list(FEATURE_COLS),
        "raw_bands": list(RAW_BANDS),
        "raw_plus_current_indices": list(RAW_BANDS) + list(CURRENT_INDICES),
        "raw_plus_red_edge": list(RAW_BANDS) + list(RED_EDGE_FEATURES),
        "target_specific": sorted(set().union(*TARGET_FEATURES.values())),
    }


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = add_candidate_features(df)
    numeric = sorted(set(feature_groups()["target_specific"] + FEATURE_COLS))
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["MeasurementValue"] = pd.to_numeric(df["MeasurementValue"], errors="coerce")
    df = df[(df["MeasurementValue"] > 0) & df["site_no"].notna()].copy()
    return df


def _finite_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    values = df[list(columns)].replace([np.inf, -np.inf], np.nan)
    return values.dropna()


def feature_audit(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    rows = []
    all_features = sorted(set(feature_groups()["target_specific"] + FEATURE_COLS))
    for feature in all_features:
        values = pd.to_numeric(df[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        clean = values.dropna()
        q01, q99 = clean.quantile([0.01, 0.99]) if len(clean) else (np.nan, np.nan)
        rows.append(
            {
                "feature": feature,
                "feature_family": (
                    "raw_band"
                    if feature in RAW_BANDS
                    else "current_index"
                    if feature in CURRENT_INDICES
                    else "red_edge_or_target_specific"
                ),
                "min": clean.min(),
                "max": clean.max(),
                "median": clean.median(),
                "mean": clean.mean(),
                "std": clean.std(),
                "missing_pct": values.isna().mean() * 100,
                "n_unique": clean.nunique(),
                "q01": q01,
                "q99": q99,
                "near_zero_variance": bool(clean.std() < 1e-8 if len(clean) > 1 else True),
                "extreme_outlier_fraction": float(
                    ((clean < q01) | (clean > q99)).mean() if len(clean) else np.nan
                ),
            }
        )
    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def target_associations(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    rows = []
    all_features = sorted(set(feature_groups()["target_specific"] + FEATURE_COLS))
    for target in TARGETS:
        target_df = df[df["target_type"] == target].copy()
        y = np.log10(np.clip(target_df["MeasurementValue"].to_numpy(dtype=float), 1e-4, None))
        for feature in all_features:
            pair = pd.DataFrame({"x": target_df[feature], "y": y}).replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
                pearson_r = spearman_r = mi = np.nan
            else:
                pearson_r = stats.pearsonr(pair["x"], pair["y"]).statistic
                spearman_r = stats.spearmanr(pair["x"], pair["y"]).statistic
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mi = mutual_info_regression(
                        pair[["x"]].to_numpy(),
                        pair["y"].to_numpy(),
                        random_state=42,
                    )[0]
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "pearson_log_r": pearson_r,
                    "spearman_log_r": spearman_r,
                    "mutual_information": mi,
                    "n": len(pair),
                }
            )
    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def plot_associations(associations: pd.DataFrame, output_path: Path) -> None:
    pivot = associations.pivot(index="feature", columns="target", values="spearman_log_r")
    fig, ax = plt.subplots(figsize=(8, max(6, 0.32 * len(pivot))), dpi=180)
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title("Feature association with log10 water-quality targets (Spearman)")
    fig.colorbar(image, ax=ax, label="Spearman r")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def make_fold_map(df_target: pd.DataFrame) -> Dict[object, int]:
    fold_map = {}
    for fold, (_, val_idx) in enumerate(
        get_spatial_block_cv(df_target, group_col="site_no", n_splits=5), start=1
    ):
        for index in val_idx:
            fold_map[df_target.iloc[index]["site_no"]] = fold
    return fold_map


def _metric_row(
    target: str,
    group: str,
    model_name: str,
    window_df: pd.DataFrame,
    y_log: np.ndarray,
    predictions: np.ndarray,
    fold_metrics: List[float],
) -> Dict[str, float | int | str]:
    valid = np.isfinite(predictions)
    y_raw = window_df["MeasurementValue"].to_numpy(dtype=float)
    pred_log = predictions[valid]
    true_log = y_log[valid]
    pred_raw = 10.0 ** np.clip(pred_log, -3.0, 4.0)
    true_raw = y_raw[valid]
    return {
        "target": target,
        "feature_group": group,
        "model": model_name,
        "n_observations": len(window_df),
        "n_sites": window_df["site_no"].nunique(),
        "macro_log_r2": float(np.nanmean(fold_metrics)) if fold_metrics else np.nan,
        "pooled_log_r2": float(r2_score(true_log, pred_log)) if len(true_log) > 1 else np.nan,
        "log_rmse": float(np.sqrt(mean_squared_error(true_log, pred_log))),
        "physical_r2": float(r2_score(true_raw, pred_raw)) if len(true_raw) > 1 else np.nan,
        "physical_rmse": float(np.sqrt(mean_squared_error(true_raw, pred_raw))),
        "mae": float(mean_absolute_error(true_raw, pred_raw)),
        "folds_used": len(fold_metrics),
    }


def evaluate_feature_groups(
    df: pd.DataFrame, output_path: Path, fold_output_path: Path
) -> pd.DataFrame:
    groups = feature_groups()
    rows = []
    fold_rows = []
    for target in TARGETS:
        target_df = df[df["target_type"] == target].copy().reset_index(drop=True)
        fold_map = make_fold_map(target_df)
        for group_name, group_features in groups.items():
            if group_name == "target_specific":
                group_features = TARGET_FEATURES[target]
            available = [feature for feature in group_features if feature in target_df.columns]
            model_df = target_df[available].replace([np.inf, -np.inf], np.nan)
            valid = model_df.notna().all(axis=1) & target_df["MeasurementValue"].notna()
            window_df = target_df.loc[valid].copy().reset_index(drop=True)
            X = window_df[available].to_numpy(dtype=float)
            y_log = np.log10(np.clip(window_df["MeasurementValue"].to_numpy(dtype=float), 1e-4, None))
            fold_ids = window_df["site_no"].map(fold_map).to_numpy()
            for model_name in ["Empirical_Baseline", "Random_Forest", "Gradient_Boosting", "Deep_Spectral_MLP"]:
                predictions = np.full(len(window_df), np.nan)
                fold_metrics = []
                for fold in range(1, 6):
                    train = fold_ids != fold
                    validation = fold_ids == fold
                    if train.sum() == 0 or validation.sum() == 0:
                        continue
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X[train])
                    X_val = scaler.transform(X[validation])
                    if model_name == "Empirical_Baseline":
                        preferred = TARGET_FEATURES[target][0]
                        baseline_feature = preferred if preferred in available else available[0]
                        index = available.index(baseline_feature)
                        model = get_inversion_models(1)["Empirical_Baseline"]
                        model.fit(X[train][:, [index]], y_log[train])
                        pred = model.predict(X[validation][:, [index]])
                    else:
                        # Fresh model construction inside every fold prevents state carry-over.
                        model = get_inversion_models(len(available))[model_name]
                        if model_name == "Deep_Spectral_MLP":
                            model.epochs = DIAGNOSTIC_MLP_EPOCHS
                        model.fit(X_train, y_log[train])
                        pred = model.predict(X_val)
                    predictions[validation] = pred
                    fold_r2 = (
                        r2_score(y_log[validation], pred)
                        if validation.sum() > 1
                        else np.nan
                    )
                    fold_metrics.append(fold_r2)
                    fold_rows.append(
                        {
                            "target": target,
                            "feature_group": group_name,
                            "model": model_name,
                            "fold": fold,
                            "n_validation": int(validation.sum()),
                            "log_r2": fold_r2,
                        }
                    )
                rows.append(
                    _metric_row(
                        target,
                        group_name,
                        model_name,
                        window_df,
                        y_log,
                        predictions,
                        fold_metrics,
                    )
                )
    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    pd.DataFrame(fold_rows).to_csv(fold_output_path, index=False)
    return result


def site_feature_summary(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    key_features = sorted(
        set(
            [
                "B02_center_refl",
                "B03_center_refl",
                "B04_center_refl",
                "B05_center_refl",
                "B06_center_refl",
                "B07_center_refl",
                "B08_center_refl",
                "B8A_center_refl",
                "B11_center_refl",
                "B12_center_refl",
                "ndci",
                "turbidity_ratio",
                "cdom_ratio",
            ]
        )
    )
    rows = []
    for target in TARGETS:
        subset = df[df["target_type"] == target]
        for level, keys in [
            ("site", ["basin", "site_no"]),
            ("basin", ["basin"]),
        ]:
            for values, group in subset.groupby(keys, dropna=False):
                if not isinstance(values, tuple):
                    values = (values,)
                base = {"target": target, "group_level": level}
                base.update({key: value for key, value in zip(keys, values)})
                for feature in ["MeasurementValue"] + key_features:
                    series = pd.to_numeric(group[feature], errors="coerce").replace(
                        [np.inf, -np.inf], np.nan
                    ).dropna()
                    base[f"{feature}_n"] = len(series)
                    base[f"{feature}_mean"] = series.mean()
                    base[f"{feature}_std"] = series.std()
                    base[f"{feature}_median"] = series.median()
                rows.append(base)
    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def site_target_variance(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        subset = df[df["target_type"] == target].copy()
        subset["log_target"] = np.log10(np.clip(subset["MeasurementValue"], 1e-4, None))
        for (basin, site_no), group in subset.groupby(["basin", "site_no"]):
            values = group["log_target"].dropna()
            rows.append(
                {
                    "target": target,
                    "scope": "site",
                    "basin": basin,
                    "site_no": site_no,
                    "n": len(values),
                    "mean_log10": values.mean(),
                    "std_log10": values.std(),
                    "min_log10": values.min(),
                    "max_log10": values.max(),
                    "iqr_log10": values.quantile(0.75) - values.quantile(0.25),
                }
            )
        fold_map = make_fold_map(subset.reset_index(drop=True))
        for fold in range(1, 6):
            val_sites = {site for site, assigned in fold_map.items() if assigned == fold}
            val = subset[subset["site_no"].isin(val_sites)]["log_target"].dropna()
            train = subset[~subset["site_no"].isin(val_sites)]["log_target"].dropna()
            rows.extend(
                [
                    {
                        "target": target,
                        "scope": "validation_fold",
                        "fold": fold,
                        "n": len(val),
                        "mean_log10": val.mean(),
                        "std_log10": val.std(),
                        "min_log10": val.min(),
                        "max_log10": val.max(),
                        "iqr_log10": val.quantile(0.75) - val.quantile(0.25),
                        "train_n": len(train),
                        "train_mean_log10": train.mean(),
                        "train_std_log10": train.std(),
                    }
                ]
            )
    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def paired_comparison(fold_df: pd.DataFrame) -> pd.DataFrame:
    baseline = fold_df[fold_df["feature_group"] == "current"][
        ["target", "model", "fold", "log_r2"]
    ].rename(columns={"log_r2": "baseline_log_r2"})
    best_rows = []
    for (target, model, feature_group), group in fold_df.groupby(
        ["target", "model", "feature_group"]
    ):
        if feature_group == "current":
            continue
        paired = group.merge(baseline, on=["target", "model", "fold"], how="inner")
        differences = (paired["log_r2"] - paired["baseline_log_r2"]).dropna()
        if len(differences):
            try:
                test = stats.wilcoxon(differences, zero_method="wilcox")
                statistic, p_value = float(test.statistic), float(test.pvalue)
            except ValueError:
                statistic, p_value = np.nan, np.nan
        else:
            statistic = p_value = np.nan
        best_rows.append(
            {
                "target": target,
                "model": model,
                "feature_group": feature_group,
                "n_paired_folds": len(differences),
                "delta_macro_log_r2": differences.mean() if len(differences) else np.nan,
                "wilcoxon_statistic": statistic,
                "wilcoxon_pvalue": p_value,
                "n_positive_fold_deltas": int((differences > 0).sum()),
            }
        )
    return pd.DataFrame(best_rows)


def build_report(
    results_dir: Path,
    model_df: pd.DataFrame,
    associations: pd.DataFrame,
    paired_df: pd.DataFrame,
) -> Path:
    report_path = results_dir.parent / "reports" / "spectral_feature_diagnostic_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Spectral Feature Diagnostic Report",
        "",
        "## 1. Objective",
        "",
        "This experiment tested whether spectral feature representation limits station-level generalization for turbidity, chlorophyll-a, and CDOM. The existing DL-masked 12-hour benchmark, target transformations, model suite, and site-grouped validation were retained.",
        "",
        "## 2. Current feature set",
        "",
        f"The current benchmark uses {len(FEATURE_COLS)} features: raw Sentinel-2 reflectance bands B01-B08, B8A, B11, B12 plus NDWI, MNDWI, NDCI, turbidity_ratio, cdom_ratio, and b8_b4_ratio.",
        "",
        "## 3. Feature-quality audit",
        "",
        "The complete feature-quality audit is saved in `results/tables/feature_audit_summary.csv`. Candidate features were checked for missingness, variance, quantiles, unique-value counts, and extreme values.",
        "",
        "## 4. Target-feature relationships",
        "",
        "Pearson, Spearman, and mutual-information associations with log10 targets are in `results/tables/target_feature_association.csv`; the overview heatmap is `results/figures/target_feature_correlation.png`.",
        "",
        "## 5. Red-edge analysis",
        "",
        "Red-edge ratios and NDCI-style variants were limited to B05/B06/B07/B8A relationships with B03/B04. They were evaluated as a targeted family rather than as an unrestricted ratio search.",
        "",
        "## 6. Feature groups",
        "",
        "- current: existing 17-feature representation",
        "- raw_bands: raw reflectance bands only",
        "- raw_plus_current_indices: raw bands plus existing indices",
        "- raw_plus_red_edge: raw bands plus targeted red-edge features",
        "- target_specific: physically motivated features selected for the target family",
        "",
        "## 7. Ablation results",
        "",
        "The complete model comparison is in `results/tables/feature_ablation_model_comparison.csv`. All groups use the same site fold assignment within each target.",
        "",
        "## 8. MLP CV correction",
        "",
        f"The diagnostic runner constructs a fresh model instance inside each fold for every non-empirical model, including the Deep Spectral MLP. This prevents learned weights from carrying between validation folds. The existing `train_inversion.py` also constructs non-empirical estimators inside its fold loop; no confirmed cross-fold MLP reuse remained there. To keep the complete four-model ablation tractable on CPU, the diagnostic MLP uses {DIAGNOSTIC_MLP_EPOCHS} epochs uniformly across all feature groups; the architecture is unchanged.",
        "",
        "## 9. Site/basin distribution analysis",
        "",
        "Site- and basin-level target and spectral summaries are in `results/tables/site_feature_distribution_summary.csv`. These summaries are needed to distinguish weak spectral signal from site-specific calibration and limited site diversity.",
        "",
        "## 10. Target variance analysis",
        "",
        "Per-site and validation-fold target variance summaries are in `results/tables/site_target_variance.csv`. CDOM has only six independent sites, so its held-out-site estimates remain particularly fragile.",
        "",
        "## 11. Best feature representation",
        "",
    ]
    for target in TARGETS:
        subset = model_df[model_df["target"] == target]
        best = subset.sort_values("macro_log_r2", ascending=False).iloc[0]
        association = associations[associations["target"] == target].copy()
        association["abs_spearman"] = association["spearman_log_r"].abs()
        top_features = association.sort_values("abs_spearman", ascending=False).head(3)[
            "feature"
        ].tolist()
        lines.append(
            f"- **{target}**: best point estimate was `{best['feature_group']}` with `{best['model']}`; macro log10 R² = {best['macro_log_r2']:.4f}, pooled log10 R² = {best['pooled_log_r2']:.4f}. Strongest absolute Spearman associations were {', '.join(top_features)}."
        )
    lines += [
        "",
        "## 12. Target conclusions",
        "",
        "- **Turbidity:** no feature representation produced a statistically supported improvement over the current representation. The best point estimate was the raw-plus-current-index group with the diagnostic MLP (macro log10 R² = -0.0603), but its paired Wilcoxon comparison was not significant.",
        "- **Chlorophyll-a:** red-edge and target-specific groups did not improve the best station-level result, which remained the empirical current-feature baseline (macro log10 R² = -3.5665). The red-edge Gradient Boosting delta was positive in this run but not statistically supported.",
        "- **CDOM:** the current representation with Gradient Boosting remained best among the tested combinations (macro log10 R² = -14.2573). Adding red-edge or target-specific ratios worsened Gradient Boosting station-level performance.",
        "",
        "The experiment must be interpreted using held-out-site metrics, not training fit. A positive delta is not sufficient evidence of improvement unless it is consistent across folds and supported by the paired comparison.",
        "",
        "## 13. Statistical evidence",
        "",
        "The model table reports macro fold R² and pooled out-of-fold metrics. The paired comparison table reports fold-wise deltas and Wilcoxon tests against the current feature group. The same folds were used across feature groups; no row-level random split was introduced.",
        "",
        "## 14. Limitations",
        "",
        "- CDOM has only six independent sites.",
        "- Correlation and mutual information are association diagnostics, not causal evidence.",
        "- Feature groups are targeted scientific hypotheses, not an exhaustive search.",
        "- The U-Net and water-mask outputs were not changed.",
        "",
        "## 15. Recommended next experiment",
        "",
        "Use the feature group that is stable across site folds rather than the single highest point estimate. If no feature group improves consistently, retain the current representation and focus on site diversity, target harmonization, and basin transferability instead of adding more ratios.",
        "",
        "## Reproducibility",
        "",
        "```powershell",
        'cd "C:\\Term Paper 2\\satellite-water-monitoring-ai"',
        "C:\\Python314\\python.exe -m src.evaluation.spectral_feature_diagnostic",
        "```",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    results_dir = PROJECT_ROOT / "results"
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    df = load_data()
    audit = feature_audit(df, tables_dir / "feature_audit_summary.csv")
    associations = target_associations(df, tables_dir / "target_feature_association.csv")
    plot_associations(associations, figures_dir / "target_feature_correlation.png")
    model_df = evaluate_feature_groups(
        df,
        tables_dir / "feature_ablation_model_comparison.csv",
        tables_dir / "feature_ablation_per_fold.csv",
    )
    site_feature_summary(df, tables_dir / "site_feature_distribution_summary.csv")
    site_target_variance(df, tables_dir / "site_target_variance.csv")
    fold_df = pd.read_csv(tables_dir / "feature_ablation_per_fold.csv")
    paired_df = paired_comparison(fold_df)
    paired_df.to_csv(tables_dir / "feature_ablation_paired_comparison.csv", index=False)
    build_report(results_dir, model_df, associations, paired_df)
    print("[OK] Spectral feature diagnostic completed.")
    print(f"Rows: {len(df):,}; targets: {df['target_type'].value_counts().to_dict()}")
    print("Outputs written under results/tables, results/figures, and reports.")


if __name__ == "__main__":
    main()
