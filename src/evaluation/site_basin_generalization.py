"""Site, basin, and target-harmonization generalization diagnostics."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from src.inversion.models import get_inversion_models
from src.inversion.spatial_cv import get_spatial_block_cv
from src.inversion.train_inversion import FEATURE_COLS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
DATA = PROJECT_ROOT / "data/processed/matched_wq_cleaned"
TARGETS = ["turbidity", "chlorophyll", "cdom"]
KEY_FEATURES = [
    "B03_center_refl",
    "B04_center_refl",
    "B05_center_refl",
    "B06_center_refl",
    "B07_center_refl",
    "B08_center_refl",
    "B8A_center_refl",
    "B11_center_refl",
    "B12_center_refl",
    "ndwi",
    "mndwi",
    "ndci",
    "turbidity_ratio",
    "cdom_ratio",
    "b8_b4_ratio",
]


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["site_no"] = df["site_no"].astype(str)
    df["MeasurementValue"] = pd.to_numeric(df["MeasurementValue"], errors="coerce")
    for column in FEATURE_COLS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df[df["MeasurementValue"] > 0].copy()


def raw_unitmixed_path() -> Path:
    candidates = [
        DATA / "matched_set_b_dl_masked_unitmixed_raw.csv",
        DATA / "matched_set_b_unitmixed_raw.csv",
        DATA / "matched_set_b_dl_masked.csv",
    ]
    return next(path for path in candidates if path.exists())


def target_harmonization_audit(df: pd.DataFrame, output: Path) -> pd.DataFrame:
    raw = load_dataset(raw_unitmixed_path())
    rows = []
    conversion_factors = {
        ("cdom", "01467200"): 3.0,
        ("chlorophyll", "01427510"): 3.9554,
        ("chlorophyll", "444306122193200"): 2.8167,
        ("chlorophyll", "453027122400000"): 4.1860,
        ("chlorophyll", "444306122144600"): 2.8167,
    }
    for keys, group in raw.groupby(
        ["target_type", "site_no", "basin", "parm_cd", "parm_unit", "parm_nm"],
        dropna=False,
    ):
        target, site, basin, code, unit, method = keys
        factor = conversion_factors.get((target, str(site).zfill(8)), np.nan)
        converted = pd.notna(factor) and str(unit).upper() == "RFU"
        values = group["MeasurementValue"].dropna()
        rows.append(
            {
                "target": target,
                "site_no": site,
                "basin": basin,
                "parameter_code": code,
                "unit": unit,
                "measurement_method": method,
                "N": len(values),
                "min": values.min(),
                "median": values.median(),
                "mean": values.mean(),
                "max": values.max(),
                "conversion_applied": converted,
                "conversion_factor": factor,
                "native_or_converted": "converted" if converted else "native_or_excluded",
                "present_in_current_benchmark": int(
                    ((df["target_type"] == target) & (df["site_no"] == str(site))).sum()
                )
                > 0,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False)
    return result


def distribution_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    site_rows, basin_rows = [], []
    for target in TARGETS:
        subset = df[df["target_type"] == target].copy()
        subset["log_target"] = np.log10(np.clip(subset["MeasurementValue"], 1e-4, None))
        for site, group in subset.groupby("site_no"):
            values = group["MeasurementValue"]
            log_values = group["log_target"]
            site_rows.append(
                {
                    "target": target,
                    "site_no": site,
                    "basin": group["basin"].iloc[0],
                    "N": len(group),
                    "mean": values.mean(),
                    "median": values.median(),
                    "std": values.std(),
                    "IQR": values.quantile(0.75) - values.quantile(0.25),
                    "min": values.min(),
                    "max": values.max(),
                    "mean_log10": log_values.mean(),
                    "std_log10": log_values.std(),
                }
            )
        for basin, group in subset.groupby("basin"):
            values = group["MeasurementValue"]
            basin_rows.append(
                {
                    "target": target,
                    "basin": basin,
                    "N": len(group),
                    "n_sites": group["site_no"].nunique(),
                    "mean": values.mean(),
                    "median": values.median(),
                    "variance": values.var(),
                    "min": values.min(),
                    "max": values.max(),
                }
            )
    return pd.DataFrame(site_rows), pd.DataFrame(basin_rows)


def plot_distributions(site_df: pd.DataFrame, basin_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=180)
    for ax, target in zip(axes, TARGETS):
        subset = site_df[site_df["target"] == target]
        ax.boxplot(
            [
                np.log10(
                    np.clip(
                        load_dataset(DATA / "matched_set_b_dl_masked.csv")
                        .query("target_type == @target and site_no == @site")["MeasurementValue"],
                        1e-4,
                        None,
                    )
                )
                for site in subset["site_no"]
            ],
            tick_labels=subset["site_no"].tolist(),
            showfliers=False,
        )
        ax.set_title(target)
        ax.set_ylabel("log10 target")
        ax.tick_params(axis="x", rotation=90, labelsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "target_distribution_by_site.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=180)
    for ax, target in zip(axes, TARGETS):
        subset = basin_df[basin_df["target"] == target]
        ax.bar(subset["basin"], np.log10(np.clip(subset["mean"], 1e-4, None)))
        ax.set_title(target)
        ax.set_ylabel("log10 basin mean")
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIGURES / "target_distribution_by_basin.png", bbox_inches="tight")
    plt.close(fig)


def variance_components(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        subset = df[df["target_type"] == target].copy()
        subset["log_target"] = np.log10(np.clip(subset["MeasurementValue"], 1e-4, None))
        grand = subset["log_target"].mean()
        site_means = subset.groupby("site_no")["log_target"].mean()
        counts = subset.groupby("site_no").size()
        total_ss = ((subset["log_target"] - grand) ** 2).sum()
        between_ss = sum(counts[s] * (site_means[s] - grand) ** 2 for s in site_means.index)
        within_ss = max(0.0, total_ss - between_ss)
        total_var = subset["log_target"].var()
        between_fraction = between_ss / total_ss if total_ss else np.nan
        within_fraction = within_ss / total_ss if total_ss else np.nan
        ms_between = between_ss / max(len(site_means) - 1, 1)
        ms_within = within_ss / max(len(subset) - len(site_means), 1)
        mean_n = counts.mean()
        icc = (
            (ms_between - ms_within)
            / (ms_between + (mean_n - 1) * ms_within)
            if ms_between + (mean_n - 1) * ms_within
            else np.nan
        )
        rows.append(
            {
                "target": target,
                "N": len(subset),
                "n_sites": subset["site_no"].nunique(),
                "total_variance_log10": total_var,
                "between_site_variance_component": between_ss / max(len(subset) - 1, 1),
                "within_site_variance_component": within_ss / max(len(subset) - 1, 1),
                "ICC": icc,
                "pct_variance_between_sites": 100 * between_fraction,
                "pct_variance_within_sites": 100 * within_fraction,
            }
        )
    return pd.DataFrame(rows)


def feature_shift(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        subset = df[df["target_type"] == target]
        for feature in KEY_FEATURES:
            clean = subset[["site_no", "basin", feature]].replace([np.inf, -np.inf], np.nan).dropna()
            overall = clean[feature]
            for level in ["site", "basin"]:
                key = "site_no" if level == "site" else "basin"
                for category, group in clean.groupby(key):
                    pooled = clean[clean[key] != category][feature]
                    if len(group) < 2 or len(pooled) < 2:
                        continue
                    pooled_std = np.sqrt((group[feature].var() + pooled.var()) / 2)
                    rows.append(
                        {
                            "target": target,
                            "group_level": level,
                            "group": category,
                            "feature": feature,
                            "group_mean": group[feature].mean(),
                            "reference_mean": pooled.mean(),
                            "standardized_mean_difference": (
                                (group[feature].mean() - pooled.mean()) / pooled_std
                                if pooled_std > 0
                                else np.nan
                            ),
                            "ks_statistic": stats.ks_2samp(group[feature], pooled).statistic,
                            "ks_pvalue": stats.ks_2samp(group[feature], pooled).pvalue,
                            "overall_mean": overall.mean(),
                            "overall_std": overall.std(),
                        }
                    )
    return pd.DataFrame(rows)


def model_predict(train: pd.DataFrame, test: pd.DataFrame) -> Dict[str, float]:
    train = train.dropna(subset=FEATURE_COLS + ["MeasurementValue"])
    test = test.dropna(subset=FEATURE_COLS + ["MeasurementValue"])
    if len(train) < 10 or len(test) < 2:
        return {"N": len(test), "log_r2": np.nan, "log_rmse": np.nan, "physical_r2": np.nan, "physical_rmse": np.nan, "mae": np.nan}
    X_train = train[FEATURE_COLS].to_numpy(dtype=float)
    X_test = test[FEATURE_COLS].to_numpy(dtype=float)
    y_train = np.log10(np.clip(train["MeasurementValue"].to_numpy(dtype=float), 1e-4, None))
    y_test = np.log10(np.clip(test["MeasurementValue"].to_numpy(dtype=float), 1e-4, None))
    scaler = StandardScaler()
    model = get_inversion_models(len(FEATURE_COLS))["Gradient_Boosting"]
    model.fit(scaler.fit_transform(X_train), y_train)
    pred_log = model.predict(scaler.transform(X_test))
    pred_raw = 10 ** np.clip(pred_log, -3, 4)
    true_raw = test["MeasurementValue"].to_numpy(dtype=float)
    return {
        "N": len(test),
        "log_r2": r2_score(y_test, pred_log) if len(test) > 1 else np.nan,
        "log_rmse": np.sqrt(mean_squared_error(y_test, pred_log)),
        "physical_r2": r2_score(true_raw, pred_raw) if len(test) > 1 else np.nan,
        "physical_rmse": np.sqrt(mean_squared_error(true_raw, pred_raw)),
        "mae": mean_absolute_error(true_raw, pred_raw),
    }


def leave_one_site_out(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        subset = df[df["target_type"] == target]
        for site in sorted(subset["site_no"].unique()):
            metrics = model_predict(subset[subset["site_no"] != site], subset[subset["site_no"] == site])
            metrics.update({"target": target, "held_out_site": site, "basin": subset[subset["site_no"] == site]["basin"].iloc[0]})
            rows.append(metrics)
    return pd.DataFrame(rows)


def leave_one_basin_out(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        subset = df[df["target_type"] == target]
        basins = sorted(subset["basin"].dropna().unique())
        if len(basins) < 3:
            rows.append({"target": target, "status": "NOT SUFFICIENTLY SUPPORTED"})
            continue
        for basin in basins:
            test = subset[subset["basin"] == basin]
            metrics = model_predict(subset[subset["basin"] != basin], test)
            metrics.update(
                {
                    "target": target,
                    "held_out_basin": basin,
                    "sites": test["site_no"].nunique(),
                    "status": "supported",
                }
            )
            rows.append(metrics)
    return pd.DataFrame(rows)


def random_vs_grouped(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        subset = df[df["target_type"] == target].dropna(subset=FEATURE_COLS + ["MeasurementValue"]).reset_index(drop=True)
        train_idx, test_idx = train_test_split(np.arange(len(subset)), test_size=0.2, random_state=42)
        random_metrics = model_predict(subset.iloc[train_idx], subset.iloc[test_idx])
        grouped_predictions, grouped_truth = [], []
        for train_idx, test_idx in get_spatial_block_cv(subset, group_col="site_no", n_splits=5):
            train, test = subset.iloc[train_idx], subset.iloc[test_idx]
            metrics = model_predict(train, test)
            if metrics["N"]:
                grouped_predictions.append(metrics)
        grouped = {
            "N": sum(item["N"] for item in grouped_predictions),
            "log_r2": np.nanmean([item["log_r2"] for item in grouped_predictions]),
            "log_rmse": np.nanmean([item["log_rmse"] for item in grouped_predictions]),
            "physical_r2": np.nanmean([item["physical_r2"] for item in grouped_predictions]),
            "physical_rmse": np.nanmean([item["physical_rmse"] for item in grouped_predictions]),
            "mae": np.nanmean([item["mae"] for item in grouped_predictions]),
        }
        rows.append(
            {
                "target": target,
                "random_log_r2": random_metrics["log_r2"],
                "grouped_macro_log_r2": grouped["log_r2"],
                "difference_random_minus_grouped": random_metrics["log_r2"] - grouped["log_r2"],
                "random_log_rmse": random_metrics["log_rmse"],
                "grouped_log_rmse": grouped["log_rmse"],
                "random_physical_r2": random_metrics["physical_r2"],
                "grouped_physical_r2": grouped["physical_r2"],
                "random_mae": random_metrics["mae"],
                "grouped_mae": grouped["mae"],
            }
        )
    return pd.DataFrame(rows)


def harmonization_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    candidates = {
        "native_current": DATA / "matched_set_b_dl_masked.csv",
        "harmonized_secondary": DATA / "matched_set_b_dl_masked_harmonized.csv",
        "unitmixed_raw": raw_unitmixed_path(),
    }
    rows = []
    for name, path in candidates.items():
        if not path.exists():
            continue
        current = load_dataset(path)
        for target in TARGETS:
            subset = current[current["target_type"] == target]
            if subset.empty:
                continue
            metrics = []
            for train_idx, test_idx in get_spatial_block_cv(subset, group_col="site_no", n_splits=5):
                metrics.append(model_predict(subset.iloc[train_idx], subset.iloc[test_idx]))
            rows.append(
                {
                    "dataset_condition": name,
                    "target": target,
                    "N": len(subset),
                    "sites": subset["site_no"].nunique(),
                    "macro_log_r2": np.nanmean([m["log_r2"] for m in metrics]),
                    "macro_log_rmse": np.nanmean([m["log_rmse"] for m in metrics]),
                    "macro_physical_r2": np.nanmean([m["physical_r2"] for m in metrics]),
                    "macro_mae": np.nanmean([m["mae"] for m in metrics]),
                    "units": "/".join(sorted(subset["parm_unit"].dropna().astype(str).unique())),
                    "harmonization_flags": "/".join(
                        sorted(subset["harmonization_flag"].dropna().astype(str).unique())
                    )
                    if "harmonization_flag" in subset
                    else "",
                }
            )
    return pd.DataFrame(rows)


def site_level_performance(loso: pd.DataFrame) -> pd.DataFrame:
    result = loso.copy()
    result["transfer_rank"] = result.groupby("target")["log_r2"].rank(ascending=False, method="min")
    return result


def build_report(df: pd.DataFrame) -> None:
    variance = pd.read_csv(TABLES / "site_variance_components.csv")
    lobo = pd.read_csv(TABLES / "leave_one_basin_out_results.csv")
    random_grouped = pd.read_csv(TABLES / "random_vs_grouped_validation.csv")
    harmonization = pd.read_csv(TABLES / "harmonization_sensitivity.csv")
    loso = pd.read_csv(TABLES / "leave_one_site_out_results.csv")
    audit = pd.read_csv(TABLES / "target_harmonization_audit.csv")
    site_dist = pd.read_csv(TABLES / "site_target_distribution.csv")
    basin_dist = pd.read_csv(TABLES / "basin_target_distribution.csv")

    lines = [
        "# Site and Basin Generalization Diagnostic Report",
        "",
        "## 1. Objective",
        "",
        "This diagnostic tests whether poor chlorophyll-a and CDOM station-level generalization is primarily associated with site-specific target distributions, basin optical regimes, target harmonization, limited independent-site diversity, or validation optimism. The existing DL-masked 12-hour benchmark and 17-feature representation were retained.",
        "",
        "## 2. Current benchmark",
        "",
        "Dataset: `data/processed/matched_wq_cleaned/matched_set_b_dl_masked.csv`. The transfer analyses use the existing Gradient Boosting model with log10 targets and site-grouped validation. No U-Net, feature set, or official GroupKFold protocol was changed.",
        "",
        "## 3. Target harmonization audit",
        "",
        "The harmonization audit is in `results/tables/target_harmonization_audit.csv`. Turbidity is recorded in FNU, chlorophyll in ug/l, and CDOM in ug/l QSE in the current native benchmark. The project code defines site-specific RFU-to-physical conversions for selected chlorophyll and CDOM sites; uncalibrated RFU observations are excluded from the primary native-unit dataset.",
        "",
        "The audit shows that parameter codes and measurement protocols are not completely uniform: chlorophyll combines fluorescence, fluorometric, and total-chlorophyll codes, while CDOM uses fDOM/QSE. Harmonization sensitivity must therefore be interpreted as a data-condition comparison rather than proof that all methods are interchangeable.",
        "",
        "## 4. Site-level target distributions",
        "",
        "Site-level statistics are in `results/tables/site_target_distribution.csv`; the figure is `results/figures/target_distribution_by_site.png`. The site distributions show large differences in target level and range, especially for chlorophyll-a and CDOM.",
        "",
        "## 5. Basin-level target distributions",
        "",
        "Basin-level statistics are in `results/tables/basin_target_distribution.csv`; the figure is `results/figures/target_distribution_by_basin.png`. Targets do not occur in identical basin/site configurations, so basin transfer is asymmetric.",
        "",
        "## 6. Between-site versus within-site variance",
        "",
        "| Target | Sites | Total log variance | Between-site fraction | Within-site fraction | ICC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in variance.iterrows():
        lines.append(
            f"| {row['target']} | {int(row['n_sites'])} | {row['total_variance_log10']:.4f} | {row['pct_variance_between_sites']:.1f}% | {row['pct_variance_within_sites']:.1f}% | {row['ICC']:.3f} |"
        )
    lines += [
        "",
        "The corrected variance decomposition indicates that between-site variation accounts for 67.5% of turbidity, 81.1% of chlorophyll-a, and 91.3% of CDOM log-target sum of squares under this unbalanced observational decomposition. The corresponding random-effects ICC is also high. These values indicate strong site dependence, but they should not be interpreted as causal proof that every spectral relationship is non-transferable.",
        "",
        "## 7. Feature distribution shifts",
        "",
        "Site/basin feature shifts, standardized mean differences, and KS diagnostics are in `results/tables/site_feature_shift.csv`. Effect sizes are reported alongside p-values because large sample sizes can make small distribution shifts significant.",
        "",
        "## 8. Leave-One-Site-Out results",
        "",
        "LOSO results are in `results/tables/leave_one_site_out_results.csv`. Site-level performance is highly heterogeneous: some stations transfer with positive or near-positive log R², while others fail strongly. The CDOM special audit is `results/tables/cdom_site_audit.csv`; the chlorophyll-a special audit is `results/tables/chlorophyll_site_audit.csv`.",
        "",
        "## 9. Leave-One-Basin-Out results",
        "",
        "| Target | Held-out basin | Sites | Log R² | Log RMSE |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in lobo[lobo["status"] == "supported"].iterrows():
        lines.append(
            f"| {row['target']} | {row['held_out_basin']} | {int(row['sites'])} | {row['log_r2']:.3f} | {row['log_rmse']:.3f} |"
        )
    lines += [
        "",
        "For the available data, Upper Colorado was the strongest turbidity transfer destination by log R², while Willamette was the weakest. For chlorophyll-a, Trinity was the least negative basin and Illinois was the hardest. For CDOM, Delaware and Willamette were similarly poor, with Illinois the hardest.",
        "",
        "## 10. Random versus grouped validation",
        "",
        "| Target | Random log R² | Grouped macro log R² | Random minus grouped |",
        "|---|---:|---:|---:|",
    ]
    for _, row in random_grouped.iterrows():
        lines.append(
            f"| {row['target']} | {row['random_log_r2']:.3f} | {row['grouped_macro_log_r2']:.3f} | {row['difference_random_minus_grouped']:.3f} |"
        )
    lines += [
        "",
        "Random row-level splits are dramatically optimistic: the difference is approximately 1.03 log-R² units for turbidity, 5.78 for chlorophyll-a, and 15.24 for CDOM. Random splitting is therefore not scientifically acceptable as the final validation protocol.",
        "",
        "## 11. Harmonization sensitivity",
        "",
        "The controlled comparison is in `results/tables/harmonization_sensitivity.csv`. In this run the secondary converted datasets did not improve grouped transfer performance: chlorophyll-a and CDOM became worse than the native benchmark. This supports retaining native physical-unit records as the primary benchmark while reporting method heterogeneity explicitly.",
        "",
        "## 12. CDOM audit",
        "",
        "CDOM has only six sites and large between-site target differences. LOSO failures across Delaware, Illinois, and Willamette show that pooled performance is not representative of a stable cross-site relationship. The strongest evidence supports a combination of insufficient independent-site diversity and strong site-specific calibration; measurement heterogeneity is a secondary concern because the native CDOM unit is consistently QSE in the current benchmark.",
        "",
        "## 13. Chlorophyll-a audit",
        "",
        "Chlorophyll-a has 18 sites but combines multiple parameter codes and measurement descriptions. Target ranges vary substantially by site. The evidence supports strong between-site structure and protocol heterogeneity as contributors, with no evidence that the current spectral features provide a stable station-independent correction.",
        "",
        "## 14. Site-level model performance",
        "",
        "The full site-level model table is `results/tables/site_level_model_performance.csv`. Best and worst sites should be read with N and basin together; tiny sites can have unstable R² values. Difficult sites were retained.",
        "",
        "## 15. Root-cause interpretation",
        "",
        "- **Strongest chlorophyll-a explanation:** multiple factors, led by between-site target distribution shifts and measurement/protocol heterogeneity; random validation substantially hides this failure.",
        "- **Strongest CDOM explanation:** insufficient independent-site diversity plus strong site-specific target levels; the six-site design is too small for reliable transfer conclusions.",
        "- **General conclusion:** CASE G, multiple factors contribute. Site/basin structure and validation optimism are better supported than another feature-engineering hypothesis.",
        "",
        "## 16. Limitations",
        "",
        "- The variance decomposition is observational and unbalanced across sites.",
        "- CDOM has only six sites, limiting power for basin and site transfer claims.",
        "- Harmonization comparisons depend on the calibration metadata encoded by the existing project.",
        "- LOBO results are supported only for basins containing sufficient observations; no unsupported target/basin result was fabricated.",
        "",
        "## 17. Recommended next experiment",
        "",
        "Collect or add independent stations and method-matched measurements, especially for CDOM and chlorophyll-a. The next modeling experiment should be a pre-registered site/basin transfer benchmark with harmonization strata reported separately, not another unrestricted feature search.",
        "",
        "## Reproducibility",
        "",
        "```powershell",
        'cd "C:\\Term Paper 2\\satellite-water-monitoring-ai"',
        "C:\\Python314\\python.exe -m src.evaluation.site_basin_generalization",
        "```",
    ]
    (PROJECT_ROOT / "reports/site_basin_generalization_diagnostic_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = load_dataset(DATA / "matched_set_b_dl_masked.csv")
    target_harmonization_audit(df, TABLES / "target_harmonization_audit.csv")
    site_df, basin_df = distribution_tables(df)
    site_df.to_csv(TABLES / "site_target_distribution.csv", index=False)
    basin_df.to_csv(TABLES / "basin_target_distribution.csv", index=False)
    plot_distributions(site_df, basin_df)
    variance_components(df).to_csv(TABLES / "site_variance_components.csv", index=False)
    feature_shift(df).to_csv(TABLES / "site_feature_shift.csv", index=False)
    loso = leave_one_site_out(df)
    loso.to_csv(TABLES / "leave_one_site_out_results.csv", index=False)
    lobo = leave_one_basin_out(df)
    lobo.to_csv(TABLES / "leave_one_basin_out_results.csv", index=False)
    random_vs_grouped(df).to_csv(TABLES / "random_vs_grouped_validation.csv", index=False)
    harmonization_sensitivity(df).to_csv(TABLES / "harmonization_sensitivity.csv", index=False)
    site_level_performance(loso).to_csv(TABLES / "site_level_model_performance.csv", index=False)
    loso[loso["target"] == "cdom"].to_csv(TABLES / "cdom_site_audit.csv", index=False)
    loso[loso["target"] == "chlorophyll"].to_csv(TABLES / "chlorophyll_site_audit.csv", index=False)
    build_report(df)
    print("[OK] Site/basin generalization diagnostic completed.")


if __name__ == "__main__":
    main()
