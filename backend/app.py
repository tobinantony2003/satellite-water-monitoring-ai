from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import tifffile as tiff
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sklearn.preprocessing import StandardScaler

from src.inversion.models import get_inversion_models
from src.inversion.train_inversion import FEATURE_COLS
from src.segmentation.unet import WaterUNet

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_PATH = RESULTS_DIR / "checkpoints" / "water_unet_best.pt"
SEGMENTATION_METRICS_PATH = RESULTS_DIR / "tables" / "stage1_segmentation_metrics.csv"
FINAL_SUMMARY_PATH = PROJECT_ROOT / "results" / "FINAL_RESULTS" / "final_metrics_summary.csv"
WQ_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "matched_wq_cleaned" / "matched_set_b_dl_masked.csv"

app = FastAPI(title="Satellite Water Monitoring AI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_CACHE: Dict[str, dict] = {}


def _read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _clean_jsonable_frame(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []
    records: list[dict] = []
    for row in frame.to_dict(orient="records"):
        cleaned = {}
        for key, value in row.items():
            if pd.isna(value) or (isinstance(value, float) and (np.isinf(value) or np.isnan(value))):
                cleaned[key] = None
            else:
                cleaned[key] = value
        records.append(cleaned)
    return records


def read_project_metrics() -> Dict[str, object]:
    summary = _read_csv_optional(FINAL_SUMMARY_PATH)
    stage1 = _read_csv_optional(SEGMENTATION_METRICS_PATH)
    stage3 = _read_csv_optional(PROJECT_ROOT / "results" / "tables" / "stage3_consolidated_inversion_benchmarks.csv")
    return {
        "project_name": "Satellite Water Monitoring AI",
        "project_root": str(PROJECT_ROOT),
        "segmentation_metrics": _clean_jsonable_frame(stage1),
        "canonical_metrics": _clean_jsonable_frame(summary),
        "latest_stage3_metrics": _clean_jsonable_frame(stage3),
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "project": "Satellite Water Monitoring AI",
        "backend": "FastAPI",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_availability": {
            "segmentation": CHECKPOINT_PATH.exists(),
            "water_quality_dataset": WQ_DATASET_PATH.exists(),
            "final_metrics_summary": FINAL_SUMMARY_PATH.exists(),
        },
    }


@app.get("/api/project-info")
def project_info():
    return read_project_metrics()


@app.get("/api/results")
def results_index():
    files: list[dict] = []
    for root, _, filenames in os.walk(PROJECT_ROOT):
        for name in sorted(filenames):
            if name.lower().endswith((".csv", ".png", ".md", ".tif", ".tiff")):
                full = Path(root) / name
                files.append({"name": name, "path": str(full.relative_to(PROJECT_ROOT))})
    return {"files": files[:200]}


@app.post("/api/segmentation")
async def segmentation_endpoint(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file was uploaded.")
    if not file.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a 6-band GeoTIFF (.tif or .tiff).")

    temp_path = PROJECT_ROOT / "tmp_uploads" / f"segmentation_{abs(hash(file.filename + str(datetime.now().timestamp())))}.tif"
    temp_path.parent.mkdir(exist_ok=True)
    temp_path.write_bytes(await file.read())

    try:
        image = tiff.imread(temp_path)
        image = np.asarray(image, dtype=np.float32)
        if image.ndim == 2:
            raise ValueError("Expected a 6-band image stack. 2D arrays are not supported.")
        if image.ndim == 3 and image.shape[-1] == 6:
            image = np.moveaxis(image, -1, 0)
        elif image.ndim == 3 and image.shape[0] != 6:
            raise ValueError("Expected a 6-band image with shape (6, H, W) or (H, W, 6).")
        if image.shape[0] != 6:
            raise ValueError("Water segmentation requires exactly 6 spectral bands.")

        normalized = np.zeros_like(image)
        for band_idx in range(image.shape[0]):
            band = image[band_idx]
            mean = float(np.mean(band))
            std = float(np.std(band)) + 1e-6
            normalized[band_idx] = (band - mean) / std

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = WaterUNet(in_channels=6, out_channels=1, base_filters=16).to(device)
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"Missing segmentation checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(str(CHECKPOINT_PATH), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        with torch.no_grad():
            inp = torch.from_numpy(normalized).unsqueeze(0).float().to(device)
            logits = model(inp)
            prob = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
        mask = (prob >= 0.5).astype(np.uint8)
        water_fraction = float(np.mean(mask))
        result = {
            "water_fraction": water_fraction,
            "mask_shape": [int(mask.shape[0]), int(mask.shape[1])],
            "mask_preview": mask.astype(int).tolist()[:80][:80],
            "band_summary": {
                "B02": float(np.mean(image[0])),
                "B03": float(np.mean(image[1])),
                "B04": float(np.mean(image[2])),
                "B08": float(np.mean(image[3])),
                "B11": float(np.mean(image[4])),
                "B12": float(np.mean(image[5])),
            },
        }
        return {"status": "ok", "result": result, **result}
    except Exception as exc:  # pragma: no cover - explicit API error path
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


@app.post("/api/water-quality")
async def water_quality_endpoint(payload: dict):
    try:
        target = str(payload.get("target", "")).lower().strip()
        features = payload.get("features", {})
        if target not in {"turbidity", "chlorophyll", "cdom"}:
            raise ValueError("Target must be one of: turbidity, chlorophyll, cdom")
        if not features:
            raise ValueError("A feature dictionary is required for water-quality prediction.")

        target_key = target if target in {"turbidity", "chlorophyll", "cdom"} else target
        if target_key in MODEL_CACHE:
            model_bundle = MODEL_CACHE[target_key]
        else:
            if not WQ_DATASET_PATH.exists():
                raise FileNotFoundError(f"Missing dataset: {WQ_DATASET_PATH}")
            df = pd.read_csv(WQ_DATASET_PATH, low_memory=False)
            df_target = df[df["target_type"] == target_key].copy()
            if df_target.empty:
                raise ValueError(f"No data available for target '{target_key}'.")
            X = df_target[FEATURE_COLS].copy()
            for col in FEATURE_COLS:
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
            y_raw = pd.to_numeric(df_target["MeasurementValue"], errors="coerce").fillna(1.0).to_numpy()
            y_log = np.log10(np.clip(y_raw, 1e-4, None))

            if target_key == "chlorophyll":
                model = __import__("sklearn.linear_model", fromlist=["Ridge"]).Ridge(alpha=1.0)
                feat = ["ndci"]
                model.fit(X[feat].values, y_log)
                model_bundle = {"model": model, "feature_names": feat, "model_name": "Empirical_Baseline"}
            else:
                model = __import__("sklearn.ensemble", fromlist=["GradientBoostingRegressor"]).GradientBoostingRegressor(random_state=42)
                scaler = StandardScaler()
                scaled = scaler.fit_transform(X.values)
                model.fit(scaled, y_log)
                model_bundle = {"model": model, "feature_names": FEATURE_COLS, "model_name": "Gradient_Boosting", "scaler": scaler}
            MODEL_CACHE[target_key] = model_bundle

        model_bundle = MODEL_CACHE[target_key]
        feature_names = model_bundle["feature_names"]
        missing = [name for name in feature_names if name not in features]
        if missing:
            raise ValueError(f"Missing required features for {target_key}: {missing}")

        if model_bundle["model_name"] == "Empirical_Baseline":
            value = float(features["ndci"])
            log_pred = float(model_bundle["model"].predict(np.array([[value]], dtype=np.float32))[0])
        else:
            row = np.array([[float(features.get(name, 0.0)) for name in FEATURE_COLS]], dtype=np.float32)
            scaled = model_bundle["scaler"].transform(row)
            log_pred = float(model_bundle["model"].predict(scaled)[0])

        raw_pred = float(10.0 ** np.clip(log_pred, -3.0, 4.0))
        return {
            "status": "ok",
            "prediction": {
                "target": target_key,
                "model_name": model_bundle["model_name"],
                "log_prediction": log_pred,
                "raw_prediction": raw_pred,
                "unit": {"turbidity": "FNU", "chlorophyll": "ug/L", "cdom": "ug/L"}[target_key],
            },
        }
    except Exception as exc:  # pragma: no cover - explicit API error path
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/temporal-comparison")
async def temporal_comparison(files: list[UploadFile] = File(...)):
    if len(files) != 2:
        raise HTTPException(status_code=400, detail="Please upload exactly two 6-band TIFF files for temporal comparison.")

    results = []
    for upl in files:
        if not upl.filename or not upl.filename.lower().endswith((".tif", ".tiff")):
            raise HTTPException(status_code=400, detail="Temporal comparison requires two valid 6-band GeoTIFF images.")
        temp_path = PROJECT_ROOT / "tmp_uploads" / f"temporal_{abs(hash(upl.filename + str(datetime.now().timestamp())))}.tif"
        temp_path.parent.mkdir(exist_ok=True)
        temp_path.write_bytes(await upl.read())
        try:
            image = np.asarray(tiff.imread(temp_path), dtype=np.float32)
            if image.ndim == 3 and image.shape[-1] == 6:
                image = np.moveaxis(image, -1, 0)
            if image.shape[0] != 6:
                raise ValueError("Temporal comparison requires 6-band imagery.")
            result = await segmentation_endpoint(upl)
            results.append(result)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    source_a = results[0].get("result", {})
    source_b = results[1].get("result", {})
    water_a = float(source_a.get("water_fraction", 0.0))
    water_b = float(source_b.get("water_fraction", 0.0))
    delta = water_b - water_a
    relative_change = 0.0 if abs(water_a) < 1e-9 else 100.0 * delta / water_a
    return {
        "status": "ok",
        "water_extent": {
            "scene_1_fraction": water_a,
            "scene_2_fraction": water_b,
            "delta_fraction": delta,
            "relative_change_percent": relative_change,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8001, reload=False)
