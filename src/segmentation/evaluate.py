import numpy as np
import torch
from typing import Dict, Tuple

def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Computes standard remote sensing binary segmentation metrics:
    - IoU (Intersection over Union / Jaccard Index)
    - Precision
    - Recall
    - F1-score (Dice coefficient)
    - Overall Accuracy (OA)
    - Cohen's Kappa (Kappa)
    
    y_true: ground truth binary mask {0, 1}
    y_pred: predicted probability in [0, 1] or binary {0, 1}
    """
    if y_pred.dtype == np.float32 or y_pred.dtype == np.float64:
        pred_bin = (y_pred >= threshold).astype(np.uint8)
    else:
        pred_bin = (y_pred > 0).astype(np.uint8)
        
    true_bin = (y_true > 0).astype(np.uint8)
    
    # Flatten
    y_t = true_bin.ravel()
    y_p = pred_bin.ravel()
    
    total = len(y_t)
    if total == 0:
        return {"iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "oa": 0.0, "kappa": 0.0}
        
    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))
    
    # Metrics
    iou = tp / (tp + fp + fn + 1e-12)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    oa = (tp + tn) / total
    
    # Cohen's Kappa
    # Po = observed agreement (oa)
    # Pe = expected chance agreement
    p_actual_pos = (tp + fn) / total
    p_actual_neg = (fp + tn) / total
    p_pred_pos = (tp + fp) / total
    p_pred_neg = (fn + tn) / total
    p_e = (p_actual_pos * p_pred_pos) + (p_actual_neg * p_pred_neg)
    
    if abs(1.0 - p_e) < 1e-12:
        kappa = 1.0 if abs(oa - p_e) < 1e-12 else 0.0
    else:
        kappa = (oa - p_e) / (1.0 - p_e)
        
    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "oa": float(oa),
        "kappa": float(kappa),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }

def format_metrics_table(metrics_dict: Dict[str, float]) -> str:
    """Formats metrics dictionary into a clean markdown table string."""
    headers = ["Metric", "Value", "Percentage"]
    rows = [
        ["Intersection over Union (IoU)", f"{metrics_dict['iou']:.4f}", f"{metrics_dict['iou']*100:.2f}%"],
        ["Precision", f"{metrics_dict['precision']:.4f}", f"{metrics_dict['precision']*100:.2f}%"],
        ["Recall", f"{metrics_dict['recall']:.4f}", f"{metrics_dict['recall']*100:.2f}%"],
        ["F1-Score (Dice)", f"{metrics_dict['f1']:.4f}", f"{metrics_dict['f1']*100:.2f}%"],
        ["Overall Accuracy (OA)", f"{metrics_dict['oa']:.4f}", f"{metrics_dict['oa']*100:.2f}%"],
        ["Cohen's Kappa (Kappa)", f"{metrics_dict['kappa']:.4f}", f"{metrics_dict['kappa']*100:.2f}%"],
    ]
    
    col_w = [30, 10, 12]
    line = "+-" + "-+-".join(["-" * w for w in col_w]) + "-+"
    header_line = "| " + " | ".join([h.ljust(col_w[i]) for i, h in enumerate(headers)]) + " |"
    
    md_lines = ["| Metric | Value | Percentage |", "| :--- | :--- | :--- |"]
    for row in rows:
        md_lines.append(f"| {row[0]} | {row[1]} | {row[2]} |")
        
    return "\n".join(md_lines)
