import os
# pyrefly: ignore [missing-import]
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import torch
from torch.utils.data import DataLoader

from src.segmentation.unet import WaterUNet
from src.segmentation.dataset import get_dataloaders, get_scene_truth_pairs
from src.segmentation.evaluate import compute_binary_metrics, format_metrics_table

def evaluate_saved_model(
    checkpoint_path: str = 'results/checkpoints/water_unet_best.pt',
    results_dir: str = 'results'
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading checkpoint from {checkpoint_path} on {device}...", flush=True)
    
    os.makedirs(os.path.join(results_dir, 'tables'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'figures'), exist_ok=True)
    
    model = WaterUNet(in_channels=6, out_channels=1, base_filters=16).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    print(f"Checkpoint loaded! Best validation IoU during training: {ckpt.get('val_iou', 0.0):.4f}", flush=True)
    
    # Load ALL validation patches (no subsampling)
    _, val_loader = get_dataloaders(
        batch_size=16,
        max_train_patches=10,
        max_val_patches=None
    )
    
    print(f"Evaluating across all {len(val_loader.dataset)} validation patches...", flush=True)
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            val_preds.append(probs)
            val_targets.append(masks.numpy())
            
    y_pred = np.concatenate(val_preds, axis=0)
    y_true = np.concatenate(val_targets, axis=0)
    
    final_metrics = compute_binary_metrics(y_true, y_pred, threshold=0.5)
    print("\nStage 1 Segmentation Final Metrics (Complete Validation Split):", flush=True)
    print(format_metrics_table(final_metrics), flush=True)
    
    # Save CSV and Markdown
    csv_path = os.path.join(results_dir, 'tables', 'stage1_segmentation_metrics.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("metric,value\n")
        for k, v in final_metrics.items():
            f.write(f"{k},{v}\n")
            
    md_path = os.path.join(results_dir, 'tables', 'stage1_segmentation_metrics.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Stage 1: Water Extraction Model Validation Metrics\n\n")
        f.write(format_metrics_table(final_metrics))
        f.write("\n")
    print(f"Metrics saved to {csv_path} and {md_path}", flush=True)
    
    # Generate 4-panel comparison visualization
    generate_comparison_visualization(model, device, results_dir)

def generate_comparison_visualization(model, device, results_dir: str):
    """Generates publication-grade 4-panel comparison figure."""
    val_pairs = get_scene_truth_pairs(
        'data/raw/dset-s2/dset-s2/val_scene',
        'data/raw/dset-s2/dset-s2/val_truth'
    )
    if not val_pairs:
        return
        
    selected_scene = None
    selected_truth = None
    for s, t in val_pairs:
        m = tiff.imread(t)
        if 0.15 < np.mean(m) < 0.60:
            selected_scene, selected_truth = s, t
            break
    if selected_scene is None:
        selected_scene, selected_truth = val_pairs[0]
        
    raw_img = tiff.imread(selected_scene).astype(np.float32)  # (6, H, W)
    raw_mask = tiff.imread(selected_truth).astype(np.uint8)   # (H, W)
    
    if raw_img.ndim == 3 and raw_img.shape[0] != 6 and raw_img.shape[2] == 6:
        raw_img = np.transpose(raw_img, (2, 0, 1))
        
    # RGB composite (B4, B3, B2 -> channels 2, 1, 0)
    rgb = np.stack([raw_img[2], raw_img[1], raw_img[0]], axis=-1)
    p2, p98 = np.percentile(rgb, (2, 98))
    rgb_vis = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)
    
    # Naive NDWI: (B3 - B8) / (B3 + B8)
    b3 = raw_img[1]
    b8 = raw_img[3]
    ndwi = (b3 - b8) / (b3 + b8 + 1e-6)
    ndwi_mask = (ndwi > 0.0).astype(np.uint8)
    
    # DL U-Net Prediction
    img_norm = np.zeros_like(raw_img)
    for c in range(6):
        band = raw_img[c]
        mu = np.mean(band)
        sigma = np.std(band) + 1e-6
        img_norm[c] = (band - mu) / sigma
        
    H, W = raw_mask.shape
    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32
    if pad_h > 0 or pad_w > 0:
        img_input = np.pad(img_norm, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
    else:
        img_input = img_norm
        
    tensor_input = torch.from_numpy(img_input).unsqueeze(0).float().to(device)
    model.eval()
    with torch.no_grad():
        logits = model(tensor_input)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        
    if pad_h > 0 or pad_w > 0:
        prob = prob[:H, :W]
    dl_mask = (prob >= 0.5).astype(np.uint8)
    
    ndwi_metrics = compute_binary_metrics(raw_mask, ndwi_mask)
    dl_metrics = compute_binary_metrics(raw_mask, dl_mask)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=200)
    
    axes[0].imshow(rgb_vis)
    axes[0].set_title("Sentinel-2 RGB Composite\n(B4, B3, B2)", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(raw_mask, cmap='Blues', vmin=0, vmax=1)
    axes[1].set_title(f"Ground Truth Mask\n(Water Coverage: {np.mean(raw_mask)*100:.1f}%)", fontsize=11, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(ndwi_mask, cmap='Blues', vmin=0, vmax=1)
    axes[2].set_title(f"Naive NDWI Threshold (>0)\nIoU: {ndwi_metrics['iou']:.3f} | F1: {ndwi_metrics['f1']:.3f}", fontsize=11, fontweight='bold')
    axes[2].axis('off')

    axes[3].imshow(dl_mask, cmap='Blues', vmin=0, vmax=1)
    axes[3].set_title(f"Water U-Net (Ours)\nIoU: {dl_metrics['iou']:.3f} | F1: {dl_metrics['f1']:.3f}", fontsize=11, fontweight='bold')
    axes[3].axis('off')

    plt.suptitle(f"Surface Water Extraction Comparison: Validation Scene {os.path.basename(selected_scene)}",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    comp_path = os.path.join(results_dir, 'figures', 'water_extraction_comparison.png')
    plt.savefig(comp_path, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison figure to {comp_path}", flush=True)

if __name__ == '__main__':
    evaluate_saved_model()
