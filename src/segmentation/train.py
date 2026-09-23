import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.segmentation.unet import WaterUNet
from src.segmentation.losses import BCEDiceLoss
from src.segmentation.dataset import get_dataloaders, get_scene_truth_pairs
from src.segmentation.evaluate import compute_binary_metrics, format_metrics_table

def train_segmentation(
    epochs: int = 8,
    batch_size: int = 16,
    lr: float = 1e-3,
    patch_size: int = 256,
    max_train_patches: int = 320,
    max_val_patches: int = 160,
    save_checkpoint_dir: str = 'results/checkpoints',
    results_dir: str = 'results'
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using compute device: {device}", flush=True)
    
    os.makedirs(save_checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'tables'), exist_ok=True)

    print("\n--- Initializing DataLoaders ---", flush=True)
    train_loader, val_loader = get_dataloaders(
        patch_size=patch_size,
        batch_size=batch_size,
        max_train_patches=max_train_patches,
        max_val_patches=max_val_patches
    )

    model = WaterUNet(in_channels=6, out_channels=1, base_filters=16).to(device)
    criterion = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_iou = 0.0
    best_checkpoint_path = os.path.join(save_checkpoint_dir, 'water_unet_best.pt')

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_iou': [],
        'val_f1': []
    }

    print("\n--- Starting Training ---", flush=True)
    start_total_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Resample active indices for fresh balanced coverage
        train_loader.dataset.resample_active_indices()
        
        model.train()
        running_loss = 0.0
        
        for batch_idx, (imgs, masks) in enumerate(train_loader):
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        scheduler.step()

        # Validation
        model.eval()
        val_running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)

                logits = model(imgs)
                loss = criterion(logits, masks)
                val_running_loss += loss.item() * imgs.size(0)

                probs = torch.sigmoid(logits).cpu().numpy()
                targets = masks.cpu().numpy()

                all_preds.append(probs)
                all_targets.append(targets)

        val_loss = val_running_loss / len(val_loader.dataset)
        all_preds_cat = np.concatenate(all_preds, axis=0)
        all_targets_cat = np.concatenate(all_targets, axis=0)

        epoch_metrics = compute_binary_metrics(all_targets_cat, all_preds_cat, threshold=0.5)
        val_iou = epoch_metrics['iou']
        val_f1 = epoch_metrics['f1']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['val_f1'].append(val_f1)

        epoch_duration = time.time() - epoch_start
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_duration:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val IoU: {val_iou:.4f} | Val F1: {val_f1:.4f} | Val OA: {epoch_metrics['oa']:.4f}", flush=True)

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_f1': val_f1,
                'metrics': epoch_metrics
            }, best_checkpoint_path)
            print(f"  --> Saved new best checkpoint to {best_checkpoint_path} (IoU: {best_val_iou:.4f})", flush=True)

    total_duration = time.time() - start_total_time
    print(f"\nTraining completed in {total_duration / 60:.2f} minutes. Best Val IoU: {best_val_iou:.4f}", flush=True)

    # Load best checkpoint for final evaluation
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Final Evaluation across ALL validation patches
    print("\n--- Evaluating Best Model on Complete Validation Split ---", flush=True)
    val_loader.dataset.max_patches_per_epoch = None
    val_loader.dataset.resample_active_indices()
    
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
    print("\nStage 1 Segmentation Final Metrics:", flush=True)
    print(format_metrics_table(final_metrics), flush=True)

    # Save metrics to CSV and Markdown
    csv_path = os.path.join(results_dir, 'tables', 'stage1_segmentation_metrics.csv')
    with open(csv_path, 'w') as f:
        f.write("metric,value\n")
        for k, v in final_metrics.items():
            f.write(f"{k},{v}\n")

    md_path = os.path.join(results_dir, 'tables', 'stage1_segmentation_metrics.md')
    with open(md_path, 'w') as f:
        f.write("# Stage 1: Water Extraction Model Validation Metrics\n\n")
        f.write(format_metrics_table(final_metrics))
        f.write("\n")

    # Plot training curves
    plt.figure(figsize=(10, 4), dpi=150)
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history['train_loss'], label='Train Loss', color='#1f77b4', lw=2)
    plt.plot(range(1, epochs + 1), history['val_loss'], label='Val Loss', color='#ff7f0e', lw=2)
    plt.title('Loss Curves (BCE + Dice)', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history['val_iou'], label='Val IoU', color='#2ca02c', lw=2)
    plt.plot(range(1, epochs + 1), history['val_f1'], label='Val F1 (Dice)', color='#d62728', lw=2)
    plt.title('Validation Accuracy Metrics', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    curve_path = os.path.join(results_dir, 'figures', 'stage1_training_history.png')
    plt.savefig(curve_path)
    plt.close()
    print(f"Saved training history plot to {curve_path}", flush=True)

    # Generate Full-Scene Comparison Figure
    generate_comparison_visualization(model, device, results_dir)

    return final_metrics

def generate_comparison_visualization(model, device, results_dir: str):
    """Generates a publication-grade 4-panel comparison on a validation scene."""
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
        if 0.10 < np.mean(m) < 0.60:
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

    # 4-Panel Visualization
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
    train_segmentation()
