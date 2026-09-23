import os
import glob
import random
import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional

def get_scene_truth_pairs(scene_dir: str, truth_dir: str) -> List[Tuple[str, str]]:
    """Matches Sentinel-2 6-band scene files with corresponding binary ground-truth masks."""
    scenes = sorted(glob.glob(os.path.join(scene_dir, '*.tif')))
    pairs = []
    for s in scenes:
        base = os.path.basename(s)
        expected_t = base.replace('_6Bands_', '_').replace('.tif', '_Truth.tif')
        t_path = os.path.join(truth_dir, expected_t)
        if os.path.exists(t_path):
            pairs.append((s, t_path))
    return pairs

class Sentinel2WaterPatchDataset(Dataset):
    """
    High-efficiency PyTorch Dataset for Sentinel-2 6-band surface water patches.
    Scenes are stored normalized in memory; patches are sliced on-the-fly via coordinate indices.
    Augmentations: Random horizontal/vertical flip, random 90-degree rotations only.
    No color or radiometric augmentation to preserve spectral integrity.
    """
    def __init__(
        self,
        scene_dir: str,
        truth_dir: str,
        patch_size: int = 256,
        stride: int = 256,
        augment: bool = False,
        max_patches_per_epoch: Optional[int] = None
    ):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.augment = augment
        self.max_patches_per_epoch = max_patches_per_epoch
        self.pairs = get_scene_truth_pairs(scene_dir, truth_dir)
        self.scenes = []  # List of (img_norm: (6, H, W) float32, mask: (H, W) uint8)
        self.patch_info = []  # List of (scene_idx, y, x, water_fraction)
        
        print(f"Loading {len(self.pairs)} scene-truth pairs from {scene_dir}...", flush=True)
        for scene_idx, (scene_path, truth_path) in enumerate(self.pairs):
            raw_img = tiff.imread(scene_path).astype(np.float32)
            raw_mask = tiff.imread(truth_path).astype(np.uint8)
            
            if raw_img.ndim == 3 and raw_img.shape[0] != 6 and raw_img.shape[2] == 6:
                raw_img = np.transpose(raw_img, (2, 0, 1))
                
            # Per-scene standardization per band
            C, H, W = raw_img.shape
            img_norm = np.zeros_like(raw_img)
            for c in range(6):
                band = raw_img[c]
                mu = np.mean(band)
                sigma = np.std(band) + 1e-6
                img_norm[c] = (band - mu) / sigma
                
            self.scenes.append((img_norm, raw_mask))
            
            # Generate tile grid coordinates
            y_steps = list(range(0, H - patch_size + 1, stride))
            if y_steps[-1] != H - patch_size:
                y_steps.append(H - patch_size)
                
            x_steps = list(range(0, W - patch_size + 1, stride))
            if x_steps[-1] != W - patch_size:
                x_steps.append(W - patch_size)
                
            for y in y_steps:
                for x in x_steps:
                    patch_m = raw_mask[y : y + patch_size, x : x + patch_size]
                    wf = float(np.mean(patch_m))
                    self.patch_info.append((scene_idx, y, x, wf))
                    
        self.active_indices = list(range(len(self.patch_info)))
        self.resample_active_indices()
        print(f"Dataset initialized: {len(self.scenes)} scenes, {len(self.patch_info)} total patches ({len(self.active_indices)} active).", flush=True)

    def resample_active_indices(self):
        """Optionally subsamples a balanced set of patches for efficient training epochs."""
        if self.max_patches_per_epoch is None or self.max_patches_per_epoch >= len(self.patch_info):
            self.active_indices = list(range(len(self.patch_info)))
            return

        # Separate into water, edge, and land patches
        edge_idx = [i for i, info in enumerate(self.patch_info) if 0.05 <= info[3] <= 0.95]
        water_idx = [i for i, info in enumerate(self.patch_info) if info[3] > 0.95]
        land_idx = [i for i, info in enumerate(self.patch_info) if info[3] < 0.05]

        # Allocate balanced proportions: 50% edge/mixed, 25% water, 25% land
        n = self.max_patches_per_epoch
        n_edge = min(len(edge_idx), int(n * 0.50))
        n_water = min(len(water_idx), int(n * 0.25))
        n_land = min(len(land_idx), n - n_edge - n_water)

        sampled = (
            random.sample(edge_idx, n_edge) +
            random.sample(water_idx, n_water) +
            random.sample(land_idx, n_land)
        )
        random.shuffle(sampled)
        self.active_indices = sampled

    def __len__(self) -> int:
        return len(self.active_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real_idx = self.active_indices[idx]
        scene_idx, y, x, _ = self.patch_info[real_idx]
        img_norm, raw_mask = self.scenes[scene_idx]
        
        patch_img = img_norm[:, y : y + self.patch_size, x : x + self.patch_size].copy()
        patch_mask = raw_mask[y : y + self.patch_size, x : x + self.patch_size].copy()
        
        if self.augment:
            if random.random() > 0.5:
                patch_img = np.flip(patch_img, axis=2)
                patch_mask = np.flip(patch_mask, axis=1)
            if random.random() > 0.5:
                patch_img = np.flip(patch_img, axis=1)
                patch_mask = np.flip(patch_mask, axis=0)
            k = random.randint(0, 3)
            if k > 0:
                patch_img = np.rot90(patch_img, k=k, axes=(1, 2))
                patch_mask = np.rot90(patch_mask, k=k, axes=(0, 1))
                
        img_tensor = torch.from_numpy(np.ascontiguousarray(patch_img)).float()
        mask_tensor = torch.from_numpy(np.ascontiguousarray(patch_mask)).float().unsqueeze(0)
        
        return img_tensor, mask_tensor

def get_dataloaders(
    tra_scene_dir: str = 'data/raw/dset-s2/dset-s2/tra_scene',
    tra_truth_dir: str = 'data/raw/dset-s2/dset-s2/tra_truth',
    val_scene_dir: str = 'data/raw/dset-s2/dset-s2/val_scene',
    val_truth_dir: str = 'data/raw/dset-s2/dset-s2/val_truth',
    patch_size: int = 256,
    tra_stride: int = 256,
    val_stride: int = 256,
    batch_size: int = 16,
    max_train_patches: Optional[int] = 320,
    max_val_patches: Optional[int] = 160,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = Sentinel2WaterPatchDataset(
        tra_scene_dir,
        tra_truth_dir,
        patch_size=patch_size,
        stride=tra_stride,
        augment=True,
        max_patches_per_epoch=max_train_patches
    )
    val_dataset = Sentinel2WaterPatchDataset(
        val_scene_dir,
        val_truth_dir,
        patch_size=patch_size,
        stride=val_stride,
        augment=False,
        max_patches_per_epoch=max_val_patches
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader
