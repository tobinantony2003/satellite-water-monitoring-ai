import os
import glob
import numpy as np
import tifffile as tiff
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, Dict

class DeepSpectralWaterClassifier(nn.Module):
    """
    Deep neural network for classifying 6-band Sentinel-2 pixel spectra into water vs. non-water.
    Input: 6 spectral bands (B2, B3, B4, B8, B11, B12) normalized.
    Architecture: 4-layer MLP with BatchNorm, LeakyReLU, and Dropout.
    """
    def __init__(self, in_features: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.1),

            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def sample_pixels_from_dset(
    scene_dir: str,
    truth_dir: str,
    pixels_per_scene: int = 5000,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Samples balanced water and non-water pixel spectra from scenes."""
    np.random.seed(random_seed)
    scenes = sorted(glob.glob(os.path.join(scene_dir, '*.tif')))
    
    all_X = []
    all_y = []
    
    for s in scenes:
        base = os.path.basename(s)
        expected_t = base.replace('_6Bands_', '_').replace('.tif', '_Truth.tif')
        t_path = os.path.join(truth_dir, expected_t)
        if not os.path.exists(t_path):
            continue
            
        img = tiff.imread(s).astype(np.float32)  # (6, H, W)
        mask = tiff.imread(t_path).astype(np.uint8)  # (H, W)
        
        if img.ndim == 3 and img.shape[0] != 6 and img.shape[2] == 6:
            img = np.transpose(img, (2, 0, 1))
            
        # Convert to (H*W, 6)
        C, H, W = img.shape
        flat_img = img.reshape(C, -1).T
        flat_mask = mask.reshape(-1)
        
        water_idx = np.where(flat_mask == 1)[0]
        land_idx = np.where(flat_mask == 0)[0]
        
        n_sample = min(len(water_idx), len(land_idx), pixels_per_scene // 2)
        if n_sample > 0:
            sel_w = np.random.choice(water_idx, n_sample, replace=False)
            sel_l = np.random.choice(land_idx, n_sample, replace=False)
            sel = np.concatenate([sel_w, sel_l])
            all_X.append(flat_img[sel])
            all_y.append(flat_mask[sel])
            
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    return X, y

def train_spectral_water_classifier(
    dset_tra_scene: str = 'data/raw/dset-s2/dset-s2/tra_scene',
    dset_tra_truth: str = 'data/raw/dset-s2/dset-s2/tra_truth',
    checkpoint_path: str = 'results/checkpoints/spectral_water_model.pt',
    epochs: int = 10,
    batch_size: int = 512
) -> DeepSpectralWaterClassifier:
    """Trains and saves the deep spectral water extraction classifier."""
    print("Sampling pixel spectra from dset-s2 training split...", flush=True)
    X_train, y_train = sample_pixels_from_dset(dset_tra_scene, dset_tra_truth, pixels_per_scene=4000)
    
    # Standardize features using robust statistics (median and IQR or mean/std)
    mean = np.mean(X_train, axis=0, keepdims=True)
    std = np.std(X_train, axis=0, keepdims=True) + 1e-6
    
    X_norm = (X_train - mean) / std
    
    dataset = TensorDataset(torch.from_numpy(X_norm).float(), torch.from_numpy(y_train).float().unsqueeze(1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepSpectralWaterClassifier(in_features=6).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    
    print(f"Training DeepSpectralWaterClassifier on {len(X_train)} spectra across {epochs} epochs...", flush=True)
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
            
        epoch_loss = running_loss / len(X_train)
        print(f"Spectral Model Epoch [{epoch:02d}/{epochs:02d}] Loss: {epoch_loss:.4f}", flush=True)
        
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'mean': mean,
        'std': std
    }, checkpoint_path)
    print(f"Saved spectral water classifier checkpoint to {checkpoint_path}", flush=True)
    return model
