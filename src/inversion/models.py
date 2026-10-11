import os
import numpy as np

# Suppress joblib loky wmic CPU detection warning on modern Windows 11
if 'LOKY_MAX_CPU_COUNT' not in os.environ:
    os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count() or 4)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from typing import Dict, Any, Tuple

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

class SpectralMLP(nn.Module):
    """Deep Multi-Layer Perceptron for spectral water quality inversion."""
    def __init__(self, in_features: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

class PyTorchMLPRegressor:
    """Scikit-Learn style wrapper for PyTorch SpectralMLP with gradient clipping and prediction bounding."""
    def __init__(self, in_features: int, epochs: int = 35, lr: float = 1e-3, batch_size: int = 256):
        self.in_features = in_features
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SpectralMLP(in_features).to(self.device)
        self.y_min = -2.0
        self.y_max = 3.5

    def fit(self, X: np.ndarray, y: np.ndarray):
        # A new estimator is created for each CV fold by the caller. Rebuild the
        # network here as an additional guard if fit() is called more than once.
        self.model = SpectralMLP(self.in_features).to(self.device)
        self.y_min = float(np.min(y))
        self.y_max = float(np.max(y))
        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-3)
        
        self.model.train()
        for epoch in range(self.epochs):
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                pred = self.model(bx)
                loss = criterion(pred, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            tx = torch.from_numpy(X).float().to(self.device)
            preds = self.model(tx).cpu().numpy()
            # Bound log predictions strictly within training target range to prevent exponential blowup on held-out blocks
            preds = np.clip(preds, self.y_min, self.y_max)
        return preds

def get_inversion_models(in_features: int) -> Dict[str, Any]:
    """Returns the suite of 4 water quality inversion model architectures."""
    models = {
        'Empirical_Baseline': Ridge(alpha=1.0),
        'Random_Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=3,
            n_jobs=4,
            random_state=42
        ),
        'Gradient_Boosting': (
            xgb.XGBRegressor(
                n_estimators=120,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.8,
                random_state=42,
                n_jobs=4
            ) if HAS_XGBOOST else
            HistGradientBoostingRegressor(
                max_iter=120,
                max_depth=6,
                learning_rate=0.08,
                random_state=42
            )
        ),
        'Deep_Spectral_MLP': PyTorchMLPRegressor(in_features=in_features, epochs=30)
    }
    return models
