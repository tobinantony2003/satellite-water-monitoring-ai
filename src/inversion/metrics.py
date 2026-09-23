import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard water quality retrieval metrics:
    - R²
    - RMSE
    - MAE
    - MAPE (%)
    - Bias (MBE)
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    # Clip negative predictions to zero for physical water parameters
    y_pred = np.clip(y_pred, 0, None)
    
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE (safe division avoiding divide by zero)
    mask = (y_true > 1e-4)
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
    else:
        mape = 0.0
        
    bias = np.mean(y_pred - y_true)
    
    return {
        'r2': float(r2),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'bias': float(bias)
    }
