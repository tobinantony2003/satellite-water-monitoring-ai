import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from typing import Generator, Tuple

def get_spatial_block_cv(
    df: pd.DataFrame,
    group_col: str = 'site_no',
    n_splits: int = 5,
    random_state: int = 42
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generates spatial block cross-validation train/test index splits based on USGS monitoring site groups.
    Strictly ensures that all observations from a given monitoring gauge belong exclusively to either
    the training fold or test fold, eliminating spatial-autocorrelation leakage.
    """
    groups = df[group_col].values
    gkf = GroupKFold(n_splits=n_splits)
    
    for train_idx, val_idx in gkf.split(df, groups=groups):
        # Verification assertion: zero site overlap between train and val
        train_sites = set(groups[train_idx])
        val_sites = set(groups[val_idx])
        assert len(train_sites.intersection(val_sites)) == 0, "Spatial leakage detected between folds!"
        yield train_idx, val_idx
