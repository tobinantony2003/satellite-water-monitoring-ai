"""
Module: splits.py
Provides centralized, deterministic train/test site splitting utilities.
Prevents non-deterministic partitioning caused by Python's set-hash randomization.
All analysis scripts MUST import and use `get_test_sites()` rather than recomputing splits locally.
"""

import json
from pathlib import Path
from typing import List, Set, Union
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CANONICAL_SPLITS_DIR = PROJECT_ROOT / 'results' / 'canonical_splits'

def get_test_sites(target_type: str = 'turbidity', return_type: str = 'set') -> Union[Set[int], List[int]]:
    """
    Loads the canonical, persisted list of held-out test monitoring site IDs.

    Args:
        target_type: Target parameter ('turbidity', 'chlorophyll', 'cdom').
        return_type: 'set' or 'list'.

    Returns:
        Canonical set or list of integer USGS monitoring site IDs.
    """
    target = target_type.lower().replace('-', '_')
    split_file = CANONICAL_SPLITS_DIR / f'{target}_test_sites.json'

    if not split_file.exists():
        # Fallback for chlorophyll variants
        if 'chlorophyll' in target:
            split_file = CANONICAL_SPLITS_DIR / 'chlorophyll_test_sites.json'
        elif 'cdom' in target:
            split_file = CANONICAL_SPLITS_DIR / 'cdom_test_sites.json'
        elif 'turbidity' in target:
            split_file = CANONICAL_SPLITS_DIR / 'turbidity_test_sites.json'

    if not split_file.exists():
        raise FileNotFoundError(f"Canonical split file not found for target '{target_type}': {split_file}")

    with open(split_file, 'r') as f:
        data = json.load(f)

    sites = [int(s) for s in data['canonical_test_sites']]
    if return_type == 'set':
        return set(sites)
    elif return_type == 'list':
        return sorted(sites)
    else:
        raise ValueError(f"Unknown return_type: {return_type}")

def split_by_canonical_sites(
    df: pd.DataFrame,
    target_type: str = 'turbidity',
    site_col: str = 'site_no'
) -> (pd.DataFrame, pd.DataFrame):
    """
    Splits a DataFrame into (train_df, test_df) using the canonical held-out test sites.

    Ensures robust type matching whether site_no is stored as int, float, or string.
    """
    test_sites = get_test_sites(target_type, return_type='set')
    
    # Cast site_col to int for matching
    # Handle possible string / float formatting
    site_series = df[site_col].astype(str).str.split('.').str[0]
    site_series_int = pd.to_numeric(site_series, errors='coerce')
    
    test_mask = site_series_int.isin(test_sites)
    
    test_df = df[test_mask].copy()
    train_df = df[~test_mask].copy()
    
    return train_df, test_df
