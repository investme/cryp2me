"""
src/data/dataset.py — cryp2me.ai Phase 5
PyTorch Dataset and fold splitters.

Key fixes vs Phase 4:
  - make_fold() has correct signature matching walk_forward_cv.py
  - No bash commands accidentally appended
  - Augmentation only on technical features (not regime/macro)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple

from src.data.features import FEATURE_COLS, TARGET_REG_COLS, TARGET_CLS_COLS


class CryptoSequenceDataset(Dataset):
    """
    Sequence dataset for crypto OHLCV + feature data.

    Each sample is a (seq_len, n_features) window with targets:
        x      : (seq_len, 35) float32 feature tensor
        y_reg  : (3,) float32  — ret1, ret2, ret3
        y_cls  : (3,) float32  — dir1, dir2, dir3
        price  : scalar float32 — closing price at sequence end
    """

    def __init__(
        self,
        dfs:     List[pd.DataFrame],
        seq_len: int  = 168,
        augment: bool = False,
    ):
        self.seq_len = seq_len
        self.augment = augment
        self.samples: List[Tuple] = []

        for df in dfs:
            # Ensure all feature columns exist
            for col in FEATURE_COLS:
                if col not in df.columns:
                    df[col] = 0.0

            X      = np.clip(df[FEATURE_COLS].values.astype(np.float32), -10, 10)
            y_reg  = df[TARGET_REG_COLS].values.astype(np.float32)
            y_cls  = df[TARGET_CLS_COLS].values.astype(np.float32)
            prices = df["close"].values.astype(np.float32)

            for i in range(seq_len, len(df)):
                if np.any(np.isnan(y_reg[i])) or np.any(np.isnan(y_cls[i])):
                    continue
                if np.any(np.isnan(X[i-seq_len:i])):
                    continue
                self.samples.append((
                    X[i-seq_len:i].copy(),
                    y_reg[i].copy(),
                    y_cls[i].copy(),
                    float(prices[i-1]),
                ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y_reg, y_cls, price = self.samples[idx]
        if self.augment:
            x = x.copy()
            # Only augment technical features (first 22), not macro/regime
            noise = np.random.normal(0, 0.002, (x.shape[0], 22)).astype(np.float32)
            x[:, :22] += noise
            x[:, :22] *= np.random.uniform(0.99, 1.01)
        return (
            torch.from_numpy(x),
            torch.from_numpy(y_reg),
            torch.from_numpy(y_cls),
            torch.tensor(price, dtype=torch.float32),
        )


def make_fold(
    all_features: Dict[str, pd.DataFrame],
    fold_idx:     int,
    n_folds:      int,
    seq_len:      int   = 168,
    val_ratio:    float = 0.1,   # kept for API compatibility, not used
    gap_days:     int   = 3,
    batch_size:   int   = 64,
) -> Tuple[CryptoSequenceDataset, CryptoSequenceDataset]:
    """
    Walk-forward fold splitter.

    Splits each ticker's data chronologically:
        [0 ... train_end] [gap] [val_start ... val_end]

    Args:
        all_features: {ticker: feature_df}
        fold_idx:     0-based fold index (0 = most recent fold)
        n_folds:      total number of folds
        seq_len:      sequence length in hours
        val_ratio:    unused, kept for API compatibility
        gap_days:     days gap between train and val
        batch_size:   unused, kept for API compatibility

    Returns:
        (train_dataset, val_dataset)
    """
    train_dfs: List[pd.DataFrame] = []
    val_dfs:   List[pd.DataFrame] = []

    for ticker, df in all_features.items():
        n       = len(df)
        fold_sz = n // n_folds
        if fold_sz < seq_len + 50:
            continue

        val_end   = n - fold_idx * fold_sz
        val_start = val_end - fold_sz
        gap_hours = gap_days * 24
        train_end = max(0, val_start - gap_hours)

        if train_end < seq_len + 50:
            continue

        train_dfs.append(df.iloc[:train_end].reset_index(drop=True))
        val_dfs.append(df.iloc[val_start:val_end].reset_index(drop=True))

    train_ds = CryptoSequenceDataset(train_dfs, seq_len, augment=True)
    val_ds   = CryptoSequenceDataset(val_dfs,   seq_len, augment=False)
    return train_ds, val_ds


def make_fold_loaders(
    train_dfs:    List[pd.DataFrame],
    val_dfs:      List[pd.DataFrame],
    seq_len:      int            = 168,
    batch_size:   int            = 64,
    feature_cols: Optional[List] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create DataLoaders from pre-split dataframe lists."""
    train_ds = CryptoSequenceDataset(train_dfs, seq_len, augment=True)
    val_ds   = CryptoSequenceDataset(val_dfs,   seq_len, augment=False)
    return (
        DataLoader(train_ds, batch_size=batch_size,   shuffle=True,  num_workers=0, pin_memory=False),
        DataLoader(val_ds,   batch_size=batch_size*2, shuffle=False, num_workers=0, pin_memory=False),
    )
