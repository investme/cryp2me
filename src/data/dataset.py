"""
src/data/dataset.py
───────────────────
PyTorch Dataset that creates sliding-window sequences from feature DataFrames.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple


FEATURE_COLS = [
    "open_pct", "high_pct", "low_pct", "close_pct",
    "volume_norm", "volume_ratio",
    "ema10_dist", "ema20_dist", "ema34_dist",
    "rsi14", "macd", "macd_signal", "macd_hist",
    "atr14_norm", "bb_width",
    "adx14", "plus_di", "minus_di",
]

TARGET_REG_COLS = ["ret1", "ret2", "ret3"]   # regression targets
TARGET_CLS_COLS = ["dir1", "dir2", "dir3"]   # classification targets


class CryptoSequenceDataset(Dataset):
    """
    Sliding window dataset.

    Each sample:
      x  — (seq_len, n_features)   float32
      y_reg — (3,)   future % returns
      y_cls — (3,)   future directions (0/1)
      price — scalar   closing price at sequence end (for inverse transform)
    """

    def __init__(
        self,
        dfs: List[pd.DataFrame],
        seq_len: int = 60,
        augment: bool = False,
    ):
        self.seq_len  = seq_len
        self.augment  = augment
        self.samples: List[Tuple] = []

        for df in dfs:
            X   = df[FEATURE_COLS].values.astype(np.float32)
            y_r = df[TARGET_REG_COLS].values.astype(np.float32)
            y_c = df[TARGET_CLS_COLS].values.astype(np.float32)
            prices = df["close"].values.astype(np.float32)

            # Clip extreme values to reduce outlier impact
            X = np.clip(X, -10, 10)

            for i in range(seq_len, len(df)):
                # Labels are at position i (predict from window ending at i-1)
                if np.any(np.isnan(y_r[i])) or np.any(np.isnan(y_c[i])):
                    continue
                self.samples.append((
                    X[i - seq_len : i],   # (seq_len, 18)
                    y_r[i],               # (3,)
                    y_c[i],               # (3,)
                    prices[i - 1],        # last close price
                ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y_r, y_c, price = self.samples[idx]

        if self.augment:
            # Light Gaussian noise augmentation
            noise = np.random.normal(0, 0.002, x.shape).astype(np.float32)
            x = x + noise
            # Random scale jitter ±1%
            scale = np.random.uniform(0.99, 1.01)
            x = x * scale

        return (
            torch.from_numpy(x),
            torch.from_numpy(y_r),
            torch.from_numpy(y_c),
            torch.tensor(price, dtype=torch.float32),
        )


def split_df_chronological(
    df: pd.DataFrame,
    val_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split — no shuffling."""
    split = int(len(df) * (1 - val_ratio))
    return df.iloc[:split], df.iloc[split:]


cat >> src/data/dataset.py << 'EOF'


def make_fold(
    all_features: dict,
    fold_idx:     int,
    n_folds:      int,
    seq_len:      int = 168,
    val_ratio:    float = 0.1,
    gap_days:     int = 30,
    batch_size:   int = 64,
):
    """
    Walk-forward fold splitter — compatible with existing walk_forward_cv.py.
    Splits each ticker's data chronologically, returns train/val Datasets.
    """
    from torch.utils.data import DataLoader
    import math

    train_dfs = []
    val_dfs   = []

    for ticker, df in all_features.items():
        n       = len(df)
        fold_sz = n // n_folds
        if fold_sz < seq_len + 50:
            continue

        val_end   = n - fold_idx * fold_sz
        val_start = val_end - fold_sz
        gap       = gap_days * 24   # hourly candles
        train_end = max(0, val_start - gap)

        if train_end < seq_len + 50:
            continue

        train_dfs.append(df.iloc[:train_end].reset_index(drop=True))
        val_dfs.append(df.iloc[val_start:val_end].reset_index(drop=True))

    train_ds = CryptoSequenceDataset(train_dfs, seq_len, augment=True)
    val_ds   = CryptoSequenceDataset(val_dfs,   seq_len, augment=False)
    return train_ds, val_ds
EOF
echo "Done"
