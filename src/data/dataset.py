import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict

FEATURE_COLS = [
    "open_pct","high_pct","low_pct","close_pct",
    "volume_norm","volume_ratio","vol_divergence","buy_sell_ratio","vol_momentum","obv_norm",
    "ema10_dist","ema20_dist","ema34_dist",
    "rsi14","macd","macd_signal","macd_hist",
    "atr14_norm","bb_width",
    "adx14","plus_di","minus_di",
    "dxy_pct","gold_pct","spx_pct","yield_10y_chg","yield_curve","vix_level","vix_chg",
    "funding_rate","oi_change",
    "regime_0","regime_1","regime_2","regime_3",
]
TARGET_REG_COLS = ["ret1","ret2","ret3"]
TARGET_CLS_COLS = ["dir1","dir2","dir3"]

class CryptoSequenceDataset(Dataset):
    def __init__(self, dfs, seq_len=168, augment=False):
        self.seq_len = seq_len
        self.augment = augment
        self.arrays  = []
        self.index   = []
        for df_idx, df in enumerate(dfs):
            for col in FEATURE_COLS:
                if col not in df.columns: df[col] = 0.0
            X      = np.clip(df[FEATURE_COLS].values.astype(np.float32), -10, 10)
            y_reg  = df[TARGET_REG_COLS].values.astype(np.float32)
            y_cls  = df[TARGET_CLS_COLS].values.astype(np.float32)
            prices = df["close"].values.astype(np.float32)
            self.arrays.append((X, y_reg, y_cls, prices))
            for i in range(seq_len, len(df)):
                if np.any(np.isnan(y_reg[i])) or np.any(np.isnan(y_cls[i])): continue
                if np.any(np.isnan(X[i-seq_len:i])): continue
                self.index.append((df_idx, i))

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        df_idx, i = self.index[idx]
        X, y_reg, y_cls, prices = self.arrays[df_idx]
        x = X[i-self.seq_len:i].copy()
        if self.augment:
            x[:,:22] += np.random.normal(0, 0.002, (x.shape[0], 22)).astype(np.float32)
        return (
            torch.from_numpy(x),
            torch.from_numpy(y_reg[i].copy()),
            torch.from_numpy(y_cls[i].copy()),
            torch.tensor(float(prices[i-1]), dtype=torch.float32),
        )

def make_fold(all_features, fold_idx, n_folds, seq_len=168, val_ratio=0.1, gap_days=3, batch_size=64):
    train_dfs, val_dfs = [], []
    gap = gap_days * 24
    for ticker, df in all_features.items():
        n = len(df)
        fold_sz = n // n_folds
        if fold_sz < seq_len + 50: continue
        val_end   = n - fold_idx * fold_sz
        val_start = val_end - fold_sz
        train_end = val_start - gap
        if train_end < seq_len + 50: continue
        if val_start < 0 or val_end < 0: continue
        train_dfs.append(df.iloc[:train_end].reset_index(drop=True))
        val_dfs.append(df.iloc[val_start:val_end].reset_index(drop=True))
    return (
        CryptoSequenceDataset(train_dfs, seq_len, augment=True),
        CryptoSequenceDataset(val_dfs,   seq_len, augment=False),
    )

def make_fold_loaders(train_dfs, val_dfs, seq_len=168, batch_size=64, feature_cols=None):
    train_ds = CryptoSequenceDataset(train_dfs, seq_len, augment=True)
    val_ds   = CryptoSequenceDataset(val_dfs,   seq_len, augment=False)
    return (
        DataLoader(train_ds, batch_size=batch_size,     shuffle=True,  num_workers=0),
        DataLoader(val_ds,   batch_size=batch_size * 2, shuffle=False, num_workers=0),
    )
