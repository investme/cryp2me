"""
src/training/walk_forward_cv.py — cryp2me.ai Phase 5
Walk-forward cross-validation. Clean rewrite.

Key fixes vs Phase 4:
  - Imports make_fold directly (no alias make_fold_loaders)
  - Returns oof_preds and oof_labels for meta-learner training
  - Skips folds gracefully if insufficient data
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Type
from rich.console import Console
from rich.progress import track

from src.data.dataset import make_fold          # ← correct import, no alias
from src.training.trainer import train_model

console = Console()


def run_walk_forward_cv(
    all_features:   Dict,
    ModelClass:     Type,
    model_kwargs:   Dict,
    device:         str,
    n_folds:        int   = 3,
    seq_len:        int   = 168,
    batch_size:     int   = 64,
    epochs:         int   = 100,
    lr:             float = 1e-3,
    patience:       int   = 8,
    gap_days:       int   = 3,
    model_name:     str   = "model",
    checkpoint_dir: Path  = Path("models/checkpoints"),
) -> Dict[str, Any]:
    """
    Walk-forward cross-validation.

    Returns dict with:
        mean_dir_acc, std_dir_acc, min_dir_acc, max_dir_acc,
        n_folds (actual completed),
        oof_preds, oof_labels (for meta-learner)
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    fold_accs  = []
    oof_preds  = []
    oof_labels = []

    for fold_idx in range(n_folds):
        console.print(f"\n  [cyan]── Fold {fold_idx + 1}/{n_folds}[/cyan]")

        train_ds, val_ds = make_fold(
            all_features = all_features,
            fold_idx     = fold_idx,
            n_folds      = n_folds,
            seq_len      = seq_len,
            gap_days     = gap_days,
            batch_size   = batch_size,
        )

        if len(train_ds) < 200 or len(val_ds) < 50:
            console.print(f"  [yellow]⚠  Fold {fold_idx} skipped: "
                          f"train={len(train_ds)} val={len(val_ds)} (too small)[/yellow]")
            continue

        console.print(f"     train={len(train_ds):,} samples  val={len(val_ds):,} samples")

        train_loader = DataLoader(train_ds, batch_size=batch_size,   shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size*2, shuffle=False, num_workers=0)

        # Instantiate fresh model for each fold
        model = ModelClass(**model_kwargs)
        model = model.to(device)

        ckpt_path = checkpoint_dir / f"{model_name}_fold{fold_idx}.pt"

        fold_result = train_model(
            model        = model,
            train_loader = train_loader,
            val_loader   = val_loader,
            device       = device,
            epochs       = epochs,
            lr           = lr,
            patience     = patience,
            checkpoint_path = ckpt_path,
            model_kwargs    = model_kwargs,
            fold_idx        = fold_idx,
        )

        dir_acc = fold_result.get("best_dir_acc", 0.5)
        fold_accs.append(dir_acc)
        console.print(f"     [green]✓  Fold {fold_idx} dir_acc: {dir_acc*100:.2f}%[/green]")

        # Collect OOF predictions for meta-learner
        if "oof_preds" in fold_result:
            oof_preds.append(fold_result["oof_preds"])
            oof_labels.append(fold_result["oof_labels"])

    if not fold_accs:
        console.print("[red]✗  All folds were skipped! Check your data size.[/red]")
        return {
            "mean_dir_acc": 0.5, "std_dir_acc": 0.0,
            "min_dir_acc":  0.5, "max_dir_acc": 0.5,
            "n_folds":      0,   "oof_preds":   [], "oof_labels": [],
        }

    return {
        "mean_dir_acc": float(np.mean(fold_accs)),
        "std_dir_acc":  float(np.std(fold_accs)),
        "min_dir_acc":  float(np.min(fold_accs)),
        "max_dir_acc":  float(np.max(fold_accs)),
        "n_folds":      len(fold_accs),
        "oof_preds":    oof_preds,
        "oof_labels":   oof_labels,
    }
