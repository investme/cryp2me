"""
src/training/walk_forward_cv.py
────────────────────────────────
Walk-forward (time-series) cross-validation.

Why walk-forward and not k-fold?
  Standard k-fold shuffles data randomly — this creates look-ahead bias
  in time series (training on future data to predict the past).
  Walk-forward keeps temporal order: train on past, test on future.

Structure:
  ┌────────────────────────────────────────────────────────────────────┐
  │  Fold 0: ████████░░  train=[0..40%] ──gap── test=[40+gap..55%]   │
  │  Fold 1: ████████████░░  train=[0..55%] ──gap── test=[55+gap..70%]│
  │  Fold 2: ██████████████░░  train=[0..70%] ──── test=[70+gap..85%] │
  └────────────────────────────────────────────────────────────────────┘
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from rich.console import Console
from pathlib import Path
import json

from src.data.dataset import make_fold_loaders as make_fold
from src.training.trainer import train_model, eval_epoch, CombinedLoss, compute_directional_accuracy

console = Console()


def run_walk_forward_cv(
    all_features:    dict,
    model_class,
    model_kwargs:    dict,
    device:          str,
    n_folds:         int   = 5,
    seq_len:         int   = 60,
    batch_size:      int   = 64,
    epochs:          int   = 100,
    lr:              float = 1e-3,
    patience:        int   = 15,
    gap_days:        int   = 30,
    checkpoint_dir:  Path  = Path("models/checkpoints"),
    model_name:      str   = "model",
) -> Dict:
    """
    Runs full walk-forward CV and saves per-fold checkpoints + OOF predictions.

    Returns:
        cv_results: dict with per-fold metrics and aggregate statistics
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fold_results   = []
    oof_preds_all  = []   # out-of-fold predictions (for meta-learner training)
    oof_labels_all = []

    console.print(f"\n[bold magenta]🔁 Walk-Forward CV — {n_folds} folds — {model_name}[/bold magenta]")

    for fold_idx in range(n_folds):
        console.print(f"\n[cyan]── Fold {fold_idx + 1}/{n_folds} ──[/cyan]")

        train_ds, val_ds = make_fold(
            all_features, fold_idx, n_folds, seq_len,
            val_ratio=0.1, gap_days=gap_days,
        )

        if len(train_ds) < 100 or len(val_ds) < 20:
            console.print(f"  [yellow]⚠  Skipping fold {fold_idx}: insufficient data[/yellow]")
            continue

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

        # Fresh model each fold
        model = model_class(**model_kwargs)

        history = train_model(
            model, train_loader, val_loader, device,
            epochs=epochs, lr=lr, patience=patience,
            model_name=f"{model_name}_fold{fold_idx}",
        )

        # Save fold checkpoint
        ckpt_path = checkpoint_dir / f"{model_name}_fold{fold_idx}.pt"
        torch.save({
            "fold": fold_idx,
            "model_state": model.state_dict(),
            "model_kwargs": model_kwargs,
            "history": history,
        }, ckpt_path)

        # Evaluate on hold-out fold
        criterion = CombinedLoss()
        val_metrics = eval_epoch(model, val_loader, criterion, device)

        # Collect OOF predictions for meta-learner
        oof_preds, oof_labels = _collect_oof(model, val_loader, device)
        oof_preds_all.append(oof_preds)
        oof_labels_all.append(oof_labels)

        fold_result = {
            "fold":       fold_idx,
            "val_loss":   val_metrics["total"],
            "dir_acc":    val_metrics.get("dir_acc", 0.0),
            "n_train":    len(train_ds),
            "n_val":      len(val_ds),
        }
        fold_results.append(fold_result)
        console.print(f"  [green]Fold {fold_idx} dir_acc: {fold_result['dir_acc']*100:.2f}%[/green]")

    # Aggregate metrics
    if fold_results:
        accs = [f["dir_acc"] for f in fold_results]
        cv_results = {
            "model":         model_name,
            "folds":         fold_results,
            "mean_dir_acc":  float(np.mean(accs)),
            "std_dir_acc":   float(np.std(accs)),
            "min_dir_acc":   float(np.min(accs)),
            "max_dir_acc":   float(np.max(accs)),
            "oof_preds":     np.concatenate(oof_preds_all, axis=0).tolist() if oof_preds_all else [],
            "oof_labels":    np.concatenate(oof_labels_all, axis=0).tolist() if oof_labels_all else [],
        }

        console.print(
            f"\n[bold green]✓  {model_name} CV Summary[/bold green]\n"
            f"   Mean dir_acc: {cv_results['mean_dir_acc']*100:.2f}% "
            f"± {cv_results['std_dir_acc']*100:.2f}%\n"
            f"   Range: {cv_results['min_dir_acc']*100:.1f}% – {cv_results['max_dir_acc']*100:.1f}%"
        )
    else:
        cv_results = {"model": model_name, "folds": [], "mean_dir_acc": 0.0}

    return cv_results


@torch.no_grad()
def _collect_oof(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    """Collect all classification predictions + labels for meta-learner training."""
    model.eval()
    preds, labels = [], []
    for x, _, y_cls, _ in loader:
        x = x.to(device)
        _, cls_out = model(x)
        preds.append(cls_out.cpu().numpy())
        labels.append(y_cls.numpy())
    return (
        np.concatenate(preds,  axis=0),
        np.concatenate(labels, axis=0),
    )
