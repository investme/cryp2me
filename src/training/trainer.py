"""
src/training/trainer.py — cryp2me.ai Phase 5
Single-model training loop with early stopping and mixed loss.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console

console = Console()


def combined_loss(pred_reg, pred_cls, y_reg, y_cls, mse_w=0.4, bce_w=0.6):
    mse = nn.functional.mse_loss(pred_reg, y_reg)
    bce = nn.functional.binary_cross_entropy(pred_cls.clamp(1e-6, 1-1e-6), y_cls)
    return mse_w * mse + bce_w * bce


def train_model(
    model, train_loader, val_loader, device,
    epochs=100, lr=1e-3, patience=8, weight_decay=1e-4,
    grad_clip=1.0, mse_weight=0.4, bce_weight=0.6,
    checkpoint_path=None, model_kwargs=None, fold_idx=0,
):
    model = model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    best_val_loss, best_dir_acc, patience_count = float("inf"), 0.0, 0
    history, best_state = [], None

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for x, y_reg, y_cls, _ in train_loader:
            x, y_reg, y_cls = x.to(device), y_reg.to(device), y_cls.to(device)
            optimiser.zero_grad()
            pred_reg, pred_cls = model(x)
            loss = combined_loss(pred_reg, pred_cls, y_reg, y_cls, mse_weight, bce_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimiser.step()
            train_losses.append(loss.item())
        scheduler.step()

        model.eval()
        val_losses, all_cls_pred, all_cls_true = [], [], []
        with torch.no_grad():
            for x, y_reg, y_cls, _ in val_loader:
                x, y_reg, y_cls = x.to(device), y_reg.to(device), y_cls.to(device)
                pred_reg, pred_cls = model(x)
                loss = combined_loss(pred_reg, pred_cls, y_reg, y_cls, mse_weight, bce_weight)
                val_losses.append(loss.item())
                all_cls_pred.append(pred_cls.cpu().numpy())
                all_cls_true.append(y_cls.cpu().numpy())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        preds      = np.concatenate(all_cls_pred, axis=0)
        labels     = np.concatenate(all_cls_true, axis=0)
        dir_acc    = (((preds > 0.5).astype(float)) == labels).mean()
        history.append({"epoch": epoch, "train_loss": float(train_loss), "val_loss": float(val_loss), "dir_acc": float(dir_acc)})

        if epoch % 5 == 0 or epoch < 3:
            console.print(f"     Ep {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  dir_acc={dir_acc*100:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss, best_dir_acc, patience_count = val_loss, float(dir_acc), 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"fold": fold_idx, "model_state": best_state, "model_kwargs": model_kwargs or {}, "history": history}, checkpoint_path)
        else:
            patience_count += 1
            if patience_count >= patience:
                console.print(f"     [yellow]Early stop at epoch {epoch}[/yellow]")
                break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    oof_preds, oof_labels = [], []
    with torch.no_grad():
        for x, _, y_cls, _ in val_loader:
            _, pred_cls = model(x.to(device))
            oof_preds.append(pred_cls.cpu().numpy())
            oof_labels.append(y_cls.numpy())

    console.print(f"     [bold green]Best: val_loss={best_val_loss:.4f}  dir_acc={best_dir_acc*100:.2f}%[/bold green]")
    return {
        "best_dir_acc": best_dir_acc, "best_val_loss": best_val_loss,
        "history": history,
        "oof_preds":  np.concatenate(oof_preds,  axis=0) if oof_preds  else np.array([]),
        "oof_labels": np.concatenate(oof_labels, axis=0) if oof_labels else np.array([]),
    }
