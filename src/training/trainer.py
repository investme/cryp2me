"""
src/training/trainer.py
────────────────────────
Training engine for both LSTM and Transformer models.
Uses combined MSE + BCE loss, AdamW + cosine LR schedule, early stopping.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Dict, List
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
import time

console = Console()


class CombinedLoss(nn.Module):
    """
    Focal Loss for classification (pushes probabilities away from 0.5)
    + MSE for regression.
    """
    def __init__(self, mse_weight: float = 0.4, bce_weight: float = 0.6, gamma: float = 2.0):
        super().__init__()
        self.mse   = nn.MSELoss()
        self.gamma = 2.0
        self.mse_w = mse_weight
        self.bce_w = bce_weight

    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred    = pred.clamp(1e-6, 1 - 1e-6)
        bce     = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        pt      = torch.where(target == 1, pred, 1 - pred)
        focal   = ((1 - pt) ** self.gamma) * bce
        return focal.mean()

    def forward(self, reg_pred, reg_true, cls_pred, cls_true):
        mse_loss = self.mse(reg_pred, reg_true)
        # Focal loss — penalises confident wrong predictions and wishy-washy 0.5 outputs
        pred  = cls_pred.clamp(1e-6, 1 - 1e-6)
        bce   = -(cls_true * torch.log(pred) + (1 - cls_true) * torch.log(1 - pred))
        pt    = torch.where(cls_true == 1, pred, 1 - pred)
        focal = ((1 - pt) ** self.gamma) * bce
        focal_loss = focal.mean()
        total = self.mse_w * mse_loss + self.bce_w * focal_loss
        return total, {"mse": mse_loss.item(), "bce": focal_loss.item(), "total": total.item()} 

class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-5):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("-inf")
        self.counter    = 0
        self.best_state = None

    def __call__(self, val_loss: float, model: nn.Module, val_acc: float = 0.0) -> bool:
        score = val_acc - val_loss * 0.1
        if score > self.best_loss - self.min_delta:
            self.best_loss  = score
            self.counter    = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

def compute_directional_accuracy(cls_pred: np.ndarray, cls_true: np.ndarray) -> float:
    """Binary accuracy — predicted direction matches actual direction."""
    pred_dir = (cls_pred > 0.5).astype(int)
    return float((pred_dir == cls_true.astype(int)).mean())


def train_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  CombinedLoss,
    device:     str,
    grad_clip:  float = 1.0,
) -> Dict[str, float]:
    model.train()
    total_losses = {"mse": 0., "bce": 0., "total": 0.}
    n_batches = 0

    for x, y_reg, y_cls, _ in loader:
        x, y_reg, y_cls = x.to(device), y_reg.to(device), y_cls.to(device)

        optimizer.zero_grad()
        reg_out, cls_out = model(x)
        loss, losses = criterion(reg_out, y_reg, cls_out, y_cls)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        for k, v in losses.items():
            total_losses[k] += v
        n_batches += 1

    return {k: v / n_batches for k, v in total_losses.items()}


@torch.no_grad()
def eval_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: CombinedLoss,
    device:    str,
) -> Dict[str, float]:
    model.eval()
    total_losses = {"mse": 0., "bce": 0., "total": 0.}
    all_cls_pred, all_cls_true = [], []
    n_batches = 0

    for x, y_reg, y_cls, _ in loader:
        x, y_reg, y_cls = x.to(device), y_reg.to(device), y_cls.to(device)
        reg_out, cls_out = model(x)
        loss, losses = criterion(reg_out, y_reg, cls_out, y_cls)

        for k, v in losses.items():
            total_losses[k] += v

        all_cls_pred.append(cls_out.cpu().numpy())
        all_cls_true.append(y_cls.cpu().numpy())
        n_batches += 1

    avg_losses = {k: v / n_batches for k, v in total_losses.items()}
    if all_cls_pred:
        pred = np.concatenate(all_cls_pred, axis=0)
        true = np.concatenate(all_cls_true, axis=0)
        avg_losses["dir_acc"] = compute_directional_accuracy(pred, true)

    return avg_losses


def train_model(
    model:          nn.Module,
    train_loader:   DataLoader,
    val_loader:     DataLoader,
    device:         str,
    epochs:         int        = 100,
    lr:             float      = 1e-3,
    weight_decay:   float      = 1e-4,
    patience:       int        = 15,
    grad_clip:      float      = 1.0,
    warmup_epochs:  int        = 5,
    mse_weight:     float      = 0.6,
    bce_weight:     float      = 0.4,
    model_name:     str        = "model",
) -> Dict[str, List]:
    """
    Full training loop with early stopping, LR warmup + cosine decay.
    Returns history dict with train/val losses and accuracy per epoch.
    """
    model = model.to(device)
    criterion = CombinedLoss(mse_weight, bce_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # Cosine annealing with warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    stopper   = EarlyStopping(patience=patience)

    history = {"train_loss": [], "val_loss": [], "val_dir_acc": [], "lr": []}

    console.print(f"\n[bold cyan]🏋  Training {model_name}[/bold cyan]  "
                  f"[dim]({sum(p.numel() for p in model.parameters() if p.requires_grad):,} params)[/dim]")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[cyan]{task.completed}/{task.total}[/cyan]"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Training", total=epochs)

        for epoch in range(epochs):
            t0 = time.time()

            train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, grad_clip)
            val_metrics   = eval_epoch(model, val_loader, criterion, device)
            scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            history["train_loss"].append(train_metrics["total"])
            history["val_loss"].append(val_metrics["total"])
            history["val_dir_acc"].append(val_metrics.get("dir_acc", 0.0))
            history["lr"].append(current_lr)

            elapsed = time.time() - t0
            progress.update(task, advance=1,
                description=f"Ep {epoch+1:3d} | "
                            f"loss {train_metrics['total']:.4f}→{val_metrics['total']:.4f} | "
                            f"dir_acc {val_metrics.get('dir_acc', 0)*100:.1f}% | "
                            f"{elapsed:.1f}s")

            if stopper(val_metrics["total"], model, val_metrics.get("dir_acc", 0.0)):
                console.print(f"  [yellow]Early stop at epoch {epoch+1}[/yellow]")
                break

    stopper.restore_best(model)
    best_acc = max(history["val_dir_acc"])
    console.print(f"  [bold green]✓  Best val dir_acc: {best_acc*100:.2f}%[/bold green]\n")
    return history
