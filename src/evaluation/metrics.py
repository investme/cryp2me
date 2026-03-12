"""
src/evaluation/metrics.py — cryp2me.ai Phase 5 Final
Accuracy report generator. Fixed: no duplicate n_horizons kwarg.
Confirmed working: full threshold analysis, JSON report.
"""
import numpy as np
from datetime import datetime
from typing import Dict, List
from rich.console import Console

console = Console()


def generate_accuracy_report(
    lstm_cv:      Dict,
    tf_cv:        Dict,
    meta_model,
    all_features: Dict,
    seq_len:      int   = 168,
    device:       str   = "cpu",
    thresholds:   List  = [0.55, 0.60, 0.65, 0.70, 0.75],
) -> Dict:
    import torch
    from torch.utils.data import DataLoader
    from src.data.dataset import CryptoSequenceDataset

    # Use last 20% of data as evaluation set
    dfs     = list(all_features.values())
    split   = int(len(dfs[0]) * 0.8)
    val_dfs = [df.iloc[split:].reset_index(drop=True)
               for df in dfs if len(df) > split + seq_len + 100]
    val_ds  = CryptoSequenceDataset(val_dfs, seq_len, augment=False)

    if len(val_ds) == 0:
        return {
            "generated_at": str(datetime.now()),
            "models": {}, "ensemble": {},
            "threshold_analysis": [], "summary": {},
        }

    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    # Load best checkpoints — use model_kwargs from checkpoint (no extra n_horizons)
    lstm_ckpts = sorted(__import__("pathlib").Path("models/checkpoints").glob("lstm*fold*.pt"))
    tf_ckpts   = sorted(__import__("pathlib").Path("models/checkpoints").glob("transformer*fold*.pt"))

    all_lstm_preds, all_tf_preds, all_labels = [], [], []

    if lstm_ckpts and tf_ckpts:
        from src.models.lstm_model import LSTMModel
        from src.models.transformer_model import TransformerModel

        # Load LSTM — use model_kwargs exactly as saved (already has n_horizons)
        ckpt = torch.load(lstm_ckpts[-1], weights_only=False, map_location="cpu")
        lstm = LSTMModel(**ckpt["model_kwargs"])
        lstm.load_state_dict(ckpt["model_state"])
        lstm.eval()

        # Load Transformer
        tf_c = torch.load(tf_ckpts[-1], weights_only=False, map_location="cpu")
        tf   = TransformerModel(**tf_c["model_kwargs"])
        tf.load_state_dict(tf_c["model_state"])
        tf.eval()

        with torch.no_grad():
            for x, y_reg, y_cls, _ in val_loader:
                _, lp = lstm(x)
                _, tp = tf(x)
                all_lstm_preds.append(lp.numpy())
                all_tf_preds.append(tp.numpy())
                all_labels.append(y_cls.numpy())

    if not all_labels:
        return {
            "generated_at": str(datetime.now()),
            "models": {}, "ensemble": {},
            "threshold_analysis": [], "summary": {},
        }

    lp  = np.concatenate(all_lstm_preds)
    tp  = np.concatenate(all_tf_preds)
    lbl = np.concatenate(all_labels)
    ep  = (lp + tp) / 2  # simple ensemble average

    lstm_acc = float(((lp > 0.5).astype(float) == lbl).mean())
    tf_acc   = float(((tp > 0.5).astype(float) == lbl).mean())
    ens_acc  = float(((ep > 0.5).astype(float) == lbl).mean())

    # Threshold analysis — filter by confidence
    threshold_analysis = []
    for thresh in thresholds:
        conf = np.maximum(ep, 1 - ep)          # confidence = distance from 0.5
        mask = (conf > thresh).any(axis=-1)    # at least one horizon is confident
        if mask.sum() == 0:
            threshold_analysis.append({
                "threshold": thresh, "precision": 0.0,
                "coverage": 0.0, "n_signals": 0,
            })
            continue
        prec = float(((ep[mask] > 0.5).astype(float) == lbl[mask]).mean())
        threshold_analysis.append({
            "threshold": thresh,
            "precision": prec,
            "coverage":  float(mask.mean()),
            "n_signals": int(mask.sum()),
        })

    # Horizon breakdown
    horizon_accuracy = {}
    for i, label in enumerate(["T+1d", "T+2d", "T+3d"]):
        acc = float(((ep[:, i] > 0.5).astype(float) == lbl[:, i]).mean())
        horizon_accuracy[label] = acc

    report = {
        "generated_at": str(datetime.now()),
        "models": {
            "LSTM": {
                "walk_forward_cv": {
                    **{k: v for k, v in lstm_cv.items() if k not in ("oof_preds", "oof_labels")},
                    "mean_dir_acc": lstm_acc,
                }
            },
            "Transformer": {
                "walk_forward_cv": {
                    **{k: v for k, v in tf_cv.items() if k not in ("oof_preds", "oof_labels")},
                    "mean_dir_acc": tf_acc,
                }
            },
        },
        "ensemble": {
            "base_accuracy":    ens_acc,
            "horizon_accuracy": horizon_accuracy,
        },
        "threshold_analysis": threshold_analysis,
        "summary": {
            "lstm_baseline_acc":        f"{lstm_acc*100:.2f}%",
            "transformer_baseline_acc": f"{tf_acc*100:.2f}%",
            "ensemble_no_threshold":    f"{ens_acc*100:.2f}%",
            "recommended_threshold": next(
                (t for t in threshold_analysis if t["precision"] >= 0.65 and t["n_signals"] > 100),
                threshold_analysis[0] if threshold_analysis else {},
            ),
        },
    }
    return report


def print_accuracy_report(report: Dict):
    console.print("\n[bold cyan]📊 Accuracy Report[/bold cyan]")
    for name, data in report.get("models", {}).items():
        acc = data["walk_forward_cv"].get("mean_dir_acc", 0)
        console.print(f"  {name}: [green]{acc*100:.2f}%[/green]")
    ens = report.get("ensemble", {}).get("base_accuracy", 0)
    console.print(f"  Ensemble: [green]{ens*100:.2f}%[/green]")

    hz = report.get("ensemble", {}).get("horizon_accuracy", {})
    if hz:
        console.print(f"\n  Horizon breakdown:")
        for label, acc in hz.items():
            console.print(f"    {label}: {acc*100:.1f}%")

    console.print("\n  Threshold Analysis:")
    for t in report.get("threshold_analysis", []):
        if t["n_signals"] > 0:
            console.print(
                f"    {t['threshold']:.2f} → "
                f"{t['precision']*100:.1f}% precision  "
                f"{t['n_signals']:,} signals  "
                f"({t['coverage']*100:.0f}% coverage)"
            )
