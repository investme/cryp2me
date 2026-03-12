"""
scripts/train.py — cryp2me.ai Phase 5
Master training script. Clean rewrite. No legacy cruft.

Usage:
    python scripts/train.py                   # full run, all 27 tickers
    python scripts/train.py --skip-download   # use cached data
    python scripts/train.py --fast            # 3 tickers, 10 epochs (smoke test)
    python scripts/train.py --tickers BTC ETH SOL  # custom ticker list
    python scripts/train.py --epochs 50       # override epoch count

Phase 5 fixes vs Phase 4:
    - Targets correctly use 24/48/72h horizons (not 1/2/3h)
    - walk_forward_cv imports make_fold directly (no alias)
    - All config fields present (no AttributeError)
    - ONNX export uses dynamo=False (no CPU/CUDA device conflict)
    - fold_gap_days=3 ensures all folds have sufficient data
"""

import sys
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import cfg
from src.data.collector import collect_all
from src.data.features import build_all, N_FEATURES, FEATURE_COLS
from src.data.dataset import CryptoSequenceDataset, make_fold
from src.models.lstm_model import LSTMModel
from src.models.transformer_model import TransformerModel
from src.models.ensemble import MetaLearner
from src.training.walk_forward_cv import run_walk_forward_cv
from src.training.trainer import train_model
from src.evaluation.metrics import generate_accuracy_report, print_accuracy_report
from src.export.onnx_export import export_all_models
from torch.utils.data import DataLoader

console = Console()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="cryp2me.ai Phase 5 Training")
    p.add_argument("--fast",          action="store_true", help="Smoke test: 3 tickers, 10 epochs")
    p.add_argument("--skip-download", action="store_true", help="Use cached data only")
    p.add_argument("--tickers",       nargs="+",           help="Override ticker list")
    p.add_argument("--epochs",        type=int,            help="Override epoch count")
    p.add_argument("--device",        type=str,            help="Override device (cpu/cuda)")
    return p.parse_args()


def print_header(device, tickers, epochs, seq_len, n_folds):
    console.print(Panel(
        f"[bold white]cryp2me.ai — Phase 5 Training[/bold white]\n"
        f"[cyan]{N_FEATURES} features: Technical + Volume + Macro + Regime[/cyan]\n"
        f"[green]Device: {device}[/green]  |  "
        f"[yellow]Tickers: {len(tickers)}[/yellow]  |  "
        f"[magenta]Epochs: {epochs}[/magenta]\n"
        f"Sequence: {seq_len}h  |  Folds: {n_folds}  |  "
        f"Horizons: T+24h / T+48h / T+72h",
        border_style="cyan",
    ))


def main():
    args   = parse_args()
    set_seed(cfg.seed)
    device = args.device or cfg.resolve_device()

    # ── Config overrides ──────────────────────────────────────────────────────
    tickers    = args.tickers or cfg.data.tickers
    epochs     = args.epochs  or (10 if args.fast else cfg.training.epochs)
    n_folds    = 2 if args.fast else cfg.training.n_folds
    batch_size = cfg.training.batch_size
    seq_len    = cfg.data.sequence_length
    gap_days   = cfg.training.fold_gap_days

    if args.fast:
        tickers = tickers[:3]

    print_header(device, tickers, epochs, seq_len, n_folds)

    # ── Create output directories ─────────────────────────────────────────────
    for d in [cfg.paths.checkpoints, cfg.paths.onnx, cfg.paths.reports, cfg.paths.logs]:
        d.mkdir(parents=True, exist_ok=True)

    # ════════════════════════════════════════════════════════════════
    # STEP 1 — Data Collection
    # ════════════════════════════════════════════════════════════════
    console.rule("[bold cyan]STEP 1 — OHLCV Data Collection[/bold cyan]")
    raw_data = collect_all(
        tickers          = tickers,
        interval         = cfg.data.interval,
        lookback_days    = cfg.data.lookback_days,
        cache_dir        = cfg.data.cache_dir,
        skip_download    = args.skip_download,
    )
    if not raw_data:
        console.print("[bold red]✗  No data collected — aborting[/bold red]")
        return
    console.print(f"  ✓  {len(raw_data)} tickers loaded")

    # ════════════════════════════════════════════════════════════════
    # STEP 2 — Feature Engineering
    # ════════════════════════════════════════════════════════════════
    console.rule(f"[bold cyan]STEP 2 — Feature Engineering ({N_FEATURES} features)[/bold cyan]")
    all_features = build_all(raw_data)

    if not all_features:
        console.print("[bold red]✗  Feature engineering failed — aborting[/bold red]")
        return

    total_samples = sum(len(v) for v in all_features.values())
    console.print(f"  ✓  {total_samples:,} total samples | {N_FEATURES} features")

    # Verify fold sizes before training
    console.print("\n  [cyan]Fold size check:[/cyan]")
    sample_ticker = list(all_features.values())[0]
    n = len(sample_ticker)
    all_folds_ok = True
    for fold_idx in range(n_folds):
        fold_sz   = n // n_folds
        val_end   = n - fold_idx * fold_sz
        val_start = val_end - fold_sz
        train_end = max(0, val_start - gap_days * 24)
        ok = train_end >= seq_len + 50
        status = "[green]✓[/green]" if ok else "[red]✗ SKIP[/red]"
        console.print(f"    Fold {fold_idx}: train={train_end:,}  val={fold_sz:,}  {status}")
        if not ok:
            all_folds_ok = False
    if not all_folds_ok:
        console.print("  [yellow]⚠  Some folds will be skipped (insufficient data)[/yellow]")

    # ════════════════════════════════════════════════════════════════
    # STEP 3 — LSTM Walk-Forward CV
    # ════════════════════════════════════════════════════════════════
    console.rule("[bold cyan]STEP 3 — LSTM Walk-Forward CV[/bold cyan]")

    lstm_kwargs = {
        "n_features":   N_FEATURES,
        "hidden_sizes": [128, 64, 32],
        "n_horizons":   cfg.data.n_horizons,
        "dropout":      cfg.lstm.dropout,
    }

    lstm_cv = run_walk_forward_cv(
        all_features = all_features,
        ModelClass   = LSTMModel,
        model_kwargs = lstm_kwargs,
        device       = device,
        n_folds      = n_folds,
        seq_len      = seq_len,
        batch_size   = batch_size,
        epochs       = epochs,
        lr           = cfg.training.lr,
        patience     = cfg.training.patience,
        gap_days     = gap_days,
        model_name   = "lstm_p5",
        checkpoint_dir = cfg.paths.checkpoints,
    )
    console.print(f"\n  ✓  LSTM mean dir_acc: "
                  f"[bold green]{lstm_cv['mean_dir_acc']*100:.2f}%[/bold green] "
                  f"± {lstm_cv['std_dir_acc']*100:.2f}%")

    # ════════════════════════════════════════════════════════════════
    # STEP 4 — Transformer Walk-Forward CV
    # ════════════════════════════════════════════════════════════════
    console.rule("[bold cyan]STEP 4 — Transformer Walk-Forward CV[/bold cyan]")

    tf_kwargs = {
        "n_features":     N_FEATURES,
        "d_model":        cfg.transformer.d_model,
        "nhead":          cfg.transformer.nhead,
        "num_layers":     cfg.transformer.num_layers,
        "dim_feedforward":cfg.transformer.dim_feedforward,
        "n_horizons":     cfg.data.n_horizons,
        "dropout":        cfg.transformer.dropout,
    }

    tf_cv = run_walk_forward_cv(
        all_features = all_features,
        ModelClass   = TransformerModel,
        model_kwargs = tf_kwargs,
        device       = device,
        n_folds      = n_folds,
        seq_len      = seq_len,
        batch_size   = batch_size,
        epochs       = epochs,
        lr           = cfg.training.lr,
        patience     = cfg.training.patience,
        gap_days     = gap_days,
        model_name   = "transformer_p5",
        checkpoint_dir = cfg.paths.checkpoints,
    )
    console.print(f"\n  ✓  Transformer mean dir_acc: "
                  f"[bold green]{tf_cv['mean_dir_acc']*100:.2f}%[/bold green] "
                  f"± {tf_cv['std_dir_acc']*100:.2f}%")

    # ════════════════════════════════════════════════════════════════
    # STEP 5 — Meta-Learner Ensemble
    # ════════════════════════════════════════════════════════════════
    console.rule("[bold cyan]STEP 5 — Train Meta-Learner (Ensemble)[/bold cyan]")

    # Collect OOF predictions from both models
    lstm_oof = lstm_cv.get("oof_preds", [])
    tf_oof   = tf_cv.get("oof_preds",  [])

    if lstm_oof and tf_oof:
        lstm_oof_arr = np.concatenate(lstm_oof, axis=0)
        tf_oof_arr   = np.concatenate(tf_oof,   axis=0)
        oof_labels   = np.concatenate(lstm_cv.get("oof_labels", []), axis=0)

        meta_input = np.hstack([lstm_oof_arr, tf_oof_arr])
        meta       = MetaLearner(n_horizons=cfg.data.n_horizons)
        meta_ds    = torch.utils.data.TensorDataset(
            torch.FloatTensor(meta_input),
            torch.FloatTensor(oof_labels),
        )
        meta_loader = DataLoader(meta_ds, batch_size=256, shuffle=True)

        meta_opt = torch.optim.Adam(meta.parameters(), lr=1e-3)
        meta.train()
        for ep in range(30):
            for xb, yb in meta_loader:
                meta_opt.zero_grad()
                pred = meta(xb)
                loss = torch.nn.functional.binary_cross_entropy(
                    pred.clamp(1e-6, 1-1e-6), yb
                )
                loss.backward()
                meta_opt.step()

        meta_path = cfg.paths.checkpoints / "meta_p5.pt"
        torch.save({"model_state": meta.state_dict()}, meta_path)
        console.print(f"  ✓  Meta-learner trained on {len(meta_input):,} OOF samples")
    else:
        console.print("  ⚠  Insufficient OOF data — using simple weighted ensemble")
        meta = MetaLearner(n_horizons=cfg.data.n_horizons)

    # ════════════════════════════════════════════════════════════════
    # STEP 6 — Accuracy Evaluation
    # ════════════════════════════════════════════════════════════════
    console.rule("[bold cyan]STEP 6 — Accuracy Evaluation[/bold cyan]")

    report = generate_accuracy_report(
        lstm_cv     = lstm_cv,
        tf_cv       = tf_cv,
        meta_model  = meta,
        all_features= all_features,
        seq_len     = seq_len,
        device      = device,
        thresholds  = [0.55, 0.60, 0.65, 0.70, 0.75],
    )
    print_accuracy_report(report)

    report_path = cfg.paths.reports / "accuracy_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    console.print(f"  ✓  Report saved → {report_path}")

    # ════════════════════════════════════════════════════════════════
    # STEP 7 — ONNX Export
    # ════════════════════════════════════════════════════════════════
    console.rule("[bold cyan]STEP 7 — Model Export[/bold cyan]")

    # Load best checkpoints — try p5 naming first, fall back to any lstm/transformer
    lstm_ckpts = sorted(cfg.paths.checkpoints.glob("lstm_p5_fold*.pt"))
    if not lstm_ckpts:
        lstm_ckpts = sorted(cfg.paths.checkpoints.glob("lstm*fold*.pt"))
    tf_ckpts = sorted(cfg.paths.checkpoints.glob("transformer_p5_fold*.pt"))
    if not tf_ckpts:
        tf_ckpts = sorted(cfg.paths.checkpoints.glob("transformer*fold*.pt"))

    if not lstm_ckpts or not tf_ckpts:
        console.print("  ⚠  No checkpoints found — skipping ONNX export")
        return

    # Use model_kwargs exactly as saved — already contains n_horizons
    ckpt      = torch.load(lstm_ckpts[-1], weights_only=False, map_location="cpu")
    best_lstm = LSTMModel(**ckpt["model_kwargs"])
    best_lstm.load_state_dict(ckpt["model_state"])
    best_lstm.eval().cpu()

    tf_ckpt  = torch.load(tf_ckpts[-1], weights_only=False, map_location="cpu")
    best_tf  = TransformerModel(**tf_ckpt["model_kwargs"])
    best_tf.load_state_dict(tf_ckpt["model_state"])
    best_tf.eval().cpu()
    meta.eval().cpu()

    try:
        onnx_paths = export_all_models(
            best_lstm, best_tf, meta,
            seq_len    = seq_len,
            n_features = N_FEATURES,
            onnx_dir   = cfg.paths.onnx,
        )
        console.print(f"  ✓  ONNX models → {cfg.paths.onnx}")
    except Exception as e:
        console.print(f"  ⚠  ONNX export failed: {e}")
        console.print("  ℹ  PT checkpoints saved — use those for inference")

    # ════════════════════════════════════════════════════════════════
    # DONE
    # ════════════════════════════════════════════════════════════════
    lstm_acc = report.get("models", {}).get("LSTM", {}).get("walk_forward_cv", {}).get("mean_dir_acc", lstm_cv["mean_dir_acc"])
    tf_acc   = report.get("models", {}).get("Transformer", {}).get("walk_forward_cv", {}).get("mean_dir_acc", tf_cv["mean_dir_acc"])
    ens_acc  = report.get("ensemble", {}).get("base_accuracy", 0)
    best_t   = report.get("summary", {}).get("recommended_threshold", {})

    console.print(Panel(
        f"[bold green]✅  Phase 5 Training Complete![/bold green]\n\n"
        f"  LSTM:        {lstm_acc*100:.2f}%\n"
        f"  Transformer: {tf_acc*100:.2f}%\n"
        f"  Ensemble:    {ens_acc*100:.2f}%\n"
        + (f"  Best threshold: {best_t.get('threshold', 0.65)} → "
           f"{best_t.get('precision', 0)*100:.1f}% precision  "
           f"{best_t.get('n_signals', 0):,} signals\n" if best_t else "")
        + f"\n  ONNX models →     models/onnx/\n"
        f"  Accuracy report → reports/accuracy_report.json\n\n"
        f"  Copy to backend:\n"
        f"  cp models/onnx/*.onnx ../cryp2me-backend/models/onnx/",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
