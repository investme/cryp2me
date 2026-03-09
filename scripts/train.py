"""
scripts/train.py
────────────────
Master training script. Run this to execute the full Phase 2 pipeline:

  1. Download data (or use cache)
  2. Engineer features
  3. Walk-forward CV for LSTM
  4. Walk-forward CV for Transformer
  5. Train meta-learner on OOF predictions
  6. Evaluate ensemble + generate accuracy report
  7. Export all models to ONNX

Usage:
  python scripts/train.py                    # full run
  python scripts/train.py --fast             # reduced epochs/data for testing
  python scripts/train.py --skip-download    # use cached data only
"""

import sys
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from rich.console import Console

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import cfg
from src.data.collector import collect_all
from src.data.features import build_all
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
    p = argparse.ArgumentParser(description="cryp2me.ai — Phase 2 Training Pipeline")
    p.add_argument("--fast",           action="store_true", help="Quick test run (reduced epochs & data)")
    p.add_argument("--skip-download",  action="store_true", help="Use cached data only")
    p.add_argument("--tickers",        nargs="+",           help="Override ticker list (e.g. BTC ETH SOL)")
    p.add_argument("--epochs",         type=int,            help="Override epoch count")
    p.add_argument("--device",         type=str,            help="Override device (cpu/cuda)")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(cfg.seed)

    device = args.device or cfg.resolve_device()
    console.print(f"\n[bold cyan]🚀 cryp2me.ai — Phase 2 Training[/bold cyan]")
    console.print(f"   Device   : [green]{device}[/green]")
    console.print(f"   Time     : {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    # ── Override config for fast mode ─────────────────────────────────────────
    tickers    = args.tickers or cfg.data.tickers
    epochs     = args.epochs  or (10 if args.fast else cfg.training.epochs)
    n_folds    = 2 if args.fast else cfg.training.n_folds
    batch_size = cfg.training.batch_size

    if args.fast:
        tickers = tickers[:5]
        console.print(f"   [yellow]⚡ Fast mode: {tickers}, {epochs} epochs, {n_folds} folds[/yellow]")

    # Create output directories
    for d in [cfg.paths.checkpoints, cfg.paths.onnx, cfg.paths.reports, cfg.data.cache_dir, cfg.data.processed_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 1 — DATA COLLECTION
    # ═══════════════════════════════════════════════════════════════════════════
    console.print("\n[bold]STEP 1 — Data Collection[/bold]")
    if args.skip_download:
        console.print("  [yellow]Skipping download, loading from cache...[/yellow]")
        import pandas as pd
        raw_data = {}
        for t in tickers:
            p = cfg.data.cache_dir / f"{t}_1d.parquet"
            if p.exists():
                raw_data[t] = pd.read_parquet(p)
    else:
        raw_data = collect_all(
            tickers,
            interval=cfg.data.interval,
            lookback_days=cfg.data.lookback_days,
            cache_dir=cfg.data.cache_dir,
        )

    if not raw_data:
        console.print("[red]✗  No data collected. Check your network connection.[/red]")
        sys.exit(1)

    console.print(f"  [green]✓  {len(raw_data)} tickers loaded[/green]")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 2 — FEATURE ENGINEERING
    # ═══════════════════════════════════════════════════════════════════════════
    console.print("\n[bold]STEP 2 — Feature Engineering[/bold]")
    all_features = build_all(raw_data)

    total_samples = sum(len(df) for df in all_features.values())
    console.print(f"  [green]✓  {total_samples:,} total samples across {len(all_features)} tickers[/green]")

    seq_len = cfg.data.sequence_length

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 3 — LSTM WALK-FORWARD CV
    # ═══════════════════════════════════════════════════════════════════════════
    console.print("\n[bold]STEP 3 — LSTM Walk-Forward CV[/bold]")
    lstm_kwargs = {
        "n_features":   cfg.features.n_features,
        "hidden_sizes": [cfg.lstm.hidden_size, cfg.lstm.hidden_size // 2, cfg.lstm.hidden_size // 4],
        "dropout":      cfg.lstm.dropout,
    }
    lstm_cv = run_walk_forward_cv(
        all_features, LSTMModel, lstm_kwargs, device,
        n_folds=n_folds, seq_len=seq_len,
        batch_size=batch_size, epochs=epochs,
        lr=cfg.training.learning_rate,
        patience=cfg.training.patience,
        gap_days=cfg.training.fold_gap_days,
        checkpoint_dir=cfg.paths.checkpoints,
        model_name="lstm",
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 4 — TRANSFORMER WALK-FORWARD CV
    # ═══════════════════════════════════════════════════════════════════════════
    console.print("\n[bold]STEP 4 — Transformer Walk-Forward CV[/bold]")
    tf_kwargs = {
        "n_features":       cfg.features.n_features,
        "d_model":          cfg.transformer.d_model,
        "nhead":            cfg.transformer.nhead,
        "num_layers":       cfg.transformer.num_layers,
        "dim_feedforward":  cfg.transformer.dim_feedforward,
        "dropout":          cfg.transformer.dropout,
    }
    tf_cv = run_walk_forward_cv(
        all_features, TransformerModel, tf_kwargs, device,
        n_folds=n_folds, seq_len=seq_len,
        batch_size=batch_size, epochs=epochs,
        lr=cfg.training.learning_rate,
        patience=cfg.training.patience,
        gap_days=cfg.training.fold_gap_days,
        checkpoint_dir=cfg.paths.checkpoints,
        model_name="transformer",
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 5 — TRAIN META-LEARNER ON OOF PREDICTIONS
    # ═══════════════════════════════════════════════════════════════════════════
    console.print("\n[bold]STEP 5 — Train Meta-Learner (Ensemble)[/bold]")

    lstm_oof  = np.array(lstm_cv.get("oof_preds", []))
    tf_oof    = np.array(tf_cv.get("oof_preds", []))
    oof_labels = np.array(lstm_cv.get("oof_labels", []))

    meta = MetaLearner(n_horizons=3)
    ensemble_probs = None

    if len(lstm_oof) > 100 and lstm_oof.shape == tf_oof.shape:
        import torch.optim as optim
        meta = meta.to(device)
        meta.train()

        optimizer = optim.Adam(meta.parameters(), lr=5e-4)
        criterion = torch.nn.BCELoss()

        # Mini training loop for meta-learner
        X_lstm  = torch.tensor(lstm_oof,   dtype=torch.float32).to(device)
        X_tf    = torch.tensor(tf_oof,     dtype=torch.float32).to(device)
        y       = torch.tensor(oof_labels, dtype=torch.float32).to(device)

        for epoch in range(200):
            optimizer.zero_grad()
            out  = meta(X_lstm, X_tf)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        torch.save(meta.state_dict(), cfg.paths.checkpoints / "meta_learner.pt")
        console.print(f"  [green]✓  Meta-learner trained on {len(lstm_oof):,} OOF samples[/green]")

        meta.eval()
        with torch.no_grad():
            ensemble_probs = meta(X_lstm, X_tf).cpu().numpy()
    else:
        console.print("  [yellow]⚠  Insufficient OOF data for meta-learner — using equal weights[/yellow]")
        if len(lstm_oof) > 0 and len(tf_oof) > 0:
            ensemble_probs = (lstm_oof + tf_oof) / 2.0

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 6 — ACCURACY EVALUATION REPORT
    # ═══════════════════════════════════════════════════════════════════════════
    console.print("\n[bold]STEP 6 — Accuracy Evaluation[/bold]")
    report = generate_accuracy_report(
        lstm_cv, tf_cv,
        ensemble_probs, oof_labels,
        output_dir=cfg.paths.reports,
    )
    print_accuracy_report(report)

    # Save full CV results
    with open(cfg.paths.reports / "cv_results.json", "w") as f:
        json.dump({"lstm": lstm_cv, "transformer": tf_cv}, f, indent=2, default=str)

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 7 — EXPORT TO ONNX
    # ═══════════════════════════════════════════════════════════════════════════
    console.print("\n[bold]STEP 7 — ONNX Export[/bold]")

    # Load best checkpoints
    final_lstm = LSTMModel(**lstm_kwargs)
    final_tf   = TransformerModel(**tf_kwargs)

    lstm_ckpts = sorted(cfg.paths.checkpoints.glob("lstm_fold*.pt"))
    tf_ckpts   = sorted(cfg.paths.checkpoints.glob("transformer_fold*.pt"))

    if lstm_ckpts:
        ckpt = torch.load(lstm_ckpts[-1], map_location="cpu", weights_only=False)
        final_lstm.load_state_dict(ckpt["model_state"])
        console.print(f"  Loaded LSTM from {lstm_ckpts[-1].name}")
    if tf_ckpts:
        ckpt = torch.load(tf_ckpts[-1], map_location="cpu", weights_only=False)
        final_tf.load_state_dict(ckpt["model_state"])
        console.print(f"  Loaded Transformer from {tf_ckpts[-1].name}")

    onnx_paths = export_all_models(
        final_lstm, final_tf, meta,
        seq_len=seq_len,
        n_features=cfg.features.n_features,
        onnx_dir=cfg.paths.onnx,
    )

    # Save paths for backend
    with open(cfg.paths.onnx / "manifest.json", "w") as f:
        json.dump(onnx_paths, f, indent=2)

    # ═══════════════════════════════════════════════════════════════════════════
    # DONE
    # ═══════════════════════════════════════════════════════════════════════════
    console.print(f"\n[bold green]═══════════════════════════════════════════[/bold green]")
    console.print(f"[bold green]  ✅  Phase 2 Complete![/bold green]")
    console.print(f"[bold green]═══════════════════════════════════════════[/bold green]")
    console.print(f"  ONNX models → [cyan]{cfg.paths.onnx}/[/cyan]")
    console.print(f"  Accuracy report → [cyan]{cfg.paths.reports}/accuracy_report.json[/cyan]")
    console.print(f"\n  [bold]Next:[/bold] Copy ONNX models to cryp2me-backend/models/onnx/")
    console.print(f"  Copy inference_service.py → cryp2me-backend/app/services/")
    console.print(f"  Copy predict_router_v2.py → cryp2me-backend/app/routers/predict.py\n")


if __name__ == "__main__":
    main()
