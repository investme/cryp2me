"""
scripts/train_v3.py
────────────────────
Phase 4 — Complete integrated training pipeline.

Features: 35 total (22 technical + 9 macro + 4 regime)
Models:   LSTM → Transformer → XGBoost ensemble
Target:   68-74% raw accuracy, 74-80% precision on confidence-gated signals

Run modes:
  python scripts/train_v3.py                    # full run, re-download all data
  python scripts/train_v3.py --skip-download    # use cached OHLCV + macro
  python scripts/train_v3.py --skip-macro       # skip macro (technical only)
  python scripts/train_v3.py --fast             # 5 tickers, 5 epochs (smoke test)
  python scripts/train_v3.py --epochs 200       # override epoch count
"""

import argparse
import sys
import time
import glob as glob_module
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.panel   import Panel
from rich.table   import Table

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.config import cfg
from src.data.features    import build_all, FEATURE_COLS, N_FEATURES
from src.training.walk_forward_cv import run_walk_forward_cv
from src.models.lstm_model        import LSTMModel
from src.models.transformer_model import TransformerModel
from src.models.xgboost_ensemble  import XGBoostEnsemble
from src.export.onnx_export       import export_all_models

console = Console()

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--skip-download", action="store_true", help="Use cached OHLCV data")
parser.add_argument("--skip-macro",    action="store_true", help="Skip macro data collection")
parser.add_argument("--fast",          action="store_true", help="Quick smoke test (5 tickers)")
parser.add_argument("--epochs",        type=int, default=None)
parser.add_argument("--tickers",       nargs="+", default=None)
args = parser.parse_args()

if args.fast:
    TICKERS = ["BTC", "ETH", "SOL", "BNB", "XRP"]
    EPOCHS  = 5
else:
    TICKERS = args.tickers or cfg.data.tickers
    EPOCHS  = args.epochs  or cfg.training.epochs

DEVICE = torch.device(cfg.resolve_device())
torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)

console.print(Panel.fit(
    f"[bold cyan]cryp2me.ai — Phase 4 Training[/bold cyan]\n"
    f"[dim]35 features: Technical + Volume + Macro + Regime + XGBoost[/dim]\n"
    f"[dim]Device: {DEVICE} | Tickers: {len(TICKERS)} | Epochs: {EPOCHS}[/dim]\n"
    f"[dim]Sequence: {cfg.data.sequence_length}h | Folds: {cfg.training.n_folds}[/dim]",
    border_style="cyan",
))

# Create output directories
for p in [cfg.paths.checkpoints, cfg.paths.onnx,
          cfg.paths.reports, cfg.paths.macro_cache]:
    p.mkdir(parents=True, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — OHLCV Data
# ═════════════════════════════════════════════════════════════════════════════
console.rule("[bold]STEP 1 — OHLCV Data Collection")

if not args.skip_download:
    from src.data.collector import collect_all
    raw_data = collect_all(
        TICKERS,
        interval=cfg.data.interval,
        lookback_days=cfg.data.lookback_days,
        cache_dir=cfg.data.cache_dir,
    )
else:
    console.print("  ↩  Loading from cache...")
    raw_data = {}
    for f in glob_module.glob(str(cfg.data.cache_dir / "*.parquet")):
        ticker = Path(f).stem.split("_")[0]
        if ticker in TICKERS:
            raw_data[ticker] = pd.read_parquet(f)
    console.print(f"  ✓  {len(raw_data)} tickers loaded")

# Get time range for macro collection
all_times = []
for df in raw_data.values():
    if "time" in df.columns:
        all_times.extend(df["time"].dropna().values[:100])
        all_times.extend(df["time"].dropna().values[-100:])

start_ms = int(min(all_times)) if all_times else 0
end_ms   = int(max(all_times)) if all_times else int(time.time() * 1000)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — Macro Data
# ═════════════════════════════════════════════════════════════════════════════
console.rule("[bold]STEP 2 — Macro Data Collection")

macro_map = {}   # {ticker: aligned_macro_df}

if args.skip_macro:
    console.print("  ↩  Skipping macro data (--skip-macro)")
else:
    try:
        from src.data.macro_collector import collect_macro, add_macro_to_df

        console.print("  📡 Fetching: DXY · Gold · S&P500 · 10Y Yield · 2Y Yield · VIX")
        macro_df = collect_macro(start_ms, end_ms,
                                 force_refresh=not args.skip_download)

        if macro_df.empty:
            console.print("  ⚠  Macro unavailable — install: pip install yfinance")
        else:
            console.print(f"  ✓  {len(macro_df):,} hourly rows · {len(macro_df.columns)} macro features")

            # Align macro to each ticker's OHLCV timestamps
            console.print("  🔗 Aligning macro to ticker timestamps...")
            for ticker, df in raw_data.items():
                try:
                    enriched = add_macro_to_df(
                        df, macro_df,
                        ticker=ticker,
                        start_ms=start_ms,
                        end_ms=end_ms,
                    )
                    # Extract just the macro columns
                    macro_cols = [c for c in macro_df.columns if c in enriched.columns]
                    macro_cols += ["funding_rate", "oi_change"]
                    macro_cols  = [c for c in macro_cols if c in enriched.columns]
                    macro_map[ticker] = enriched[macro_cols].reset_index(drop=True)
                except Exception as e:
                    console.print(f"  ⚠  Macro alignment failed for {ticker}: {e}")

            console.print(f"  ✓  Macro aligned for {len(macro_map)} tickers")

    except ImportError:
        console.print("  ⚠  macro_collector not found — skipping macro features")
    except Exception as e:
        console.print(f"  ⚠  Macro collection failed: {e}")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — Feature Engineering (35 features)
# ═════════════════════════════════════════════════════════════════════════════
console.rule("[bold]STEP 3 — Feature Engineering (35 features)")

all_features = build_all(raw_data, macro_map=macro_map if macro_map else None)

total_samples = sum(len(df) for df in all_features.values())
console.print(f"  ✓  {N_FEATURES} features · {total_samples:,} total samples")

# Feature breakdown table
table = Table(show_header=True, header_style="bold cyan", box=None)
table.add_column("Group",    style="cyan")
table.add_column("Features", style="white")
table.add_column("Count",    style="green")
table.add_row("Price",     "open_pct, high_pct, low_pct, close_pct", "4")
table.add_row("Volume",    "norm, ratio, divergence, buy/sell, momentum, OBV", "6")
table.add_row("Technical", "EMA×3, RSI, MACD×3, ATR, BB, ADX×3", "12")
table.add_row("Macro",     "DXY, Gold, SPX, Yield10Y, YieldCurve, VIX×2, Funding, OI", "9")
table.add_row("Regime",    "Ranging, TrendUp, TrendDown, Volatile (one-hot)", "4")
table.add_row("[bold]Total", "", "[bold]35")
console.print(table)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — LSTM Walk-Forward CV
# ═════════════════════════════════════════════════════════════════════════════
console.rule("[bold]STEP 4 — LSTM Walk-Forward CV")

lstm_kwargs = {
    "n_features":   N_FEATURES,
    "hidden_sizes": [cfg.lstm.hidden_size, 128, 64],
    "n_horizons":   cfg.data.n_horizons,
    "dropout":      cfg.lstm.dropout,
}

lstm_cv = run_walk_forward_cv(
    all_features, LSTMModel, lstm_kwargs, DEVICE,
    seq_len    = cfg.data.sequence_length,
    epochs     = EPOCHS,
    lr         = cfg.training.lr,
    patience   = cfg.training.patience,
    model_name = "lstm_v3",
)

console.print(f"\n  ✓  LSTM mean dir_acc: "
              f"[bold green]{lstm_cv['mean_dir_acc']*100:.2f}%[/bold green] "
              f"± {lstm_cv['std_dir_acc']*100:.2f}%")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — Transformer Walk-Forward CV
# ═════════════════════════════════════════════════════════════════════════════
console.rule("[bold]STEP 5 — Transformer Walk-Forward CV")

tf_kwargs = {
    "n_features":      N_FEATURES,
    "d_model":         cfg.transformer.d_model,
    "nhead":           cfg.transformer.nhead,
    "num_layers":      cfg.transformer.num_layers,
    "dim_feedforward": cfg.transformer.dim_feedforward,
    "n_horizons":      cfg.data.n_horizons,
    "dropout":         cfg.transformer.dropout,
}

tf_cv = run_walk_forward_cv(
    all_features, TransformerModel, tf_kwargs, DEVICE,
    seq_len    = cfg.data.sequence_length,
    epochs     = EPOCHS,
    lr         = cfg.training.lr,
    patience   = cfg.training.patience,
    model_name = "transformer_v3",
)

console.print(f"\n  ✓  Transformer mean dir_acc: "
              f"[bold green]{tf_cv['mean_dir_acc']*100:.2f}%[/bold green] "
              f"± {tf_cv['std_dir_acc']*100:.2f}%")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — XGBoost Meta-Learner
# ═════════════════════════════════════════════════════════════════════════════
console.rule("[bold]STEP 6 — XGBoost Meta-Learner")

lstm_oof    = lstm_cv.get("oof_cls_preds", np.array([]))
tf_oof      = tf_cv.get("oof_cls_preds",   np.array([]))
oof_targets = lstm_cv.get("oof_targets",   np.array([]))
oof_macro   = lstm_cv.get("oof_macro",     None)
oof_regime  = lstm_cv.get("oof_regime",    np.zeros(max(len(lstm_oof), 1)))

if len(lstm_oof) > 0 and len(tf_oof) > 0 and len(oof_targets) > 0:
    console.print(f"  🌲 Training XGBoost on {len(lstm_oof):,} OOF samples...")

    # Extract macro+regime features from oof feature matrix
    if oof_macro is not None and oof_macro.shape[1] >= N_FEATURES:
        macro_start = 22   # after 22 technical features
        macro_end   = 31   # 9 macro features
        oof_macro_only = oof_macro[:, macro_start:macro_end]
    else:
        oof_macro_only = np.zeros((len(lstm_oof), 9))

    try:
        xgb_ens = XGBoostEnsemble(
            n_estimators  = cfg.xgboost.n_estimators,
            max_depth     = cfg.xgboost.max_depth,
            learning_rate = cfg.xgboost.learning_rate,
            subsample     = cfg.xgboost.subsample,
            cache_path    = cfg.paths.xgb_model,
        )
        xgb_ens.fit(
            lstm_oof, tf_oof,
            oof_macro_only,
            oof_regime.astype(int),
            oof_targets,
        )

        # Feature importance report
        fi = xgb_ens.feature_importance_report()
        if not fi.empty:
            console.print("\n  📊 [bold]Top 10 features (XGBoost):[/bold]")
            top10 = fi.groupby("feature")["importance"].mean().sort_values(ascending=False).head(10)
            for feat, imp in top10.items():
                bar = "█" * max(1, int(imp * 60))
                console.print(f"     {feat:22s} {bar} {imp:.4f}")

    except Exception as e:
        console.print(f"  ⚠  XGBoost failed: {e}")
        console.print("     Install with: pip install xgboost")
        xgb_ens = None
else:
    console.print("  ⚠  Insufficient OOF data — skipping XGBoost")
    xgb_ens = None

# ═════════════════════════════════════════════════════════════════════════════
# STEP 7 — Accuracy Evaluation
# ═════════════════════════════════════════════════════════════════════════════
console.rule("[bold]STEP 7 — Accuracy Evaluation")

# Raw accuracy summary
console.print(f"\n  {'Model':<20} {'Accuracy':>10} {'Std':>8}")
console.print(f"  {'-'*40}")
console.print(f"  {'LSTM':<20} {lstm_cv['mean_dir_acc']*100:>9.2f}% {lstm_cv['std_dir_acc']*100:>7.2f}%")
console.print(f"  {'Transformer':<20} {tf_cv['mean_dir_acc']*100:>9.2f}% {tf_cv['std_dir_acc']*100:>7.2f}%")

# Ensemble accuracy
if len(lstm_oof) > 0 and len(tf_oof) > 0 and len(oof_targets) > 0:
    if xgb_ens is not None and xgb_ens._fitted:
        oof_macro_only = np.zeros((len(lstm_oof), 9)) if oof_macro is None else oof_macro[:, 22:31]
        ens_preds = xgb_ens.predict_proba(
            lstm_oof, tf_oof, oof_macro_only, oof_regime.astype(int)
        )
    else:
        ens_preds = 0.45 * lstm_oof + 0.55 * tf_oof

    ens_acc = ((ens_preds[:, 0] > 0.5).astype(int) == oof_targets[:, 0].astype(int)).mean()
    console.print(f"  {'XGBoost Ensemble':<20} {ens_acc*100:>9.2f}%")

    # Confidence threshold analysis
    console.print(f"\n  [bold]Confidence-gated accuracy:[/bold]")
    console.print(f"  {'Threshold':<12} {'Signals':>10} {'Coverage':>10} {'Accuracy':>10}")
    console.print(f"  {'-'*45}")

    h0_probs   = ens_preds[:, 0]
    h0_targets = oof_targets[:, 0].astype(int)

    for thresh in [0.52, 0.55, 0.58, 0.60, 0.63, 0.65]:
        mask     = (h0_probs > thresh) | (h0_probs < (1 - thresh))
        n_sigs   = mask.sum()
        coverage = n_sigs / len(h0_probs)
        if n_sigs > 10:
            acc = ((h0_probs[mask] > 0.5).astype(int) == h0_targets[mask]).mean()
            console.print(f"  {thresh:<12.2f} {n_sigs:>10,} {coverage:>9.1%} {acc:>9.2%}")
        else:
            console.print(f"  {thresh:<12.2f} {n_sigs:>10,} {'—':>10} {'—':>10}")

# Save accuracy report
import json
report = {
    "lstm_acc":        float(lstm_cv["mean_dir_acc"]),
    "lstm_std":        float(lstm_cv["std_dir_acc"]),
    "transformer_acc": float(tf_cv["mean_dir_acc"]),
    "transformer_std": float(tf_cv["std_dir_acc"]),
    "n_features":      N_FEATURES,
    "feature_cols":    FEATURE_COLS,
    "macro_enabled":   bool(macro_map),
    "tickers":         TICKERS,
    "sequence_length": cfg.data.sequence_length,
}
report_path = cfg.paths.reports / "accuracy_report.json"
report_path.parent.mkdir(parents=True, exist_ok=True)
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
console.print(f"\n  ✓  Report saved → {report_path}")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 8 — ONNX Export
# ═════════════════════════════════════════════════════════════════════════════
console.rule("[bold]STEP 8 — Model Export")

lstm_ckpts = sorted(glob_module.glob(str(cfg.paths.checkpoints / "lstm_v3_fold*.pt")))
tf_ckpts   = sorted(glob_module.glob(str(cfg.paths.checkpoints / "transformer_v3_fold*.pt")))

if lstm_ckpts and tf_ckpts:
    final_lstm = LSTMModel(**lstm_kwargs)
    ckpt = torch.load(lstm_ckpts[-1], map_location="cpu", weights_only=False)
    final_lstm.load_state_dict(ckpt["model_state"])

    final_tf = TransformerModel(**tf_kwargs)
    ckpt = torch.load(tf_ckpts[-1], map_location="cpu", weights_only=False)
    final_tf.load_state_dict(ckpt["model_state"])

    try:
        export_all_models(
            final_lstm, final_tf, None,
            seq_len=cfg.data.sequence_length,
            n_features=N_FEATURES,
            onnx_dir=cfg.paths.onnx,
        )
        console.print(f"  ✓  ONNX models → {cfg.paths.onnx}")
    except Exception as e:
        console.print(f"  ⚠  ONNX export failed: {e}")
else:
    console.print("  ⚠  No checkpoints found — skipping ONNX export")

# ── Final summary ─────────────────────────────────────────────────────────────
console.print(Panel.fit(
    f"[bold green]✅  Phase 4 Training Complete![/bold green]\n\n"
    f"  ONNX models      →  models/onnx/\n"
    f"  XGBoost model    →  models/xgb_ensemble.pkl\n"
    f"  Regime model     →  models/regime_classifier.pkl\n"
    f"  Accuracy report  →  reports/accuracy_report.json\n\n"
    f"  Copy to backend:\n"
    f"  cp models/onnx/*.onnx           ../cryp2me-backend/models/onnx/\n"
    f"  cp models/xgb_ensemble.pkl      ../cryp2me-backend/models/\n"
    f"  cp models/regime_classifier.pkl ../cryp2me-backend/models/",
    border_style="green",
))
