"""
cryp2me-ml / configs/config.py
──────────────────────────────
Single source of truth for all hyperparameters, paths, and settings.
Change things here — don't scatter magic numbers through training code.
"""

from pydantic_settings import BaseSettings
from pydantic import BaseModel
from pathlib import Path
from typing import List

ROOT = Path(__file__).parent.parent


# ── Data ──────────────────────────────────────────────────────────────────────

class DataConfig(BaseModel):
    tickers: List[str] = [
        "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX",
        "LINK", "DOT", "MATIC", "UNI", "ATOM", "LTC", "NEAR",
        "OP",  "ARB", "INJ",  "SUI", "APT", "FIL", "AAVE",
        "MKR", "SNX", "CRV",  "GRT", "1INCH",
    ]
    interval: str          = "1h"           # Binance interval
    lookback_days: int     = 4 * 365        # ~4 years training data
    sequence_length: int   = 168            # timesteps fed to model
    prediction_horizons: List[int] = [24, 48, 72]   # T+1d, T+2d, T+3d
    val_size: float        = 0.1            # hold-out within each fold
    cache_dir: Path        = ROOT / "data" / "cache"
    processed_dir: Path    = ROOT / "data" / "processed"


# ── Features ─────────────────────────────────────────────────────────────────

class FeatureConfig(BaseModel):
    # Raw OHLCV (normalised)
    price_features:  List[str] = ["open_pct", "high_pct", "low_pct", "close_pct"]
    volume_features: List[str] = ["volume_norm", "volume_ratio"]
    # Technical indicators
    ema_features:    List[str] = ["ema10_dist", "ema20_dist", "ema34_dist"]
    momentum:        List[str] = ["rsi14", "macd", "macd_signal", "macd_hist"]
    volatility:      List[str] = ["atr14_norm", "bb_width"]
    trend:           List[str] = ["adx14", "plus_di", "minus_di"]

    @property
    def all_features(self) -> List[str]:
        return (
            self.price_features + self.volume_features +
            self.ema_features   + self.momentum +
            self.volatility     + self.trend
        )

    @property
    def n_features(self) -> int:
        return len(self.all_features)   # 18


# ── LSTM ─────────────────────────────────────────────────────────────────────

class LSTMConfig(BaseModel):
    hidden_size:   int   = 256
    num_layers:    int   = 3
    dropout:       float = 0.2
    bidirectional: bool  = False


# ── Transformer ───────────────────────────────────────────────────────────────

class TransformerConfig(BaseModel):
    d_model:     int   = 128
    nhead:       int   = 8
    num_layers:  int   = 4
    dim_feedforward: int = 512
    dropout:     float = 0.1


# ── Training ─────────────────────────────────────────────────────────────────

class TrainConfig(BaseModel):
    batch_size:      int   = 64
    epochs:          int   = 100
    learning_rate:   float = 1e-3
    weight_decay:    float = 1e-4
    patience:        int   = 15         # early stopping
    grad_clip:       float = 1.0
    warmup_epochs:   int   = 5
    # Loss weights: regression vs classification
    mse_weight:      float = 0.6
    bce_weight:      float = 0.4
    # Walk-forward CV
    n_folds:         int   = 5
    fold_gap_days:   int   = 30         # gap between train end and test start (prevent leakage)


# ── Ensemble ─────────────────────────────────────────────────────────────────

class EnsembleConfig(BaseModel):
    confidence_threshold: float = 0.55  # below this → LOW_CONFIDENCE
    lstm_weight:          float = 0.5   # initial weight (tuned by meta-learner)
    transformer_weight:   float = 0.5


# ── Paths ─────────────────────────────────────────────────────────────────────

class PathConfig(BaseModel):
    checkpoints: Path = ROOT / "models" / "checkpoints"
    onnx:        Path = ROOT / "models" / "onnx"
    reports:     Path = ROOT / "reports"
    logs:        Path = ROOT / "logs"


# ── Master config ─────────────────────────────────────────────────────────────

class Config(BaseModel):
    data:        DataConfig        = DataConfig()
    features:    FeatureConfig     = FeatureConfig()
    lstm:        LSTMConfig        = LSTMConfig()
    transformer: TransformerConfig = TransformerConfig()
    training:    TrainConfig       = TrainConfig()
    ensemble:    EnsembleConfig    = EnsembleConfig()
    paths:       PathConfig        = PathConfig()
    seed:        int               = 42
    device:      str               = "auto"   # "auto" → cuda if available, else cpu

    def resolve_device(self) -> str:
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


# Singleton
cfg = Config()
