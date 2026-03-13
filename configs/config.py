"""
configs/config.py — cryp2me.ai Phase 5
Clean, complete config. All fields present. No missing attributes.
"""
from pydantic import BaseModel
from pathlib import Path
from typing import List

ROOT = Path(__file__).parent.parent


class DataConfig(BaseModel):
    tickers: List[str] = [
        "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX",
        "LINK", "DOT", "MATIC", "UNI", "ATOM", "LTC", "NEAR",
        "OP", "ARB", "INJ", "SUI", "APT", "FIL", "AAVE",
        "MKR", "SNX", "CRV", "GRT", "1INCH",
    ]
    interval:            str       = "1h"          # hourly candles
    lookback_days:       int       = 4 * 365       # 4 years
    sequence_length:     int       = 48           # 1 week lookback
    prediction_horizons: List[int] = [24, 48, 72]  # T+1d, T+2d, T+3d
    n_horizons:          int       = 3
    val_size:            float     = 0.1
    cache_dir:           Path      = ROOT / "data" / "cache"
    processed_dir:       Path      = ROOT / "data" / "processed"


class FeatureConfig(BaseModel):
    price_features:  List[str] = ["open_pct", "high_pct", "low_pct", "close_pct"]
    volume_features: List[str] = ["volume_norm", "volume_ratio", "vol_divergence",
                                   "buy_sell_ratio", "vol_momentum", "obv_norm"]
    ema_features:    List[str] = ["ema10_dist", "ema20_dist", "ema34_dist"]
    momentum:        List[str] = ["rsi14", "macd", "macd_signal", "macd_hist"]
    volatility:      List[str] = ["atr14_norm", "bb_width"]
    trend:           List[str] = ["adx14", "plus_di", "minus_di"]
    macro_features:  List[str] = ["dxy_pct", "gold_pct", "spx_pct",
                                   "yield_10y_chg", "yield_curve",
                                   "vix_level", "vix_chg",
                                   "funding_rate", "oi_change"]
    regime_features: List[str] = ["regime_0", "regime_1", "regime_2", "regime_3"]

    @property
    def all_features(self) -> List[str]:
        return (self.price_features + self.volume_features + self.ema_features +
                self.momentum + self.volatility + self.trend +
                self.macro_features + self.regime_features)

    @property
    def n_features(self) -> int:
        return len(self.all_features)


class LSTMConfig(BaseModel):
    hidden_sizes: List[int] = [128, 64, 32]  # stacked LSTM layers
    dropout:      float     = 0.5


class TransformerConfig(BaseModel):
    d_model:         int   = 128
    nhead:           int   = 8
    num_layers:      int   = 4
    dim_feedforward: int   = 512
    dropout:         float = 0.1


class TrainConfig(BaseModel):
    batch_size:    int   = 16
    epochs:        int   = 100
    lr:            float = 1e-3
    learning_rate: float = 1e-3   # alias for lr
    weight_decay:  float = 1e-4
    patience:      int   = 8      # early stopping patience
    grad_clip:     float = 1.0
    warmup_epochs: int   = 5
    mse_weight:    float = 0.4    # regression loss weight
    bce_weight:    float = 0.6    # classification loss weight
    n_folds:       int   = 3      # walk-forward folds
    fold_gap_days: int   = 3      # gap between train/val in days


class EnsembleConfig(BaseModel):
    confidence_threshold: float = 0.65  # precision-optimized threshold
    lstm_weight:          float = 0.45
    transformer_weight:   float = 0.55


class PathConfig(BaseModel):
    checkpoints:  Path = ROOT / "models" / "checkpoints"
    onnx:         Path = ROOT / "models" / "onnx"
    reports:      Path = ROOT / "reports"
    logs:         Path = ROOT / "logs"
    macro_cache:  Path = ROOT / "data" / "cache" / "macro"
    regime_model: Path = ROOT / "models" / "regime_classifier.pkl"


class Config(BaseModel):
    data:        DataConfig        = DataConfig()
    features:    FeatureConfig     = FeatureConfig()
    lstm:        LSTMConfig        = LSTMConfig()
    transformer: TransformerConfig = TransformerConfig()
    training:    TrainConfig       = TrainConfig()
    ensemble:    EnsembleConfig    = EnsembleConfig()
    paths:       PathConfig        = PathConfig()
    seed:        int               = 42
    device:      str               = "auto"

    def resolve_device(self) -> str:
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


cfg = Config()
