"""
src/models/ensemble.py
──────────────────────
Ensemble that combines LSTM + Transformer predictions.

Architecture:
  LSTM cls output   (batch, 3) ─┐
                                 ├→ concat (batch, 6) → Logistic meta-learner → final probs
  Transformer cls   (batch, 3) ─┘

Confidence thresholding:
  max(P(up), P(down)) < threshold → signal = LOW_CONFIDENCE (don't trade)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Literal

Signal = Literal["BULLISH", "BEARISH", "NEUTRAL", "LOW_CONFIDENCE"]


class MetaLearner(nn.Module):
    """
    Lightweight meta-learner trained on out-of-fold predictions.
    Takes concatenated direction probabilities from both base models.
    """

    def __init__(self, n_horizons: int = 3, dropout: float = 0.1):
        super().__init__()
        in_dim = n_horizons * 2   # LSTM probs + Transformer probs

        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, n_horizons),
            nn.Sigmoid(),
        )

    def forward(
        self,
        lstm_probs: torch.Tensor,        # (batch, 3)
        transformer_probs: torch.Tensor, # (batch, 3)
    ) -> torch.Tensor:                   # (batch, 3)
        x = torch.cat([lstm_probs, transformer_probs], dim=-1)
        return self.net(x)


class EnsemblePredictor:
    """
    Inference wrapper: wraps trained LSTM, Transformer, and MetaLearner.
    Handles confidence thresholding and signal generation.
    """

    def __init__(
        self,
        lstm_model,
        transformer_model,
        meta_learner: MetaLearner,
        confidence_threshold: float = 0.65,
        device: str = "cpu",
    ):
        self.lstm_model        = lstm_model.to(device).eval()
        self.transformer_model = transformer_model.to(device).eval()
        self.meta_learner      = meta_learner.to(device).eval()
        self.confidence_threshold = confidence_threshold
        self.device = device

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,  # (batch, seq_len, n_features)
    ) -> Dict:
        """
        Returns a dict with:
          probabilities: (batch, 3)   — P(up) for each horizon
          signal:        list[Signal]
          confidence:    list[float]
        """
        x = x.to(self.device)

        lstm_probs = self.lstm_model.predict_proba(x)
        tf_probs   = self.transformer_model.predict_proba(x)
        final_probs = self.meta_learner(lstm_probs, tf_probs)  # (batch, 3)

        probs_np = final_probs.cpu().numpy()

        signals     = []
        confidences = []

        for row in probs_np:
            # Use T+1d probability as primary signal
            p_up   = float(row[0])
            p_down = 1.0 - p_up
            conf   = max(p_up, p_down)

            if conf < self.confidence_threshold:
                signals.append("LOW_CONFIDENCE")
            elif p_up > p_down:
                signals.append("BULLISH")
            else:
                signals.append("BEARISH")

            confidences.append(conf * 100)   # → percentage

        return {
            "probabilities": probs_np,   # raw
            "signals":       signals,
            "confidences":   confidences,
        }

    @torch.no_grad()
    def predict_prices(
        self,
        x: torch.Tensor,
        current_prices: torch.Tensor,    # (batch,)
    ) -> np.ndarray:
        """
        Returns predicted prices for each horizon.
        Uses regression head from LSTM (higher quality on regression).
        Shape: (batch, 3)
        """
        x = x.to(self.device)
        reg_out, _ = self.lstm_model(x)   # (batch, 3) — % returns
        reg_np = reg_out.cpu().numpy()
        prices = current_prices.cpu().numpy().reshape(-1, 1)

        # Predicted price = current * (1 + predicted_return)
        predicted = prices * (1.0 + reg_np)
        return predicted
