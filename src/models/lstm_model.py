"""
src/models/lstm_model.py
────────────────────────
Stacked Bidirectional LSTM with dual output heads:
  - Regression head:      predicts T+1d, T+2d, T+3d % returns
  - Classification head:  predicts direction probability (0=down, 1=up)
"""

import torch
import torch.nn as nn
from typing import Tuple


class LSTMModel(nn.Module):
    """
    Architecture:
      Input  (batch, seq_len, n_features)
        │
      LSTM Layer 1: hidden=256, dropout=0.2
        │
      LSTM Layer 2: hidden=128, dropout=0.2
        │
      LSTM Layer 3: hidden=64
        │
      Last hidden state → Linear → LayerNorm
        │
      ┌──────────────────┐
      │ Regression head  │  → (batch, 3) — % return predictions
      │ Classification   │  → (batch, 3) — direction probabilities
      └──────────────────┘
    """

    def __init__(
        self,
        n_features:   int   = 18,
        hidden_sizes: list  = [256, 128, 64],
        n_horizons:   int   = 3,
        dropout:      float = 0.2,
    ):
        super().__init__()
        self.n_features   = n_features
        self.n_horizons   = n_horizons
        self.hidden_sizes = hidden_sizes

        # ── Stacked LSTM ─────────────────────────────────────────────────────
        self.lstm_layers = nn.ModuleList()
        in_size = n_features
        for i, hidden in enumerate(hidden_sizes):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=in_size,
                    hidden_size=hidden,
                    num_layers=1,
                    batch_first=True,
                    dropout=0.0,
                )
            )
            in_size = hidden

        self.dropout = nn.Dropout(dropout)
        final_hidden = hidden_sizes[-1]

        # ── Shared projection ─────────────────────────────────────────────────
        self.projection = nn.Sequential(
            nn.Linear(final_hidden, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # ── Output heads ─────────────────────────────────────────────────────
        # Regression: predicts % returns
        self.reg_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, n_horizons),
        )

        # Classification: predicts direction probability
        self.cls_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, n_horizons),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            reg_out: (batch, n_horizons) — predicted % returns
            cls_out: (batch, n_horizons) — direction probabilities [0,1]
        """
        h = x
        for lstm in self.lstm_layers:
            h, _ = lstm(h)
            h = self.dropout(h)

        # Take last timestep
        last = h[:, -1, :]                  # (batch, final_hidden)
        proj = self.projection(last)        # (batch, 128)
        return self.reg_head(proj), self.cls_head(proj)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns direction probabilities only — used in ensemble."""
        _, cls = self.forward(x)
        return cls

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
