"""
src/models/transformer_model.py
────────────────────────────────
Temporal Transformer with positional encoding and causal masking.
Stronger at capturing long-range dependencies across the 60-step window.
Complements the LSTM's local pattern recognition in the ensemble.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    Tells the attention mechanism the order of timesteps,
    since Transformers are inherently order-agnostic.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe     = torch.zeros(max_len, d_model)
        pos    = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div    = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)   # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Architecture:
      Input  (batch, seq_len, n_features)
        │
      Input projection → d_model=128
        │
      Positional Encoding
        │
      4× Transformer Encoder layers
        (8 heads, dim_ff=512, GELU, causal mask)
        │
      Global Average Pooling across time
        │
      ┌──────────────────┐
      │ Regression head  │  → (batch, 3)
      │ Classification   │  → (batch, 3) probabilities
      └──────────────────┘
    """

    def __init__(
        self,
        n_features:      int   = 18,
        d_model:         int   = 128,
        nhead:           int   = 8,
        num_layers:      int   = 4,
        dim_feedforward: int   = 512,
        n_horizons:      int   = 3,
        dropout:         float = 0.1,
        max_seq_len:     int   = 512,
    ):
        super().__init__()
        self.d_model    = d_model
        self.n_horizons = n_horizons

        # ── Input projection ─────────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
        )

        # ── Positional encoding ───────────────────────────────────────────────
        self.pos_enc = PositionalEncoding(d_model, max_seq_len, dropout)

        # ── Transformer encoder ───────────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,    # Pre-LN — more stable training
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # ── Output heads ─────────────────────────────────────────────────────
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_horizons),
        )

        self.cls_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_horizons),
            nn.Sigmoid(),
        )

 
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            reg_out: (batch, n_horizons)
            cls_out: (batch, n_horizons) probabilities
        """
        # Project to d_model
        h = self.input_proj(x)              # (batch, seq_len, d_model)
        h = self.pos_enc(h)

        # Causal mask
        # NEW — pass is_causal=True directly, skip mask entirely
        seq = h.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(seq, device=h.device)
        h = self.encoder(h, mask=mask, is_causal=True)

        # Global average pooling (more stable than taking last token only)
        pooled = h.mean(dim=1)
        return self.reg_head(pooled), self.cls_head(pooled)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        _, cls = self.forward(x)
        return cls

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
