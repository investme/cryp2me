"""
src/models/ensemble.py — cryp2me.ai Phase 5 Final
MetaLearner MLP ensemble. Trained on OOF predictions from LSTM + Transformer.
Confirmed working: ONNX export clean.
"""
import torch
import torch.nn as nn


class MetaLearner(nn.Module):
    def __init__(self, n_horizons: int = 3, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_horizons * 2, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_horizons), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
