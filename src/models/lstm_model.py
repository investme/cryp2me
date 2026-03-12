"""
src/models/lstm_model.py — cryp2me.ai Phase 5 Final
Stacked LSTM with regression + classification heads.
Confirmed working: ONNX export 2.7x speedup, dir_acc 52-55% local / 54% Kaggle
"""
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        n_features:   int       = 35,
        hidden_sizes: list      = [128, 64, 32],
        n_horizons:   int       = 3,
        dropout:      float     = 0.5,
    ):
        super().__init__()
        self.lstm_layers = nn.ModuleList()
        in_size = n_features
        for hs in hidden_sizes:
            self.lstm_layers.append(nn.LSTM(in_size, hs, batch_first=True))
            in_size = hs

        self.dropout    = nn.Dropout(dropout)
        self.projection = nn.Sequential(nn.Linear(hidden_sizes[-1], 64), nn.ReLU())
        self.reg_head   = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_horizons))
        self.cls_head   = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_horizons), nn.Sigmoid())

    def forward(self, x):
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
            x = self.dropout(x)
        out = self.projection(x[:, -1, :])
        return self.reg_head(out), self.cls_head(out)
