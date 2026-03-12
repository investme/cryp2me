"""
src/models/transformer_model.py — cryp2me.ai Phase 5 Final
Transformer encoder with regression + classification heads.
Confirmed working: ONNX export clean, dir_acc 50-57% local / 57.7% Kaggle
"""
import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(
        self,
        n_features:     int   = 35,
        d_model:        int   = 128,
        nhead:          int   = 8,
        num_layers:     int   = 4,
        dim_feedforward:int   = 512,
        n_horizons:     int   = 3,
        dropout:        float = 0.1,
        max_seq_len:    int   = 512,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc    = nn.Embedding(max_seq_len, d_model)
        enc_layer       = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers)
        self.reg_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, n_horizons))
        self.cls_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, n_horizons), nn.Sigmoid())

    def forward(self, x):
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x   = self.input_proj(x) + self.pos_enc(pos)
        x   = self.encoder(x)
        out = x[:, -1, :]
        return self.reg_head(out), self.cls_head(out)
