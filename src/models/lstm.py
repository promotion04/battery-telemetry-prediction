# src/models/lstm.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class LSTMBackbone(nn.Module):
    """
    입력: x (B,T,F)
    출력: g (B,H), H (B,H,T)
    - bidirectional 사용 안 함(인과성 유지)
    """
    def __init__(self, in_feat: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=(dropout if num_layers > 1 else 0.0),
            batch_first=True,
            bidirectional=False,
        )
        self.hidden = int(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: (B,T,F)
        y, _ = self.lstm(x)        # (B,T,H)
        y = self.dropout(y)
        g = y.mean(dim=1)          # (B,H) 전역 평균 풀링
        H = y.transpose(1, 2)      # (B,H,T) -> heads에서 Conv1d(1x1) 사용
        return g, H
