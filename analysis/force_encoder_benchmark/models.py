"""프레임 단위 3-class(free/sliding/collision) 분류를 하는 두 encoder.

둘 다 입력 (B, T, 6) -> 출력 (B, T, 3) 로 시퀀스 길이를 유지한다.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUEncoder(nn.Module):
    """Model A: 2-layer GRU + MLP head."""

    def __init__(self, in_dim: int = 6, hidden: int = 64, num_layers: int = 2,
                 num_classes: int = 3):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)          # (B, T, hidden)
        return self.head(out)         # (B, T, num_classes)


class CausalConv1d(nn.Module):
    """왼쪽에만 padding=(kernel_size-1)*dilation 을 넣어 미래 누설을 막는 conv."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T)
        x = F.pad(x, (self.left_pad, 0))  # 왼쪽(과거)만 패딩
        return self.conv(x)


class CausalConvEncoder(nn.Module):
    """Model B: dilated causal 1D-CNN 3층 + 1x1 conv head."""

    def __init__(self, in_dim: int = 6, channels=(32, 64, 64), kernel_size: int = 3,
                 dilations=(1, 2, 4), num_classes: int = 3):
        super().__init__()
        assert len(channels) == len(dilations)
        layers = []
        c_in = in_dim
        for c_out, d in zip(channels, dilations):
            layers.append(CausalConv1d(c_in, c_out, kernel_size, dilation=d))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
            c_in = c_out
        self.net = nn.Sequential(*layers)
        self.head = nn.Conv1d(c_in, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, C)
        h = self.net(x.transpose(1, 2))     # (B, C', T)
        logits = self.head(h)               # (B, num_classes, T)
        return logits.transpose(1, 2)       # (B, T, num_classes)
