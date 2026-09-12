"""Ablation baselines that share WGSA's (B, C, H, W) -> (B, 1, H, W) interface."""

import torch
import torch.nn as nn


class CBAMAttention(nn.Module):
    """CBAM-style spatial attention with no wavelet path (Table 6, C2).

    Channel-pool mean and max, concatenate to 2 maps, 7x7 conv, sigmoid.
    Drops into `WGN(attention=CBAMAttention())` unchanged. The fusion 1x1 conv
    still dominates the parameter budget, so the full model stays at ~5.76 M.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so padding is centred.")
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        avg = feat.mean(dim=1, keepdim=True)
        mx = feat.amax(dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
