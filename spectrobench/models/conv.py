"""Local-convolution baselines: the CNN side of the comparison.

These use only small (3x3 / kernel-3) local kernels, exactly the regime where a
CNN is cheap and hardware-optimized but has a *local* receptive field and is
tied to the training grid spacing. They are the honest CNN opponent for the
Fourier operators.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1d(nn.Module):
    """(B, N, in_ch) -> (B, N, out_ch), stack of dilated conv blocks."""
    def __init__(self, in_ch=1, out_ch=1, width=48, n_layers=6):
        super().__init__()
        self.inp = nn.Conv1d(in_ch, width, 1)
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            d = 2 ** (i % 4)
            self.blocks.append(nn.Conv1d(width, width, 3, padding=d, dilation=d, padding_mode="circular"))
        self.out = nn.Conv1d(width, out_ch, 1)

    def forward(self, x):
        h = self.inp(x.permute(0, 2, 1))
        for b in self.blocks:
            h = F.gelu(b(h) + h)
        return self.out(h).permute(0, 2, 1)


class UNet2d(nn.Module):
    """Small U-Net over (B, H, W, in_ch) -> (B, H, W, out_ch). H, W should be
    divisible by 4 (two down/up stages)."""
    def __init__(self, in_ch=1, out_ch=1, width=24):
        super().__init__()
        w = width
        self.e1 = self._blk(in_ch, w)
        self.e2 = self._blk(w, 2 * w)
        self.e3 = self._blk(2 * w, 4 * w)
        self.u2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, stride=2)
        self.d2 = self._blk(4 * w, 2 * w)
        self.u1 = nn.ConvTranspose2d(2 * w, w, 2, stride=2)
        self.d1 = self._blk(2 * w, w)
        self.out = nn.Conv2d(w, out_ch, 1)
        self.pool = nn.MaxPool2d(2)

    @staticmethod
    def _blk(i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1, padding_mode="circular"), nn.GELU(),
            nn.Conv2d(o, o, 3, padding=1, padding_mode="circular"), nn.GELU())

    def forward(self, x):  # (B, H, W, in_ch)
        h = x.permute(0, 3, 1, 2)
        e1 = self.e1(h)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        d2 = self.d2(torch.cat([self.u2(e3), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        return self.out(d1).permute(0, 2, 3, 1)
