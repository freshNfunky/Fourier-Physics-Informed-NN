"""Fourier Neural Operator layers (1D and 2D).

The spectral convolution multiplies a truncated set of Fourier modes by learned
complex weights: a *global* convolution in one layer at O(N log N). The mode
cutoff `modes` is the sensitive knob the spec asks to ablate (H1 caveat): too
many modes and the operator overfits high-frequency noise and its derivative in
a PDE residual blows up; too few and it low-passes the target.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class SpectralConv1d(nn.Module):
    def __init__(self, in_c: int, out_c: int, modes: int):
        super().__init__()
        self.out_c = out_c
        self.modes = modes
        scale = 1.0 / (in_c * out_c)
        self.weight = nn.Parameter(
            scale * torch.rand(in_c, out_c, modes, dtype=torch.cfloat))

    def forward(self, x):  # x: (B, C, N)
        B, C, N = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        m = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(B, self.out_c, x_ft.shape[-1],
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :m] = torch.einsum("bim,iom->bom", x_ft[:, :, :m], self.weight[:, :, :m])
        return torch.fft.irfft(out_ft, n=N, dim=-1)


class SpectralConv2d(nn.Module):
    def __init__(self, in_c: int, out_c: int, modes1: int, modes2: int):
        super().__init__()
        self.out_c = out_c
        self.modes1, self.modes2 = modes1, modes2
        scale = 1.0 / (in_c * out_c)
        self.w1 = nn.Parameter(scale * torch.rand(in_c, out_c, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(in_c, out_c, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x):  # x: (B, C, H, W)
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        m1 = min(self.modes1, H)
        m2 = min(self.modes2, x_ft.shape[-1])
        out_ft = torch.zeros(B, self.out_c, H, x_ft.shape[-1], dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :m1, :m2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], self.w1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], self.w2[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(H, W), dim=(-2, -1))


class FNO1d(nn.Module):
    """Operator: (B, N, in_ch) -> (B, N, out_ch). Grid-based, resolution-agnostic."""
    def __init__(self, in_ch=1, out_ch=1, width=32, modes=16, n_layers=4):
        super().__init__()
        self.lift = nn.Linear(in_ch + 1, width)  # +1 for coordinate channel
        self.spectral = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(n_layers)])
        self.local = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        self.proj = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, out_ch))

    def forward(self, x):  # x: (B, N, in_ch)
        B, N, _ = x.shape
        grid = torch.linspace(0, 1, N, device=x.device).reshape(1, N, 1).repeat(B, 1, 1)
        h = self.lift(torch.cat([x, grid], dim=-1)).permute(0, 2, 1)  # (B, width, N)
        for sp, lc in zip(self.spectral, self.local):
            h = torch.nn.functional.gelu(sp(h) + lc(h))
        return self.proj(h.permute(0, 2, 1))


class FNO2d(nn.Module):
    """Operator: (B, H, W, in_ch) -> (B, H, W, out_ch)."""
    def __init__(self, in_ch=1, out_ch=1, width=24, modes=12, n_layers=4):
        super().__init__()
        self.lift = nn.Linear(in_ch + 2, width)
        self.spectral = nn.ModuleList([SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.proj = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, out_ch))

    def forward(self, x):  # x: (B, H, W, in_ch)
        B, H, W, _ = x.shape
        gx = torch.linspace(0, 1, H, device=x.device).reshape(1, H, 1, 1).repeat(B, 1, W, 1)
        gy = torch.linspace(0, 1, W, device=x.device).reshape(1, 1, W, 1).repeat(B, H, 1, 1)
        h = self.lift(torch.cat([x, gx, gy], dim=-1)).permute(0, 3, 1, 2)  # (B, width, H, W)
        for sp, lc in zip(self.spectral, self.local):
            h = torch.nn.functional.gelu(sp(h) + lc(h))
        return self.proj(h.permute(0, 2, 3, 1))
