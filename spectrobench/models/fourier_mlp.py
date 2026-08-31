"""Coordinate networks: Fourier-feature MLP vs plain MLP.

This is the second face of "Fourier in ML" and the one the video is about: the
Fourier *series / feature* basis as the input encoding of a coordinate network.
A plain MLP has a spectral bias (learns low frequencies first, Rahaman 2019); the
Fourier-feature encoding (Tancik 2020) front-loads a high-frequency basis and
reshapes the NTK so all frequencies are learned at comparable rates
(Wang & Perdikaris 2021). `sigma` is the sensitive bandwidth knob to sweep.

Used for the reconstruction / function-fitting and pattern tasks, where the
input is a coordinate grid rather than a full input field.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int, n_features: int = 128, sigma: float = 10.0):
        super().__init__()
        # Fixed random Gaussian frequencies (not trained), the classic Tancik map.
        B = torch.randn(in_dim, n_features) * sigma
        self.register_buffer("B", B)

    def forward(self, coords):  # (..., in_dim)
        proj = 2 * torch.pi * coords @ self.B
        return torch.cat([proj.sin(), proj.cos()], dim=-1)  # (..., 2*n_features)


class FourierFeatureMLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=1, n_features=128, sigma=10.0, width=128, depth=4):
        super().__init__()
        self.ff = FourierFeatures(in_dim, n_features, sigma)
        layers = [nn.Linear(2 * n_features, width), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.GELU()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        return self.net(self.ff(coords))


class PlainMLP(nn.Module):
    """Plain coordinate MLP: the spectral-bias baseline the Fourier features fix."""
    def __init__(self, in_dim=2, out_dim=1, width=256, depth=5):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.GELU()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        return self.net(coords)
