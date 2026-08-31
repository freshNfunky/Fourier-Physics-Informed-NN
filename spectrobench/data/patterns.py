"""Track B data: pattern matching, to test the thesis that pattern matching is
"indirectly Fourier".

Two grounded sub-tasks:

1. Grating frequency regression / classification. Each image is a 2D sinusoidal
   grating (plus optional noise) with a controlled spatial frequency. Predicting
   the frequency is a purely spectral question. A local CNN must integrate over a
   large receptive field to see a global periodic pattern; a Fourier-mixing model
   reads it off one transform. Sweeping the frequency toward Nyquist is the knob.

2. Template localization. A known template is placed at a random shift in a noisy
   field; the target is its location. This is *literally* cross-correlation, which
   by the correlation theorem is a multiplication in the Fourier domain. It makes
   the thesis quantitative: classical FFT correlation vs learned local vs learned
   spectral, accuracy vs SNR and vs compute.
"""
from __future__ import annotations
import torch
import math


def grating_dataset(batch, n=64, f_min=2.0, f_max=20.0, noise=0.1, device="cpu", seed=0):
    """2D gratings with random frequency/orientation/phase. Target = frequency.

    Returns img (batch, n, n, 1) and freq (batch, 1) normalized to [0, 1].
    """
    g = torch.Generator(device=device); g.manual_seed(seed)
    f = f_min + (f_max - f_min) * torch.rand(batch, 1, 1, generator=g, device=device)
    theta = math.pi * torch.rand(batch, 1, 1, generator=g, device=device)
    phase = 2 * math.pi * torch.rand(batch, 1, 1, generator=g, device=device)
    xs = torch.linspace(0, 1, n, device=device)
    X, Y = torch.meshgrid(xs, xs, indexing="ij")
    X = X.unsqueeze(0); Y = Y.unsqueeze(0)
    arg = 2 * math.pi * f * (X * torch.cos(theta) + Y * torch.sin(theta)) + phase
    img = torch.sin(arg) + noise * torch.randn(batch, n, n, generator=g, device=device)
    freq = (f.reshape(batch, 1) - f_min) / (f_max - f_min)
    return img.unsqueeze(-1), freq


def template_localization_dataset(batch, n=48, tsize=9, snr=1.0, device="cpu", seed=0):
    """Place a fixed Gaussian-bump template at a random location in noise.

    Returns field (batch, n, n, 1), target_xy (batch, 2) in [0,1], and the
    template (tsize, tsize) so a classical correlation baseline can use it.
    """
    g = torch.Generator(device=device); g.manual_seed(seed)
    ts = torch.linspace(-1, 1, tsize, device=device)
    TX, TY = torch.meshgrid(ts, ts, indexing="ij")
    template = torch.exp(-(TX ** 2 + TY ** 2) / 0.3)
    field = (1.0 / max(snr, 1e-6)) * torch.randn(batch, n, n, generator=g, device=device)
    cx = torch.randint(tsize, n - tsize, (batch,), generator=g, device=device)
    cy = torch.randint(tsize, n - tsize, (batch,), generator=g, device=device)
    h = tsize // 2
    for b in range(batch):
        field[b, cx[b] - h:cx[b] + h + 1, cy[b] - h:cy[b] + h + 1] += template
    target = torch.stack([cx.float() / n, cy.float() / n], dim=-1)
    return field.unsqueeze(-1), target, template


def fft_correlation_localize(field, template):
    """Classical baseline: locate the template by FFT cross-correlation.

    Demonstrates the thesis directly: the 'pattern matcher' is a multiply in the
    frequency domain. Returns predicted (x, y) in [0, 1] per sample.
    """
    B, H, W, _ = field.shape
    f = field.squeeze(-1)
    t = torch.zeros(H, W, device=field.device)
    th, tw = template.shape
    t[:th, :tw] = template
    corr = torch.fft.irfft2(torch.fft.rfft2(f) * torch.conj(torch.fft.rfft2(t)[None]), s=(H, W))
    idx = corr.reshape(B, -1).argmax(dim=-1)
    x = (idx // W).float() / H
    y = (idx % W).float() / W
    # correlation peak sits at the template origin; shift to its center
    x = (x + (th // 2) / H) % 1.0
    y = (y + (tw // 2) / W) % 1.0
    return torch.stack([x, y], dim=-1)
