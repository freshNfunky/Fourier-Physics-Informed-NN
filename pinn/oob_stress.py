#!/usr/bin/env python3
"""Out-of-band stress test for the Fourier-feature PINN.

Same model and training as pinn_spectral_bias.py. The Fourier embedding is held
fixed at K harmonics, [sin 2pi k x, cos 2pi k x] for k = 1..K. We then move the
target's fine mode m across the band edge (in band for m <= K, out of band for
m > K) and, for each m, compare three networks at a matched budget:

  plain          base periodic encoding only  (spectral-bias baseline)
  fourier(K)     fixed K-harmonic embedding    (the model under test)
  fourier(wide)  K widened to cover the mode   (control: does covering help?)

Problem: recover u_theta(x) from the first-order residual u'(x) = g(x) on
x in [0,1) periodic, with one anchor u(0) = 0, for the multi-scale target

    u*(x) = sin(2 pi x) + (1/m) sin(2 pi m x).

The 1/m amplitude makes both source modes weigh equally in the first-derivative
residual, so the demo isolates the representation from the operator's own
spectral weighting. Metric: |amplitude gap| at the fine mode (exactly the
quantity a reviewer asking about out-of-band forcing cares about), plus the
global relative L2.

    python pinn/oob_stress.py

CPU, a few minutes. Deterministic (fixed seeds).
"""
import math, time, torch, torch.nn as nn

K = 16          # fixed embedding bandwidth: harmonics k = 1..K
MODES = [8, 16, 20, 26]
STEPS = 1800


def targets(m):
    A = 1.0 / m
    u = lambda x: torch.sin(2 * math.pi * x) + A * torch.sin(2 * math.pi * m * x)
    g = lambda x: 2 * math.pi * torch.cos(2 * math.pi * x) + 2 * math.pi * torch.cos(2 * math.pi * m * x)
    return u, g, A


class CoordNet(nn.Module):
    def __init__(self, modes=1, width=64, depth=4):
        super().__init__(); self.modes = modes
        layers, d = [], 2 * modes
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]; d = width
        layers += [nn.Linear(width, 1)]; self.net = nn.Sequential(*layers)

    def encode(self, x):
        ks = torch.arange(1, self.modes + 1, dtype=x.dtype)
        ang = 2 * math.pi * x[:, None] * ks[None, :]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)

    def forward(self, x):
        return self.net(self.encode(x)).squeeze(-1)


def du(model, x):
    x = x.clone().requires_grad_(True); u = model(x)
    return torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]


def train(model, g, u, steps=STEPS, n_f=1024, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    xa = torch.zeros(1)
    for _ in range(steps):
        xf = torch.rand(n_f); opt.zero_grad()
        res = du(model, xf) - g(xf)
        anc = model(xa) - u(xa)
        (res.pow(2).mean() + 10.0 * anc.pow(2).mean()).backward()
        opt.step(); sch.step()


@torch.no_grad()
def scores(model, m, u):
    xg = torch.linspace(0, 1, 512 + 1)[:-1]
    fp = torch.fft.rfft(model(xg)) / xg.numel()
    fr = torch.fft.rfft(u(xg)) / xg.numel()
    errM = (fp[m] - fr[m]).abs().item()
    rl = (torch.linalg.norm(model(xg) - u(xg)) / torch.linalg.norm(u(xg))).item()
    return errM, rl


def fit(modes, m, u, g):
    torch.manual_seed(1); net = CoordNet(modes); train(net, g, u); return scores(net, m, u)


if __name__ == "__main__":
    print(f"fixed embedding band: harmonics k = 1..{K}   (mode m > {K} is OUT OF BAND)\n")
    hdr = f"{'m':>4} {'band':>5} | {'plain errM':>11} {'fourK errM':>11} {'wide errM':>10} | {'plain relL2':>12} {'fourK relL2':>12} | {'trueA':>7}"
    print(hdr); print("-" * len(hdr))
    t0 = time.time()
    for m in MODES:
        u, g, A = targets(m)
        ep, rp = fit(1, m, u, g)
        ef, rf = fit(K, m, u, g)
        ew, rw = fit(max(K, m + 2), m, u, g)
        band = "in" if m <= K else "OUT"
        print(f"{m:>4} {band:>5} | {ep:>11.5f} {ef:>11.5f} {ew:>10.5f} | {rp:>12.5f} {rf:>12.5f} | {A:>7.4f}", flush=True)
    print(f"\n[{time.time()-t0:.0f}s]  errM = |amplitude gap| at the fine mode; true amplitude 1/m; relL2 = global rel. L2")
