#!/usr/bin/env python3
"""PINN demo: a Fourier-feature ansatz beats the spectral bias of a PINN.

Spectral bias (Rahaman 2019) is a *rate* phenomenon: a plain network learns low
frequencies fast and high frequencies exponentially slowly. It is not that a
plain tanh network cannot represent a high mode (its nonlinearity synthesizes
harmonics), it is that reaching it takes far longer. So the honest demonstration
is a convergence comparison at a fixed budget.

We pose a clean, balanced physics-informed problem: recover u_theta(x) from the
first-order residual of the ODE

    u'(x) = g(x)   on x in [0,1), periodic,   with   u(0) = 0,

for the multi-scale target

    u*(x) = sin(2 pi x) + (1/m) sin(2 pi m x),        m = 15,

so g(x) = u*'(x) = 2 pi cos(2 pi x) + 2 pi cos(2 pi m x). The 1/m amplitude makes
the two source modes weigh *equally* in the residual (a first-derivative operator
scales as k, so 1/m cancels the m), which removes the loss imbalance that a
Poisson (k^2) operator would introduce. Training uses only the residual and one
anchor point, no solution labels in between.

Two backbones at matched budget:
  - "plain":   base periodic encoding  [sin 2pi x, cos 2pi x]        (spectral-bias baseline)
  - "fourier": multi-frequency encoding [sin 2pi k x, cos 2pi k x, k=1..K]

Both fit the dominant low mode; the question is the high mode m=15. We track the
per-mode error over training. The plain net crawls toward m=15; the Fourier net
reaches it almost immediately (Tancik 2020; Wang & Perdikaris 2021, NTK).

    python pinn/pinn_spectral_bias.py --plot

Small CPU demo (about a minute). First increment of the separate PINN track.
"""
import os, math, time, argparse
import torch
import torch.nn as nn

torch.manual_seed(0)
M = 15
A = 1.0 / M


def u_star(x):
    return torch.sin(2 * math.pi * x) + A * torch.sin(2 * math.pi * M * x)


def g_src(x):  # u*'(x): both modes carry amplitude 2*pi -> balanced residual
    return 2 * math.pi * torch.cos(2 * math.pi * x) + 2 * math.pi * torch.cos(2 * math.pi * M * x)


class CoordNet(nn.Module):
    """Periodic coordinate MLP. modes=1 -> plain base encoding; modes=K -> Fourier features."""
    def __init__(self, modes=1, width=64, depth=4):
        super().__init__()
        self.modes = modes
        layers, d = [], 2 * modes
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]; d = width
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def encode(self, x):
        ks = torch.arange(1, self.modes + 1, device=x.device, dtype=x.dtype)
        ang = 2 * math.pi * x[:, None] * ks[None, :]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)

    def forward(self, x):
        return self.net(self.encode(x)).squeeze(-1)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def first_derivative(model, x):
    x = x.clone().requires_grad_(True)
    u = model(x)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    return u_x


@torch.no_grad()
def mode_error(model, xg, uref, ks=(1, M)):
    fp = torch.fft.rfft(model(xg)) / xg.numel()
    fr = torch.fft.rfft(uref) / xg.numel()
    return {k: (fp[k] - fr[k]).abs().item() for k in ks}


def train(model, steps=5000, n_f=1024, lr=3e-3, seed=0, track_every=200):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    x_anchor = torch.zeros(1)
    xg = torch.linspace(0, 1, 256 + 1)[:-1]
    uref = u_star(xg)
    track = []  # (step, err_k1, err_kM)
    for s in range(steps):
        xf = torch.rand(n_f)
        opt.zero_grad()
        res = first_derivative(model, xf) - g_src(xf)
        anchor = model(x_anchor) - u_star(x_anchor)
        loss = (res ** 2).mean() + 10.0 * (anchor ** 2).mean()
        loss.backward(); opt.step(); sched.step()
        if s % track_every == 0 or s == steps - 1:
            e = mode_error(model, xg, uref)
            track.append((s, e[1], e[M]))
    return track


def run(steps=5000, modes=16, width=64):
    xg = torch.linspace(0, 1, 256 + 1)[:-1]
    uref = u_star(xg)
    out = {}
    t0 = time.time()
    for name, m in [("plain", 1), ("fourier", modes)]:
        net = CoordNet(modes=m, width=width)
        track = train(net, steps=steps)
        with torch.no_grad():
            pred = net(xg)
        rl = (torch.linalg.norm(pred - uref) / torch.linalg.norm(uref)).item()
        out[name] = dict(pred=pred, rel_l2=rl, params=count_params(net), track=track)
        e1, eM = track[-1][1], track[-1][2]
        print(f"{name:8s} params={count_params(net):5d}  rel_l2={rl:.4f}  "
              f"final err@k=1={e1:.5f}  err@k={M}={eM:.5f}  (true A={A:.4f})")
    print(f"[{time.time()-t0:.1f}s]")
    return xg, uref, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--modes", type=int, default=16)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    xg, uref, out = run(args.steps, args.modes, args.width)
    if args.plot:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from make_pinn_figure import plot
        plot(xg, uref, out, M, A)
