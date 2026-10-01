#!/usr/bin/env python3
"""Causal ablation: is out-of-band recovery of m=26 caused by the presence of
harmonics that can SYNTHESIZE 26 via intermodulation? Hold the feature count
fixed at N=8 (identical parameter count and budget) and vary only WHICH
frequencies are embedded. Routes to 26 at low order: self-double of k=13
(2*13=26), pair sums 10+16, 11+15. A matched N=8 set with none of these should
fail while sets that have one should recover. Target u*(x)=sin(2pi x)+(1/26)sin(2pi 26 x),
first-order residual u'=g, one anchor."""
import math, json, time, numpy as np, torch, torch.nn as nn

M = 26; A = 1.0 / M; STEPS = 1800
u_star = lambda x: torch.sin(2*math.pi*x) + A*torch.sin(2*math.pi*M*x)
g_src = lambda x: 2*math.pi*torch.cos(2*math.pi*x) + 2*math.pi*torch.cos(2*math.pi*M*x)

SETS = {
    "plain base {1}":             [1],
    "K=16 full {1..16}":          list(range(1, 17)),
    "N8 low {1..8}":              [1, 2, 3, 4, 5, 6, 7, 8],
    "N8 low+13 (2x13=26)":        [1, 2, 3, 4, 5, 6, 7, 13],
    "N8 low+{10,16} (10+16)":     [1, 2, 3, 4, 5, 6, 10, 16],
    "N8 low+{11,15} (11+15)":     [1, 2, 3, 4, 5, 6, 11, 15],
    "N8 shifted {9..16} no 1-route": [9, 10, 11, 12, 13, 14, 15, 16],
}


class FreqNet(nn.Module):
    def __init__(self, freqs, width=64, depth=4):
        super().__init__()
        self.register_buffer("freqs", torch.tensor(freqs, dtype=torch.float32))
        L, d = [], 2*len(freqs)
        for _ in range(depth): L += [nn.Linear(d, width), nn.Tanh()]; d = width
        L += [nn.Linear(width, 1)]; self.net = nn.Sequential(*L)
    def forward(self, x):
        ang = 2*math.pi*x[:, None]*self.freqs[None, :]
        return self.net(torch.cat([torch.sin(ang), torch.cos(ang)], 1)).squeeze(-1)


def du(m, x):
    x = x.clone().requires_grad_(True); u = m(x)
    return torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]


def train(model, steps=STEPS):
    opt = torch.optim.Adam(model.parameters(), 3e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps); xa = torch.zeros(1)
    for _ in range(steps):
        xf = torch.rand(1024); opt.zero_grad()
        (( du(model, xf)-g_src(xf))**2).mean().add(10*((model(xa)-u_star(xa))**2).mean()).backward()
        opt.step(); sch.step()


@torch.no_grad()
def score(model):
    xg = torch.linspace(0, 1, 512+1)[:-1]
    fp = torch.fft.rfft(model(xg))/xg.numel(); fr = torch.fft.rfft(u_star(xg))/xg.numel()
    errM = (fp[M]-fr[M]).abs().item()
    rl = (torch.linalg.norm(model(xg)-u_star(xg))/torch.linalg.norm(u_star(xg))).item()
    return errM, rl


def nparams(m): return sum(p.numel() for p in m.parameters())


t0 = time.time(); rows = []
print(f"target fine mode m={M}, true amplitude 1/m={A:.4f}\n")
print(f"{'embedding':<30} {'N':>3} {'params':>7} {'route to 26':>12} {'errM@26':>9} {'relL2':>8}")
for name, fr in SETS.items():
    torch.manual_seed(1); net = FreqNet(fr); train(net); e, rl = score(net)
    # low-order route flag: self-double (13 in set) or a pair summing to 26
    s = set(fr); reachable = any((M-a) in s for a in s)  # self-double (13) or pair summing to 26
    route = "yes" if reachable else "no"
    rows.append(dict(name=name, N=len(fr), params=nparams(net), route=route, errM=e, relL2=rl))
    print(f"{name:<30} {len(fr):>3} {nparams(net):>7} {route:>12} {e:>9.4f} {rl:>8.4f}", flush=True)
json.dump(rows, open("/tmp/claude-0/-home-claude/6154dd51-51fa-5db9-826e-cd8edd2541f6/scratchpad/ablation_numbers.json", "w"), indent=1)
print(f"\n[{time.time()-t0:.0f}s]  N=8 rows are parameter-matched; only the embedded frequencies differ")
