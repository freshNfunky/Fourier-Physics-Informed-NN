#!/usr/bin/env python3
"""Causal ablation, corrected diagnostic. The N=8 frequency-subset test showed
every parameter-matched embedding still recovers m=26: a deep tanh stack reaches
it through many redundant higher-order intermodulation routes, so removing one
low-order combination does not bite. The variable that actually gates recovery is
the MINIMUM intermodulation ORDER needed to synthesize 26 from the embedded band,
a recursion of the spectral-bias rate law. Here we hold capacity fixed at N=2
embeddings {1, q} and raise that minimum order by choosing q, then measure whether
recovery of m=26 degrades monotonically."""
import math, json, time, numpy as np, torch, torch.nn as nn
from collections import deque

M = 26; A = 1.0 / M; STEPS = 2200
u_star = lambda x: torch.sin(2*math.pi*x) + A*torch.sin(2*math.pi*M*x)
g_src = lambda x: 2*math.pi*torch.cos(2*math.pi*x) + 2*math.pi*torch.cos(2*math.pi*M*x)


def min_order(freqs, target=M, cap=40, bound=80):
    """fewest signed terms (+/- f, f in freqs) summing to target."""
    seen = {0: 0}; dq = deque([0])
    while dq:
        v = dq.popleft(); o = seen[v]
        if o >= cap: continue
        for f in freqs:
            for nv in (v+f, v-f):
                if abs(nv) <= bound and nv not in seen:
                    seen[nv] = o+1
                    if nv == target: return o+1
                    dq.append(nv)
    return seen.get(target, cap)


SETS = {
    "{1,13}":  [1, 13],
    "{1,9}":   [1, 9],
    "{1,7}":   [1, 7],
    "{1,5}":   [1, 5],
    "{1,4}":   [1, 4],
    "{1,3}":   [1, 3],
    "{1,2}":   [1, 2],
    "plain {1}": [1],
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
    return (fp[M]-fr[M]).abs().item(), (torch.linalg.norm(model(xg)-u_star(xg))/torch.linalg.norm(u_star(xg))).item()


t0 = time.time(); rows = []
print(f"target m={M}, true amplitude 1/m={A:.4f}\n")
print(f"{'embedding':<12} {'min order to 26':>16} {'errM@26':>9} {'relL2':>8}")
for name, fr in SETS.items():
    mo = min_order(fr)
    torch.manual_seed(1); net = FreqNet(fr); train(net); e, rl = score(net)
    rows.append(dict(name=name, min_order=mo, errM=e, relL2=rl))
    print(f"{name:<12} {mo:>16} {e:>9.4f} {rl:>8.4f}", flush=True)
json.dump(rows, open("/tmp/claude-0/-home-claude/6154dd51-51fa-5db9-826e-cd8edd2541f6/scratchpad/order_numbers.json", "w"), indent=1)
print(f"\n[{time.time()-t0:.0f}s]  N=2 rows parameter-matched; only q (hence the min order to reach 26) differs")
