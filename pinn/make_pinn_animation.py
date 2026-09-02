#!/usr/bin/env python3
"""Intuitive animation: watch the Fourier-feature PINN catch the fine detail that
the plain PINN stays blind to.

Left panel: the reconstructed function u(x) evolving during training. The target
is a smooth wave with a fine high-frequency ripple riding on it. The Fourier-
feature net grows the ripple within a few hundred steps; the plain net stays a
smooth low-frequency curve and never gets there (spectral bias).
Right panel: the frequency content, as animated bars. The target has a tall bar
at the high wavenumber; the Fourier net fills it, the plain net does not.

    python pinn/make_pinn_animation.py          # writes gif + mp4 + preview png

The message is meant to be graspable in one glance, the way an explainer video is.
"""
import os, sys, math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pinn_spectral_bias import CoordNet

torch.manual_seed(0)
M, A = 14, 0.30                      # high mode + amplitude, tuned to be clearly visible
C_PLAIN, C_FOUR, C_TGT = "#D55E00", "#0072B2", "0.55"


def u_star(x): return torch.sin(2 * math.pi * x) + A * torch.sin(2 * math.pi * M * x)
def g_src(x):  return 2 * math.pi * torch.cos(2 * math.pi * x) + A * 2 * math.pi * M * torch.cos(2 * math.pi * M * x)


def train_capture(model, steps=2600, n_f=1024, lr=3e-3, frames=52, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    xg = torch.linspace(0, 1, 400 + 1)[:-1]
    x_anchor = torch.zeros(1)
    # denser capture early, where the interesting dynamics happen
    cps = sorted(set(int(round(s)) for s in np.unique(np.concatenate([
        np.linspace(0, steps * 0.25, frames // 2), np.linspace(steps * 0.25, steps - 1, frames // 2)]))))
    snaps, snap_steps = [], []
    ci = 0
    for s in range(steps):
        xf = torch.rand(n_f)
        opt.zero_grad()
        x = xf.clone().requires_grad_(True)
        u = model(x)
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        res = u_x - g_src(xf)
        anchor = model(x_anchor) - u_star(x_anchor)
        (res ** 2).mean().add(10.0 * (anchor ** 2).mean()).backward()
        opt.step(); sched.step()
        if ci < len(cps) and s == cps[ci]:
            with torch.no_grad():
                snaps.append(model(xg).numpy())
            snap_steps.append(s); ci += 1
    return xg.numpy(), snaps, snap_steps


def spectrum(y, kmax=18):
    f = np.abs(np.fft.rfft(y) / len(y))
    return f[:kmax + 1]


def main():
    xg, snap_p, steps_p = train_capture(CoordNet(modes=1, width=64), seed=0)
    xg, snap_f, steps_f = train_capture(CoordNet(modes=16, width=64), seed=0)
    n = min(len(snap_p), len(snap_f))
    snap_p, snap_f, steps = snap_p[:n], snap_f[:n], steps_p[:n]
    tgt = u_star(torch.tensor(xg)).numpy()
    tgt_spec = spectrum(tgt)
    ks = np.arange(len(tgt_spec))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.2, 3.3))
    fig.suptitle("Watch the Fourier network catch the fine detail the plain one misses",
                 fontsize=11, y=0.99)

    # left: function
    axL.plot(xg, tgt, color=C_TGT, lw=3, label="target", zorder=1)
    (lp,) = axL.plot([], [], color=C_PLAIN, lw=1.8, label="plain PINN")
    (lf,) = axL.plot([], [], color=C_FOUR, lw=1.8, label="Fourier PINN")
    axL.set_xlim(0, 1); axL.set_ylim(tgt.min() - 0.25, tgt.max() + 0.25)
    axL.set_xlabel("x"); axL.set_ylabel("u(x)"); axL.set_title("reconstruction")
    axL.legend(loc="upper right", frameon=False, fontsize=8)
    step_txt = axL.text(0.02, 0.04, "", transform=axL.transAxes, fontsize=9, color="0.3")

    # right: spectrum bars
    w = 0.4
    axR.bar(ks, tgt_spec, width=0.8, color=C_TGT, alpha=0.35, label="target", zorder=1)
    bp = axR.bar(ks - w / 2, np.zeros_like(tgt_spec), width=w, color=C_PLAIN, label="plain")
    bf = axR.bar(ks + w / 2, np.zeros_like(tgt_spec), width=w, color=C_FOUR, label="Fourier")
    axR.axvline(M, color="0.6", lw=0.8, ls=":")
    axR.text(M, tgt_spec.max() * 1.02, f"fine detail (k={M})", color="0.4", fontsize=8, ha="center", va="bottom")
    axR.set_xlim(-0.6, len(ks) - 0.4); axR.set_ylim(0, tgt_spec.max() * 1.2)
    axR.set_xlabel("wavenumber k"); axR.set_ylabel("|amplitude|"); axR.set_title("frequency content")
    axR.legend(loc="center", frameon=False, fontsize=8)

    def frame(i):
        lp.set_data(xg, snap_p[i]); lf.set_data(xg, snap_f[i])
        sp, sf = spectrum(snap_p[i]), spectrum(snap_f[i])
        for b, h in zip(bp, sp): b.set_height(h)
        for b, h in zip(bf, sf): b.set_height(h)
        step_txt.set_text(f"training step {steps[i]}")
        return [lp, lf, *bp, *bf, step_txt]

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    # hold the last frame a moment by repeating it
    order = list(range(n)) + [n - 1] * 8
    anim = FuncAnimation(fig, frame, frames=order, blit=False)
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "media")
    os.makedirs(d, exist_ok=True)
    gif = os.path.join(d, "spectral_bias.gif")
    anim.save(gif, writer=PillowWriter(fps=12))
    try:
        anim.save(gif.replace(".gif", ".mp4"), writer="ffmpeg", fps=12, dpi=130)
    except Exception as e:
        print("mp4 skipped:", e)
    frame(n - 1); fig.savefig(os.path.join(d, "spectral_bias_final.png"), dpi=130)
    print("wrote", gif, "(+ mp4, final png)")
    print(f"final plain err@k={M}:", f"{abs(spectrum(snap_p[-1])[M]-tgt_spec[M]):.4f}",
          f"| fourier err@k={M}:", f"{abs(spectrum(snap_f[-1])[M]-tgt_spec[M]):.4f}",
          f"| true A={A}")


if __name__ == "__main__":
    main()
