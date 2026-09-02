"""Figure for the PINN spectral-bias demo. Imported by pinn_spectral_bias.py --plot."""
import os, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "pdf.fonttype": 42, "figure.dpi": 200, "axes.grid": True,
    "grid.alpha": 0.3, "grid.linewidth": 0.5,
})
C_PLAIN, C_FOUR = "#D55E00", "#0072B2"  # plain = vermillion, fourier = blue


def plot(xg, uref, out, M, A):
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.5))

    # (a) convergence of the high-mode error vs training step
    a = ax[0]
    for name, c in [("plain", C_PLAIN), ("fourier", C_FOUR)]:
        tr = out[name]["track"]
        steps = [t[0] for t in tr]
        e1 = [t[1] for t in tr]
        eM = [t[2] for t in tr]
        a.plot(steps, eM, color=c, lw=1.6, label=f"{name}: mode m={M}")
        a.plot(steps, e1, color=c, lw=1.0, ls=":", alpha=0.8, label=f"{name}: mode 1")
    a.axhline(A, color="0.5", lw=0.7, ls="--")
    a.text(steps[-1], A * 1.15, f"true amplitude A=1/{M}", color="0.4",
           fontsize=6, ha="right", va="bottom")
    a.set_yscale("log"); a.set_xlabel("training step"); a.set_ylabel("per-mode error |amplitude|")
    a.set_title("(a) spectral bias: high mode converges slowly")
    a.legend(frameon=False, fontsize=6, loc="lower left")

    # (b) final per-wavenumber error
    b = ax[1]
    kmax = 20
    ks = torch.arange(kmax + 1)
    fr = torch.fft.rfft(uref) / xg.numel()
    width = 0.4
    for i, (name, c) in enumerate([("plain", C_PLAIN), ("fourier", C_FOUR)]):
        fp = torch.fft.rfft(out[name]["pred"]) / xg.numel()
        err = (fp[:kmax + 1] - fr[:kmax + 1]).abs()
        b.bar(ks.numpy() + (i - 0.5) * width, err.numpy(), width=width,
              color=c, label=name)
    b.axvline(M, color="0.6", lw=0.7, ls=":")
    b.text(M, b.get_ylim()[1], f"m={M}", color="0.4", fontsize=6, ha="center", va="top")
    b.set_yscale("log"); b.set_xlabel("wavenumber k"); b.set_ylabel("final error |amplitude|")
    b.set_title("(b) plain misses the high mode; Fourier does not")
    b.legend(frameon=False, fontsize=7, loc="upper right")

    fig.tight_layout(pad=0.4, w_pad=1.3)
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
    os.makedirs(d, exist_ok=True)
    out_pdf = os.path.join(d, "spectral_bias.pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.replace(".pdf", ".png"), bbox_inches="tight", dpi=220)
    print("wrote", out_pdf, "and .png")
