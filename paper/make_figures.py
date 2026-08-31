#!/usr/bin/env python3
"""Generate the whitepaper result figures from paper/results.json.

Produces paper/figs/results.pdf, a full-width three panel figure:
  (a) in-distribution vs zero-shot 2x-resolution relative L2 across viscosity,
  (b) resolution degradation ratio (2x / in) across viscosity,
  (c) Track B template localization error vs SNR.
Vector PDF, colorblind-safe (Wong) palette, fonts embedded for LaTeX.

    python paper/make_figures.py
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 200,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
})
C_FNO, C_CNN = "#0072B2", "#D55E00"  # Wong blue / vermillion

nus = sorted([float(k) for k in R["track_a"].keys()])
def col(model, field, i):
    return R["track_a"][f"{nus[i]:.0e}"][model][field]

fno_in = [col("FNO1d", "rel_l2_in", i)[0] for i in range(len(nus))]
fno_in_s = [col("FNO1d", "rel_l2_in", i)[1] for i in range(len(nus))]
fno_2x = [col("FNO1d", "rel_l2_2x", i)[0] for i in range(len(nus))]
fno_2x_s = [col("FNO1d", "rel_l2_2x", i)[1] for i in range(len(nus))]
cnn_in = [col("CNN1d", "rel_l2_in", i)[0] for i in range(len(nus))]
cnn_in_s = [col("CNN1d", "rel_l2_in", i)[1] for i in range(len(nus))]
cnn_2x = [col("CNN1d", "rel_l2_2x", i)[0] for i in range(len(nus))]
cnn_2x_s = [col("CNN1d", "rel_l2_2x", i)[1] for i in range(len(nus))]

fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.25))

# (a) in vs 2x
a = ax[0]
a.errorbar(nus, fno_in, yerr=fno_in_s, color=C_FNO, marker="o", ms=4, lw=1.4, label="FNO in-dist")
a.errorbar(nus, fno_2x, yerr=fno_2x_s, color=C_FNO, marker="s", ms=4, lw=1.4, ls="--", label="FNO 2x-res")
a.errorbar(nus, cnn_in, yerr=cnn_in_s, color=C_CNN, marker="o", ms=4, lw=1.4, label="CNN in-dist")
a.errorbar(nus, cnn_2x, yerr=cnn_2x_s, color=C_CNN, marker="s", ms=4, lw=1.4, ls="--", label="CNN 2x-res")
a.set_xscale("log"); a.set_xlabel(r"viscosity $\nu$"); a.set_ylabel(r"relative $L^2$ error")
a.set_title("(a) accuracy: in-dist vs 2x resolution")
a.legend(frameon=False, loc="center left", fontsize=6.5)
a.invert_xaxis()

# (b) degradation ratio
b = ax[1]
b.plot(nus, [fno_2x[i]/fno_in[i] for i in range(len(nus))], color=C_FNO, marker="o", ms=4, lw=1.4, label="FNO")
b.plot(nus, [cnn_2x[i]/cnn_in[i] for i in range(len(nus))], color=C_CNN, marker="s", ms=4, lw=1.4, label="CNN")
b.axhline(1.0, color="k", lw=0.7, ls=":")
b.set_xscale("log"); b.set_yscale("log"); b.set_xlabel(r"viscosity $\nu$")
b.set_ylabel(r"degradation ratio  $2\times$ / in")
b.set_title("(b) resolution fragility")
b.legend(frameon=False, loc="center right"); b.invert_xaxis()

# (c) Track B
c = ax[2]
snrs = sorted([float(k) for k in R["track_b"].keys()])
corr = [R["track_b"][str(s)]["fft_corr"] for s in snrs]
base = R["track_b"][str(snrs[0])]["center_baseline"]
c.plot(snrs, corr, color="#009E73", marker="o", ms=4, lw=1.4, label="FFT correlation")
c.axhline(base, color="k", lw=0.9, ls="--", label="center-guess baseline")
c.axhline(1/48, color="0.5", lw=0.7, ls=":", label="1 pixel")
c.set_xscale("log"); c.set_xlabel("signal-to-noise ratio"); c.set_ylabel("localization error")
c.set_title("(c) pattern matching = FFT correlation")
c.legend(frameon=False, loc="upper right", fontsize=6.5)

fig.tight_layout(pad=0.4, w_pad=1.2)
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
out = os.path.join(HERE, "figs", "results.pdf")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
