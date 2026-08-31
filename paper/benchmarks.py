#!/usr/bin/env python3
"""Full benchmark runner that produces the numbers documented in the whitepaper.

Runs Track A (Burgers viscosity sweep, FNO vs CNN at matched budget, multiple
seeds: in-distribution error, zero-shot 2x-resolution error, top-band absolute
spectral error, log-spectral distance) and Track B (template localization by FFT
cross-correlation across SNR). Writes paper/results.json.

CPU-runnable. Deliberately a small, reproducible scale -- see the whitepaper's
limitations section. Scale up seeds / steps / n_train for a heavier run.

    python paper/benchmarks.py
"""
import os, sys, json, time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spectrobench.data.pde import burgers_dataset
from spectrobench.data.patterns import template_localization_dataset, fft_correlation_localize
from spectrobench.models.fno import FNO1d
from spectrobench.models.conv import CNN1d
from spectrobench.budget import count_params, match_width
from spectrobench.train import train_supervised, evaluate_field

torch.set_num_threads(max(1, os.cpu_count() or 1))

CFG = dict(
    nus=[3e-2, 1e-2, 5e-3, 2e-3],
    N=128, n_train=300, n_test=60,
    steps=400, seeds=3, target_params=45000, solver_steps=150, n_bands=6,
)


def mean_std(xs):
    t = torch.tensor(xs)
    return [t.mean().item(), (t.std().item() if len(xs) > 1 else 0.0)]


def build(name, target):
    if name == "FNO1d":
        return match_width(lambda w: FNO1d(width=w, modes=16, n_layers=4), target)
    return match_width(lambda w: CNN1d(width=w, n_layers=6), target)


def track_a():
    c = CFG
    results = {}
    for nu in c["nus"]:
        x_tr, y_tr = burgers_dataset(c["n_train"], n=c["N"], nu=nu, n_steps=c["solver_steps"], seed=1)
        x_te, y_te = burgers_dataset(c["n_test"], n=c["N"], nu=nu, n_steps=c["solver_steps"], seed=99)
        x_hi, y_hi = burgers_dataset(c["n_test"], n=2 * c["N"], nu=nu, n_steps=c["solver_steps"], seed=99)
        cell = {}
        for name in ["FNO1d", "CNN1d"]:
            rin, rhi, tband, lspec = [], [], [], []
            for s in range(c["seeds"]):
                m = build(name, c["target_params"])
                m = train_supervised(m, x_tr, y_tr, steps=c["steps"], bs=16, seed=s)
                mi = evaluate_field(m, x_te, y_te, n_bands=c["n_bands"])
                mh = evaluate_field(m, x_hi, y_hi, n_bands=c["n_bands"])
                rin.append(mi["rel_l2"]); rhi.append(mh["rel_l2"])
                # highest band whose target energy fraction is non-negligible
                ef = mi["band_energy_frac"]; ab = mi["band_abs"]
                top = max(i for i, e in enumerate(ef) if e > 1e-4)
                tband.append(ab[top]); lspec.append(mi["log_spec_dist"])
            cell[name] = dict(params=count_params(build(name, c["target_params"])),
                              rel_l2_in=mean_std(rin), rel_l2_2x=mean_std(rhi),
                              top_band_abs=mean_std(tband), log_spec=mean_std(lspec))
        results[f"{nu:.0e}"] = cell
        fi = cell["FNO1d"]; ci = cell["CNN1d"]
        print(f"  nu={nu:.0e}  FNO in={fi['rel_l2_in'][0]:.4f} 2x={fi['rel_l2_2x'][0]:.4f} | "
              f"CNN in={ci['rel_l2_in'][0]:.4f} 2x={ci['rel_l2_2x'][0]:.4f}", flush=True)
    return results


def track_b():
    out = {}
    for snr in [4.0, 2.0, 1.0, 0.5, 0.25]:
        field, target, template = template_localization_dataset(256, n=48, snr=snr, seed=7)
        pred = fft_correlation_localize(field, template)
        err = (pred - target).norm(dim=-1).mean().item()
        base = (torch.full_like(target, 0.5) - target).norm(dim=-1).mean().item()
        out[f"{snr}"] = dict(fft_corr=err, center_baseline=base)
        print(f"  SNR={snr:>4}  fft_corr={err:.3f}  center={base:.3f}", flush=True)
    return out


if __name__ == "__main__":
    t0 = time.time()
    print("Track A (Burgers viscosity sweep)...", flush=True)
    a = track_a()
    print("Track B (template localization vs SNR)...", flush=True)
    b = track_b()
    out = dict(config=CFG, track_a=a, track_b=b, seconds=round(time.time() - t0, 1))
    with open(os.path.join(os.path.dirname(__file__), "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone in {out['seconds']}s -> paper/results.json", flush=True)
