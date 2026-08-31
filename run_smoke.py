#!/usr/bin/env python3
"""End-to-end smoke test for SpectroBench.

Runs a *tiny* slice of both tracks so you can confirm the pipeline works on a CPU
in a minute or two, and see the headline metrics in action. It is NOT the real
benchmark run (budgets and dataset sizes are deliberately small) -- see
run_experiment.py and SPEC.md for the controlled protocol.

    python run_smoke.py
"""
import torch
from spectrobench.data.pde import burgers_dataset
from spectrobench.data.patterns import template_localization_dataset, fft_correlation_localize
from spectrobench.models.fno import FNO1d
from spectrobench.models.conv import CNN1d
from spectrobench.budget import count_params
from spectrobench.train import train_supervised, evaluate_field

torch.manual_seed(0)
DEV = "cpu"


def fmt_bands(b):
    return "[" + ", ".join(f"{x:.2f}" for x in b) + "]"


def track_a():
    print("\n=== Track A: Burgers operator learning (steep fronts, nu=6e-3) ===")
    N, NU = 128, 6e-3
    x_tr, y_tr = burgers_dataset(200, n=N, nu=NU, n_steps=120, seed=1, device=DEV)
    x_te, y_te = burgers_dataset(40, n=N, nu=NU, n_steps=120, seed=99, device=DEV)
    # zero-shot resolution generalization: same physics, finer grid
    x_hi, y_hi = burgers_dataset(40, n=2 * N, nu=NU, n_steps=120, seed=99, device=DEV)

    fno = FNO1d(width=24, modes=16, n_layers=4)
    cnn = CNN1d(width=48, n_layers=6)
    print(f"  params: FNO1d={count_params(fno):,}   CNN1d={count_params(cnn):,}")

    for name, model in [("FNO1d", fno), ("CNN1d", cnn)]:
        model = train_supervised(model, x_tr, y_tr, steps=150, bs=16, device=DEV, seed=0)
        m_in = evaluate_field(model, x_te, y_te, n_bands=6, device=DEV)
        m_hi = evaluate_field(model, x_hi, y_hi, n_bands=6, device=DEV)
        print(f"  {name:6s} in-dist  rel_l2={m_in['rel_l2']:.4f}  "
              f"log_spec={m_in['log_spec_dist']:.3f}")
        print(f"  {name:6s}   band_abs (robust)   ={fmt_bands(m_in['band_abs'])}")
        print(f"  {name:6s}   band_energy_frac    ={fmt_bands(m_in['band_energy_frac'])}")
        print(f"  {name:6s} 2x-res   rel_l2={m_hi['rel_l2']:.4f}  "
              f"(zero-shot resolution generalization)")


def track_b():
    print("\n=== Track B: template localization = FFT cross-correlation ===")
    print("  (pattern matching is a multiply in the Fourier domain)")
    for snr in [2.0, 1.0, 0.5, 0.25]:
        field, target, template = template_localization_dataset(64, n=48, snr=snr, seed=7, device=DEV)
        pred = fft_correlation_localize(field, template)
        err = (pred - target).norm(dim=-1).mean().item()
        # naive baseline: always guess image center
        base = (torch.full_like(target, 0.5) - target).norm(dim=-1).mean().item()
        print(f"  SNR={snr:4.2f}  FFT-corr loc-error={err:.3f}   center-guess baseline={base:.3f}")


if __name__ == "__main__":
    track_a()
    track_b()
    print("\nSmoke test complete.")
