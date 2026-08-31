#!/usr/bin/env python3
"""Configurable driver for the controlled SpectroBench runs.

This is the fuller grid behind the smoke test: a viscosity sweep (H1d regime
boundary) and a resolution-generalization sweep (H1b), each at matched parameter
budget and averaged over seeds. Still CPU-runnable but slower than run_smoke.py;
scale N_TRAIN / SEEDS / STEPS up for a real run.

    python run_experiment.py

Results are printed as tables and saved to results.json.
"""
import json
import argparse
import torch

from spectrobench.data.pde import burgers_dataset
from spectrobench.models.fno import FNO1d
from spectrobench.models.conv import CNN1d
from spectrobench.budget import count_params, match_width
from spectrobench.train import train_supervised, evaluate_field


def build_matched(target_params):
    fno = match_width(lambda w: FNO1d(width=w, modes=16, n_layers=4), target_params)
    cnn = match_width(lambda w: CNN1d(width=w, n_layers=6), target_params)
    return {"FNO1d": fno, "CNN1d": cnn}


def run_cell(nu, n_train, n_test, N, steps, seeds, target_params, n_steps_solver):
    x_tr, y_tr = burgers_dataset(n_train, n=N, nu=nu, n_steps=n_steps_solver, seed=1)
    x_te, y_te = burgers_dataset(n_test, n=N, nu=nu, n_steps=n_steps_solver, seed=99)
    x_hi, y_hi = burgers_dataset(n_test, n=2 * N, nu=nu, n_steps=n_steps_solver, seed=99)
    out = {}
    for name in ["FNO1d", "CNN1d"]:
        rl_in, rl_hi = [], []
        for s in range(seeds):
            model = build_matched(target_params)[name]
            model = train_supervised(model, x_tr, y_tr, steps=steps, bs=16, seed=s)
            rl_in.append(evaluate_field(model, x_te, y_te, n_bands=6)["rel_l2"])
            rl_hi.append(evaluate_field(model, x_hi, y_hi, n_bands=6)["rel_l2"])
        def mean_std(xs):
            t = torch.tensor(xs)
            return (t.mean().item(), (t.std().item() if len(xs) > 1 else 0.0))
        out[name] = {
            "params": count_params(build_matched(target_params)[name]),
            "rel_l2_in": mean_std(rl_in),
            "rel_l2_2x": mean_std(rl_hi),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nus", type=float, nargs="+", default=[3e-2, 1e-2, 6e-3, 3e-3])
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--n_train", type=int, default=300)
    ap.add_argument("--n_test", type=int, default=60)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--params", type=int, default=45000)
    ap.add_argument("--solver_steps", type=int, default=150)
    args = ap.parse_args()

    print(f"Regime sweep over nu (H1d) and resolution generalization (H1b)")
    print(f"matched budget ~{args.params:,} params, {args.seeds} seeds\n")
    print(f"{'nu':>8} {'model':>6} {'rel_l2 in':>18} {'rel_l2 @2x':>18} {'2x/in':>7}")
    results = {}
    for nu in args.nus:
        cell = run_cell(nu, args.n_train, args.n_test, args.N, args.steps,
                        args.seeds, args.params, args.solver_steps)
        results[f"{nu:.0e}"] = cell
        for name, r in cell.items():
            mi, si = r["rel_l2_in"]; mh, sh = r["rel_l2_2x"]
            ratio = mh / max(mi, 1e-9)
            print(f"{nu:>8.0e} {name:>6} {mi:>8.4f}+-{si:<7.4f} "
                  f"{mh:>8.4f}+-{sh:<7.4f} {ratio:>6.2f}x")
        print()
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results.json")
    print("\nRead: a large '2x/in' ratio = the model fails to generalize to finer\n"
          "resolution. Expect FNO ~1 (flat), CNN >> 1, and the smooth-vs-sharp\n"
          "trend across nu to map the regime boundary (H1d).")


if __name__ == "__main__":
    main()
