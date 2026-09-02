# PINN track: does Fourier beat the spectral bias?

This is the separate PINN application built on the kernel benchmark. First
increment: a small, clean demo of *why* a Fourier-feature ansatz helps a
physics-informed net.

## The demo

`pinn_spectral_bias.py` trains a physics-informed coordinate network on a
deliberately simple, balanced problem so the spectral-bias effect is isolated
rather than tangled with solver or loss-weighting artifacts.

Problem: recover u(x) from the first-order residual of

    u'(x) = g(x),   x in [0,1) periodic,   u(0) = 0,

for the multi-scale target u*(x) = sin(2 pi x) + (1/m) sin(2 pi m x), m = 15.
Training uses only the PDE residual (via autodiff) plus one anchor point, no
solution labels in between. Two backbones at matched budget:

- `plain`: base periodic encoding `[sin 2pi x, cos 2pi x]`, the spectral-bias baseline.
- `fourier`: multi-frequency encoding `[sin 2pi k x, cos 2pi k x, k=1..K]`.

Why this exact problem: a plain tanh network is *not* frequency-limited (its
nonlinearity synthesizes harmonics), so spectral bias is a convergence-*rate*
effect, not a representational wall. And a first-derivative operator scales as k,
so the 1/m amplitude makes both source modes weigh equally in the residual,
removing the loss imbalance a Poisson (k^2) operator would introduce. That leaves
a clean question: at a fixed budget, does each net reach the high mode m=15?

## Result (CPU, ~1 minute, one seed)

Both nets fit the dominant low mode. The high mode is where they split:

- `plain`: rel_l2 ~ 0.067, and the error at mode m=15 stalls at ~0.033, i.e. it
  captures only about half the high-frequency component within the budget.
- `fourier`: rel_l2 ~ 0.000, error at mode m=15 ~ 0; it reaches the high mode
  almost immediately.

`figs/spectral_bias.png` shows it: (a) the per-mode error over training, the
plain net's high mode flat/stalled while the Fourier net's decays steadily; (b)
the final per-wavenumber error, a single tall plain spike at k=15 against a flat
Fourier floor.

## Run

```
python pinn/pinn_spectral_bias.py --plot
```

## Relation to the rest

This confirms the mechanism the paper attributes to Fourier features (Tancik
2020; Wang & Perdikaris 2021, the eigenvector-bias / NTK argument) in a
physics-informed setting. Next per `../BACKLOG.md`: move to time-dependent,
compressible (finite-speed) Burgers with the residual-derivative stability sweep
(the sigma / mode bandwidth is a sharp knob once the residual differentiates the
basis), soft vs hard conservation constraints, and the kernel's band-resolved and
regime-boundary metrics.
