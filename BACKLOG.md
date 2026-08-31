# Backlog

## Next up: extend to the PINN domain
Add a physics-informed training track and benchmark whether the Fourier ansatz
improves PINN performance.

- Add a PINN training mode: minimize the PDE residual via autodiff (no or few
  solution labels), on the same Burgers / diffusion / advection tasks.
- Compare trial-function backbones at matched budget:
  plain-MLP PINN (spectral-bias baseline) vs Fourier-feature MLP (Tancik /
  Wang-Perdikaris) vs a physics-informed spectral operator (PINO-style).
- Hypothesis (H1a/H1c applied to PINNs): Fourier features fix the spectral-bias
  failure of plain PINNs on multi-scale / stiff PDEs and converge in fewer steps.
- Score with the existing metrics: band-resolved error + energy fraction,
  zero-shot resolution generalization, and the viscosity regime sweep.
- Watch the known caveat: differentiating a high-bandwidth Fourier basis inside
  the residual amplifies high modes (|k|^order); sweep the feature bandwidth sigma.

## Also open (from the paper's future-work section)
- Full nu-sweep with error bars to pin the regime boundary nu*.
- Learned Track B comparisons (grating-frequency regression, global phase).
- Data-efficiency curves (H1c).
- 2D Navier-Stokes for energy-spectrum fidelity (radial PSD, log-spectral dist).
- Ablations: FNO mode count, Fourier-feature bandwidth sigma.
