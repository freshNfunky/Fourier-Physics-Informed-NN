# Backlog

This repository is the **kernel**: an isolated, data-driven benchmark of the
Fourier-versus-local representation question. The PINN work below is a
**separate application** that builds on this kernel. It is framed and published
as its own PINN study, not merged into this paper.

## Next (separate study): Fourier ansatz inside a PINN
Question: does a Fourier representation improve physics-informed training over
the standard backbone?

- Mapping note: a classic PINN uses neither a CNN nor an FNO but a coordinate
  MLP trained on the PDE residual. The two Fourier directions to test are
  (i) a Fourier-feature MLP ansatz (mesh-free, aimed at the spectral-bias
  failure of plain PINNs) and (ii) a physics-informed spectral operator
  (PINO = FNO + residual loss).
- Baselines at matched budget: plain-MLP PINN, Fourier-feature MLP PINN, PINO.
- Caveat this kernel did NOT stress: the residual loss differentiates the
  network, so a high-bandwidth Fourier basis gets amplified by |k|^order and can
  destabilize training. Sweeping the feature bandwidth sigma is the open risk,
  and is why the PINN result does not follow automatically from this kernel.
- Reuse the kernel metrics (band-resolved error, resolution generalization,
  regime sweep) and add PINN-specific axes: convergence speed and residual
  stability.
- Deliver as a separate repo/paper that cites this kernel as its foundation.

## Also open (this kernel's own future work)
- Full nu-sweep with error bars to pin the regime boundary nu*.
- Learned Track B comparisons (grating-frequency regression, global phase).
- Data-efficiency curves (H1c).
- 2D Navier-Stokes for energy-spectrum fidelity (radial PSD, log-spectral dist).
- Ablations: FNO mode count, Fourier-feature bandwidth sigma.
