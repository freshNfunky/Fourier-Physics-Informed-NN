"""Track A data: PDE operator-learning tasks spanning smooth -> shock.

The design intent: one dataset family with a single knob that moves the target
from smooth (Fourier's home turf) to near-discontinuous (Gibbs territory), so the
benchmark can *locate the crossover* rather than just declare a winner. For
viscous Burgers that knob is the viscosity nu: as nu -> 0 the solution forms
sharp shocks.

Everything is generated on the fly with torch FFT, no downloads, CPU-friendly.
"""
from __future__ import annotations
import torch


def gaussian_random_field_1d(batch, n, alpha=4.0, device="cpu", seed=None):
    """Random periodic fields with power spectrum ~ |k|^-alpha.

    Larger alpha => smoother field (energy concentrated at low k). alpha is a
    secondary smoothness knob for the pure reconstruction sub-task.
    """
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    k = torch.fft.rfftfreq(n, d=1.0 / n, device=device).clamp_min(1.0)
    amp = k ** (-alpha / 2.0)
    real = torch.randn(batch, k.numel(), generator=g, device=device)
    imag = torch.randn(batch, k.numel(), generator=g, device=device)
    spec = (real + 1j * imag) * amp
    u = torch.fft.irfft(spec, n=n, dim=-1)
    u = u - u.mean(dim=-1, keepdim=True)
    u = u / u.std(dim=-1, keepdim=True).clamp_min(1e-8)
    return u  # (batch, n)


def _make_nonlinear(k, dealias):
    ik = 1j * k

    def Nfun(v, n):
        u = torch.fft.irfft(v, n=n, dim=-1)
        f = torch.fft.rfft(u * u, dim=-1) * dealias
        return -0.5 * ik * f  # conservative flux form of -u u_x
    return Nfun


def burgers_dataset(batch, n=256, nu=1e-2, t_final=0.35, n_steps=200,
                    alpha=4.0, device="cpu", seed=0):
    """Operator-learning pairs (u0, u_T) for viscous Burgers at viscosity nu.

    Integrating-factor RK4 (Cox-Matthews) in Fourier space: the diffusion term is
    integrated *exactly* via exp(-nu k^2 dt), which removes the stiffness that
    blows up a naive explicit scheme; a 2/3 dealiasing mask kills the aliasing of
    the quadratic nonlinearity. Returns u0, uT each shaped (batch, n, 1).

    nu is the regime knob: nu ~ 5e-2 smooth, nu ~ 5e-3 steep resolved fronts,
    nu -> 1e-3 needs higher n to stay shock-resolved (that resolution demand is
    itself part of the regime-boundary finding in the spec).
    """
    u = gaussian_random_field_1d(batch, n, alpha=alpha, device=device, seed=seed)
    k = torch.fft.rfftfreq(n, d=1.0 / n, device=device)
    kmax = k.max()
    dealias = (k <= (2.0 / 3.0) * kmax).to(u.dtype)
    L = -nu * k ** 2
    dt = t_final / n_steps
    E = torch.exp(dt * L)
    E2 = torch.exp(dt / 2 * L)
    Nfun = _make_nonlinear(k, dealias)

    v = torch.fft.rfft(u, dim=-1)
    u0 = u.clone()
    for _ in range(n_steps):
        k1 = Nfun(v, n)
        k2 = Nfun(E2 * v + dt / 2 * E2 * k1, n)
        k3 = Nfun(E2 * v + dt / 2 * k2, n)
        k4 = Nfun(E * v + dt * E2 * k3, n)
        v = E * v + dt / 6.0 * (E * k1 + 2 * E2 * (k2 + k3) + k4)
    uT = torch.fft.irfft(v, n=n, dim=-1)
    return u0.unsqueeze(-1), uT.unsqueeze(-1)


def diffusion_dataset(batch, n=256, nu=5e-3, t_final=0.2, alpha=3.0, device="cpu", seed=0):
    """Smooth control task: pure heat equation u_t = nu u_xx (exact in Fourier).

    Deliberately smooth and low-frequency: the regime where Fourier should win
    only mildly and any large claimed advantage is suspect. A sanity anchor.
    """
    u0 = gaussian_random_field_1d(batch, n, alpha=alpha, device=device, seed=seed)
    k = torch.fft.rfftfreq(n, d=1.0 / n, device=device)
    decay = torch.exp(-nu * (k ** 2) * t_final)
    uT = torch.fft.irfft(decay * torch.fft.rfft(u0, dim=-1), n=n, dim=-1)
    return u0.unsqueeze(-1), uT.unsqueeze(-1)
