"""Spectral metrics for SpectroBench.

The headline idea: a single scalar error (MSE / relative L2) is dominated by the
low-frequency content, where most signal energy sits. It hides whether a model
reconstructs the high-frequency tail (edges, texture, turbulent cascade). Every
metric here is designed to expose the *frequency-resolved* story.

All functions accept torch tensors shaped (B, ...spatial) with no channel axis
(single scalar field per sample), unless noted. Batched, differentiable-safe.
"""
from __future__ import annotations
import torch


def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample relative L2 error, averaged over the batch.

    This is the standard operator-learning metric and the reference number.
    Deliberately reported *alongside* the band-resolved error, never alone.
    """
    B = pred.shape[0]
    p = pred.reshape(B, -1)
    t = target.reshape(B, -1)
    num = torch.linalg.norm(p - t, dim=1)
    den = torch.linalg.norm(t, dim=1).clamp_min(1e-12)
    return (num / den).mean()


def _radial_bins(shape, n_bands: int, device):
    """Return (bin_index_per_freq, n_bands) for an fftshift-ed spectrum of `shape`.

    Works for 1D and 2D. Radius is normalized so the Nyquist corner maps to ~1.
    """
    grids = []
    for n in shape:
        f = torch.fft.fftshift(torch.fft.fftfreq(n, d=1.0 / n, device=device))
        # fftfreq with d=1/n gives integer wavenumbers -n/2..n/2
        grids.append(f)
    mesh = torch.meshgrid(*grids, indexing="ij")
    radius = torch.sqrt(sum(g ** 2 for g in mesh))
    r_max = radius.max().clamp_min(1e-12)
    idx = torch.floor(radius / r_max * (n_bands - 1e-6)).long().clamp_(0, n_bands - 1)
    return idx, mesh


def band_resolved_error(pred: torch.Tensor, target: torch.Tensor, n_bands: int = 8):
    """Frequency-band-resolved error, the metric that decides H1a.

    Returns a dict with three (n_bands,) tensors:
      - 'relative': per-band ||F(err)|| / ||F(target)||. Sensitive but ILL-
        CONDITIONED where a band carries almost no target energy (relative error
        explodes on ~empty high bands); read it together with 'energy_frac'.
      - 'absolute': per-band ||F(err)|| normalized by the *total* target norm, so
        bands are comparable and empty bands stay near zero. The robust headline.
      - 'energy_frac': share of the target's spectral energy in each band, i.e.
        how much a band even matters / how trustworthy its relative error is.
    Band 0 = lowest frequencies, band n_bands-1 = highest.
    """
    dims = tuple(range(1, pred.ndim))
    spatial = pred.shape[1:]
    err_ft = torch.fft.fftshift(torch.fft.fftn(pred - target, dim=dims), dim=dims)
    tgt_ft = torch.fft.fftshift(torch.fft.fftn(target, dim=dims), dim=dims)
    idx, _ = _radial_bins(spatial, n_bands, pred.device)
    idx_flat = idx.reshape(-1)
    err_pow = (err_ft.abs() ** 2).reshape(pred.shape[0], -1)
    tgt_pow = (tgt_ft.abs() ** 2).reshape(pred.shape[0], -1)
    total_tgt = tgt_pow.sum(dim=1).sqrt().clamp_min(1e-12)
    total_energy = tgt_pow.sum(dim=1).clamp_min(1e-12)
    rel = torch.zeros(n_bands, device=pred.device)
    ab = torch.zeros(n_bands, device=pred.device)
    ef = torch.zeros(n_bands, device=pred.device)
    for b in range(n_bands):
        mask = (idx_flat == b)
        if mask.sum() == 0:
            continue
        e_num = err_pow[:, mask].sum(dim=1).sqrt()
        t_den = tgt_pow[:, mask].sum(dim=1).sqrt().clamp_min(1e-12)
        rel[b] = (e_num / t_den).mean()
        ab[b] = (e_num / total_tgt).mean()
        ef[b] = (tgt_pow[:, mask].sum(dim=1) / total_energy).mean()
    return {"relative": rel, "absolute": ab, "energy_frac": ef}


def radial_psd(field: torch.Tensor, n_bands: int = 32):
    """Radially averaged power spectral density E(k), averaged over the batch.

    The physical metric for turbulence: getting E(k) right means the energy
    cascade is right. Returns (n_bands,) mean power per radial band.
    """
    spatial = field.shape[1:]
    dims = tuple(range(1, field.ndim))
    ft = torch.fft.fftshift(torch.fft.fftn(field, dim=dims), dim=dims)
    power = (ft.abs() ** 2).reshape(field.shape[0], -1).mean(dim=0)
    idx, _ = _radial_bins(spatial, n_bands, field.device)
    idx_flat = idx.reshape(-1)
    out = torch.zeros(n_bands, device=field.device)
    for b in range(n_bands):
        mask = (idx_flat == b)
        if mask.sum() > 0:
            out[b] = power[mask].mean()
    return out


def log_spectral_distance(pred: torch.Tensor, target: torch.Tensor, n_bands: int = 32):
    """Mean |log E_pred(k) - log E_target(k)| over radial bands.

    A scale-invariant summary of how well the *shape* of the energy spectrum is
    matched, independent of the overall magnitude that rel_l2 already captures.
    """
    ep = radial_psd(pred, n_bands).clamp_min(1e-12)
    et = radial_psd(target, n_bands).clamp_min(1e-12)
    return (ep.log() - et.log()).abs().mean()


def all_metrics(pred: torch.Tensor, target: torch.Tensor, n_bands: int = 8) -> dict:
    """Convenience bundle used by the training / evaluation loop."""
    b = band_resolved_error(pred, target, n_bands)
    return {
        "rel_l2": rel_l2(pred, target).item(),
        "band_rel": b["relative"].tolist(),
        "band_abs": b["absolute"].tolist(),
        "band_energy_frac": b["energy_frac"].tolist(),
        "log_spec_dist": log_spectral_distance(pred, target).item(),
    }
