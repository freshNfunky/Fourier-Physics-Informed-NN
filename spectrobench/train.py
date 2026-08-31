"""Generic training / evaluation loop with a fixed, shared protocol.

Every model in a comparison cell sees the *same* optimizer, schedule, step
budget and data. That is what isolates architecture from training recipe. If you
later tune schedules per model, report it explicitly (see SPEC.md, threats to
validity).
"""
from __future__ import annotations
import torch
from .metrics import rel_l2, all_metrics


def train_supervised(model, x, y, steps=300, bs=16, lr=1e-3, device="cpu",
                     loss="rel_l2", seed=0, verbose=False):
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    n = x.shape[0]
    for step in range(steps):
        idx = torch.randint(0, n, (bs,), device=device)
        xb, yb = x[idx], y[idx]
        opt.zero_grad()
        pred = model(xb)
        if loss == "rel_l2":
            l = rel_l2(pred.squeeze(-1), yb.squeeze(-1)) if pred.ndim > 2 else \
                torch.nn.functional.mse_loss(pred, yb)
        else:
            l = torch.nn.functional.mse_loss(pred, yb)
        l.backward()
        opt.step(); sched.step()
        if verbose and step % max(1, steps // 5) == 0:
            print(f"    step {step:4d}  loss {l.item():.4e}")
    return model


@torch.no_grad()
def evaluate_field(model, x, y, n_bands=8, device="cpu"):
    """Full spectral metric bundle for a field-to-field operator task."""
    model.eval()
    pred = model(x.to(device)).cpu()
    return all_metrics(pred.squeeze(-1), y.squeeze(-1), n_bands=n_bands)


@torch.no_grad()
def evaluate_scalar(model, x, y, device="cpu"):
    model.eval()
    pred = model(x.to(device)).cpu()
    mse = torch.nn.functional.mse_loss(pred, y).item()
    mae = (pred - y).abs().mean().item()
    return {"mse": mse, "mae": mae}
