"""Compute / parameter budget matching.

A benchmark that lets one architecture have more capacity than another measures
capacity, not architecture. Every comparison in SpectroBench is run at a matched
parameter budget (within a tolerance) and a matched training-step budget. These
helpers make that explicit and auditable.
"""
from __future__ import annotations
from typing import Callable
import torch.nn as nn


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def match_width(build: Callable[[int], nn.Module], target_params: int,
                lo: int = 4, hi: int = 512, tol: float = 0.05) -> nn.Module:
    """Binary-search a width so build(width) lands within `tol` of target_params.

    `build` maps an integer width (hidden channels) to a model. Returns the model
    whose parameter count is closest to the target. Purely structural, no
    training, so it is cheap to call once per (model, budget) cell of the grid.
    """
    best = None
    best_gap = float("inf")
    a, b = lo, hi
    while a <= b:
        mid = (a + b) // 2
        m = build(mid)
        n = count_params(m)
        gap = abs(n - target_params) / target_params
        if gap < best_gap:
            best_gap, best = gap, m
        if gap <= tol:
            return m
        if n < target_params:
            a = mid + 1
        else:
            b = mid - 1
    return best
