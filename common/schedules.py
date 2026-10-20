"""Noise schedules (the "scheduler" of the assignment), written from scratch.

Notation (Ho et al., 2020):
    beta_t      noise added at step t
    alpha_t     = 1 - beta_t
    alpha_bar_t = prod_{s<=t} alpha_s     fraction of the *original signal variance* that is left at step t
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps          (closed-form forward process)
"""
from __future__ import annotations

import math

import torch


def linear_betas(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Original DDPM schedule. It is defined for T=1000, so we rescale it for other T."""
    scale = 1000 / T
    betas = torch.linspace(beta_start * scale, beta_end * scale, T, dtype=torch.float64)
    # FIX: for T <= 20 the rescaled beta reaches 1.0 (T=20) or more (T<20): alpha_bar becomes 0 / negative and
    # sqrt(1/alpha_bar) = inf -> NaN samples. Clip like the cosine schedule does (no effect for T >= 30, incl. T=200/1000).
    return betas.clamp(max=0.999)


def cosine_betas(T: int, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    """Nichol & Dhariwal (2021): define alpha_bar directly with a cosine so that the signal is destroyed
    smoothly instead of too fast at the end (linear schedule wastes many steps on almost-pure noise)."""
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(max=max_beta)


class NoiseSchedule:
    """Pre-computes every coefficient needed by training and sampling (cached on the right device)."""

    _TENSORS = ["betas", "alphas", "alpha_bar", "alpha_bar_prev", "sqrt_ab", "sqrt_1m_ab", "sqrt_recip_ab",
                "sqrt_recipm1_ab", "post_var", "post_logvar", "post_coef_x0", "post_coef_xt", "snr"]

    def __init__(self, name: str, T: int):
        if name == "linear":
            betas = linear_betas(T)
        elif name == "cosine":
            betas = cosine_betas(T)
        else:
            raise ValueError(f"unknown schedule {name}")
        self.name, self.T = name, T
        alphas = 1.0 - betas
        ab = torch.cumprod(alphas, 0)
        ab_prev = torch.cat([torch.ones(1, dtype=torch.float64), ab[:-1]])
        post_var = betas * (1 - ab_prev) / (1 - ab)           # variance of q(x_{t-1} | x_t, x_0)
        t = {
            "betas": betas, "alphas": alphas, "alpha_bar": ab, "alpha_bar_prev": ab_prev,
            "sqrt_ab": ab.sqrt(), "sqrt_1m_ab": (1 - ab).sqrt(),
            "sqrt_recip_ab": (1 / ab).sqrt(), "sqrt_recipm1_ab": (1 / ab - 1).sqrt(),
            "post_var": post_var, "post_logvar": post_var.clamp(min=1e-20).log(),
            "post_coef_x0": betas * ab_prev.sqrt() / (1 - ab),              # mean of q(x_{t-1}|x_t,x_0) = c0*x0 + ct*xt
            "post_coef_xt": (1 - ab_prev) * alphas.sqrt() / (1 - ab),
            "snr": ab / (1 - ab),                                            # signal-to-noise ratio at step t
        }
        for k, v in t.items():
            setattr(self, k, v.float())

    def to(self, device) -> "NoiseSchedule":
        for k in self._TENSORS:
            setattr(self, k, getattr(self, k).to(device))
        return self

    def extract(self, name: str, t: torch.Tensor, ndim: int) -> torch.Tensor:
        """Gather coefficient[t] and reshape to (B,1,1,...) so it broadcasts over images AND videos."""
        return getattr(self, name)[t].view(-1, *([1] * (ndim - 1)))