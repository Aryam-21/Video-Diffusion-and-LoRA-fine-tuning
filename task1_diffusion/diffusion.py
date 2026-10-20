"""The diffusion process itself: forward noising, training loss, and two reverse samplers (DDPM and DDIM).

Written from scratch on top of `common.schedules.NoiseSchedule`. Works unchanged for images (B,C,H,W) and
videos (B,C,T,H,W) because every coefficient is broadcast with `schedule.extract`.

Improvements over a bare-bones DDPM (each one is switchable and compared in `experiments.py`):
    * Min-SNR loss weighting (Hang et al., 2023)    -> faster, more balanced convergence
    * Classifier-free guidance (Ho & Salimans, 2022) -> samples follow the requested label much more closely
    * DDIM sampler (Song et al., 2021)               -> 20-50 steps instead of 1000 with similar quality
    * x_0 clipping during sampling                   -> prevents error accumulation / saturated pixels
"""
from __future__ import annotations

import torch

from common.schedules import NoiseSchedule


class GaussianDiffusion:
    def __init__(self, schedule: NoiseSchedule, num_classes: int = 0, cond_drop_prob: float = 0.1,
                 min_snr_gamma: float | None = None):
        self.sch = schedule
        self.T = schedule.T
        self.num_classes = num_classes
        self.cond_drop_prob = cond_drop_prob
        self.min_snr_gamma = min_snr_gamma

    # ------------------------------------------------------------------ forward process
    def q_sample(self, x0, t, noise=None):
        """x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps   (jump straight to step t in one shot)."""
        noise = torch.randn_like(x0) if noise is None else noise
        return (self.sch.extract("sqrt_ab", t, x0.ndim) * x0
                + self.sch.extract("sqrt_1m_ab", t, x0.ndim) * noise)

    def predict_x0(self, xt, t, eps):
        """Invert the forward equation:  x_0 = (x_t - sqrt(1-ab) eps) / sqrt(ab)."""
        return (self.sch.extract("sqrt_recip_ab", t, xt.ndim) * xt
                - self.sch.extract("sqrt_recipm1_ab", t, xt.ndim) * eps)

    # ------------------------------------------------------------------ training
    def training_losses(self, model, x0, y=None):
        """One training step's loss.  Returns (loss_to_backprop, plain_mse_for_logging).

        L_simple = E_{x0, t, eps} || eps - eps_theta(x_t, t) ||^2     with t ~ U{0..T-1}, eps ~ N(0, I)
        """
        B = x0.size(0)
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)

        if y is not None and self.num_classes > 0 and self.cond_drop_prob > 0:      # label dropout -> learns the
            drop = torch.rand(B, device=x0.device) < self.cond_drop_prob            # unconditional model too (for CFG)
            y = torch.where(drop, torch.full_like(y, self.num_classes), y)

        pred = model(xt, t, y).float()
        mse = (pred - noise.float()).square().flatten(1).mean(1)                    # per-sample MSE
        weight = torch.ones_like(mse)
        if self.min_snr_gamma:
            snr = self.sch.snr[t]
            weight = snr.clamp(max=self.min_snr_gamma) / snr
        return (weight * mse).mean(), mse.mean().detach()

    # ------------------------------------------------------------------ eps prediction (+ guidance)
    def model_eps(self, model, x, t, y=None, guidance: float = 1.0):
        """Classifier-free guidance:  eps = eps_uncond + w * (eps_cond - eps_uncond).   w=1 -> plain conditional."""
        if y is None or guidance == 1.0 or self.num_classes == 0:
            return model(x, t, y)
        null = torch.full_like(y, self.num_classes)
        e_c, e_u = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, null])).chunk(2)
        return e_u + guidance * (e_c - e_u)

    @staticmethod
    def _randn(shape, device, generator=None):
        return torch.randn(shape, device=device, generator=generator)

    # ------------------------------------------------------------------ DDPM (ancestral) sampling
    @torch.no_grad()
    def ddpm_sample(self, model, shape, y=None, guidance=1.0, generator=None, clip_x0=True):
        """Full T-step stochastic reverse process: x_{t-1} ~ N(posterior_mean(x_t, x0_hat), posterior_var)."""
        device = next(model.parameters()).device
        x = self._randn(shape, device, generator)
        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            eps = self.model_eps(model, x, t, y, guidance)
            x0 = self.predict_x0(x, t, eps)
            if clip_x0:
                x0 = x0.clamp(-1, 1)
            mean = self.sch.extract("post_coef_x0", t, x.ndim) * x0 + self.sch.extract("post_coef_xt", t, x.ndim) * x
            if i > 0:
                x = mean + (0.5 * self.sch.extract("post_logvar", t, x.ndim)).exp() * self._randn(shape, device, generator)
            else:
                x = mean
        return x

    # ------------------------------------------------------------------ DDIM sampling
    @torch.no_grad()
    def ddim_sample(self, model, shape, steps=50, eta=0.0, y=None, guidance=1.0, generator=None,
                    x_T=None, t_start=None, clip_x0=True):
        """Deterministic (eta=0) / partially stochastic (eta>0) sampler on a sub-sequence of `steps` timesteps.

        `x_T` + `t_start` let us start from a *partially noised real sample* (used by the denoising test)."""
        device = next(model.parameters()).device
        t_start = self.T - 1 if t_start is None else t_start
        x = self._randn(shape, device, generator) if x_T is None else x_T
        seq = torch.linspace(t_start, 0, steps).round().long().unique_consecutive().tolist()
        ab = self.sch.alpha_bar
        for k, i in enumerate(seq):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            eps = self.model_eps(model, x, t, y, guidance)
            x0 = self.predict_x0(x, t, eps)
            if clip_x0:
                x0 = x0.clamp(-1, 1)
                eps = (x - ab[i].sqrt() * x0) / (1 - ab[i]).sqrt()                   # keep eps consistent with clipped x0
            ab_t = ab[i]
            ab_prev = ab[seq[k + 1]] if k + 1 < len(seq) else torch.ones((), device=device)
            sigma = eta * ((1 - ab_prev) / (1 - ab_t)).sqrt() * (1 - ab_t / ab_prev).sqrt()
            x = ab_prev.sqrt() * x0 + (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps
            if eta > 0 and k + 1 < len(seq):
                x = x + sigma * self._randn(shape, device, generator)
        return x


# ---------------------------------------------------------------------- evaluation helpers
@torch.no_grad()
def validation_loss(diffusion: GaussianDiffusion, model, x, y=None, batch_size=128, seed=123) -> float:
    """Deterministic validation MSE: the SAME timesteps and noise every epoch, so curves are comparable
    and any change comes from the model, not from sampling luck. No label dropout, no loss weighting."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    total, n = 0.0, 0
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size].to(device)
        yb = y[i:i + batch_size].to(device) if y is not None else None
        t = torch.randint(0, diffusion.T, (len(xb),), device=device, generator=g)
        noise = torch.randn(xb.shape, device=device, generator=g)
        pred = model(diffusion.q_sample(xb, t, noise), t, yb)
        total += (pred.float() - noise).square().mean().item() * len(xb)
        n += len(xb)
    model.train(was_training)
    return total / n


@torch.no_grad()
def loss_by_timestep(diffusion: GaussianDiffusion, model, x, y=None, buckets=10, batch_size=128, seed=7):
    """MSE as a function of the noise level. Small t (little noise): eps is hard to tell from fine image detail.
    Large t (almost pure noise): eps is easy to read off x_t. This curve explains the shape of the training loss."""
    device = next(model.parameters()).device
    model.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    out = []
    for b in range(buckets):
        lo, hi = b * diffusion.T // buckets, (b + 1) * diffusion.T // buckets
        tot, n = 0.0, 0
        for i in range(0, len(x), batch_size):
            xb = x[i:i + batch_size].to(device)
            yb = y[i:i + batch_size].to(device) if y is not None else None
            t = torch.randint(lo, hi, (len(xb),), device=device, generator=g)
            noise = torch.randn(xb.shape, device=device, generator=g)
            pred = model(diffusion.q_sample(xb, t, noise), t, yb)
            tot += (pred.float() - noise).square().mean().item() * len(xb)
            n += len(xb)
        out.append(dict(t_lo=lo, t_hi=hi, mse=tot / n))
    return out
