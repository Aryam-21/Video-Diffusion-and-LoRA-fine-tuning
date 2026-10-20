"""Save / load helpers. A checkpoint stores weights, EMA weights, optimizer state AND the full config,
so that any experiment can be re-created from the file alone."""
from __future__ import annotations

import torch

from common.schedules import NoiseSchedule
from task1_diffusion.diffusion import GaussianDiffusion
from task1_diffusion.models import build_unet


def build_from_config(cfg: dict, device):
    model = build_unet(cfg["task"], cfg["num_classes"], cfg["base"], cfg["mults"]).to(device)
    schedule = NoiseSchedule(cfg["schedule"], cfg["timesteps"]).to(device)
    diffusion = GaussianDiffusion(schedule, cfg["num_classes"], cfg.get("cond_drop", 0.0), cfg.get("min_snr") or None)
    return model, diffusion


def load_trained(path, device, use_ema: bool = True):
    ck = torch.load(path, map_location=device, weights_only=False)
    model, diffusion = build_from_config(ck["config"], device)
    model.load_state_dict(ck["ema" if use_ema else "model"])
    return model.eval(), diffusion, ck["config"], ck
