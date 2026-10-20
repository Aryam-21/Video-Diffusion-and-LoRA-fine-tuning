"""LoRA (Low-Rank Adaptation) implemented from scratch - no `peft` needed.

For a frozen linear layer with weight W0 (out x in) we learn a low-rank update:

        y = W0 x  +  (alpha / r) * B A x          A: (r x in), B: (out x r),  r << min(in, out)

* B is initialised to ZERO  -> at step 0 the model is *exactly* the pretrained model.
* A is initialised randomly (Kaiming) so that gradients can flow into B.
* Only A and B are trained: with r=16 that is ~0.5-1% of the parameters -> far less optimizer memory
  (Adam keeps 2 extra tensors per trainable parameter) and tiny checkpoints (a few MB instead of GBs).
* After training the update can be merged: W = W0 + (alpha/r) B A  -> zero extra inference cost.
* `multiplier` rescales the update at inference time (0 = original model, 1 = as trained, >1 = exaggerate).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Attention projections of every (spatial AND temporal) transformer block of the diffusers UNet3D.
ATTN_TARGETS = ("to_q", "to_k", "to_v", "to_out.0")
# Optionally also the feed-forward layers (more capacity, ~2x more trainable parameters).
FF_TARGETS = ("ff.net.0.proj", "ff.net.2")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.r, self.alpha, self.multiplier = r, alpha, 1.0
        dev = base.weight.device
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

    @property
    def scaling(self) -> float:
        return self.alpha / self.r

    def forward(self, x):
        out = self.base(x)
        if self.merged or self.multiplier == 0:
            return out
        delta = F.linear(F.linear(self.drop(x).to(self.A.dtype), self.A), self.B)       # B (A x)
        return out + (self.multiplier * self.scaling) * delta.to(out.dtype)

    @torch.no_grad()
    def merge(self):
        """Fold the update into the base weight (inference only; cannot be trained afterwards)."""
        if not self.merged:
            self.base.weight += (self.scaling * self.multiplier * (self.B @ self.A)).to(self.base.weight.dtype)
            self.merged = True


def inject_lora(model: nn.Module, targets=ATTN_TARGETS, r: int = 16, alpha: float = 16.0, dropout: float = 0.0):
    """Freeze `model` and replace every nn.Linear whose name ends with one of `targets` by a LoRALinear."""
    model.requires_grad_(False)
    replaced = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and any(full == t or full.endswith("." + t) for t in targets):
                setattr(module, child_name, LoRALinear(child, r, alpha, dropout))
                replaced.append(full)
    if not replaced:
        raise RuntimeError(f"no Linear layer matched {targets}; print model.named_modules() and adjust the targets")
    return replaced


def lora_modules(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def lora_parameters(model: nn.Module):
    return [p for m in lora_modules(model) for p in (m.A, m.B)]


def set_lora_scale(model: nn.Module, multiplier: float) -> None:
    for m in lora_modules(model):
        m.multiplier = multiplier


def merge_lora(model: nn.Module) -> None:
    for m in lora_modules(model):
        m.merge()


def lora_state_dict(model: nn.Module) -> dict:
    """Only the A/B matrices, keyed by the layer's name (checkpoints are a few MB)."""
    return {f"{n}.{k}": v.detach().cpu() for n, m in model.named_modules() if isinstance(m, LoRALinear)
            for k, v in (("A", m.A), ("B", m.B))}


def load_lora_state_dict(model: nn.Module, state: dict) -> None:
    mods = {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}
    for key, v in state.items():
        name, k = key.rsplit(".", 1)
        getattr(mods[name], k).data.copy_(v.to(getattr(mods[name], k).device))
