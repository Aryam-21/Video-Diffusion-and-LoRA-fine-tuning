"""Denoising networks.

ONE UNet class serves both data types - only `dims` changes:
    dims=2  images  (B, C, H, W)      Conv2d  -> each pixel sees its spatial neighbours
    dims=3  videos  (B, C, T, H, W)   Conv3d  -> each pixel sees spatial AND temporal neighbours ("3D kernels connect
                                                 neighbouring frames")
Down/up-sampling only touches H and W, never T, so the number of frames is preserved end to end.

Conditioning
    * timestep t  -> sinusoidal features -> MLP -> added to every ResBlock as FiLM (scale & shift)   ("inject timestep features")
    * class label -> learned embedding added to the timestep embedding (index `num_classes` = "no label",
      used for classifier-free guidance)

Output has exactly the input shape: one predicted noise value per input value.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(dims: int, cin: int, cout: int, k: int = 3, stride=1) -> nn.Module:
    C = nn.Conv2d if dims == 2 else nn.Conv3d
    return C(cin, cout, k, stride=stride, padding=k // 2)


def _gn(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(8, ch), ch)


class SinusoidalEmbedding(nn.Module):
    """Same idea as transformer positional encoding: t -> [sin(t*w_i), cos(t*w_i)] with geometric frequencies."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        a = t.float()[:, None] * freqs[None]
        return torch.cat([a.sin(), a.cos()], dim=-1)


class ResBlock(nn.Module):
    """GroupNorm-SiLU-Conv x2 with FiLM conditioning and a residual connection."""

    def __init__(self, cin: int, cout: int, emb_dim: int, dims: int, dropout: float = 0.0):
        super().__init__()
        self.dims = dims
        self.n1, self.c1 = _gn(cin), _conv(dims, cin, cout)
        self.emb = nn.Linear(emb_dim, 2 * cout)
        self.n2, self.drop, self.c2 = _gn(cout), nn.Dropout(dropout), _conv(dims, cout, cout)
        nn.init.zeros_(self.c2.weight)       # block starts as identity -> stable early training
        nn.init.zeros_(self.c2.bias)
        self.skip = nn.Identity() if cin == cout else _conv(dims, cin, cout, 1)

    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.emb(F.silu(emb)).view(x.size(0), -1, *([1] * self.dims)).chunk(2, dim=1)
        h = self.n2(h) * (1 + scale) + shift
        return self.c2(self.drop(F.silu(h))) + self.skip(x)


class SelfAttention(nn.Module):
    """Global attention over all positions (space for images, space-time for video) - used at the lowest resolution."""

    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.norm = _gn(ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x):
        b, c, *sp = x.shape
        h = self.norm(x).flatten(2).transpose(1, 2)               # (B, tokens, C)
        h, _ = self.attn(h, h, h, need_weights=False)
        return x + h.transpose(1, 2).reshape(b, c, *sp)


class Downsample(nn.Module):
    def __init__(self, ch: int, dims: int):
        super().__init__()
        self.conv = _conv(dims, ch, ch, 3, stride=2 if dims == 2 else (1, 2, 2))   # never downsample time

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, cin: int, cout: int, dims: int):
        super().__init__()
        self.dims = dims
        self.conv = _conv(dims, cin, cout)

    def forward(self, x):
        scale = 2.0 if self.dims == 2 else (1.0, 2.0, 2.0)
        return self.conv(F.interpolate(x, scale_factor=scale, mode="nearest"))


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, base: int = 32, mults=(1, 2, 4), dims: int = 2,
                 num_classes: int = 0, dropout: float = 0.0):
        super().__init__()
        assert dims in (2, 3)
        self.dims, self.num_classes = dims, num_classes
        emb_dim = base * 4
        chs = [base * m for m in mults]
        L = len(chs)

        self.time_mlp = nn.Sequential(SinusoidalEmbedding(base), nn.Linear(base, emb_dim), nn.SiLU(),
                                      nn.Linear(emb_dim, emb_dim))
        self.class_emb = nn.Embedding(num_classes + 1, emb_dim) if num_classes > 0 else None   # +1 = "null" label

        self.in_conv = _conv(dims, in_channels, base)
        self.enc = nn.ModuleList([ResBlock(base if i == 0 else chs[i - 1], chs[i], emb_dim, dims, dropout) for i in range(L)])
        self.down = nn.ModuleList([Downsample(chs[i], dims) for i in range(L - 1)])

        self.mid1 = ResBlock(chs[-1], chs[-1], emb_dim, dims, dropout)
        self.mid_attn = SelfAttention(chs[-1])
        self.mid2 = ResBlock(chs[-1], chs[-1], emb_dim, dims, dropout)

        self.dec = nn.ModuleList([ResBlock(2 * chs[i], chs[i], emb_dim, dims, dropout) for i in range(L)])   # 2x: skip concat
        self.up = nn.ModuleList([Upsample(chs[i], chs[i - 1], dims) for i in range(1, L)])

        self.out = nn.Sequential(_gn(chs[0]), nn.SiLU(), _conv(dims, chs[0], in_channels))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        assert x.ndim == 2 + self.dims, f"expected {2 + self.dims}-D input, got {tuple(x.shape)}"
        emb = self.time_mlp(t)
        if self.class_emb is not None:
            if y is None:
                y = torch.full((x.size(0),), self.num_classes, device=x.device, dtype=torch.long)
            emb = emb + self.class_emb(y)

        h = self.in_conv(x)
        skips = []
        for i, block in enumerate(self.enc):
            h = block(h, emb)
            skips.append(h)
            if i < len(self.down):
                h = self.down[i](h)

        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)

        for i in reversed(range(len(self.dec))):
            h = self.dec[i](torch.cat([h, skips[i]], dim=1), emb)
            if i > 0:
                h = self.up[i - 1](h)
        return self.out(h)


def build_unet(task: str, num_classes: int = 0, base: int | None = None, mults=(1, 2, 4), dropout: float = 0.0) -> UNet:
    dims = 2 if task == "image" else 3
    base = base or (64 if dims == 2 else 32)
    return UNet(1, base, tuple(mults), dims, num_classes, dropout)


class Classifier(nn.Module):
    """Small CNN used ONLY for evaluation: (a) does a generated sample show the requested class? (b) its penultimate
    features give a Frechet distance (a mini "FID" computed in the feature space of a net we trained ourselves)."""

    def __init__(self, num_classes: int, dims: int, base: int = 32):
        super().__init__()
        s2 = 2 if dims == 2 else (2, 2, 2)           # video: also pool over time once, so motion direction is visible
        s1 = 2 if dims == 2 else (1, 2, 2)
        self.features_net = nn.Sequential(
            _conv(dims, 1, base), _gn(base), nn.SiLU(),
            _conv(dims, base, 2 * base, 3, s2), _gn(2 * base), nn.SiLU(),
            _conv(dims, 2 * base, 4 * base, 3, s1), _gn(4 * base), nn.SiLU(),
            _conv(dims, 4 * base, 4 * base), _gn(4 * base), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1) if dims == 2 else nn.AdaptiveAvgPool3d(1), nn.Flatten())
        self.head = nn.Linear(4 * base, num_classes)

    def features(self, x):
        return self.features_net(x)

    def forward(self, x):
        return self.head(self.features(x))
