"""Small helpers shared by Task 1 and Task 2 (seeding, logging, EMA, image/video saving)."""
from __future__ import annotations

import copy
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# ----------------------------------------------------------------------------- reproducibility
def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch so that runs are repeatable ("record random seeds")."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def count_params(model: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


# ----------------------------------------------------------------------------- GPU bookkeeping
def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> float:
    return torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ----------------------------------------------------------------------------- logging
class CSVLogger:
    """Append-only CSV log. The header is taken from the first row that is logged."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()
        self._fields = None

    def log(self, **row) -> None:
        first = self._fields is None
        if first:
            self._fields = list(row)
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._fields)
            if first:
                w.writeheader()
            w.writerow(row)


def read_csv(path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


# ----------------------------------------------------------------------------- EMA
class EMA:
    """Exponential moving average of the weights.

    Sampling from the averaged weights is almost always cleaner than from the raw weights because
    SGD noise is averaged out. A warm-up on the decay avoids the EMA being dominated by the random init.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.num_updates = 0
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.num_updates += 1
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.lerp_(p.detach(), 1.0 - d)


# ----------------------------------------------------------------------------- image / video output
def to_uint8(x: torch.Tensor, value_range=(-1.0, 1.0)) -> np.ndarray:
    lo, hi = value_range
    x = (x.detach().float().cpu().clamp(lo, hi) - lo) / (hi - lo)
    return (x * 255).round().byte().numpy()


def make_grid(x: torch.Tensor, nrow: int = 8, pad: int = 2, scale: int = 1, value_range=(-1.0, 1.0)) -> np.ndarray:
    """(N,C,H,W) float tensor -> one uint8 image with the N images tiled."""
    imgs = to_uint8(x, value_range)
    n, c, h, w = imgs.shape
    ncol = min(nrow, n)
    nrows = math.ceil(n / ncol)
    canvas = np.full((nrows * (h + pad) + pad, ncol * (w + pad) + pad, c), 255, np.uint8)
    for i in range(n):
        r, k = divmod(i, ncol)
        y0, x0 = pad + r * (h + pad), pad + k * (w + pad)
        canvas[y0:y0 + h, x0:x0 + w] = imgs[i].transpose(1, 2, 0)
    if scale > 1:
        canvas = canvas.repeat(scale, 0).repeat(scale, 1)
    return canvas[..., 0] if c == 1 else canvas


def save_grid(x, path, nrow: int = 8, scale: int = 2, **kw) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(make_grid(x, nrow=nrow, scale=scale, **kw)).save(path)


def save_filmstrip(video: torch.Tensor, path, scale: int = 2, **kw) -> None:
    """(N,C,T,H,W): one clip per row, one frame per column - the best static view of a video."""
    n, c, t, h, w = video.shape
    frames = video.permute(0, 2, 1, 3, 4).reshape(n * t, c, h, w)
    save_grid(frames, path, nrow=t, scale=scale, **kw)


def save_gif(video: torch.Tensor, path, nrow: int = 4, fps: int = 6, scale: int = 3, **kw) -> None:
    """(N,C,T,H,W) -> animated gif in which the N clips play side by side."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frames = [Image.fromarray(make_grid(video[:, :, t], nrow=nrow, scale=scale, **kw)) for t in range(video.shape[2])]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)


def save_samples(x: torch.Tensor, path_stem, nrow: int = 8) -> None:
    """Dispatch on rank: images -> png grid; videos -> png filmstrip + gif."""
    path_stem = str(path_stem)
    if x.ndim == 4:
        save_grid(x, path_stem + ".png", nrow=nrow)
    else:
        save_filmstrip(x[:8], path_stem + "_filmstrip.png")
        save_gif(x[:16], path_stem + ".gif", nrow=4)
