"""Image / video preprocessing, implemented by hand (the assignment asks for this explicitly).

Pipeline for one image:   uint8 array -> float [0,1] -> grayscale -> centre-crop square -> resize -> [-1, 1]

Why [-1, 1]?  The forward process adds N(0, 1) noise. If the data also has roughly zero mean and unit-ish
scale, the signal-to-noise ratio is controlled by alpha_bar alone and the network sees well-scaled inputs.
All functions accept a leading batch dimension (..., C, H, W) so images and video frames share one code path.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def to_grayscale(x: torch.Tensor) -> torch.Tensor:
    """ITU-R BT.601 luminance: Y = 0.299 R + 0.587 G + 0.114 B.  Input (..., C, H, W) in [0, 1]."""
    if x.shape[-3] == 1:
        return x
    w = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype, device=x.device).view(3, 1, 1)
    return (x[..., :3, :, :] * w).sum(dim=-3, keepdim=True)


def center_crop_square(x: torch.Tensor) -> torch.Tensor:
    h, w = x.shape[-2:]
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    return x[..., top:top + s, left:left + s]


def resize(x: torch.Tensor, size: int) -> torch.Tensor:
    """Bilinear resize to (size, size). `antialias` low-pass filters first when shrinking, avoiding aliasing."""
    lead = x.shape[:-3]
    flat = x.reshape(-1, *x.shape[-3:]).float()
    if flat.shape[-2:] != (size, size):
        shrinking = size < min(flat.shape[-2:])
        flat = F.interpolate(flat, size=(size, size), mode="bilinear", align_corners=False, antialias=shrinking)
    return flat.reshape(*lead, *flat.shape[-3:])


def normalize(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def preprocess_batch(x: torch.Tensor, size: int, grayscale: bool = True) -> torch.Tensor:
    """x: float (..., C, H, W) in [0,1]  ->  (..., C', size, size) in [-1, 1]."""
    if grayscale:
        x = to_grayscale(x)
    return normalize(resize(center_crop_square(x), size))


def preprocess_image(img, size: int, grayscale: bool = True) -> torch.Tensor:
    """PIL image / HxW(xC) numpy array -> (C, size, size) tensor in [-1, 1]."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = arr[..., None]
    x = torch.from_numpy(arr.copy()).permute(2, 0, 1).float()
    x = x / 255.0 if arr.dtype == np.uint8 else x
    return preprocess_batch(x, size, grayscale)


def load_image_file(path, size: int, grayscale: bool = True) -> torch.Tensor:
    return preprocess_image(Image.open(path).convert("RGB"), size, grayscale)


def preprocess_video_frames(frames: np.ndarray, size: int, num_frames: int, grayscale: bool = True) -> torch.Tensor:
    """frames: uint8 (T, H, W, C).  Evenly sub-sample `num_frames` frames, then preprocess each one.
    Returns (C, num_frames, size, size) - channels first, time second, the layout Conv3d expects."""
    idx = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
    x = torch.from_numpy(frames[idx].copy()).permute(0, 3, 1, 2).float() / 255.0      # (T, C, H, W)
    x = preprocess_batch(x, size, grayscale)
    return x.permute(1, 0, 2, 3).contiguous()
