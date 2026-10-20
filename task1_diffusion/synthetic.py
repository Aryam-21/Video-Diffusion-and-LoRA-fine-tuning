"""Synthetic "moving shapes" video generator (pure numpy, no downloads, fully reproducible).

Each clip shows one shape (circle / square / diamond) travelling in one of 4 directions. The start position is chosen so the shape never hits a wall,
hence the motion label is unambiguous in every frame.
The label (= our "caption") is  shape * 4 + direction  -> 12 classes.

Why synthetic video? You know the ground truth, so you can *measure* whether the generated clip has the right
shape AND the right motion - impossible with natural video at this scale.
"""
from __future__ import annotations

import numpy as np

SHAPES = ["circle", "square", "diamond"]
DIRECTIONS = ["right", "left", "down", "up"]
CLASS_NAMES = [f"{s}-{d}" for s in SHAPES for d in DIRECTIONS]
_VX = np.array([1.0, -1.0, 0.0, 0.0])
_VY = np.array([0.0, 0.0, 1.0, -1.0])


def generate_moving_shapes(n: int, num_frames: int = 8, size: int = 32, seed: int = 0, chunk: int = 500):
    """Returns videos float32 (n, 1, T, H, W) in [0, 1] and labels int64 (n,)."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, len(CLASS_NAMES), n)
    vids = np.empty((n, 1, num_frames, size, size), np.float32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    t = np.arange(num_frames, dtype=np.float32)
    for s in range(0, n, chunk):
        y = labels[s:s + chunk]
        m = len(y)
        shape, d = (y // 4)[:, None, None, None], y % 4
        r = rng.uniform(4.0, 7.0, (m, 1)).astype(np.float32)
        r = np.minimum(r, 0.22 * size)                                        # small canvases: keep the shape small enough to move
        speed = rng.uniform(1.0, 2.0, (m, 1)).astype(np.float32)
        # FIX: long clips used to run out of the frame (61% of 24-frame clips) -> cap the speed so the whole path fits
        speed = np.minimum(speed, 0.9 * (size - 1 - 2 * r) / max(num_frames - 1, 1))
        lo, hi = r, size - 1 - r
        travel = speed * (num_frames - 1)                                     # total distance covered by the clip
        sx, sy = _VX[d][:, None], _VY[d][:, None]                             # direction signs (-1, 0, +1)

        def start(sign):                                                      # leave room to travel without hitting a wall
            avail = (hi - lo) - travel * np.abs(sign)
            return (lo + travel * (sign < 0) + rng.uniform(0, 1, (m, 1)) * avail).astype(np.float32)

        cx = start(sx) + speed * sx * t                                       # (m, T)
        cy = start(sy) + speed * sy * t
        dx = xx[None, None] - cx[:, :, None, None]
        dy = yy[None, None] - cy[:, :, None, None]
        r4 = r[:, :, None, None]
        dist = np.where(shape == 0, np.sqrt(dx ** 2 + dy ** 2),
                        np.where(shape == 1, np.maximum(np.abs(dx), np.abs(dy)), (np.abs(dx) + np.abs(dy)) / 1.35))
        mask = 1.0 / (1.0 + np.exp(-(r4 - dist) / 0.6))                      # soft (anti-aliased) edge
        bright = rng.uniform(0.7, 1.0, (m, 1, 1, 1)).astype(np.float32)
        v = mask * bright + rng.normal(0, 0.02, mask.shape).astype(np.float32)
        vids[s:s + m, 0] = np.clip(v, 0, 1)
    return vids, labels.astype(np.int64)
