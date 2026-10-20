"""Create a small captioned video dataset for LoRA fine-tuning (no download, fully reproducible).

    python -m task2_lora.make_toy_dataset --out data/toy_videos --n_train 48 --n_val 8

Clips show ONE coloured shape moving in ONE direction on a black background; the caption describes exactly that
("a red circle moving from left to right on a black background"). A base text-to-video model is poor at such
precise motion/colour control, so fine-tuning has a clear, measurable effect.

Held-out concept: by default every "yellow square" combination is excluded from training, so `evaluate_lora.py` can test
whether the LoRA composes seen attributes (yellow + square + motion) in a new way.

You can use your own data instead: a folder with clips + metadata.jsonl, one JSON per line:
    {"file": "clips/a.mp4", "caption": "a dog running on a beach", "split": "train"}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

COLORS = {"red": (1.0, 0.1, 0.1), "green": (0.1, 0.9, 0.2), "blue": (0.15, 0.3, 1.0), "yellow": (1.0, 0.9, 0.1)}
SHAPES = ["circle", "square"]
DIRS = {"from left to right": (1, 0), "from right to left": (-1, 0),
        "from top to bottom": (0, 1), "from bottom to top": (0, -1)}


def render(color, shape, direction, size=256, frames=16, rng=None):
    rng = rng or np.random.default_rng(0)
    r = rng.uniform(0.10, 0.16) * size
    speed = rng.uniform(0.45, 0.65) * (size - 2 * r) / (frames - 1)
    dx, dy = DIRS[direction]
    travel = speed * (frames - 1)
    cx = size / 2 + rng.uniform(-0.15, 0.15) * size * (1 - abs(dx))
    cy = size / 2 + rng.uniform(-0.15, 0.15) * size * (1 - abs(dy))
    cx = cx - dx * travel / 2
    cy = cy - dy * travel / 2
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    col = np.array(COLORS[color], np.float32)
    out = []
    for t in range(frames):
        px, py = cx + dx * speed * t, cy + dy * speed * t
        d = np.sqrt((xx - px) ** 2 + (yy - py) ** 2) if shape == "circle" else np.maximum(abs(xx - px), abs(yy - py))
        m = 1 / (1 + np.exp(-(r - d) / 1.2))
        out.append((m[..., None] * col * 255).astype(np.uint8))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/toy_videos")
    p.add_argument("--n_train", type=int, default=48)
    p.add_argument("--n_val", type=int, default=8)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--holdout", default="yellow square", help="'<color> <shape>' excluded from train/val ('' = none)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    import imageio
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    combos = [(c, s, d) for c in COLORS for s in SHAPES for d in DIRS if f"{c} {s}" != a.holdout]
    meta = []
    for split, n in (("train", a.n_train), ("val", a.n_val)):
        for i in range(n):
            c, s, d = combos[(i if split == "train" else rng.integers(len(combos))) % len(combos)]
            f = f"clips/{split}_{i:03d}.mp4"
            imageio.mimsave(out / f, render(c, s, d, a.size, a.frames, rng), fps=8)
            meta.append(dict(file=f, caption=f"a {c} {s} moving {d} on a black background", split=split))
    with open(out / "metadata.jsonl", "w") as fh:
        fh.write("\n".join(json.dumps(m) for m in meta) + "\n")

    seen = [f"a {c} {s} moving {d} on a black background" for (c, s, d) in [("red", "circle", "from left to right"),
            ("blue", "square", "from top to bottom"), ("green", "circle", "from right to left")]]
    unseen = [f"a yellow square moving {d} on a black background" for d in ("from left to right", "from bottom to top")]
    json.dump({"seen": seen, "unseen": unseen}, open(out / "eval_prompts.json", "w"), indent=2)
    print(f"wrote {len(meta)} clips + metadata.jsonl + eval_prompts.json to {out}")


if __name__ == "__main__":
    main()
