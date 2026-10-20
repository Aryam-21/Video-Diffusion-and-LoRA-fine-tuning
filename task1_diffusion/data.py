"""Dataset preparation for Task 1.

Image datasets : fashion_mnist | mnist (auto-download) | folder (your own images, sub-folder = class) | shapes (offline)
Video dataset  : moving_shapes (synthetic, generated on the fly and cached) | folder (your own .mp4/.gif clips)

Every dataset ends up as float tensors in [-1, 1] with layout
    image: (N, 1, H, W)        video: (N, 1, T, H, W)
Splits are train / val / test. Val is used for model selection, test only for the final report.
"""
from __future__ import annotations

from pathlib import Path

import torch

from task1_diffusion import preprocessing as prep
from task1_diffusion.synthetic import CLASS_NAMES as VIDEO_CLASSES, SHAPES, generate_moving_shapes

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "data" / "cache"
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VID_EXT = {".mp4", ".gif", ".avi", ".mov", ".mkv"}


# ----------------------------------------------------------------------------- images
def _torchvision_images(name: str, root: Path, size: int):
    """Download (once) and preprocess with OUR pipeline: uint8 -> [0,1] -> resize -> [-1,1]."""
    from torchvision import datasets
    cls = {"fashion_mnist": datasets.FashionMNIST, "mnist": datasets.MNIST}[name]
    tr, te = cls(root, train=True, download=True), cls(root, train=False, download=True)
    to_x = lambda d: prep.preprocess_batch(d.data.float().div(255).unsqueeze(1), size)
    return (to_x(tr), tr.targets.long()), (to_x(te), te.targets.long()), list(tr.classes)


def _folder_images(root: Path, size: int):
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)
    if not paths:
        raise FileNotFoundError(f"no images found under {root}")
    names = sorted({p.parent.name for p in paths})
    xs = torch.stack([prep.load_image_file(p, size) for p in paths])
    ys = torch.tensor([names.index(p.parent.name) for p in paths])
    return xs, ys, names


def _split_train_val(x, y, val_frac=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(x), generator=g)
    n_val = int(len(x) * val_frac)
    v, t = perm[:n_val], perm[n_val:]
    return (x[t], y[t]), (x[v], y[v])


def build_image_data(dataset: str, size: int, root: str, max_train: int | None = None):
    root = Path(root)
    if dataset in ("fashion_mnist", "mnist"):
        train_all, test, names = _torchvision_images(dataset, root, size)
        train, val = _split_train_val(*train_all)
    elif dataset == "folder":
        x, y, names = _folder_images(root, size)
        (rest, test) = _split_train_val(x, y, 0.1, seed=1)
        (train, val) = _split_train_val(*rest, 0.1, seed=2)
    elif dataset == "shapes":      # offline fallback: first frame of synthetic clips, class = shape
        def make(n, seed):
            v, lab = generate_moving_shapes(n, 1, size, seed)
            return torch.from_numpy(v[:, :, 0]) * 2 - 1, torch.from_numpy(lab // 4)
        train, val, test = make(6000, 0), make(600, 1), make(600, 2)
        names = list(SHAPES)
    else:
        raise ValueError(dataset)
    if max_train:
        train = (train[0][:max_train], train[1][:max_train])
    meta = dict(task="image", dataset=dataset, num_classes=len(names), class_names=names, sample_shape=[1, size, size])
    return {"train": train, "val": val, "test": test}, meta


# ----------------------------------------------------------------------------- videos
def _folder_videos(root: Path, size: int, num_frames: int):
    import imageio
    import numpy as np
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in VID_EXT)
    if not paths:
        raise FileNotFoundError(f"no videos found under {root}")
    names = sorted({p.parent.name for p in paths})
    clips = []
    for p in paths:
        frames = np.stack([np.asarray(f)[..., :3] for f in imageio.get_reader(str(p))])
        clips.append(prep.preprocess_video_frames(frames, size, num_frames))
    return torch.stack(clips), torch.tensor([names.index(p.parent.name) for p in paths]), names


def build_video_data(dataset: str, size: int, num_frames: int, root: str, n_train: int = 4000):
    if dataset == "moving_shapes":
        cache = CACHE_DIR / f"moving_shapes_{size}_{num_frames}_{n_train}.pt"
        if cache.exists():
            splits = torch.load(cache)
        else:
            splits = {}
            for name, n, seed in [("train", n_train, 0), ("val", max(n_train // 8, 64), 1), ("test", max(n_train // 8, 64), 2)]:
                v, lab = generate_moving_shapes(n, num_frames, size, seed)
                splits[name] = (torch.from_numpy(v) * 2 - 1, torch.from_numpy(lab))
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(splits, cache)
        names = VIDEO_CLASSES
    elif dataset == "folder":
        x, y, names = _folder_videos(Path(root), size, num_frames)
        (rest, test) = _split_train_val(x, y, 0.1, seed=1)
        (train, val) = _split_train_val(*rest, 0.1, seed=2)
        splits = {"train": train, "val": val, "test": test}
    else:
        raise ValueError(dataset)
    meta = dict(task="video", dataset=dataset, num_classes=len(names), class_names=list(names),
                sample_shape=[1, num_frames, size, size])
    return splits, meta


def get_data(task: str, dataset: str | None, size: int, num_frames: int, root: str,
             n_train: int = 4000, max_train: int | None = None):
    if task == "image":
        return build_image_data(dataset or "fashion_mnist", size, root, max_train)
    return build_video_data(dataset or "moving_shapes", size, num_frames, root, n_train)
