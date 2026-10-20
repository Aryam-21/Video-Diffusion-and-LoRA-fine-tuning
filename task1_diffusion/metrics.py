"""Evaluation metrics, implemented by hand.

Reconstruction metrics (compare a denoised sample with its clean original):
    MSE   mean squared error                        lower is better
    PSNR  10*log10(range^2 / MSE)  in dB             higher is better (every +3 dB = half the error)
    SSIM  structural similarity (luminance, contrast, structure in local windows)    1.0 = identical

Generative metrics (compare a *set* of generated samples with real data):
    class accuracy   a classifier we trained says the sample shows the requested class  -> "prompt alignment"
    Frechet distance distance between Gaussian fits of classifier features (same maths as FID, own features)
    temporal stats   mean |frame_{t+1} - frame_t| : too small = frozen video, too large = flicker
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from task1_diffusion.models import Classifier


# ------------------------------------------------------------------ reconstruction
def mse(x, y):
    return (x.float() - y.float()).square().flatten(1).mean(1)


def psnr(x, y, data_range: float = 2.0):
    return 10 * torch.log10(data_range ** 2 / mse(x, y).clamp_min(1e-10))


def _gauss_window(size: int, sigma: float, ch: int, device):
    c = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    g = torch.exp(-c ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(ch, 1, size, size).contiguous()


def ssim(x, y, data_range: float = 2.0, win: int = 7, sigma: float = 1.5):
    """Per-sample SSIM. Videos are scored frame by frame and averaged."""
    if x.ndim == 5:
        B, C, T, H, W = x.shape
        x, y = (v.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W) for v in (x, y))
        return ssim(x, y, data_range, win, sigma).view(B, T).mean(1)
    x, y = x.float(), y.float()
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    w = _gauss_window(win, sigma, x.shape[1], x.device)
    f = lambda a: F.conv2d(a, w, groups=x.shape[1])
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    m = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2))
    return m.flatten(1).mean(1)


# ------------------------------------------------------------------ generative
def temporal_diff(x: torch.Tensor) -> float:
    """Mean absolute difference between consecutive frames (videos only)."""
    return (x[:, :, 1:] - x[:, :, :-1]).abs().mean().item()


def frechet_distance(f1: np.ndarray, f2: np.ndarray) -> float:
    """||mu1-mu2||^2 + Tr(S1 + S2 - 2 (S1 S2)^{1/2}).  Needs a few hundred samples to be stable."""
    from scipy import linalg
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2)
    if not np.isfinite(covmean).all():                       # singular covariance (few samples): add a tiny ridge, as in the FID reference code
        ridge = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + ridge) @ (s2 + ridge))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def get_classifier(meta, train, test, device, epochs: int, cache_name: str, cache_dir: Path) -> Classifier:
    """Train (or load from cache) the evaluation classifier on the REAL training split."""
    dims = 2 if meta["task"] == "image" else 3
    clf = Classifier(meta["num_classes"], dims).to(device)
    path = Path(cache_dir) / f"clf_{cache_name}.pt"
    if path.exists():
        clf.load_state_dict(torch.load(path, map_location=device))
        return clf.eval()
    x, y = train
    opt = torch.optim.AdamW(clf.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = epochs * (len(x) // 128)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=max(steps, 1))
    clf.train()
    for ep in range(epochs):
        perm = torch.randperm(len(x))
        for i in range(0, len(x) - 127, 128):
            idx = perm[i:i + 128]
            loss = F.cross_entropy(clf(x[idx].to(device)), y[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    clf.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(clf.state_dict(), path)
    acc, _ = classify(clf, *test, device)
    print(f"[classifier] trained {epochs} epochs, real test accuracy = {acc:.3f}")
    return clf


@torch.no_grad()
def classify(clf, x, y, device, batch_size: int = 256):
    """Returns (accuracy, features[N, D] as numpy)."""
    feats, correct = [], 0
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size].to(device)
        f = clf.features(xb)
        correct += (clf.head(f).argmax(1).cpu() == y[i:i + batch_size].cpu()).sum().item()
        feats.append(f.float().cpu())
    return correct / len(x), torch.cat(feats).numpy()
