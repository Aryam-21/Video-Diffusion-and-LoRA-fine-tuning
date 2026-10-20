"""Loss-curve plotting from the CSV logs (works for Task 1 and Task 2)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.utils import read_csv


def plot_training_curves(csv_path, png_path, keys=("train_loss", "val_loss"), title="Training curves", logy=True):
    rows = read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    for k in keys:
        pts = [(int(r["epoch"]), float(r[k])) for r in rows if r.get(k) not in ("", None)]   # skip blank cells (epoch-0 row)
        if pts:
            ax.plot(*zip(*pts), marker="o", ms=3, label=k)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def plot_lines(series: dict, png_path, xlabel="x", ylabel="y", title="", logy=False, x=None):
    """series: {label: list_of_values}. Optional shared x."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, ys in series.items():
        ax.plot(x if x is not None else range(len(ys)), ys, marker="o", ms=3, label=label)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
