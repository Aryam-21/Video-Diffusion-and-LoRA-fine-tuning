"""Collect all LoRA runs -> one table + overlaid validation curves; answers "which epoch / config is best?".

    python -m task2_lora.analyze --runs_dir runs/lora
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.utils import read_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs/lora")
    a = p.parse_args()
    root = Path(a.runs_dir)
    runs = sorted(d for d in root.iterdir() if (d / "summary.json").exists())
    if not runs:
        raise SystemExit(f"no finished runs in {root}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    lines = ["| run | rank | lr | targets | trainable params | baseline val | best val | best epoch | final val | "
             "overfit? | time (min) | peak GB |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in runs:
        s = json.load(open(d / "summary.json"))
        rows = read_csv(d / "train_log.csv")
        ep = [int(r["epoch"]) for r in rows]
        val = [float(r["val_loss"]) for r in rows]
        tr = [(int(r["epoch"]), float(r["train_loss"])) for r in rows if r["train_loss"] != ""]
        ax[0].plot(ep, val, marker="o", ms=3, label=d.name)
        ax[1].plot(*zip(*tr), marker="o", ms=3, label=d.name)
        # overfitting heuristic: val loss at the end is clearly worse than at its best epoch while train keeps falling
        over = "yes" if s["final_val_loss"] > s["best_val_loss"] * 1.02 and s["best_epoch"] < ep[-1] else "no"
        lines.append(f"| {d.name} | {s['rank']} | {s['lr']} | {s['targets']} | {s['trainable_params'] / 1e6:.2f}M | "
                     f"{s['baseline_val_loss']:.5f} | {s['best_val_loss']:.5f} | {s['best_epoch']} | {s['final_val_loss']:.5f} | "
                     f"{over} | {s['total_time_s'] / 60:.1f} | {s['peak_mem_gb']:.1f} |")
    for x, t in zip(ax, ("validation loss (epoch 0 = base model)", "training loss")):
        x.set_title(t); x.set_xlabel("epoch"); x.grid(alpha=0.3); x.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root / "lora_curves.png", dpi=150)
    (root / "lora_experiments.md").write_text("# LoRA experiments\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {root / 'lora_experiments.md'} and {root / 'lora_curves.png'}")


if __name__ == "__main__":
    main()
