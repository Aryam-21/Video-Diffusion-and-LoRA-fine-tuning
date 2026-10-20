"""Run a small grid of LoRA configurations, then summarise them (best epoch per run + best run overall).

    python -m task2_lora.sweep --ranks 4 16 64 --lrs 1e-4 3e-4 --epochs 15
    python -m task2_lora.sweep --quick            # 2 epochs, 1 config: checks that the whole pipeline works

Only the swept factors change; seed, data, steps per epoch are identical -> a controlled experiment.
"""
import argparse
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/toy_videos")
    p.add_argument("--runs_dir", default="runs/lora")
    p.add_argument("--ranks", nargs="+", type=int, default=[4, 16, 64])
    p.add_argument("--lrs", nargs="+", type=float, default=[1e-4, 3e-4])
    p.add_argument("--targets", nargs="+", default=["attn"], choices=["attn", "attn_ff"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    if a.quick:
        a.ranks, a.lrs, a.epochs = [8], [1e-4], 2
    for t in a.targets:
        for r in a.ranks:
            for lr in a.lrs:
                name = f"r{r}_lr{lr:g}_{t}"
                cmd = [sys.executable, "-m", "task2_lora.train_lora", "--data_dir", a.data_dir, "--rank", str(r),
                       "--lr", str(lr), "--targets", t, "--epochs", str(a.epochs), "--out_dir", f"{a.runs_dir}/{name}",
                       "--sample_every", "0" if a.quick else "5"]
                print("\n$", " ".join(cmd))
                subprocess.run(cmd, check=True)
    subprocess.run([sys.executable, "-m", "task2_lora.analyze", "--runs_dir", a.runs_dir], check=True)


if __name__ == "__main__":
    main()
