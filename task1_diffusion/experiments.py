"""Controlled experiments: linear vs cosine schedule x plain vs Min-SNR loss, for images and/or videos.

    python -m task1_diffusion.experiments --tasks image video
    python -m task1_diffusion.experiments --tasks image --quick        # 2-epoch smoke test

Only ONE factor changes between runs (same seed, data, model, epochs), so differences can be attributed to it.
Outputs results/task1/experiment_table.md (the "experiment table with results and limitations").
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["image", "video"], choices=["image", "video"])
    p.add_argument("--schedules", nargs="+", default=["linear", "cosine"])
    p.add_argument("--min_snr", nargs="+", type=float, default=[0.0, 5.0])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--out_dir", default="results/task1")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--amp", action="store_true")
    a = p.parse_args()
    epochs = 2 if a.quick else a.epochs

    rows = []
    for task in a.tasks:
        for sch in a.schedules:
            for snr in a.min_snr:
                name = f"{task}_{sch}_{'minsnr' if snr else 'plain'}"
                cmd = [sys.executable, "-m", "task1_diffusion.train", "--task", task, "--schedule", sch,
                       "--min_snr", str(snr), "--out_dir", a.out_dir, "--run_name", name]
                if epochs:
                    cmd += ["--epochs", str(epochs)]
                if a.quick:
                    cmd += ["--n_train", "256", "--max_train", "2048", "--sample_every", "99"]
                if a.amp:
                    cmd += ["--amp"]
                run(cmd)
                ev = [sys.executable, "-m", "task1_diffusion.evaluate", "--ckpt", f"{a.out_dir}/{name}/best.pt"]
                if a.quick:
                    ev += ["--n_samples", "64", "--n_denoise", "16", "--guidance", "1.0", "2.0"]
                run(ev)
                d = Path(a.out_dir) / name
                m = json.loads((d / "eval" / "metrics.json").read_text())
                s = json.loads((d / "train_summary.json").read_text())
                g = m["generation"]
                wbest = max((k for k in g if k.startswith("guidance_")), key=lambda k: g[k]["class_acc"])
                dn = {r["frac"]: r for r in m["denoising"]}
                rows.append(dict(run=name, task=task, schedule=sch, minsnr=snr, best_epoch=s["best_epoch"],
                                 val=s["best_val_loss"], psnr_t50=dn[0.5]["ddim_psnr"], ssim_t50=dn[0.5]["ddim_ssim"],
                                 best_w=wbest, acc=g[wbest]["class_acc"], fd=g[wbest]["FD"],
                                 fd_floor=g["FD_real_train_vs_real_test(noise floor)"],
                                 sample_s=g[wbest]["sec_per_sample"], mem=s["peak_mem_mb"], time=s["train_time_s"],
                                 params=s["params"] / 1e6))

    hdr = ("| run | best epoch | val MSE | PSNR@t=.5 (dB) | SSIM@t=.5 | best guidance | class acc | FD (floor) | s/sample | "
           "train peak MB | train time (s) | params (M) |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    body = "".join(f"| {r['run']} | {r['best_epoch']} | {r['val']:.4f} | {r['psnr_t50']:.2f} | {r['ssim_t50']:.3f} | "
                   f"{r['best_w'].split('_')[1]} | {r['acc']:.3f} | {r['fd']:.2f} ({r['fd_floor']:.2f}) | {r['sample_s']:.3f} | "
                   f"{r['mem']:.0f} | {r['time']:.0f} | {r['params']:.2f} |\n" for r in rows)
    limits = ("\n**Limitations.** Single seed per configuration (differences < ~5% are within run-to-run noise); small "
              "32x32 / 8-frame data; Frechet distance uses features of our own small classifier, so it is *not* comparable "
              "with published FID/FVD values; val MSE of runs with different schedules is measured under different "
              "noise-level mixtures, so compare PSNR / class accuracy / FD across schedules, not val MSE alone.\n")
    out = Path(a.out_dir) / "experiment_table.md"
    out.write_text("# Task 1 - experiment table\n\n" + hdr + body + limits)
    print(f"\nwrote {out}\n\n" + hdr + body)


if __name__ == "__main__":
    main()
