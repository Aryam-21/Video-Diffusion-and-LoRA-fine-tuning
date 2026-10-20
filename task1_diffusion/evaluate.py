"""Evaluate a trained diffusion checkpoint.  Produces everything the assignment asks to "observe":

    python -m task1_diffusion.evaluate --ckpt results/task1/image_cosine_minsnr/best.pt

 1. Denoising test   - take REAL test samples, add noise of increasing strength, let the model remove it
                       (one-step x0 prediction  AND  multi-step DDIM from that noise level). PSNR / SSIM / MSE vs the clean input.
 2. Loss vs. noise   - MSE per timestep bucket (explains the shape of the training loss).
 3. Generation       - samples from pure noise with several guidance scales: class accuracy, Frechet distance, temporal stats.
 4. Benchmark        - parameters, DDIM vs DDPM sampling time, train-step time, peak GPU memory.
Everything is written to <ckpt dir>/eval/ (png, gif, csv, metrics.json).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from common.plotting import plot_lines
from common.utils import (CSVLogger, count_params, get_device, peak_memory_mb, reset_peak_memory, save_grid,
                          save_filmstrip, save_json, save_samples, seed_everything, sync)
from task1_diffusion import metrics as M
from task1_diffusion.checkpoint import load_trained
from task1_diffusion.data import CACHE_DIR, get_data
from task1_diffusion.diffusion import loss_by_timestep

NOISE_LEVELS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9]       # fraction of the diffusion chain (t / T)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n_denoise", type=int, default=64)
    p.add_argument("--n_samples", type=int, default=256, help="generated samples per guidance scale (>=256 for a stable FD)")
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--guidance", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    p.add_argument("--clf_epochs", type=int, default=None)
    p.add_argument("--with_ddpm", action="store_true", help="also time/benchmark the full T-step DDPM sampler")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def mid_frame(x):           # for videos, show the middle frame in the denoising grids
    return x[:, :, x.shape[2] // 2] if x.ndim == 5 else x


@torch.no_grad()
def denoising_test(model, diffusion, x0, y, args, device, out, task):
    B, T = len(x0), diffusion.T
    g = torch.Generator(device=device).manual_seed(args.seed)
    log = CSVLogger(out / "denoising_metrics.csv")
    rows = []
    for frac in NOISE_LEVELS:
        tv = min(int(frac * (T - 1)), T - 1)
        t = torch.full((B,), tv, device=device, dtype=torch.long)
        xt = diffusion.q_sample(x0, t, torch.randn(x0.shape, device=device, generator=g))
        one = diffusion.predict_x0(xt, t, diffusion.model_eps(model, xt, t, y)).clamp(-1, 1)
        multi = diffusion.ddim_sample(model, x0.shape, steps=min(args.ddim_steps, tv + 1), y=y, x_T=xt, t_start=tv)
        row = dict(t=tv, frac=frac)
        for name, est in [("noisy", xt.clamp(-1, 1)), ("onestep", one), ("ddim", multi)]:
            row[f"{name}_mse"] = M.mse(est, x0).mean().item()
            row[f"{name}_psnr"] = M.psnr(est, x0).mean().item()
            row[f"{name}_ssim"] = M.ssim(est, x0).mean().item()
        log.log(**row)
        rows.append(row)
        if frac in (0.1, 0.5, 0.9):                                         # qualitative figures
            k = 8
            stack = [x0[:k], xt[:k].clamp(-1, 1), one[:k], multi[:k]]       # rows: clean / noisy / one-step / multi-step
            save_grid(torch.cat([mid_frame(s) for s in stack]), out / f"denoise_t{tv:04d}.png", nrow=k)
            if task == "video":
                save_filmstrip(torch.stack([s[0] for s in stack]), out / f"denoise_t{tv:04d}_clip0_filmstrip.png")
    plot_lines({"noisy input": [r["noisy_psnr"] for r in rows], "one-step x0": [r["onestep_psnr"] for r in rows],
                "multi-step DDIM": [r["ddim_psnr"] for r in rows]}, out / "denoising_psnr.png",
               xlabel="noise level index (t/T = " + ", ".join(str(f) for f in NOISE_LEVELS) + ")", ylabel="PSNR (dB)",
               title="Denoising quality vs. noise level", x=[r["frac"] for r in rows])
    return rows


@torch.no_grad()
def generation_eval(model, diffusion, cfg, clf, test, train, args, device, out):
    task, K = cfg["task"], cfg["num_classes"]
    n = args.n_samples
    xr, yr = test[0][:n], test[1][:n]
    real_acc, real_feats = M.classify(clf, xr, yr, device)
    _, train_feats = M.classify(clf, train[0][:n], train[1][:n], device)
    res = {"real_test_class_acc": real_acc, "FD_real_train_vs_real_test(noise floor)": M.frechet_distance(train_feats, real_feats)}
    if task == "video":
        res["real_temporal_diff"] = M.temporal_diff(xr)
    shape = tuple(cfg["sample_shape"])
    for w in args.guidance:
        g = torch.Generator(device=device).manual_seed(args.seed + 1)
        ys = (torch.arange(n) % max(K, 1)).to(device)
        sync(device); t0 = time.time()
        xs = torch.cat([diffusion.ddim_sample(model, (len(ys[i:i + 64]), *shape), steps=args.ddim_steps,
                                              y=ys[i:i + 64] if K else None, guidance=w, generator=g)
                        for i in range(0, n, 64)])
        sync(device); dt = time.time() - t0
        acc, feats = M.classify(clf, xs, ys.cpu(), device)
        r = dict(class_acc=acc, FD=M.frechet_distance(feats, real_feats), sec_per_sample=dt / n,
                 pixel_mean=xs.mean().item(), pixel_std=xs.std().item())
        if task == "video":
            r["temporal_diff"] = M.temporal_diff(xs)
        res[f"guidance_{w}"] = r
        save_samples(xs[:32], out / f"generated_w{w}", nrow=8)
        print(f"[gen w={w}] " + "  ".join(f"{k}={v:.4f}" for k, v in r.items()))
    save_samples(xr[:32], out / "real_reference", nrow=8)
    return res


def benchmark(model, diffusion, cfg, device, with_ddpm, ddim_steps):
    shape = (16, *cfg["sample_shape"])
    y = (torch.arange(16) % max(cfg["num_classes"], 1)).to(device) if cfg["num_classes"] else None
    res = {"params_M": count_params(model) / 1e6}
    reset_peak_memory(device)
    sync(device); t0 = time.time()
    diffusion.ddim_sample(model, shape, steps=ddim_steps, y=y)
    sync(device); res[f"ddim{ddim_steps}_s_per_batch16"] = time.time() - t0
    if with_ddpm:
        sync(device); t0 = time.time()
        diffusion.ddpm_sample(model, shape, y=y)
        sync(device); res[f"ddpm{diffusion.T}_s_per_batch16"] = time.time() - t0
    res["sampling_peak_mem_MB"] = peak_memory_mb(device)
    # one training step (forward + backward) to report training cost
    m = type(model)(1, cfg["base"] or 32, tuple(cfg["mults"]), model.dims, cfg["num_classes"]).to(device).train()
    x = torch.randn(cfg["batch_size"], *cfg["sample_shape"], device=device)
    yy = torch.randint(0, max(cfg["num_classes"], 1), (len(x),), device=device) if cfg["num_classes"] else None
    reset_peak_memory(device)
    sync(device); t0 = time.time()
    loss, _ = diffusion.training_losses(m, x, yy)
    loss.backward()
    sync(device); res["train_step_s"] = time.time() - t0
    res["train_step_peak_mem_MB"] = peak_memory_mb(device)
    return res


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    device = get_device()
    model, diffusion, cfg, ck = load_trained(args.ckpt, device)
    out = Path(args.ckpt).parent / "eval"
    out.mkdir(exist_ok=True)
    print(f"[eval] {args.ckpt}  (epoch {ck['epoch']}, val {ck['val_loss']:.4f})  device={device}")

    splits, meta = get_data(cfg["task"], cfg["dataset"], cfg["size"], cfg["frames"], cfg["data_root"],
                            cfg["n_train"], cfg["max_train"])
    test, train = splits["test"], splits["train"]
    K = cfg["num_classes"]
    x0 = test[0][:args.n_denoise].to(device)
    y0 = test[1][:args.n_denoise].to(device) if K else None

    metrics = {"checkpoint": str(args.ckpt), "epoch": ck["epoch"], "val_loss": ck["val_loss"]}
    metrics["denoising"] = denoising_test(model, diffusion, x0, y0, args, device, out, cfg["task"])
    metrics["loss_by_timestep"] = loss_by_timestep(diffusion, model, test[0][:512], test[1][:512] if K else None)
    plot_lines({"val MSE": [r["mse"] for r in metrics["loss_by_timestep"]]}, out / "loss_by_timestep.png",
               xlabel=f"timestep bucket (0 = clean ... {len(metrics['loss_by_timestep']) - 1} = pure noise)", ylabel="MSE",
               title="Noise-prediction error vs. noise level", logy=True)

    clf_epochs = args.clf_epochs or (3 if cfg["task"] == "image" else 6)
    clf = M.get_classifier(meta, train, test, device, clf_epochs,
                           f"{cfg['task']}_{meta['dataset']}_{cfg['size']}_{cfg['frames']}_{cfg['n_train']}", CACHE_DIR)
    metrics["generation"] = generation_eval(model, diffusion, cfg, clf, test, train, args, device, out)
    metrics["benchmark"] = benchmark(model, diffusion, cfg, device, args.with_ddpm, args.ddim_steps)
    save_json(metrics, out / "metrics.json")
    print(f"[eval] wrote {out / 'metrics.json'}")
    return metrics


if __name__ == "__main__":
    main()
