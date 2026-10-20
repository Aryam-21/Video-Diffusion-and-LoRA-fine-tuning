"""Train the diffusion model (images OR videos).

    python -m task1_diffusion.train --task image                       # Fashion-MNIST, 2D UNet
    python -m task1_diffusion.train --task video                       # moving shapes, 3D UNet
    python -m task1_diffusion.train --task video --schedule cosine --min_snr 5

Console line per epoch (see docs/LEARNING_NOTES.md for what each number means):
    train   mean noise-prediction MSE over the epoch (what we optimise; Min-SNR weighting NOT included)
    val     the same MSE on held-out data, fixed noise/timesteps, EMA weights  -> model selection
    gnorm   mean gradient norm BEFORE clipping (spikes = instability)
    lr      current learning rate (warm-up then cosine decay)
    it/s    training throughput          mem   peak GPU memory
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from common.plotting import plot_training_curves
from common.schedules import NoiseSchedule
from common.utils import (CSVLogger, EMA, count_params, get_device, peak_memory_mb, reset_peak_memory, save_json,
                          save_samples, seed_everything)
from task1_diffusion.data import get_data
from task1_diffusion.diffusion import GaussianDiffusion, validation_loss
from task1_diffusion.models import build_unet


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=["image", "video"], default="image")
    p.add_argument("--dataset", default=None, help="image: fashion_mnist|mnist|shapes|folder  video: moving_shapes|folder")
    p.add_argument("--data_root", default="data")
    p.add_argument("--size", type=int, default=32, help="spatial resolution (must be divisible by 4)")
    p.add_argument("--frames", type=int, default=8, help="frames per clip (video)")
    p.add_argument("--n_train", type=int, default=4000, help="synthetic video: number of training clips")
    p.add_argument("--max_train", type=int, default=None, help="optionally cap the number of training images")
    # diffusion
    p.add_argument("--schedule", choices=["linear", "cosine"], default="cosine")
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--min_snr", type=float, default=5.0, help="Min-SNR gamma; 0 disables loss weighting")
    p.add_argument("--cond_drop", type=float, default=0.1, help="label dropout for classifier-free guidance")
    p.add_argument("--no_class_cond", action="store_true", help="unconditional model")
    # model / optimisation
    p.add_argument("--base", type=int, default=None, help="UNet base channels (default 64 image / 32 video)")
    p.add_argument("--mults", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--amp", action="store_true", help="mixed precision (fp16) on CUDA")
    # bookkeeping
    p.add_argument("--out_dir", default="results/task1")
    p.add_argument("--run_name", default=None)
    p.add_argument("--sample_every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    a.epochs = a.epochs or (20 if a.task == "image" else 40)
    a.batch_size = a.batch_size or (128 if a.task == "image" else 32)
    return a


def cosine_lr(step, total, warmup, min_ratio=0.05):
    if step < warmup:
        return (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def preview(model, diffusion, meta, device, path_stem, n=16, steps=50, guidance=2.0):
    """Quick qualitative check during training: one sample per class, DDIM + guidance, fixed seed."""
    g = torch.Generator(device=device).manual_seed(0)
    k = meta["num_classes"]
    y = (torch.arange(n, device=device) % k) if k else None
    x = diffusion.ddim_sample(model, (n, *meta["sample_shape"]), steps=steps, y=y, guidance=guidance, generator=g)
    save_samples(x, path_stem)


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    device = get_device()
    use_amp = args.amp and device.type == "cuda"

    # ---------------------------------------------------------------- data
    splits, meta = get_data(args.task, args.dataset, args.size, args.frames, args.data_root, args.n_train, args.max_train)
    num_classes = 0 if args.no_class_cond else meta["num_classes"]
    (xtr, ytr), (xva, yva) = splits["train"], splits["val"]
    print(f"[data] {meta['dataset']}  train {tuple(xtr.shape)}  val {tuple(xva.shape)}  classes={num_classes}  "
          f"range=[{xtr.min():.2f}, {xtr.max():.2f}]")
    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=args.batch_size, shuffle=True, drop_last=True,
                        pin_memory=device.type == "cuda")

    # ---------------------------------------------------------------- model / diffusion / optimiser
    model = build_unet(args.task, num_classes, args.base, args.mults, args.dropout).to(device)
    args.base = args.base or (64 if args.task == "image" else 32)
    ema = EMA(model, args.ema_decay)
    schedule = NoiseSchedule(args.schedule, args.timesteps).to(device)
    diffusion = GaussianDiffusion(schedule, num_classes, args.cond_drop, args.min_snr or None)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(loader)
    lr_sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_lr(s, total_steps, args.warmup_steps))
    scaler = make_scaler(use_amp)

    name = args.run_name or f"{args.task}_{args.schedule}_{'minsnr' if args.min_snr else 'plain'}"
    out = Path(args.out_dir) / name
    (out / "samples").mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "num_classes": num_classes, "sample_shape": meta["sample_shape"],
              "class_names": meta["class_names"], "params": count_params(model), "device": str(device)}
    save_json(config, out / "config.json")
    log = CSVLogger(out / "train_log.csv")
    print(f"[model] {config['params'] / 1e6:.2f}M params | device={device} | amp={use_amp} | out={out}")

    best_val, best_epoch, t_train = float("inf"), -1, time.time()
    reset_peak_memory(device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run_loss, run_gn, n, n_gn = time.time(), 0.0, 0.0, 0, 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True) if num_classes else None
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss, plain = diffusion.training_losses(model, xb, yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            lr_sched.step()
            ema.update(model)
            run_loss, n = run_loss + plain.item(), n + 1
            if math.isfinite(float(gn)):                      # AMP overflow steps (skipped by the scaler) give inf: don't average them in
                run_gn, n_gn = run_gn + float(gn), n_gn + 1

        val = validation_loss(diffusion, ema.model, xva, yva if num_classes else None)
        dt = time.time() - t0
        row = dict(epoch=epoch, train_loss=run_loss / n, val_loss=val, grad_norm=run_gn / max(n_gn, 1),
                   lr=lr_sched.get_last_lr()[0], it_per_s=n / dt, peak_mem_mb=peak_memory_mb(device), epoch_s=dt)
        log.log(**row)
        print(f"ep {epoch:03d}/{args.epochs} | train {row['train_loss']:.4f} | val {val:.4f} | gnorm {row['grad_norm']:.2f} "
              f"| lr {row['lr']:.1e} | {row['it_per_s']:.1f} it/s | mem {row['peak_mem_mb']:.0f}MB | {dt:.0f}s")

        ck = dict(config=config, epoch=epoch, val_loss=val, model=model.state_dict(), ema=ema.model.state_dict(),
                  opt=opt.state_dict())
        torch.save(ck, out / "last.pt")
        if val < best_val:
            best_val, best_epoch = val, epoch
            torch.save(ck, out / "best.pt")
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            preview(ema.model, diffusion, {**meta, "num_classes": num_classes}, device, out / "samples" / f"epoch_{epoch:03d}")

    summary = dict(best_epoch=best_epoch, best_val_loss=best_val, final_train_loss=row["train_loss"],
                   train_time_s=time.time() - t_train, peak_mem_mb=peak_memory_mb(device), params=config["params"])
    save_json(summary, out / "train_summary.json")
    plot_training_curves(out / "train_log.csv", out / "loss_curves.png", title=f"{name}: noise-prediction MSE")
    print(f"[done] best val {best_val:.4f} at epoch {best_epoch}  ->  {out}")
    return out


if __name__ == "__main__":
    main()
