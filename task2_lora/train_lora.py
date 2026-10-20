"""LoRA fine-tuning of an open-source text-to-video model (default: ModelScope T2V 1.7B from Hugging Face).

    python -m task2_lora.make_toy_dataset --out data/toy_videos
    python -m task2_lora.train_lora --data_dir data/toy_videos --rank 16 --lr 1e-4 --epochs 15

What happens
    1. load pipeline (fp16), freeze VAE / text encoder / UNet
    2. encode every clip ONCE -> cached latents + text embeddings
    3. inject our own LoRA layers (task2_lora/lora.py) into the attention projections of the UNet
    4. train with the standard diffusion objective on the latents; validate every epoch with FIXED noise/timesteps
    5. save a small LoRA checkpoint EVERY epoch + best.pt (lowest val loss) + log CSV + sample videos

Memory tricks used (each one listed in the lecture slides): fp16 weights, gradient checkpointing, cached latents,
batch 1 + gradient accumulation, only LoRA params in the optimizer.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.plotting import plot_training_curves
from common.utils import (CSVLogger, count_params, get_device, peak_memory_mb, reset_peak_memory, save_json,
                          seed_everything)
from task2_lora.lora import ATTN_TARGETS, FF_TARGETS, inject_lora, lora_parameters, lora_state_dict
from task2_lora.pipeline_utils import generate_frames, load_pipeline, save_gif
from task2_lora.video_data import LatentDataset, build_latent_cache, encode_text, read_metadata

VAL_TIMESTEPS = (100, 300, 500, 700, 900)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_id", default="damo-vilab/text-to-video-ms-1.7b")
    p.add_argument("--data_dir", default="data/toy_videos")
    p.add_argument("--out_dir", default="runs/lora/default")
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    # LoRA
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=None, help="default = rank (scaling alpha/r = 1)")
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--targets", choices=["attn", "attn_ff"], default="attn")
    # optimisation
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--caption_dropout", type=float, default=0.1)
    p.add_argument("--no_grad_ckpt", action="store_true", help="disable gradient checkpointing (faster, more VRAM)")
    # bookkeeping
    p.add_argument("--sample_every", type=int, default=5, help="generate preview clips every N epochs (0 = never)")
    p.add_argument("--sample_steps", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def lr_at(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))) * 0.95 + 0.05


@torch.no_grad()
def validate(unet, noise_sched, ds, device, dtype, seed=0):
    """Mean noise-prediction MSE over all val clips x fixed timesteps x fixed noise (deterministic -> comparable across epochs)."""
    unet.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    tot, n = 0.0, 0
    for lat, emb in ds:
        lat, emb = lat[None].to(device, dtype), emb[None].to(device, dtype)
        for t in VAL_TIMESTEPS:
            tt = torch.tensor([t], device=device)
            noise = torch.randn(lat.shape, generator=g, device=device, dtype=dtype)
            noisy = noise_sched.add_noise(lat, noise, tt)
            target = noise_sched.get_velocity(lat, noise, tt) if noise_sched.config.prediction_type == "v_prediction" else noise
            pred = unet(noisy, tt, encoder_hidden_states=emb).sample
            tot += F.mse_loss(pred.float(), target.float()).item()
            n += 1
    unet.train()
    return tot / max(n, 1)


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    device, dtype = get_device(), torch.float16
    assert device.type == "cuda", "LoRA fine-tuning of a 1.7B video UNet needs a CUDA GPU (Colab / Kaggle T4 is enough)"
    out = Path(args.out_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(exist_ok=True)

    # ------------------------------------------------------------ model + data
    pipe = load_pipeline(args.model_id, device, dtype)
    from diffusers import DDPMScheduler
    noise_sched = DDPMScheduler.from_config(pipe.scheduler.config)       # training scheduler (pipeline may sample with another)
    unet, vae, tok, te = pipe.unet, pipe.vae, pipe.tokenizer, pipe.text_encoder
    for m in (unet, vae, te):
        m.requires_grad_(False)

    records = read_metadata(args.data_dir)
    items = build_latent_cache(records, vae, tok, te, args, device, dtype)
    empty_emb = encode_text(tok, te, [""], device)[0].to(dtype)           # "no caption" embedding for caption dropout
    train_ds, val_ds = LatentDataset(items["train"]), LatentDataset(items["val"])
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"[data] train {len(train_ds)} clips, val {len(val_ds)} clips, latent {tuple(items['train'][0]['latent'].shape)}")

    # ------------------------------------------------------------ LoRA
    alpha = args.alpha if args.alpha is not None else float(args.rank)
    targets = ATTN_TARGETS + (FF_TARGETS if args.targets == "attn_ff" else ())
    replaced = inject_lora(unet, targets, args.rank, alpha, args.lora_dropout)
    params = lora_parameters(unet)
    n_train, n_total = sum(p.numel() for p in params), count_params(unet)
    print(f"[lora] {len(replaced)} layers adapted | trainable {n_train / 1e6:.2f}M / {n_total / 1e6:.0f}M "
          f"({100 * n_train / n_total:.2f}%) | r={args.rank} alpha={alpha}")
    if not args.no_grad_ckpt:
        unet.enable_gradient_checkpointing()
    unet.train()

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(loader) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_at(s, total_steps, args.warmup_steps))
    try:
        scaler = torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler()

    config = {**vars(args), "alpha": alpha, "targets_list": list(targets), "n_layers": len(replaced),
              "trainable_params": n_train, "total_params": n_total, "scheduler_prediction_type": noise_sched.config.prediction_type}
    save_json(config, out / "config.json")
    log = CSVLogger(out / "train_log.csv")

    prompts = []
    pf = Path(args.data_dir) / "eval_prompts.json"
    if args.sample_every and pf.exists():
        e = json.load(open(pf))
        prompts = [e["seen"][0], e["unseen"][0]]
    elif args.sample_every:
        prompts = [items["train"][0]["caption"]]

    # ------------------------------------------------------------ epoch 0 (= base model) baseline
    base_val = validate(unet, noise_sched, val_ds, device, dtype)
    print(f"[val] epoch 0 (untrained LoRA = base model): {base_val:.5f}")
    log.log(epoch=0, train_loss="", val_loss=base_val, lr=0.0, grad_norm="", epoch_s=0.0, peak_mem_gb=0.0)

    best_val, best_epoch, t_all = base_val, 0, time.time()
    reset_peak_memory(device)
    for epoch in range(1, args.epochs + 1):
        t0, run, gn_sum, n_loss, n_opt = time.time(), 0.0, 0.0, 0, 0
        n_gn = 0
        opt.zero_grad(set_to_none=True)
        for i, (lat, emb) in enumerate(loader):
            lat, emb = lat.to(device, dtype), emb.to(device, dtype)
            drop = torch.rand(len(emb), device=device) < args.caption_dropout
            emb = torch.where(drop[:, None, None], empty_emb[None], emb)
            noise = torch.randn_like(lat)
            t = torch.randint(0, noise_sched.config.num_train_timesteps, (len(lat),), device=device)
            noisy = noise_sched.add_noise(lat, noise, t)
            if not args.no_grad_ckpt:
                noisy.requires_grad_(True)               # lets reentrant checkpointing propagate grads to the LoRA params
            target = noise_sched.get_velocity(lat, noise, t) if noise_sched.config.prediction_type == "v_prediction" else noise
            with torch.autocast("cuda", dtype=torch.float16):
                pred = unet(noisy, t, encoder_hidden_states=emb).sample
            loss = F.mse_loss(pred.float(), target.float())
            scaler.scale(loss / args.grad_accum).backward()
            run, n_loss = run + loss.item(), n_loss + 1
            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(loader):
                scaler.unscale_(opt)
                gn = float(torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm))
                if math.isfinite(gn):                                   # fp16 overflow steps are skipped by the scaler; don't log inf
                    gn_sum, n_gn = gn_sum + gn, n_gn + 1
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
                n_opt += 1

        val = validate(unet, noise_sched, val_ds, device, dtype)
        row = dict(epoch=epoch, train_loss=run / n_loss, val_loss=val, lr=sched.get_last_lr()[0],
                   grad_norm=gn_sum / max(n_gn, 1), epoch_s=time.time() - t0, peak_mem_gb=peak_memory_mb(device) / 1024)
        log.log(**row)
        print(f"ep {epoch:03d}/{args.epochs} | train {row['train_loss']:.5f} | val {val:.5f} | gnorm {row['grad_norm']:.3f} "
              f"| lr {row['lr']:.1e} | {row['epoch_s']:.0f}s | peak {row['peak_mem_gb']:.1f}GB")

        ck = dict(lora=lora_state_dict(unet), config=config, epoch=epoch, val_loss=val, train_loss=row["train_loss"])
        torch.save(ck, out / "checkpoints" / f"epoch_{epoch:03d}.pt")
        if val < best_val:
            best_val, best_epoch = val, epoch
            torch.save(ck, out / "checkpoints" / "best.pt")

        if args.sample_every and (epoch % args.sample_every == 0 or epoch == args.epochs):
            unet.eval()
            for k, pr in enumerate(prompts):
                try:
                    fr = generate_frames(pipe, pr, 1234, args.num_frames, args.height, args.width, args.sample_steps)
                    save_gif(fr, out / "samples" / f"epoch_{epoch:03d}_prompt{k}.gif")
                except Exception as e:  # a failed preview must never kill a training run
                    print(f"[warn] preview failed: {e}")
            unet.train()

    summary = dict(best_epoch=best_epoch, best_val_loss=best_val, baseline_val_loss=base_val,
                   final_val_loss=val, final_train_loss=row["train_loss"], total_time_s=time.time() - t_all,
                   peak_mem_gb=peak_memory_mb(device) / 1024, trainable_params=n_train, rank=args.rank, lr=args.lr,
                   targets=args.targets)
    save_json(summary, out / "summary.json")
    plot_training_curves(out / "train_log.csv", out / "loss_curves.png", title="LoRA fine-tuning", logy=False)
    if best_epoch == 0:
        print("[warn] val loss never improved over the base model - lower the LR or check the data/captions")
    print(f"[done] best val {best_val:.5f} at epoch {best_epoch} (baseline {base_val:.5f}) -> {out}")


if __name__ == "__main__":
    main()
