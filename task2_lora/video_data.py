"""Load captioned clips and turn them into cached (latent, text-embedding) pairs.

The VAE and the text encoder are FROZEN, so we run them ONCE and keep the results (a few hundred KB per clip).
Training then only needs the UNet in GPU memory -> much faster and fits on a free Colab/Kaggle T4.

Latent layout for the diffusers UNet3D: (C=4, F, h, w) with h = H/8, w = W/8 (the VAE is 2D, applied frame by frame).
"""
from __future__ import annotations

import json
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def read_metadata(data_dir: str) -> list[dict]:
    d = Path(data_dir)
    rows = [json.loads(l) for l in open(d / "metadata.jsonl") if l.strip()]
    for r in rows:
        r["path"] = str(d / r["file"])
    if not any("split" in r for r in rows):             # no split given -> every 10th clip is validation
        for i, r in enumerate(rows):
            r["split"] = "val" if i % 10 == 9 else "train"
    return rows


def read_clip(path: str, num_frames: int, height: int, width: int) -> torch.Tensor:
    """-> float (3, F, H, W) in [-1, 1]: evenly sub-sample frames, cover-crop to the target aspect ratio, resize."""
    frames = [np.asarray(f)[..., :3] for f in imageio.get_reader(path)]
    idx = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
    out = []
    for i in idx:
        im = Image.fromarray(frames[i])
        scale = max(width / im.width, height / im.height)
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.BICUBIC)
        l, t = (im.width - width) // 2, (im.height - height) // 2
        out.append(np.asarray(im.crop((l, t, l + width, t + height))))
    x = torch.from_numpy(np.stack(out)).float() / 127.5 - 1.0          # (F, H, W, 3)
    return x.permute(3, 0, 1, 2).contiguous()


@torch.no_grad()
def encode_text(tokenizer, text_encoder, captions: list[str], device) -> torch.Tensor:
    ids = tokenizer(captions, padding="max_length", max_length=tokenizer.model_max_length, truncation=True,
                    return_tensors="pt").input_ids.to(device)
    return text_encoder(ids)[0]


@torch.no_grad()
def encode_video(vae, clip: torch.Tensor, device, dtype, chunk: int = 8) -> torch.Tensor:
    """(3,F,H,W) -> latent (4,F,h,w), scaled by the VAE's scaling factor. Chunked to bound VRAM."""
    frames = clip.permute(1, 0, 2, 3).to(device, dtype)                # (F, 3, H, W)
    lat = torch.cat([vae.encode(frames[i:i + chunk]).latent_dist.mean for i in range(0, len(frames), chunk)])
    return (lat * vae.config.scaling_factor).permute(1, 0, 2, 3).cpu()


class LatentDataset(Dataset):
    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        return it["latent"], it["text_emb"]


def build_latent_cache(records, vae, tokenizer, text_encoder, args, device, dtype):
    items = {"train": [], "val": []}
    for i, r in enumerate(records):
        clip = read_clip(r["path"], args.num_frames, args.height, args.width)
        items[r["split"]].append(dict(
            latent=encode_video(vae, clip, device, dtype).to(torch.float16),
            text_emb=encode_text(tokenizer, text_encoder, [r["caption"]], device)[0].cpu().to(torch.float16),
            caption=r["caption"]))
        if (i + 1) % 10 == 0 or i + 1 == len(records):
            print(f"[cache] encoded {i + 1}/{len(records)} clips")
    return items
