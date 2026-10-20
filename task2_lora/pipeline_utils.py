"""Load the diffusers text-to-video pipeline and generate clips (shared by training previews, evaluation, generate.py).

Default model: damo-vilab/text-to-video-ms-1.7b  (ModelScope T2V, UNet3D, ~1.7B params).
It is the largest open text-to-video model that can still be LoRA-tuned on a free 16 GB T4 (fp16 + gradient checkpointing
+ cached latents). Bigger models (CogVideoX, Wan2.1) need 24 GB+ or aggressive offloading - see README "Alternatives".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_pipeline(model_id: str, device, dtype=torch.float16):
    from diffusers import DiffusionPipeline
    try:
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, variant="fp16")
    except (OSError, ValueError):                       # repo without an fp16 variant
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


@torch.no_grad()
def generate_frames(pipe, prompt: str, seed: int, num_frames: int, height: int, width: int,
                    steps: int = 25, guidance: float = 9.0) -> np.ndarray:
    """-> uint8 array (F, H, W, 3)."""
    g = torch.Generator(device=pipe.device).manual_seed(seed)
    out = pipe(prompt, num_frames=num_frames, height=height, width=width, num_inference_steps=steps,
               guidance_scale=guidance, generator=g, output_type="np")
    f = out.frames
    f = f[0] if (isinstance(f, (list, tuple)) or getattr(f, "ndim", 0) == 5) else f      # drop batch dim across diffusers versions
    arr = np.stack([np.asarray(x) for x in f])
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).round().astype(np.uint8)
    return arr


def save_gif(frames: np.ndarray, path, fps: int = 8) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ims = [Image.fromarray(f) for f in frames]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0)
