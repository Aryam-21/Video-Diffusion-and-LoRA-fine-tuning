"""Generate a clip with the base model or with a LoRA checkpoint.

    python -m task2_lora.generate --prompt "a red circle moving from left to right on a black background" \
        --lora runs/lora/default/checkpoints/best.pt --scale 1.0 --out outputs/red_circle.gif
"""
import argparse

import torch

from common.utils import get_device
from task2_lora.lora import inject_lora, load_lora_state_dict, merge_lora, set_lora_scale
from task2_lora.pipeline_utils import generate_frames, load_pipeline, save_gif


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--lora", default=None, help="path to a LoRA checkpoint (.pt); omit for the base model")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--merge", action="store_true", help="fold LoRA into the weights first (no extra inference cost)")
    p.add_argument("--model_id", default="damo-vilab/text-to-video-ms-1.7b")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--out", default="outputs/sample.gif")
    a = p.parse_args()

    pipe = load_pipeline(a.model_id, get_device())
    if a.lora:
        ck = torch.load(a.lora, map_location="cpu", weights_only=False)
        c = ck["config"]
        inject_lora(pipe.unet, c["targets_list"], c["rank"], c["alpha"])
        load_lora_state_dict(pipe.unet, ck["lora"])
        set_lora_scale(pipe.unet, a.scale)
        if a.merge:
            merge_lora(pipe.unet)
    frames = generate_frames(pipe, a.prompt, a.seed, a.frames, a.size, a.size, a.steps)
    save_gif(frames, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
