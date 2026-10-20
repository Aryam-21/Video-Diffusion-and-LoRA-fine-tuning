"""Compare the base model with LoRA checkpoints on held-out prompts.

    python -m task2_lora.evaluate_lora --run_dir runs/lora/r16_lr1e-4_attn --ckpts best.pt epoch_005.pt \
           --scales 0 1.0 --prompts data/toy_videos/eval_prompts.json

Metrics (computed with a public CLIP model, higher is better):
    clip_text   mean cosine(frame embedding, prompt embedding)   -> does the video match the prompt? ("prompt alignment")
    clip_temp   mean cosine between consecutive frame embeddings -> temporal consistency (flicker / identity drift)
Scale 0 = LoRA switched off = the ORIGINAL model, so every comparison has its own baseline.
Prompts in the "unseen" group were never in the training data -> tests generalisation, not memorisation.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from common.utils import get_device, save_json, seed_everything
from task2_lora.lora import inject_lora, load_lora_state_dict, set_lora_scale
from task2_lora.pipeline_utils import generate_frames, load_pipeline, save_gif


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--ckpts", nargs="+", default=["best.pt"])
    p.add_argument("--scales", nargs="+", type=float, default=[0.0, 1.0])
    p.add_argument("--prompts", default=None, help="json {'seen': [...], 'unseen': [...]}; default: <data_dir>/eval_prompts.json")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    return p.parse_args(argv)


class ClipScorer:
    def __init__(self, name, device):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        self.model = CLIPModel.from_pretrained(name, torch_dtype=torch.float16).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(name)

    @torch.no_grad()
    def __call__(self, frames, prompt):
        px = self.proc(images=list(frames), return_tensors="pt")["pixel_values"].to(self.device, torch.float16)
        tx = self.proc.tokenizer([prompt], padding=True, truncation=True, return_tensors="pt").to(self.device)
        img = F.normalize(self.model.get_image_features(pixel_values=px).float(), dim=-1)
        txt = F.normalize(self.model.get_text_features(**tx).float(), dim=-1)
        return dict(clip_text=(img @ txt.T).mean().item(), clip_temp=(img[1:] * img[:-1]).sum(-1).mean().item())


def main(argv=None):
    args = parse_args(argv)
    run = Path(args.run_dir)
    cfg = json.load(open(run / "config.json"))
    device = get_device()
    seed_everything(0)
    prompts = json.load(open(args.prompts or Path(cfg["data_dir"]) / "eval_prompts.json"))
    out = run / "eval"
    out.mkdir(exist_ok=True)

    pipe = load_pipeline(cfg["model_id"], device)
    inject_lora(pipe.unet, cfg["targets_list"], cfg["rank"], cfg["alpha"])
    scorer = ClipScorer(args.clip_model, device)

    rows = []
    for ck_name in args.ckpts:
        ck = torch.load(run / "checkpoints" / ck_name, map_location="cpu", weights_only=False)
        load_lora_state_dict(pipe.unet, ck["lora"])
        for scale in args.scales:
            if scale == 0 and ck_name != args.ckpts[0]:
                continue                      # base model is identical for every checkpoint: generate it once
            set_lora_scale(pipe.unet, scale)
            for group, plist in prompts.items():
                for pi, prompt in enumerate(plist):
                    for seed in args.seeds:
                        fr = generate_frames(pipe, prompt, seed, cfg["num_frames"], cfg["height"], cfg["width"], args.steps)
                        tag = f"{Path(ck_name).stem}_s{scale}_{group}{pi}_seed{seed}"
                        save_gif(fr, out / "videos" / f"{tag}.gif")
                        rows.append(dict(ckpt=ck_name, epoch=ck["epoch"], scale=scale, group=group, prompt=prompt,
                                         seed=seed, **scorer(fr, prompt)))
            print(f"[eval] {ck_name} scale={scale} done")

    with open(out / "clip_scores_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    agg = defaultdict(list)
    for r in rows:
        agg[(r["ckpt"] if r["scale"] else "base model", r["scale"], r["group"])].append(r)
    lines = ["| model | LoRA scale | prompts | CLIP text-video | CLIP temporal | n |", "|---|---|---|---|---|---|"]
    summary = []
    for (name, scale, group), rs in sorted(agg.items(), key=lambda kv: (kv[0][2], kv[0][1], kv[0][0])):
        ct = sum(r["clip_text"] for r in rs) / len(rs)
        cm = sum(r["clip_temp"] for r in rs) / len(rs)
        lines.append(f"| {name} | {scale} | {group} | {ct:.4f} | {cm:.4f} | {len(rs)} |")
        summary.append(dict(model=name, scale=scale, group=group, clip_text=ct, clip_temp=cm, n=len(rs)))
    (out / "clip_scores.md").write_text("\n".join(lines) + "\n")
    save_json(summary, out / "clip_scores.json")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
