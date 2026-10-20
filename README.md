# Video Diffusion and LoRA fine-tuning


```
video_diffusion_assignment/
├── common/                  schedules (linear / cosine), EMA, logging, image+video saving, plotting
├── task1_diffusion/         TASK 1 - diffusion model written from scratch (images AND video)
│   ├── preprocessing.py     grayscale / crop / resize / normalise, written by hand
│   ├── synthetic.py         moving-shapes video generator (ground-truth shape + motion labels)
│   ├── data.py              Fashion-MNIST / MNIST / folder / synthetic video -> tensors
│   ├── models.py            ONE UNet: dims=2 (images, Conv2d) or dims=3 (video, Conv3d) + eval classifier
│   ├── diffusion.py         forward noising, loss (+Min-SNR), DDPM + DDIM samplers, classifier-free guidance
│   ├── metrics.py           MSE / PSNR / SSIM / Frechet distance / class accuracy / temporal stats (by hand)
│   ├── train.py  evaluate.py  experiments.py  tests/
├── task2_lora/              TASK 2 - LoRA fine-tuning of an open text-to-video model
│   ├── lora.py              LoRA written from scratch (no peft)
│   ├── make_toy_dataset.py  captioned clips     video_data.py  cached latents
│   ├── train_lora.py        training, per-epoch checkpoints, validation, previews
│   ├── evaluate_lora.py     base vs LoRA, CLIP prompt-alignment + temporal consistency, LoRA-scale sweep
│   ├── sweep.py  analyze.py  generate.py  tests/
├── scripts/smoke_test.sh
└── docs/LEARNING_NOTES.md  +  docs/COMPARISON.md   the theory + every metric explained + likely oral-exam questions (READ THIS)
```
Run every command **from this folder** (`python -m package.module`).

## 0. Setup
```bash
pip install -r requirements.txt
python -m pytest task1_diffusion/tests task2_lora/tests -q   # shapes, gradients, samplers, LoRA correctness (CPU, ~1 min)
bash scripts/smoke_test.sh                                   # tests + tiny end-to-end run
```
Colab/Kaggle: `!git clone <repo>; %cd <repo>/video_diffusion_assignment; !pip install -r requirements.txt`, runtime = GPU.

## 1. Task 1 - diffusion from scratch
```bash
# images (Fashion-MNIST, 32x32 grayscale, 2D UNet)        ~10 min on a T4
python -m task1_diffusion.train    --task image --amp
python -m task1_diffusion.evaluate --ckpt results/task1/image_cosine_minsnr/best.pt

# video (synthetic moving shapes, 8x32x32, 3D UNet)       ~25 min on a T4
python -m task1_diffusion.train    --task video --amp
python -m task1_diffusion.evaluate --ckpt results/task1/video_cosine_minsnr/best.pt

# controlled experiments: {linear, cosine} x {plain, Min-SNR} x {image, video} -> results/task1/experiment_table.md
python -m task1_diffusion.experiments --tasks image video --amp
```
No internet / no torchvision? `--dataset shapes` uses a fully offline image set. Own images: `--dataset folder --data_root my_imgs/`
(sub-folder name = class). Own clips: `--task video --dataset folder`.

Outputs per run in `results/task1/<run>/`: `config.json`, `train_log.csv`, `loss_curves.png`, `best.pt`/`last.pt`,
`samples/` (previews during training) and `eval/` (`denoise_t*.png` input-vs-denoised grids, `denoising_metrics.csv`,
`loss_by_timestep.png`, `generated_w*.png|gif`, `metrics.json`).

## 2. Task 2 - LoRA fine-tuning of a Hugging Face video model
Default model `damo-vilab/text-to-video-ms-1.7b` (ModelScope text-to-video, UNet3D 1.7B) - the largest open T2V model that
fits a free 16 GB T4 with fp16 + gradient checkpointing + cached latents.
```bash
python -m task2_lora.make_toy_dataset --out data/toy_videos            # 48 train / 8 val clips, "yellow square" held out
python -m task2_lora.train_lora --data_dir data/toy_videos --rank 16 --lr 1e-4 --epochs 15
python -m task2_lora.sweep --ranks 4 16 64 --lrs 1e-4 3e-4             # hyper-parameter study + table + curves
python -m task2_lora.evaluate_lora --run_dir runs/lora/r16_lr0.0001_attn --ckpts best.pt epoch_005.pt epoch_015.pt --scales 0 0.5 1 1.5
python -m task2_lora.generate --prompt "a red circle moving from left to right on a black background" \
       --lora runs/lora/r16_lr0.0001_attn/checkpoints/best.pt --out outputs/red_circle.gif
```
Out of memory? `--height 192 --width 192`, `--num_frames 6`, or keep gradient checkpointing on (default).
data: a folder with clips and `metadata.jsonl` (`{"file":..., "caption":..., "split":"train|val"}`), pass `--data_dir`.
Alternatives (need >=24 GB or offloading): `THUDM/CogVideoX-2b`, `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` - the LoRA code in
`lora.py` is model-agnostic; only the target names and the forward call in `train_lora.py` change.

## 3. What is implemented by hand vs. taken from a library
| by hand | from a library |
|---|---|
| noise schedules, forward process, loss, DDPM/DDIM samplers, guidance, Min-SNR, EMA | PyTorch (tensors, autograd, Conv/GroupNorm/MultiheadAttention) |
| UNet2D/3D, timestep + class conditioning | torchvision (only to download Fashion-MNIST) |
| preprocessing, PSNR, SSIM, Frechet distance, temporal metrics | scipy `sqrtm` (matrix square root inside Frechet distance) |
| LoRA layers, injection, merge, save/load, scale control | diffusers (pipeline, VAE, UNet3D, tokenizer/text encoder, DDPMScheduler), CLIP for scoring |

## 4. Submission checklist (from the assignment)
- [x] Runnable source + this README
- [ ] Checkpoints + configs + logs: `results/task1/*/{best.pt,config.json,train_log.csv}`, `runs/lora/*/{checkpoints/best.pt,config.json,train_log.csv}`
- [ ] Generated clips + loss graphs: `results/task1/*/eval/`, `*/loss_curves.png`, `runs/lora/lora_curves.png`
- [ ] Experiment table + limitations: `results/task1/experiment_table.md`, `runs/lora/lora_experiments.md`, `eval/clip_scores.md`
