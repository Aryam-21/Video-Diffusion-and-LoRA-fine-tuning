#!/bin/bash
# End-to-end check in a few minutes. Run from the repo root (video_diffusion_assignment/).
set -e
python -m pytest task1_diffusion/tests task2_lora/tests -q                       # unit tests (CPU, ~1 min)
python -m task1_diffusion.experiments --tasks image video --schedules cosine --min_snr 5.0 --quick   # tiny Task 1 run + eval + table
# Task 2 needs a CUDA GPU and downloads the model from Hugging Face:
if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  python -m task2_lora.make_toy_dataset --out data/toy_videos --n_train 12 --n_val 4
  python -m task2_lora.sweep --quick
else
  echo "[skip] no CUDA GPU: Task 2 smoke test skipped"
fi
