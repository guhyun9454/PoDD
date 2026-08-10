#!/bin/bash
# Resume smoke test: prove a restart continues rather than restarting, before committing a
# 6-day job to a mechanism that has never been exercised.
#
# Phase 1 runs 4 fresh epochs and checkpoints; phase 2 re-runs the identical command with a
# higher epoch cap and must pick up at epoch 4 with the optimizer state intact.
set -x

export HF_HOME=/nas2/data/jihye4118/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/nas2/data/jihye4118/torch_home
export PYTHONUNBUFFERED=1

REPO=/nas2/data/jihye4118/g/PoDD
RUN_DIR=/nas2/data/jihye4118/runs/podd_smoke
DATA=/nas2/data/jihye4118/datasets/ImageNet_nette

rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"

source /ceph_data/jihye4118/miniconda3/etc/profile.d/conda.sh
conda activate /nas2/data/jihye4118/envs/podd
cd "$REPO"

# test_freq is pushed out of range and num_train_eval dropped to 1: the periodic re-training
# evaluation costs ~20 min a block and proves nothing about resume.
COMMON="--dataset imagenet-subset-nette --root $DATA --arch convnet5 --comp_ipc 1 \
  --batch_size 400 --distill_batch_size 32 --window 30 --minwindow 0 --totwindow 80 \
  --lr 0.0015 --inner_lr 0.0015 --train_y \
  --poster_class_num_x 5 --poster_class_num_y 2 --poster_width 640 --poster_height 256 \
  --class_area_width 128 --class_area_height 128 --patch_num_x 5 --patch_num_y 4 \
  --num_train_eval 1 --test_freq 1000 --workers 4 --wandb"

echo "############ PHASE 1 — fresh run, 4 epochs ############"
python -u main.py $COMMON --epochs 4 \
  --state_ckpt "$RUN_DIR/state.pt" --state_ckpt_every 2 --resume "$RUN_DIR/state.pt"
echo "PHASE1_EXIT=$?"

python - <<'PY'
import torch
s = torch.load('/nas2/data/jihye4118/runs/podd_smoke/state.pt', map_location='cpu')
print('CKPT epoch      :', s['epoch'])
print('CKPT keys       :', sorted(s.keys()))
print('CKPT poster     :', tuple(s['poster'].shape))
print('CKPT adam steps :', [v['step'] for v in s['optimizer']['state'].values()])
PY

echo "############ PHASE 2 — resume, epochs 4..7 ############"
python -u main.py $COMMON --epochs 8 \
  --state_ckpt "$RUN_DIR/state.pt" --state_ckpt_every 2 --resume "$RUN_DIR/state.pt"
echo "PHASE2_EXIT=$?"

python - <<'PY'
import torch
s = torch.load('/nas2/data/jihye4118/runs/podd_smoke/state.pt', map_location='cpu')
print('FINAL epoch     :', s['epoch'])
print('FINAL adam steps:', [v['step'] for v in s['optimizer']['state'].values()])
PY

echo SMOKE_DONE
