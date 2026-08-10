#!/bin/bash
# PoDD on Seraph/ariel, ImageNet-Nette 128^2, the kdst-219 protocol.
#
# Paths are hardcoded here rather than passed through `sbatch --export`: on this account
# --export has silently failed to reach jobs before, and the resulting HF downloads blew the
# Ceph quota (2026-08-01). Everything writable points at NAS.
set -x

export HF_HOME=/nas2/data/jihye4118/hf_cache
export HF_HUB_OFFLINE=1          # CLIP is pre-fetched; a job must never hit the network
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/nas2/data/jihye4118/torch_home
export PYTHONUNBUFFERED=1

REPO=/nas2/data/jihye4118/g/PoDD
RUN_DIR=/nas2/data/jihye4118/runs/podd_nette_128
DATA=/nas2/data/jihye4118/datasets/ImageNet_nette

mkdir -p "$RUN_DIR"
echo "resolved HF_HOME=$HF_HOME  HF_HUB_OFFLINE=$HF_HUB_OFFLINE  RUN_DIR=$RUN_DIR"

source /ceph_data/jihye4118/miniconda3/etc/profile.d/conda.sh
conda activate /nas2/data/jihye4118/envs/podd
python -c "import torch;print('torch',torch.__version__,'gpu',torch.cuda.get_device_name(0))"

# Per-process GPU memory sampler. The kdst measurement was card-total and could not be
# attributed to PoDD; this records only our own PID.
( while true; do
    printf '%s ' "$(date -u +%FT%T)" >> "$RUN_DIR/gpu_mem_samples.txt"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
        >> "$RUN_DIR/gpu_mem_samples.txt" 2>/dev/null
    sleep 30
  done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

cd "$REPO"

# --resume and --state_ckpt point at the same file on purpose: the first submission finds
# nothing and starts fresh, every later submission continues. One command, no edits between
# links in the chain.
python -u main.py \
  --dataset imagenet-subset-nette --root "$DATA" \
  --arch convnet5 --comp_ipc 1 --batch_size 400 --distill_batch_size 32 \
  --window 30 --minwindow 0 --totwindow 80 --epochs 6000 \
  --lr 0.0015 --inner_lr 0.0015 --train_y \
  --poster_class_num_x 5 --poster_class_num_y 2 --poster_width 640 --poster_height 256 \
  --class_area_width 128 --class_area_height 128 --patch_num_x 5 --patch_num_y 4 \
  --num_train_eval 3 --workers 4 --wandb \
  --state_ckpt "$RUN_DIR/state.pt" \
  --state_ckpt_every 10 \
  --resume "$RUN_DIR/state.pt"

echo "PYTHON_EXIT=$?"
