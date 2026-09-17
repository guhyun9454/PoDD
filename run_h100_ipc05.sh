#!/bin/bash
# PoDD ImageNet-Nette 128^2 at IPC 0.5, on H100 node1 GPU 4 (borrowed KHU/SDS card).
#
# Mirrors run_podd_seraph_ipc05.sh (poster 450x180, patch 5x4, window 30/80, 6000 epochs)
# but for the H100 singularity environment. GPU 4 ONLY — GPU 5 is held by an external user
# (geji) and must not be touched. Offline CLIP (cached at /home/disc/.cache/huggingface) so
# PoCO.optimize_poster_class_order never hits the network. Writable state on /dataset (NFS,
# 94T free); the root disk / has only ~175G.
#
# R7K2 restore is NOT done here: hold.sh runs from the operator's local machine, so this
# wrapper only writes an EXIT marker on any exit and the operator restores R7K2 on GPU 4.
set -x

GPU=4
REPO=/dataset/disc/workspace/podd_verify/PoDD
RUN_DIR=/dataset/disc/run/podd_nette_ipc05_h100
LOGDIR=/dataset/disc/logs/podd_ipc05_h100
SIF=/dataset/singularity_images/pytorch271-cu128-devel.sif
DATA=/dataset/disc/data/podd/ImageNet_nette
LOG=$LOGDIR/run.out
MARKER=$LOGDIR/EXIT
mkdir -p "$RUN_DIR" "$LOGDIR"
rm -f "$MARKER"

# Per-process memory + util sampler on GPU 4 (records our own PID and card util).
( while true; do
    ts=$(date -u +%FT%T)
    util=$(nvidia-smi -i $GPU --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr '\n' ' ')
    apps=$(nvidia-smi -i $GPU --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | tr '\n' ';')
    echo "$ts util=[${util}] apps=[${apps:-none}]" >> "$RUN_DIR/gpu_mem_samples.txt"
    sleep 30
  done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

export PYTHONUSERBASE=/pyuser
export PYTHONUNBUFFERED=1
export SINGULARITYENV_HF_HOME=/home/disc/.cache/huggingface
export SINGULARITYENV_HF_HUB_OFFLINE=1
export SINGULARITYENV_TRANSFORMERS_OFFLINE=1
export SINGULARITYENV_PYTHONUSERBASE=/pyuser
export SINGULARITYENV_PYTHONUNBUFFERED=1
cd "$REPO"

env CUDA_VISIBLE_DEVICES=$GPU singularity exec --nv \
  --bind /dataset:/dataset,$HOME/.cache/singularity_pyuser:/pyuser \
  "$SIF" \
  python -u main.py \
    --dataset imagenet-subset-nette --root "$DATA" \
    --arch convnet5 --comp_ipc 1 --batch_size 400 --distill_batch_size 32 \
    --window 30 --minwindow 0 --totwindow 80 --epochs 6000 \
    --lr 0.0015 --inner_lr 0.0015 --train_y \
    --poster_class_num_x 5 --poster_class_num_y 2 --poster_width 450 --poster_height 180 \
    --class_area_width 128 --class_area_height 128 --patch_num_x 5 --patch_num_y 4 \
    --num_train_eval 3 --workers 4 --wandb \
    --state_ckpt "$RUN_DIR/state.pt" \
    --state_ckpt_every 10 \
    --resume "$RUN_DIR/state.pt" \
    >> "$LOG" 2>&1
code=$?
echo "EXIT=$code $(date -u +%FT%T)" > "$MARKER"
echo "PYTHON_EXIT=$code"
