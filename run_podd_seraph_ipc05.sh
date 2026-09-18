#!/bin/bash
# PoDD on Seraph/ariel, ImageNet-Nette 128^2 at IPC 0.5.
#
# Identical to run_podd_seraph.sh (the completed IPC 1.0 run, jobs 391739/397617/404644)
# except the poster budget: PoDD's convention scales both poster dims by sqrt(ratio), so
# 640x256 * sqrt(0.5) ~= 452x181, snapped to 450x180 = 81,000 px = 0.494 * 10 * 128^2.
# The snap to an exact 2.5:1 aspect matters: PoDDL.init_label_array only yields the correct
# (2,5) label grid when poster_h*shrink == 2 exactly (its min() treats a zero-slack candidate
# as infeasible); at 452x181 it degrades to (2,4) and one class column gets a zero-width
# label region, so that class would never be trained. Staying 1% under budget is in line
# with the paper's own LT1-90 = 0.897. Patch grid 5x4 stays valid: y steps 0/17/35/52.
#
# Paths are hardcoded rather than passed through `sbatch --export` (silently failed before;
# the resulting HF downloads blew the Ceph quota 2026-08-01). Everything writable is on NAS.
set -x

export HF_HOME=/ceph_data/jihye4118/hf_cache
export HF_HUB_OFFLINE=1          # CLIP is pre-fetched; a job must never hit the network
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/ceph_data/jihye4118/torch_home
export PYTHONUNBUFFERED=1

REPO=/ceph_data/jihye4118/g/PoDD
RUN_DIR=/ceph_data/jihye4118/runs/podd_nette_128_ipc05
# Node-local dataset staging (NAS is backup storage, 2026-09-18): first
# /data{2,3,4}/local_datasets with >=8 GB free, else Ceph. flock guards the
# two-jobs-per-node race on the extract.
# Reuse an existing node-local copy first (jh staged the subsets on several v-nodes).
DATA=""
for d in /data2 /data3 /data4; do
  for u in jihye4118 jh guhyun9454; do
    if [ -d "$d/local_datasets/$u/ImageNet_nette/train" ]; then
      DATA="$d/local_datasets/$u/ImageNet_nette"; break 2
    fi
  done
done
if [ -z "$DATA" ]; then
DATA_PARENT=""
for d in /data2 /data3 /data4; do
  if [ -d "$d/local_datasets" ]; then
    avail=$(df -Pm "$d" | awk 'NR==2{print $4}')
    if [ "${avail:-0}" -ge 8000 ]; then DATA_PARENT="$d/local_datasets/jihye4118"; break; fi
  fi
done
if [ -z "$DATA_PARENT" ]; then
  # Node without local disks (ariel-n1: /data there is an NFS user share): read the NAS
  # copy directly — user decision 2026-09-18; the Ceph dataset copy was deleted.
  DATA=/nas2/data/jihye4118/datasets/ImageNet_nette
else
  mkdir -p "$DATA_PARENT"
  (
    flock 9
    [ -d "$DATA_PARENT/ImageNet_nette/train" ] || \
      tar -xzf /nas2/data/guhyun9454/ImageNet/ImageNet_nette.tar.gz -C "$DATA_PARENT"
  ) 9>"$DATA_PARENT/.nette_stage.lock"
  DATA=$DATA_PARENT/ImageNet_nette
fi
fi
echo "resolved DATA=$DATA"

mkdir -p "$RUN_DIR"
echo "resolved HF_HOME=$HF_HOME  HF_HUB_OFFLINE=$HF_HUB_OFFLINE  RUN_DIR=$RUN_DIR"

source /ceph_data/jihye4118/miniconda3/etc/profile.d/conda.sh
# Blackwell nodes (ariel-n1, sm_120) need the torch 2.7+cu128 env; the stock podd env
# (torch 1.13, sm_86 max) crashes there — how jobs 425409/425410 died on 2026-09-17.
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in
  *Blackwell*|*"PRO 6000"*) conda activate podd_bw ;;
  *)                        conda activate podd ;;   # Ceph clone of the NAS podd env
esac
python -c "import torch;print('torch',torch.__version__,'gpu',torch.cuda.get_device_name(0))"

# Per-process GPU memory sampler (records only our own PID).
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
  --poster_class_num_x 5 --poster_class_num_y 2 --poster_width 450 --poster_height 180 \
  --class_area_width 128 --class_area_height 128 --patch_num_x 5 --patch_num_y 4 \
  --num_train_eval 3 --workers 4 --wandb \
  --state_ckpt "$RUN_DIR/state.pt" \
  --state_ckpt_every 10 \
  --resume "$RUN_DIR/state.pt"

echo "PYTHON_EXIT=$?"
