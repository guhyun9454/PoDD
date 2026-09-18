#!/bin/bash
# PoDD on Seraph/ariel, CIFAR-10 at sub-1 IPC budgets.
#
# Usage: sbatch <opts> run_podd_seraph_cifar_subipc.sh <05|04|03|02|01>
#
# Protocol = the official CIFAR-10 LT1 command from the PoDD README (window 60/200,
# curriculum cctype 2, 10,000 epochs, batch 5000, ZCA, 16x6 patches, distill batch 96,
# seed 0, no train_y), with only the poster budget changed. Poster dims follow the
# sqrt-scaling convention on the full 160x64 mosaic (IPC 1.0 = 10 * 32^2 = 10,240 px);
# paper's own LT1-90 (153x60) sits 10% under its nominal budget, ours are within 1%.
#
#   0.5 -> 113x45 (0.4966)   5x2 classes, 16x6 patches   } paper reports 49.5 / 47.1 /
#   0.4 -> 101x40 (0.3945)   5x2 classes, 16x6 patches   } 42.3 at 0.5 / 0.4 / 0.3 —
#   0.3 ->  87x35 (0.2974)   5x2 classes, 16x6 patches   } reproduce-verification runs
#   0.2 ->  64x32 (0.2000 exact)  10x1 classes, 96x1 patches
#   0.1 ->  32x32 (0.1000 exact)  10x1 classes, 96x1 patches
#
# Geometry constraints (see smoke_init_cifar_subipc.py, which validates all five):
# - Crops are 32x32, so both poster dims must be >= 32: at 0.2 the only valid shape is
#   64x32 and at 0.1 it is 32x32 — the 2.5:1 aspect cannot be kept.
# - At height 32 a 5x2 class layout collapses (both rows at y=0 -> label-identical class
#   pairs, ~50% accuracy ceiling), so 0.2/0.1 use the natural one-row 10x1 layout; the
#   patch grid becomes 96x1 (linspace(0,0,6) rows would just duplicate), keeping
#   total_patch_num = 96 = distill_batch_size identical to the official protocol.
# - 0.1 is structurally degenerate either way: the poster IS the single 32x32 crop and
#   the fixed soft label is uniform. We run it to measure the collapse honestly.
#
# Paths are hardcoded rather than passed through `sbatch --export` (silently failed
# before; the resulting HF downloads blew the Ceph quota 2026-08-01). Everything
# writable is on NAS.
set -x

IPC="$1"
case "$IPC" in
  05) W=113; H=45; CX=5;  CY=2; PX=16; PY=6 ;;
  04) W=101; H=40; CX=5;  CY=2; PX=16; PY=6 ;;
  03) W=87;  H=35; CX=5;  CY=2; PX=16; PY=6 ;;
  02) W=64;  H=32; CX=10; CY=1; PX=96; PY=1 ;;
  01) W=32;  H=32; CX=10; CY=1; PX=96; PY=1 ;;
  *) echo "unknown IPC tag: '$IPC' (want 05|04|03|02|01)"; exit 2 ;;
esac

# NAS is backup storage (admin policy, 2026-09-18): execution set lives on Ceph, and the
# dataset is staged into the node-local disk at startup so steady-state I/O is local.
export HF_HOME=/ceph_data/jihye4118/hf_cache
export HF_HUB_OFFLINE=1          # CLIP (PoCO) is pre-fetched; a job must never hit the network
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/ceph_data/jihye4118/torch_home
export PYTHONUNBUFFERED=1

REPO=/ceph_data/jihye4118/g/PoDD
RUN_DIR=/ceph_data/jihye4118/runs/podd_cifar10_ipc$IPC

# Node-local dataset root: first /data{2,3,4}/local_datasets with >=5 GB free;
# falls back to Ceph if the node has none. flock guards the two-jobs-per-node race.
DATA_ROOT=""
for d in /data2 /data3 /data4; do
  if [ -d "$d/local_datasets" ]; then
    avail=$(df -Pm "$d" | awk 'NR==2{print $4}')
    if [ "${avail:-0}" -ge 5000 ]; then DATA_ROOT="$d/local_datasets/jihye4118"; break; fi
  fi
done
[ -z "$DATA_ROOT" ] && DATA_ROOT=/ceph_data/jihye4118/datasets
mkdir -p "$DATA_ROOT/cifar10"
(
  flock 9
  [ -f "$DATA_ROOT/cifar10/cifar-10-python.tar.gz" ] || \
    cp /ceph_data/jihye4118/datasets/cifar-10-python.tar.gz "$DATA_ROOT/cifar10/"
) 9>"$DATA_ROOT/cifar10/.stage.lock"

mkdir -p "$RUN_DIR"
echo "resolved HF_HOME=$HF_HOME  HF_HUB_OFFLINE=$HF_HUB_OFFLINE  RUN_DIR=$RUN_DIR  geom ${W}x${H} cls ${CX}x${CY} patch ${PX}x${PY}"

source /ceph_data/jihye4118/miniconda3/etc/profile.d/conda.sh
# Blackwell nodes (ariel-n1, RTX PRO 6000, sm_120) need the torch 2.7+cu128 env; the
# stock podd env (torch 1.13, sm_86 max) crashes there ("numel: integer multiplication
# overflow" after the arch warning — how jobs 425409/425410 died). Autodetect per node.
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
# nothing and starts fresh, every later submission continues. One command, no edits
# between links in the chain. --wandb is store_false (disables wandb; parse the log).
python -u main.py \
  --name "PoDD-CIFAR10-IPC$IPC" \
  --dataset cifar10 --root "$DATA_ROOT" \
  --arch convnet --comp_ipc 1 \
  --distill_batch_size 96 --patch_num_x "$PX" --patch_num_y "$PY" \
  --num_train_eval 8 --update_steps 1 --batch_size 5000 \
  --ddtype curriculum --cctype 2 \
  --epochs 10000 --test_freq 10 --print_freq 10 \
  --window 60 --minwindow 0 --totwindow 200 \
  --inner_optim Adam --outer_optim Adam --inner_lr 0.001 --lr 0.001 \
  --syn_strategy flip_rotate --real_strategy flip_rotate \
  --seed 0 --zca \
  --class_area_width 32 --class_area_height 32 \
  --poster_width "$W" --poster_height "$H" \
  --poster_class_num_x "$CX" --poster_class_num_y "$CY" \
  --workers 4 --wandb \
  --state_ckpt "$RUN_DIR/state.pt" \
  --state_ckpt_every 10 \
  --resume "$RUN_DIR/state.pt"

echo "PYTHON_EXIT=$?"
