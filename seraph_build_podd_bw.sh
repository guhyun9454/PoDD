#!/bin/bash
# Build the `podd_bw` conda env: the podd stack on torch 2.7.1+cu128, for Blackwell
# (sm_120) GPUs — ariel-n1's RTX PRO 6000s reject the podd env's torch 1.13 (sm_86 max).
# CPU-only sbatch job on a standard node; the first real GPU job on n1 is the GPU test.
#
# Env lives in the Ceph miniconda3 like the account's other envs (ddif, disc_bw);
# torch/torchvision come from the cu128 index, the rest from PyPI per requirements.txt
# minus torch/torchvision (replaced) and pytorch-lightning (shimmed out in src/base.py,
# and pl 2.x imports fail with this stack anyway).
set -x
source /ceph_data/jihye4118/miniconda3/etc/profile.d/conda.sh

# conda-forge only: sidesteps the Anaconda-channel ToS gate (a licence acceptance we
# don't make on a borrowed account) — per the seraph skill's standing guidance.
conda create -y -n podd_bw -c conda-forge --override-channels python=3.9 || exit 1
conda activate podd_bw

export TMPDIR=/nas2/data/jihye4118/tmp
mkdir -p "$TMPDIR"

REQ=/nas2/data/jihye4118/g/PoDD/requirements.txt
grep -vE '^(torch|torchvision|pytorch-lightning)==' "$REQ" > "$TMPDIR/podd_req_notorch.txt"

pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128 || exit 1
pip install --no-cache-dir -r "$TMPDIR/podd_req_notorch.txt" || exit 1

python -c "
import torch, torchvision, numpy, higher, transformers, kornia, pandas, wandb
print('torch', torch.__version__, 'tv', torchvision.__version__, 'numpy', numpy.__version__)
print('sm120 in arch list:', 'sm_120' in str(torch.cuda.get_arch_list()))
print('PODD_BW_ENV_OK')
"
echo "BUILD_EXIT=$?"
