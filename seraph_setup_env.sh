#!/bin/bash
# One-off environment build for PoDD on Seraph/ariel (account jihye4118).
#
# Everything lands on NAS: /ceph_data/jihye4118 is at 96.5% of its 1.43TB quota, and the
# 2026-08-01 incident on this account was caused by jobs writing caches to Ceph.
set -x

export PIP_CACHE_DIR=/nas2/data/jihye4118/.pipcache
export HF_HOME=/nas2/data/jihye4118/hf_cache
export ENV_PREFIX=/nas2/data/jihye4118/envs/podd

mkdir -p "$PIP_CACHE_DIR"

source /ceph_data/jihye4118/miniconda3/etc/profile.d/conda.sh

# Match the kdst-219 env exactly (python 3.9.16, torch 1.13.1+cu117): `higher` is
# unmaintained and does not reliably support torch 2.x, and this run has to stay
# comparable with the one it continues.
# conda-forge only, with --override-channels: conda 26 refuses the Anaconda defaults
# channels until their Terms of Service are accepted, and this is a borrowed account -- not
# somewhere to accept a licence agreement on the owner's behalf.
conda create -y -p "$ENV_PREFIX" -c conda-forge --override-channels python=3.9.16
conda activate "$ENV_PREFIX"
command -v python || { echo "FATAL: env activation failed"; exit 1; }

python -m pip install --no-input --upgrade pip
python -m pip install --no-input torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# pytorch-lightning and kornia are deliberately omitted: seed_everything is now a local
# shim and nothing imports kornia.
python -m pip install --no-input numpy==1.23.5 pandas==1.5.3 higher==0.2.1 \
    tqdm==4.66.1 wandb==0.13.10 transformers==4.37.1 pillow==9.4.0

# PoCO builds the poster class order from CLIP text embeddings on every start, including
# every resumed job. Pre-fetch it once here so the jobs themselves never need the network.
python - <<'PY'
import os
from transformers import CLIPTokenizer, CLIPTextModel
print('HF_HOME =', os.environ.get('HF_HOME'))
CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32')
print('CLIP cached OK')
PY

python -c "import sys,torch,higher,transformers;print('py',sys.version.split()[0]);print('torch',torch.__version__,'cuda',torch.version.cuda);print('transformers',transformers.__version__);print('higher ok')"
echo SETUP_DONE
