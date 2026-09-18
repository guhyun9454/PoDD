#!/bin/bash
# One-shot migration of the active PoDD execution set from NAS (backup storage, per
# admin policy relayed 2026-09-18) to Ceph, run as a CPU-only sbatch job so the login
# node stays idle. Running jobs keep writing to NAS while this copies; their run dirs
# get a final delta-sync after scancel, before resubmission from the Ceph repo.
set -x
CEPH=/ceph_data/jihye4118
NAS=/nas2/data/jihye4118

mkdir -p "$CEPH/g" "$CEPH/runs" "$CEPH/datasets" "$CEPH/torch_home"

# 1. Repo (with .git), then fast-forward to the Ceph-path launch scripts and strip CRLF.
rsync -a "$NAS/g/PoDD/" "$CEPH/g/PoDD/" || exit 1
cd "$CEPH/g/PoDD" && git pull origin sm-full-resume && sed -i 's/\r$//' run_podd_seraph_cifar_subipc.sh run_podd_seraph_ipc05.sh run_podd_seraph_ipc025.sh seraph_migrate_ceph.sh || exit 1

# 2. Active run dirs (state.pt + samplers; small). Completed-run archives stay on NAS.
for d in podd_cifar10_ipc01 podd_cifar10_ipc02 podd_cifar10_ipc03 podd_cifar10_ipc04 \
         podd_cifar10_ipc05 podd_nette_128_ipc05 podd_nette_128_ipc025; do
  rsync -a "$NAS/runs/$d/" "$CEPH/runs/$d/"
done

# 3. Dataset staging sources on Ceph (jobs re-stage from here into node-local
#    /data{2,3,4}/local_datasets at startup, so steady-state reads never touch NAS).
cp -n "$NAS/datasets/cifar10/cifar-10-python.tar.gz" "$CEPH/datasets/"
cp -n /nas2/data/guhyun9454/ImageNet/ImageNet_nette.tar.gz "$CEPH/datasets/"

# 4. CLIP (PoCO) into the Ceph HF cache so HF_HOME can leave NAS.
mkdir -p "$CEPH/hf_cache/hub"
rsync -a "$NAS/hf_cache/hub/models--openai--clip-vit-base-patch32" "$CEPH/hf_cache/hub/"

# 5. Clone the torch-1.13 podd env into the Ceph miniconda3 as named env `podd`.
# ToS was accepted with explicit user approval 2026-09-17, but the login-node acceptance
# did not carry into batch jobs — re-accept here (idempotent) before the clone.
source "$CEPH/miniconda3/etc/profile.d/conda.sh"
conda tos accept --override-channels -c https://repo.anaconda.com/pkgs/main -c https://repo.anaconda.com/pkgs/r || true
conda create -y -n podd --clone "$NAS/envs/podd" || exit 1
conda activate podd
python -c "import torch, higher, transformers, kornia; print('clone import OK, torch', torch.__version__)" || exit 1

ls -la "$CEPH/datasets/"
echo "MIGRATE_PREP_DONE=0"
