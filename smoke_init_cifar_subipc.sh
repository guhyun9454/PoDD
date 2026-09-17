#!/bin/bash
# CPU-only smoke test for the CIFAR-10 sub-IPC poster geometries, plus a one-time
# CIFAR-10 download to NAS so the racing GPU jobs never download concurrently.
set -x
source /ceph_data/jihye4118/miniconda3/etc/profile.d/conda.sh
conda activate /nas2/data/jihye4118/envs/podd
cd /nas2/data/jihye4118/g/PoDD
python -u smoke_init_cifar_subipc.py
SMOKE_EXIT=$?
mkdir -p /nas2/data/jihye4118/datasets
python -u -c "
import torchvision
torchvision.datasets.CIFAR10(root='/nas2/data/jihye4118/datasets/cifar10', download=True)
torchvision.datasets.CIFAR10(root='/nas2/data/jihye4118/datasets/cifar10', train=False, download=True)
print('CIFAR10_STAGED_OK')
"
echo "SMOKE_EXIT=$SMOKE_EXIT DATA_EXIT=$?"
exit $SMOKE_EXIT
