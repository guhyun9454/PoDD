#!/bin/bash
# CPU-only smoke test for the IPC 0.5 / 0.25 poster init (no GPU, no dataset).
set -x
source /ceph_data/jihye4118/miniconda3/etc/profile.d/conda.sh
conda activate /nas2/data/jihye4118/envs/podd
cd /nas2/data/jihye4118/g/PoDD
python -u smoke_init_subipc.py
echo "PYTHON_EXIT=$?"
