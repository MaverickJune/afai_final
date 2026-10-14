#!/bin/bash
#SBATCH --job-name=verify_container
#SBATCH --partition=gpu_devel
#SBATCH --gpus=h200:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0:20:00
#SBATCH --output=logs/%x_%j.out
# Verification 1: imports, versions, CUDA, arch list, real GPU compute.
# For B200: sbatch --gpus=b200:1 jobs/verify_container.sbatch
set -euo pipefail
source jobs/slime_common.sh
echo "host: $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv

echo "=== cuDNN diagnosis (why slime_common.sh overrides LD_LIBRARY_PATH)"
slime_exec bash -c '
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "--- libs injected by --nv:"; ls /.singularity.d/libs 2>/dev/null | grep -iE "cudnn|cublas|nccl" || echo "(no cudnn/cublas/nccl injected)"
echo "--- libcudnn.so* in image:"; find / -xdev -name "libcudnn.so*" 2>/dev/null
echo "--- pip:"; pip list 2>/dev/null | grep -iE "nvidia-cudnn|nvidia-cublas|nvidia-nccl"
python -c "
import torch
try: torch.backends.cudnn.version()
except Exception: pass
print(\"loaded cudnn libs:\", sorted({l.split()[-1] for l in open(\"/proc/self/maps\") if \"cudnn\" in l}))
" 2>&1 | tail -3
'

echo "=== GPU checks (slime_exec, with the cuDNN LD_LIBRARY_PATH fix from slime_common.sh)"
slime_exec python jobs/lib/gpu_check.py
